use std::path::PathBuf;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio_rusqlite::Connection;
use rusqlite::{params, OptionalExtension};
use super::backend::{CacheBackend, CacheError, CacheCapabilities, CompressibleCache, EncryptableCache, EncryptionAlgorithm};
use super::types::{CacheEntry, CacheStats, CacheType, StorageConfig};
use flate2::{Compress, Decompress, FlushCompress, FlushDecompress};
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, AeadCore,
    Nonce
};
use chacha20poly1305::{
    aead::Aead as ChaChaAead,
    ChaCha20Poly1305,
};
use generic_array::GenericArray;
use tokio::time::{timeout, Duration as TokioDuration};
use futures::future::{BoxFuture, FutureExt};

const SCHEMA_VERSION: i32 = 1;
const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS cache_entries (
    key TEXT PRIMARY KEY,
    value BLOB NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER,
    size_bytes INTEGER NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed INTEGER NOT NULL,
    priority INTEGER NOT NULL DEFAULT 0,
    metadata TEXT,
    checksum TEXT
);

CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at);
CREATE INDEX IF NOT EXISTS idx_last_accessed ON cache_entries(last_accessed);
CREATE INDEX IF NOT EXISTS idx_priority ON cache_entries(priority);
"#;

/// SQLite-based cache implementation
pub struct SQLiteCache {
    /// Database connection
    conn: Arc<Mutex<Connection>>,
    
    /// Cache configuration
    config: StorageConfig,
    
    /// Cache statistics
    stats: Arc<Mutex<CacheStats>>,
    
    /// Auto-vacuum configuration
    auto_vacuum_config: Arc<Mutex<AutoVacuumConfig>>,
}

/// Auto-vacuum configuration
#[derive(Debug, Clone)]
struct AutoVacuumConfig {
    /// Auto-vacuum mode (0=NONE, 1=FULL, 2=INCREMENTAL)
    mode: i32,
    /// Number of pages to free in each incremental vacuum step
    incremental_pages: i32,
    /// Last vacuum timestamp
    last_vacuum: Option<std::time::SystemTime>,
    /// Vacuum interval in seconds
    vacuum_interval: i64,
}

impl SQLiteCache {
    /// Create a new SQLite cache
    pub async fn new(config: StorageConfig) -> Result<Self, CacheError> {
        let path = config.storage_path.clone()
            .ok_or_else(|| CacheError::InitError("Storage path not configured".into()))?;
            
        let conn = Connection::open(PathBuf::from(path))
            .await
            .map_err(|e| CacheError::InitError(e.to_string()))?;
            
        let cache = Self {
            conn: Arc::new(Mutex::new(conn)),
            config: config.clone(),
            stats: Arc::new(Mutex::new(CacheStats::default())),
            auto_vacuum_config: Arc::new(Mutex::new(AutoVacuumConfig {
                mode: 1, // FULL by default
                incremental_pages: 10,
                last_vacuum: None,
                vacuum_interval: 86400, // 24 hours
            })),
        };
        
        // Eagerly ensure the schema exists
        cache.ensure_schema_exists().await?;

        Ok(cache)
    }
    
    /// Initialize database schema
    async fn init_schema(&self) -> Result<(), CacheError> {
        let mut conn = self.conn.lock().await;
        conn.call(|conn| {
            conn.execute_batch(SCHEMA)
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
            Ok::<_, tokio_rusqlite::Error>(Ok(()))
        }).await.map_err(|e| CacheError::StorageError(e.to_string()))?
    }
    
    /// Update cache statistics
    async fn update_stats(&self) -> Result<(), CacheError> {
        self.ensure_schema_exists().await?;
        let conn = Arc::clone(&self.conn);
        let stats = Arc::clone(&self.stats);
        
        let mut conn_guard = conn.lock().await;
        conn_guard.call(move |conn| {
            let (count, size): (i64, i64) = conn.query_row(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM cache_entries",
                [],
                |row| Ok((row.get::<usize, i64>(0)?, row.get::<usize, i64>(1)?))
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            let mut stats = stats.blocking_lock();
            stats.item_count = count as usize;
            stats.total_size_bytes = size as usize;
            Ok::<_, tokio_rusqlite::Error>(Ok(()))
        }).await.map_err(|e| CacheError::StorageError(e.to_string()))?
    }
    
    /// Check storage quota
    async fn check_quota(&self, new_size: usize) -> Result<(), CacheError> {
        let stats = self.stats.lock().await;
        let total_size_mb = (stats.total_size_bytes + new_size) as f64 / 1024.0 / 1024.0;
        
        if total_size_mb > self.config.max_storage_mb as f64 {
            Err(CacheError::QuotaExceeded)
        } else {
            Ok(())
        }
    }
    
    /// Remove expired entries
    async fn remove_expired(&self) -> Result<usize, CacheError> {
        self.ensure_schema_exists().await?;
        let conn = Arc::clone(&self.conn);
        let stats = Arc::clone(&self.stats);
        
        let mut conn_guard = conn.lock().await;
        conn_guard.call(move |conn| {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs() as i64;
                
            let removed = conn.execute(
                "DELETE FROM cache_entries WHERE expires_at IS NOT NULL AND expires_at < ?1",
                params![now],
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            if removed > 0 {
                let mut stats = stats.blocking_lock();
                stats.expirations += removed as u64;
            }
            
            Ok::<_, tokio_rusqlite::Error>(Ok(removed as usize))
        }).await.map_err(|e| CacheError::StorageError(e.to_string()))?
    }

    /// Check if vacuum is needed based on interval
    async fn should_vacuum(&self) -> bool {
        let auto_vacuum = self.auto_vacuum_config.lock().await;
        if let Some(last_vacuum) = auto_vacuum.last_vacuum {
            let now = std::time::SystemTime::now();
            if let Ok(duration) = now.duration_since(last_vacuum) {
                return duration.as_secs() as i64 >= auto_vacuum.vacuum_interval;
            }
        }
        
        true // Vacuum if no previous vacuum recorded
    }

    /// Update last vacuum timestamp
    async fn update_last_vacuum(&self) -> Result<(), CacheError> {
        let mut auto_vacuum = self.auto_vacuum_config.lock().await;
        auto_vacuum.last_vacuum = Some(std::time::SystemTime::now());
        Ok(())
    }

    /// Configure auto-vacuum settings
    pub async fn configure_auto_vacuum(&self, mode: i32, incremental_pages: i32, interval: i64) -> Result<(), CacheError> {
        let mut auto_vacuum = self.auto_vacuum_config.lock().await;
        auto_vacuum.mode = mode.clamp(0, 2);
        auto_vacuum.incremental_pages = incremental_pages.max(1);
        auto_vacuum.vacuum_interval = interval.max(0);
        Ok(())
    }

    /// Perform incremental vacuum
    async fn vacuum_incremental(&self) -> Result<(), CacheError> {
        let auto_vacuum = self.auto_vacuum_config.lock().await;
        let pages = auto_vacuum.incremental_pages;
        drop(auto_vacuum);
        
        let conn = self.conn.lock().await;
        conn.call(move |conn| {
            // Start transaction for incremental vacuum
            let tx = conn.transaction()
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            // Enable auto_vacuum = INCREMENTAL if not already set
            tx.execute_batch("PRAGMA auto_vacuum = INCREMENTAL")
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            // Perform incremental vacuum for specified number of pages
            for _ in 0..pages {
                tx.execute_batch("PRAGMA incremental_vacuum(1)")
                    .map_err(|e| CacheError::StorageError(e.to_string()))?;
            }
            
            tx.commit()
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            Ok::<Result<(), rusqlite::Error>, tokio_rusqlite::Error>(Ok(()))
        })
        .await
        .map_err(|e| CacheError::StorageError(e.to_string()))?;
        Ok(())
    }

    /// Perform full vacuum with progress monitoring
    async fn vacuum_full(&self) -> Result<(), CacheError> {
        let mut conn = self.conn.lock().await;
        conn.call(|conn| {
            // Get initial database size
            let initial_size: i64 = conn.query_row(
                "SELECT page_count * page_size FROM pragma_page_count, pragma_page_size",
                [],
                |row| row.get::<usize, i64>(0),
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            // Perform vacuum
            conn.execute("VACUUM", [])
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
                
            // Get final database size
            let final_size: i64 = conn.query_row(
                "SELECT page_count * page_size FROM pragma_page_count, pragma_page_size",
                [],
                |row| row.get::<usize, i64>(0),
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            // Calculate space saved
            let space_saved = initial_size - final_size;
            tracing::info!(
                "Vacuum completed: Initial size: {} bytes, Final size: {} bytes, Saved: {} bytes",
                initial_size,
                final_size,
                space_saved
            );
            
            Ok::<Result<(), tokio_rusqlite::Error>, tokio_rusqlite::Error>(Ok(()))
        }).await.map_err(|e| CacheError::StorageError(e.to_string()))??;
        Ok(())
    }

    async fn with_connection<F, T>(&self, f: F) -> Result<T, CacheError>
    where
        F: FnOnce(&mut rusqlite::Connection) -> Result<T, CacheError> + Send + 'static,
        T: Send + 'static,
    {
        let conn = Arc::clone(&self.conn);
        let conn = conn.lock().await;
        conn.call(move |conn| {
            match f(conn) {
                Ok(result) => Ok(Ok(result)),
                Err(e) => Ok(Err(e)),
            }
        })
        .await
        .map_err(|e| CacheError::StorageError(e.to_string()))?
        .map_err(|e| CacheError::StorageError(e.to_string()))
    }

    async fn ensure_schema_exists(&self) -> Result<(), CacheError> {
        // Check if the 'cache_entries' table exists
        let exists = self.with_connection(|conn| {
            conn.query_row(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cache_entries'",
                [],
                |_| Ok(1)
            ).optional().map_err(|e| CacheError::StorageError(e.to_string()))
        }).await?;
        if exists.is_none() {
            // If the table does not exist, initialize the schema
            self.init_schema().await
        } else {
            Ok(())
        }
    }
}

impl From<CacheError> for rusqlite::Error {
    fn from(err: CacheError) -> Self {
        rusqlite::Error::InvalidParameterName(err.to_string())
    }
}

#[async_trait]
impl CacheBackend for SQLiteCache {
    fn backend_type(&self) -> CacheType {
        CacheType::SQLite
    }
    
    fn capabilities(&self) -> CacheCapabilities {
        CacheCapabilities {
            persistent: true,
            streaming: false,
            compression: true,
            encryption: true,
            transactions: true,
            pubsub: false,
        }
    }
    
    async fn initialize(&mut self, config: StorageConfig) -> Result<(), CacheError> {
        self.config = config;
        self.init_schema().await
    }
    
    async fn get(&self, key: &str) -> Result<CacheEntry, CacheError> {
        self.ensure_schema_exists().await?;
        let key_str = key.to_string();
        let key_for_fetch = key_str.clone();
        // Fetch raw data from the database with a move closure using its own cloned key
        let raw_data_opt: Option<Vec<u8>> = self.with_connection(move |conn| {
            conn.query_row(
                "SELECT value FROM cache_entries WHERE key = ?1",
                params![key_for_fetch],
                |row| row.get(0)
            ).optional().map_err(|e| CacheError::StorageError(e.to_string()))
        }).await?;

        let raw_data = match raw_data_opt {
            Some(data) => data,
            None => return Err(CacheError::NotFound),
        };

        // If compression is enabled, decompress the data
        let data = if self.config.use_compression {
            self.decompress(&raw_data)?
        } else {
            raw_data
        };

        // Deserialize the cache entry
        let entry: CacheEntry = bincode::deserialize(&data)
            .map_err(|e| CacheError::InvalidData(e.to_string()))?;

        // If the entry is expired, delete it and return an error
        if entry.metadata.is_expired() {
            let key_for_delete = key_str.clone();
            let _ = self.with_connection(move |conn| {
                conn.execute(
                    "DELETE FROM cache_entries WHERE key = ?1",
                    params![key_for_delete],
                ).map_err(|e| CacheError::StorageError(e.to_string()))
            }).await;
            Err(CacheError::Expired)
        } else {
            Ok(entry)
        }
    }
    
    async fn set(&self, entry: CacheEntry) -> Result<(), CacheError> {
        self.ensure_schema_exists().await?;
        let serialized = bincode::serialize(&entry)
            .map_err(|e| CacheError::InvalidData(e.to_string()))?;
        let final_data = if self.config.use_compression {
            self.compress(&serialized)?
        } else {
            serialized
        };

        let key = entry.key.clone();
        let created_at = entry.metadata.created_at.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;
        let expires_at = entry.metadata.expires_at.map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64);
        let size_bytes = entry.metadata.size_bytes as i64;
        let access_count = entry.metadata.access_count;
        let last_accessed = entry.metadata.last_accessed.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64;

        self.with_connection(move |conn| {
            conn.execute(
                "INSERT OR REPLACE INTO cache_entries(key, value, created_at, expires_at, size_bytes, access_count, last_accessed) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![key, final_data, created_at, expires_at, size_bytes, access_count, last_accessed]
            ).map(|_| ()).map_err(|e| CacheError::StorageError(e.to_string()))
        }).await
    }
    
    async fn delete(&self, key: &str) -> Result<(), CacheError> {
        self.ensure_schema_exists().await?;
        let key = key.to_string();
        self.with_connection(move |conn| {
            let removed = conn.execute(
                "DELETE FROM cache_entries WHERE key = ?1",
                params![key],
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            if removed == 0 {
                Err(CacheError::NotFound)
            } else {
                Ok(())
            }
        }).await
    }
    
    async fn exists(&self, key: &str) -> Result<bool, CacheError> {
        self.ensure_schema_exists().await?;
        let key = key.to_string();
        self.with_connection(move |conn| {
            let exists: bool = conn
                .query_row(
                    "SELECT 1 FROM cache_entries WHERE key = ?1",
                    params![key],
                    |_| Ok(true),
                )
                .optional()
                .map_err(|e| CacheError::StorageError(e.to_string()))?
                .unwrap_or(false);
                
            Ok(exists)
        }).await
    }
    
    async fn get_many(&self, keys: &[String]) -> Result<Vec<Option<CacheEntry>>, CacheError> {
        self.ensure_schema_exists().await?;
        let keys = keys.to_vec();
        self.with_connection(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT value FROM cache_entries WHERE key = ?1"
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            let mut results = Vec::with_capacity(keys.len());
            
            for key in keys {
                let entry = stmt
                    .query_row(params![key], |row| {
                        let data: Vec<u8> = row.get(0)?;
                        let entry: CacheEntry = bincode::deserialize(&data)
                            .map_err(|e| CacheError::InvalidData(e.to_string()))?;
                        Ok(entry)
                    })
                    .optional()
                    .map_err(|e| CacheError::StorageError(e.to_string()))?;
                    
                results.push(entry);
            }
            
            Ok(results)
        }).await
    }
    
    async fn set_many(&self, entries: Vec<CacheEntry>) -> Result<(), CacheError> {
        self.ensure_schema_exists().await?;
        self.with_connection(move |conn| {
            let tx = conn.transaction()
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            for entry in entries {
                let data = bincode::serialize(&entry)
                    .map_err(|e| CacheError::InvalidData(e.to_string()))?;

                tx.execute(
                    "INSERT OR REPLACE INTO cache_entries (key, value, created_at, expires_at, size_bytes, access_count, last_accessed) \
                    VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        entry.key,
                        data,
                        entry.metadata.created_at.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64,
                        entry.metadata.expires_at.map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64),
                        entry.metadata.size_bytes as i64,
                        entry.metadata.access_count,
                        entry.metadata.last_accessed.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs() as i64
                    ]
                ).map_err(|e| CacheError::StorageError(e.to_string()))?;
            }

            tx.commit().map_err(|e| CacheError::StorageError(e.to_string()))?;
            Ok(())
        }).await
    }
    
    async fn delete_many(&self, keys: &[String]) -> Result<(), CacheError> {
        self.ensure_schema_exists().await?;
        let keys = keys.to_vec();
        self.with_connection(move |conn| {
            let mut stmt = conn.prepare(
                "DELETE FROM cache_entries WHERE key = ?1"
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            for key in keys {
                stmt.execute(params![key])
                    .map_err(|e| CacheError::StorageError(e.to_string()))?;
            }
            
            Ok(())
        }).await
    }
    
    async fn clear(&self) -> Result<(), CacheError> {
        self.ensure_schema_exists().await?;
        self.with_connection(move |conn| {
            conn.execute("DELETE FROM cache_entries", [])
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
            Ok(())
        }).await
    }
    
    async fn stats(&self) -> Result<CacheStats, CacheError> {
        self.ensure_schema_exists().await?;
        self.with_connection(move |conn| {
            let (count, size): (i64, i64) = conn.query_row(
                "SELECT COUNT(*), COALESCE(SUM(size_bytes), 0) FROM cache_entries",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;
            
            Ok(CacheStats {
                item_count: count as usize,
                total_size_bytes: size as usize,
                hits: 0,
                misses: 0,
                evictions: 0,
                expirations: 0,
            })
        }).await
    }
    
    async fn cleanup(&self) -> Result<(), CacheError> {
        self.ensure_schema_exists().await?;
        let removed = self.remove_expired().await?;
        if removed > 0 {
            self.update_stats().await?;
        }
        Ok(())
    }
    
    async fn optimize(&self) -> Result<(), CacheError> {
        self.ensure_schema_exists().await?;
        if !self.should_vacuum().await {
            return Ok(());
        }
        
        let auto_vacuum = self.auto_vacuum_config.lock().await;
        match auto_vacuum.mode {
            0 => {
                // No auto-vacuum
                tracing::info!("Auto-vacuum disabled");
                Ok(())
            }
            1 => {
                // Full vacuum
                drop(auto_vacuum);
                self.vacuum_full().await?;
                self.update_last_vacuum().await
            }
            2 => {
                // Incremental vacuum
                drop(auto_vacuum);
                self.vacuum_incremental().await?;
                self.update_last_vacuum().await
            }
            _ => Ok(()),
        }
    }
    
    /// Update the backup() function to return Result directly
    async fn backup(&self, path: &str) -> Result<(), CacheError> {
        // If using an in-memory database, skip backup
        if self.config.storage_path.as_deref() == Some(":memory:") {
            return Ok(());
        }

        self.ensure_schema_exists().await?;
        let path = path.to_string();
        let mut conn = self.conn.lock().await;

        // Add a timeout for the entire operation
        let result = timeout(
            std::time::Duration::from_secs(10), // 10 second timeout 
            conn.call(move |source| -> Result<(), tokio_rusqlite::Error> {
                // Open destination database
                let mut dest = rusqlite::Connection::open(&path)
                    .map_err(|e| tokio_rusqlite::Error::Other(Box::new(e)))?;
                dest.busy_timeout(std::time::Duration::from_secs(5))
                    .map_err(|e| tokio_rusqlite::Error::Other(Box::new(e)))?;
                
                // Correct ordering: destination as first argument, source as second
                let mut backup = rusqlite::backup::Backup::new(&mut dest, source)
                    .map_err(|e| tokio_rusqlite::Error::Other(Box::new(e)))?;

                // Use a smaller max iterations
                let mut iterations = 0;
                let max_iterations = 100; // Reduced from 1000
                
                loop {
                    if iterations >= max_iterations {
                        return Err(tokio_rusqlite::Error::Other(Box::new(
                            rusqlite::Error::SqliteFailure(
                                rusqlite::ffi::Error::new(1),
                                Some("Backup operation timed out".to_string())
                            )
                        )));
                    }
                    iterations += 1;
                    
                    // Process more pages per step (100 -> 500)
                    match backup.step(500) {
                        Ok(rusqlite::backup::StepResult::Done) => break,
                        Ok(rusqlite::backup::StepResult::More) => {
                            std::thread::yield_now();
                            continue;
                        },
                        Ok(_) => {
                            return Err(tokio_rusqlite::Error::Other(Box::new(
                                rusqlite::Error::SqliteFailure(
                                    rusqlite::ffi::Error::new(1),
                                    Some("Backup operation failed".to_string())
                                )
                            )));
                        },
                        Err(e) => return Err(tokio_rusqlite::Error::Other(Box::new(e))),
                    }
                }

                Ok(())
            })
        ).await;

        match result {
            Ok(result) => result.map_err(|e| CacheError::StorageError(e.to_string())),
            Err(_) => Err(CacheError::StorageError("Backup operation timed out".to_string())),
        }
    }

    async fn restore(&self, path: &str) -> Result<(), CacheError> {
        // If using an in-memory database, skip restore
        if self.config.storage_path.as_deref() == Some(":memory:") {
            return Ok(());
        }

        let path = path.to_string();
        let mut conn = self.conn.lock().await;

        // Add a timeout for the entire operation
        let result = timeout(
            std::time::Duration::from_secs(10), // 10 second timeout
            conn.call(move |dest| -> Result<(), tokio_rusqlite::Error> {
                // Open source database (backup file) as mutable
                let mut source = rusqlite::Connection::open(&path)
                    .map_err(|e| tokio_rusqlite::Error::Other(Box::new(e)))?;
                source.busy_timeout(std::time::Duration::from_secs(5))
                    .map_err(|e| tokio_rusqlite::Error::Other(Box::new(e)))?;
                
                // Correct ordering: destination as first argument, source as second, passing mutable reference to source
                let mut backup = rusqlite::backup::Backup::new(dest, &mut source)
                    .map_err(|e| tokio_rusqlite::Error::Other(Box::new(e)))?;

                // Use a smaller max iterations
                let mut iterations = 0;
                let max_iterations = 100; // Reduced from 1000
                
                loop {
                    if iterations >= max_iterations {
                        return Err(tokio_rusqlite::Error::Other(Box::new(
                            rusqlite::Error::SqliteFailure(
                                rusqlite::ffi::Error::new(1),
                                Some("Restore operation timed out".to_string())
                            )
                        )));
                    }
                    iterations += 1;
                    
                    // Process more pages per step (100 -> 500)
                    match backup.step(500) {
                        Ok(rusqlite::backup::StepResult::Done) => break,
                        Ok(rusqlite::backup::StepResult::More) => {
                            std::thread::yield_now();
                            continue;
                        },
                        Ok(_) => {
                            return Err(tokio_rusqlite::Error::Other(Box::new(
                                rusqlite::Error::SqliteFailure(
                                    rusqlite::ffi::Error::new(1),
                                    Some("Restore operation failed".to_string())
                                )
                            )));
                        },
                        Err(e) => return Err(tokio_rusqlite::Error::Other(Box::new(e))),
                    }
                }

                Ok(())
            })
        ).await;

        match result {
            Ok(result) => {
                result.map_err(|e| CacheError::StorageError(e.to_string()))?;
                // Reinitialize schema after restore
                self.init_schema().await
            },
            Err(_) => Err(CacheError::StorageError("Restore operation timed out".to_string())),
        }
    }
    
    async fn health_check(&self) -> Result<(), CacheError> {
        self.ensure_schema_exists().await?;
        self.with_connection(move |conn| {
            conn.query_row("SELECT 1", [], |_| Ok(()))
                .map_err(|e| CacheError::StorageError(e.to_string()))
        }).await
    }

    async fn get_entries_batch(&self, offset: usize, limit: usize) -> Result<Vec<CacheEntry>, CacheError> {
        self.ensure_schema_exists().await?;
        self.with_connection(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT value FROM cache_entries ORDER BY last_accessed DESC LIMIT ?1 OFFSET ?2"
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;

            let entries = stmt.query_map(
                params![limit as i64, offset as i64],
                |row| {
                    let data: Vec<u8> = row.get(0)?;
                    let entry: CacheEntry = bincode::deserialize(&data)
                        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                            0,
                            rusqlite::types::Type::Blob,
                            Box::new(e)
                        ))?;
                    Ok(entry)
                },
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;

            let mut results = Vec::new();
            for entry in entries {
                results.push(entry.map_err(|e| CacheError::StorageError(e.to_string()))?);
            }

            Ok(results)
        }).await
    }

    async fn get_high_priority_entries(&self, min_priority: u32, max_items: usize) -> Result<Vec<CacheEntry>, CacheError> {
        self.ensure_schema_exists().await?;
        self.with_connection(move |conn| {
            let mut stmt = conn.prepare(
                "SELECT value FROM cache_entries WHERE priority >= ?1 ORDER BY priority DESC LIMIT ?2"
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;

            let entries = stmt.query_map(
                params![min_priority as i64, max_items as i64],
                |row| {
                    let data: Vec<u8> = row.get(0)?;
                    let entry: CacheEntry = bincode::deserialize(&data)
                        .map_err(|e| rusqlite::Error::FromSqlConversionFailure(
                            0,
                            rusqlite::types::Type::Blob,
                            Box::new(e)
                        ))?;
                    Ok(entry)
                },
            ).map_err(|e| CacheError::StorageError(e.to_string()))?;

            let mut results = Vec::new();
            for entry in entries {
                results.push(entry.map_err(|e| CacheError::StorageError(e.to_string()))?);
            }

            Ok(results)
        }).await
    }
}

#[async_trait]
impl CompressibleCache for SQLiteCache {
    fn compression_level(&self) -> u32 {
        self.config.compression_level.unwrap_or(4)
    }

    fn set_compression_level(&mut self, level: u32) {
        self.config.compression_level = Some(level.clamp(0, 9));
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CacheError> {
        let mut compressor = Compress::new(flate2::Compression::new(self.compression_level()), true);
        let mut compressed = Vec::with_capacity(data.len());
        compressor.compress_vec(
            data,
            &mut compressed,
            FlushCompress::Finish,
        ).map_err(|e| CacheError::CompressionError(e.to_string()))?;
        
        // Prepend size
        let mut size_bytes = (data.len() as u32).to_le_bytes().to_vec();
        size_bytes.extend(compressed);
        
        Ok(size_bytes)
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CacheError> {
        // Extract size from first 4 bytes
        let size = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
        let compressed = &data[4..];

        let mut decompressor = Decompress::new(true);
        let mut decompressed = Vec::with_capacity(size);
        decompressor.decompress_vec(
            compressed,
            &mut decompressed,
            FlushDecompress::Finish,
        ).map_err(|e| CacheError::DecompressionError(e.to_string()))?;

        Ok(decompressed)
    }
}

#[async_trait]
impl EncryptableCache for SQLiteCache {
    async fn encrypt(&self, data: Vec<u8>, algorithm: EncryptionAlgorithm) -> Result<Vec<u8>, CacheError> {
        // Placeholder implementation; real encryption logic to be implemented
        match algorithm {
            EncryptionAlgorithm::Aes256Gcm => {
                todo!("Aes256Gcm encryption for SQLiteCache not implemented");
            },
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                todo!("ChaCha20Poly1305 encryption for SQLiteCache not implemented");
            }
        }
    }

    async fn decrypt(&self, data: Vec<u8>, algorithm: EncryptionAlgorithm) -> Result<Vec<u8>, CacheError> {
        // Placeholder implementation; real decryption logic to be implemented
        match algorithm {
            EncryptionAlgorithm::Aes256Gcm => {
                todo!("Aes256Gcm decryption for SQLiteCache not implemented");
            },
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                todo!("ChaCha20Poly1305 decryption for SQLiteCache not implemented");
            }
        }
    }

    fn is_encryption_ready(&self) -> bool {
        // Default: encryption not ready
        false
    }

    fn supports_encryption(&self) -> bool {
        // Default: SQLiteCache does not support encryption
        false
    }

    fn supported_algorithms(&self) -> Vec<EncryptionAlgorithm> {
        // Return an empty list as encryption is not supported
        vec![]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    use std::time::{SystemTime, Duration};

    // Helper function to create a minimal StorageConfig instance.
    // Updated to include all required fields and proper types.
    fn make_storage_config() -> StorageConfig {
        StorageConfig {
            storage_path: Some(String::from(":memory:")),
            max_storage_mb: 10,
            max_memory_mb: 10,
            use_compression: false,
            compression_level: Some(4),
            use_encryption: false,
            encryption_key: None,
            encryption_algorithm: Some(EncryptionAlgorithm::Aes256Gcm),
        }
    }

    #[tokio::test]
    async fn test_health_check() {
        let config = make_storage_config();
        let cache = SQLiteCache::new(config).await.expect("Failed to create SQLiteCache");
        cache.health_check().await.expect("Health check failed");
    }

    #[tokio::test]
    async fn test_cleanup() {
        let config = make_storage_config();
        let cache = SQLiteCache::new(config).await.expect("Failed to create SQLiteCache");
        // Without any expired entries, cleanup should succeed.
        cache.cleanup().await.expect("Cleanup failed");
    }
} 