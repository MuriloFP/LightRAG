use std::sync::Arc;
use tokio::sync::Mutex;
use redis::{Client, AsyncCommands, RedisError};
use async_trait::async_trait;
use super::backend::{CacheBackend, CacheError, CacheCapabilities};
use super::types::{CacheEntry, CacheStats, CacheType, StorageConfig};
use crate::llm::LLMResponse;
use crate::util::compression::{compress_prepend_size, decompress_size_prepended};
use rand;
use aes::Aes256Gcm;
use chacha20poly1305::ChaCha20Poly1305;

/// Redis-based cache implementation
pub struct RedisCache {
    /// Redis client
    client: Arc<Client>,
    
    /// Cache configuration
    config: StorageConfig,
    
    /// Cache statistics
    stats: Arc<Mutex<CacheStats>>,
}

impl RedisCache {
    /// Create a new Redis cache
    pub async fn new(config: StorageConfig) -> Result<Self, CacheError> {
        let connection_string = config.connection_string.as_ref()
            .ok_or_else(|| CacheError::InitError("Redis connection string not configured".into()))?;
            
        let client = Client::open(connection_string.as_str())
            .map_err(|e| CacheError::InitError(format!("Failed to create Redis client: {}", e)))?;
            
        Ok(Self {
            client: Arc::new(client),
            config,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        })
    }
    
    /// Update cache statistics
    async fn update_stats(&self) -> Result<(), CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let mut stats = self.stats.lock().await;
        
        // Get total number of keys
        let count: i64 = redis::cmd("DBSIZE")
            .query_async(&mut conn)
            .await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        // Get total memory usage
        let info: String = redis::cmd("INFO")
            .arg("memory")
            .query_async(&mut conn)
            .await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let used_memory = info.lines()
            .find(|line| line.starts_with("used_memory:"))
            .and_then(|line| line.split(':').nth(1))
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
            
        stats.item_count = count as usize;
        stats.total_size_bytes = used_memory;
        
        Ok(())
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
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let mut removed = 0;
        let mut cursor = 0u64;
        
        loop {
            let scan: (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg("cache:*")
                .query_async(&mut conn)
                .await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
                
            cursor = scan.0;
            
            for key in scan.1 {
                let entry_data: Option<Vec<u8>> = conn.get(&key).await
                    .map_err(|e| CacheError::StorageError(e.to_string()))?;
                    
                if let Some(data) = entry_data {
                    if let Ok(entry) = bincode::deserialize::<CacheEntry>(&data) {
                        if entry.metadata.is_expired() {
                            conn.del(&key).await
                                .map_err(|e| CacheError::StorageError(e.to_string()))?;
                            removed += 1;
                        }
                    }
                }
            }
            
            if cursor == 0 {
                break;
            }
        }
        
        if removed > 0 {
            let mut stats = self.stats.lock().await;
            stats.expirations += removed as u64;
        }
        
        Ok(removed)
    }
    
    /// Build cache key
    fn build_key(&self, key: &str) -> String {
        format!("cache:{}", key)
    }
}

#[async_trait]
impl CacheBackend for RedisCache {
    fn backend_type(&self) -> CacheType {
        CacheType::Redis
    }
    
    fn capabilities(&self) -> CacheCapabilities {
        CacheCapabilities {
            persistent: true,
            streaming: true,
            compression: true,
            encryption: true,
            transactions: true,
            pubsub: true,
        }
    }
    
    async fn initialize(&mut self, config: StorageConfig) -> Result<(), CacheError> {
        self.config = config;
        self.update_stats().await
    }
    
    async fn get(&self, key: &str) -> Result<CacheEntry, CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let redis_key = self.build_key(key);
        let data: Option<Vec<u8>> = conn.get(&redis_key).await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let data = data.ok_or(CacheError::NotFound)?;
        let mut entry: CacheEntry = bincode::deserialize(&data)
            .map_err(|e| CacheError::InvalidData(e.to_string()))?;
            
        if entry.metadata.is_expired() {
            conn.del(&redis_key).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
            return Err(CacheError::Expired);
        }
        
        // Update access stats
        entry.metadata.access_count += 1;
        entry.metadata.last_accessed = std::time::SystemTime::now();
        
        let serialized = bincode::serialize(&entry)
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        conn.set(&redis_key, serialized).await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let mut stats = self.stats.lock().await;
        stats.hits += 1;
        
        Ok(entry)
    }
    
    async fn set(&self, entry: CacheEntry) -> Result<(), CacheError> {
        // Check quota
        self.check_quota(entry.metadata.size_bytes).await?;
        
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let redis_key = self.build_key(&entry.key);
        let serialized = bincode::serialize(&entry)
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        if let Some(ttl) = entry.metadata.ttl {
            conn.set_ex(&redis_key, serialized, ttl.as_secs() as usize).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
        } else {
            conn.set(&redis_key, serialized).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
        }
        
        self.update_stats().await?;
        Ok(())
    }
    
    async fn delete(&self, key: &str) -> Result<(), CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let redis_key = self.build_key(key);
        let removed: i32 = conn.del(&redis_key).await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        if removed == 0 {
            return Err(CacheError::NotFound);
        }
        
        self.update_stats().await?;
        Ok(())
    }
    
    async fn exists(&self, key: &str) -> Result<bool, CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let redis_key = self.build_key(key);
        let exists: bool = conn.exists(&redis_key).await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        Ok(exists)
    }
    
    async fn get_many(&self, keys: &[String]) -> Result<Vec<Option<CacheEntry>>, CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let redis_keys: Vec<String> = keys.iter()
            .map(|key| self.build_key(key))
            .collect();
            
        let values: Vec<Option<Vec<u8>>> = conn.get(&redis_keys[..]).await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let mut results = Vec::with_capacity(keys.len());
        let mut stats = self.stats.lock().await;
        
        for value in values {
            match value {
                Some(data) => {
                    match bincode::deserialize::<CacheEntry>(&data) {
                        Ok(mut entry) => {
                            if !entry.metadata.is_expired() {
                                entry.metadata.access_count += 1;
                                entry.metadata.last_accessed = std::time::SystemTime::now();
                                stats.hits += 1;
                                results.push(Some(entry));
                                continue;
                            }
                        }
                        Err(_) => {}
                    }
                }
                None => {}
            }
            
            stats.misses += 1;
            results.push(None);
        }
        
        Ok(results)
    }
    
    async fn set_many(&self, entries: Vec<CacheEntry>) -> Result<(), CacheError> {
        // Calculate total size
        let total_new_size: usize = entries.iter()
            .map(|e| e.metadata.size_bytes)
            .sum();
            
        // Check quota
        self.check_quota(total_new_size).await?;
        
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let mut pipe = redis::pipe();
        
        for entry in entries {
            let redis_key = self.build_key(&entry.key);
            let serialized = bincode::serialize(&entry)
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
                
            if let Some(ttl) = entry.metadata.ttl {
                pipe.set_ex(&redis_key, serialized, ttl.as_secs() as usize);
            } else {
                pipe.set(&redis_key, serialized);
            }
        }
        
        pipe.query_async(&mut conn).await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        self.update_stats().await?;
        Ok(())
    }
    
    async fn delete_many(&self, keys: &[String]) -> Result<(), CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let redis_keys: Vec<String> = keys.iter()
            .map(|key| self.build_key(key))
            .collect();
            
        conn.del(&redis_keys[..]).await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        self.update_stats().await?;
        Ok(())
    }
    
    async fn clear(&self) -> Result<(), CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        let mut cursor = 0u64;
        
        loop {
            let scan: (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg("cache:*")
                .query_async(&mut conn)
                .await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;
                
            cursor = scan.0;
            
            if !scan.1.is_empty() {
                conn.del(&scan.1[..]).await
                    .map_err(|e| CacheError::StorageError(e.to_string()))?;
            }
            
            if cursor == 0 {
                break;
            }
        }
        
        let mut stats = self.stats.lock().await;
        *stats = CacheStats::default();
        
        Ok(())
    }
    
    async fn stats(&self) -> Result<CacheStats, CacheError> {
        let stats = self.stats.lock().await;
        Ok(stats.clone())
    }
    
    async fn cleanup(&self) -> Result<(), CacheError> {
        let removed = self.remove_expired().await?;
        if removed > 0 {
            self.update_stats().await?;
        }
        Ok(())
    }
    
    async fn optimize(&self) -> Result<(), CacheError> {
        // Redis handles memory optimization internally
        Ok(())
    }
    
    async fn backup(&self, path: &str) -> Result<(), CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        redis::cmd("SAVE")
            .query_async(&mut conn)
            .await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        Ok(())
    }
    
    async fn restore(&self, path: &str) -> Result<(), CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        redis::cmd("RESTORE")
            .arg(path)
            .query_async(&mut conn)
            .await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        self.update_stats().await?;
        Ok(())
    }
    
    async fn health_check(&self) -> Result<(), CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        redis::cmd("PING")
            .query_async(&mut conn)
            .await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;
            
        Ok(())
    }
}

#[async_trait]
impl CompressibleCache for RedisCache {
    fn compression_level(&self) -> u32 {
        self.config.compression_level.unwrap_or(4)
    }

    fn set_compression_level(&mut self, level: u32) {
        self.config.compression_level = Some(level.clamp(0, 9));
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CacheError> {
        Ok(compress_prepend_size(data))
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CacheError> {
        decompress_size_prepended(data)
            .map_err(|e| CacheError::CompressionError(format!("Failed to decompress data: {}", e)))
    }
}

#[async_trait]
impl EncryptableCache for RedisCache {
    async fn encrypt(&self, data: &[u8], algorithm: EncryptionAlgorithm) -> Result<Vec<u8>, CacheError> {
        if !self.is_encryption_ready() {
            return Err(CacheError::EncryptionNotInitialized);
        }

        let mut nonce = vec![0u8; 12];
        rand::thread_rng().fill_bytes(&mut nonce);

        let key = self.config.encryption_key.as_ref()
            .ok_or_else(|| CacheError::EncryptionError("No encryption key provided".to_string()))?;

        let encrypted = match algorithm {
            EncryptionAlgorithm::Aes256Gcm => {
                let cipher = Aes256Gcm::new_from_slice(key)
                    .map_err(|e| CacheError::EncryptionError(e.to_string()))?;
                
                let nonce = AesNonce::from_slice(&nonce);
                cipher.encrypt(nonce, data)
                    .map_err(|e| CacheError::EncryptionError(e.to_string()))?
            }
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                let cipher = ChaCha20Poly1305::new_from_slice(key)
                    .map_err(|e| CacheError::EncryptionError(e.to_string()))?;
                
                let nonce = ChaChaPolNonce::from_slice(&nonce);
                cipher.encrypt(nonce, data)
                    .map_err(|e| CacheError::EncryptionError(e.to_string()))?
            }
        };

        let mut result = nonce;
        result.extend(encrypted);
        Ok(result)
    }

    async fn decrypt(&self, data: &[u8], algorithm: EncryptionAlgorithm) -> Result<Vec<u8>, CacheError> {
        if !self.is_encryption_ready() {
            return Err(CacheError::EncryptionNotInitialized);
        }

        if data.len() < 12 {
            return Err(CacheError::InvalidData("Data too short for nonce".into()));
        }

        let (nonce, encrypted) = data.split_at(12);
        let key = self.config.encryption_key.as_ref()
            .ok_or_else(|| CacheError::DecryptionError("No encryption key provided".to_string()))?;

        match algorithm {
            EncryptionAlgorithm::Aes256Gcm => {
                let cipher = Aes256Gcm::new_from_slice(key)
                    .map_err(|e| CacheError::DecryptionError(e.to_string()))?;
                
                let nonce = AesNonce::from_slice(nonce);
                cipher.decrypt(nonce, encrypted)
                    .map_err(|e| CacheError::DecryptionError(e.to_string()))
            }
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                let cipher = ChaCha20Poly1305::new_from_slice(key)
                    .map_err(|e| CacheError::DecryptionError(e.to_string()))?;
                
                let nonce = ChaChaPolNonce::from_slice(nonce);
                cipher.decrypt(nonce, encrypted)
                    .map_err(|e| CacheError::DecryptionError(e.to_string()))
            }
        }
    }

    fn is_encryption_ready(&self) -> bool {
        self.config.use_encryption && self.config.encryption_key.is_some()
    }

    fn supports_encryption(&self) -> bool {
        true
    }

    fn supported_algorithms(&self) -> Vec<EncryptionAlgorithm> {
        vec![
            EncryptionAlgorithm::Aes256Gcm,
            EncryptionAlgorithm::ChaCha20Poly1305
        ]
    }
} 