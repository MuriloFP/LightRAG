use std::sync::Arc;
use parking_lot::Mutex;
use async_trait::async_trait;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{IdbDatabase, IdbObjectStore, IdbTransaction, IdbTransactionMode};
use super::backend::{CacheBackend, CacheError, CacheCapabilities};
use super::types::{CacheEntry, CacheStats, CacheType, StorageConfig, CacheValue};
use crate::util::compression::{compress_prepend_size, decompress_size_prepended};
use rand;
use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce as AesNonce
};
use chacha20poly1305::{
    aead::{Aead as ChaChaAead, KeyInit as ChaChaKeyInit},
    ChaCha20Poly1305, Nonce as ChaChaPolNonce
};
use rand::RngCore;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use super::offline::{OfflineManager, OfflineOperation};

const DB_NAME: &str = "lightrag_cache";
const STORE_NAME: &str = "cache_entries";
const SCHEMA_VERSION: u32 = 1;

/// IndexedDB cache implementation for browser environments
pub struct IndexedDBCache {
    /// Database connection
    db: Arc<IdbDatabase>,
    
    /// Cache configuration
    config: StorageConfig,
    
    /// Cache statistics
    stats: Arc<Mutex<CacheStats>>,
    
    /// Offline manager
    offline_manager: Arc<RwLock<OfflineManager>>,
}

impl IndexedDBCache {
    /// Create a new IndexedDB cache
    pub async fn new(config: StorageConfig) -> Result<Self, CacheError> {
        let window = web_sys::window()
            .ok_or_else(|| CacheError::InitError("No window object available".into()))?;
            
        let idb = window.indexed_db()
            .map_err(|e| CacheError::InitError(format!("Failed to get IndexedDB: {:?}", e)))?
            .ok_or_else(|| CacheError::InitError("IndexedDB not available".into()))?;
            
        let db_req = idb.open_with_u32(DB_NAME, SCHEMA_VERSION)
            .map_err(|e| CacheError::InitError(format!("Failed to open database: {:?}", e)))?;
            
        // Handle database upgrade
        let onupgradeneeded = Closure::wrap(Box::new(move |event: web_sys::IdbVersionChangeEvent| {
            let db = event.target()
                .unwrap()
                .dyn_into::<web_sys::IdbRequest>()
                .unwrap()
                .result()
                .unwrap()
                .dyn_into::<web_sys::IdbDatabase>()
                .unwrap();
                
            if !db.object_store_names().contains(&JsValue::from_str(STORE_NAME)) {
                let store = db.create_object_store(STORE_NAME).unwrap();
                // Add indexes for offline support
                store.create_index("version", "version", web_sys::IdbIndexParameters::new().unique(false)).unwrap();
                store.create_index("last_modified", "last_modified", web_sys::IdbIndexParameters::new().unique(false)).unwrap();
            }
        }) as Box<dyn FnMut(_)>);
        
        db_req.set_onupgradeneeded(Some(onupgradeneeded.as_ref().unchecked_ref()));
        onupgradeneeded.forget();
        
        let db = JsFuture::from(db_req)
            .await
            .map_err(|e| CacheError::InitError(format!("Failed to initialize database: {:?}", e)))?
            .dyn_into::<IdbDatabase>()
            .map_err(|_| CacheError::InitError("Invalid database object".into()))?;

        // Initialize offline manager
        let offline_manager = OfflineManager::new().await?;
        
        let cache = Self {
            db: Arc::new(db),
            config,
            stats: Arc::new(Mutex::new(CacheStats::default())),
            offline_manager: Arc::new(RwLock::new(offline_manager)),
        };

        // Set up online/offline event listeners
        cache.setup_network_listeners()?;
        
        Ok(cache)
    }
    
    /// Set up network status event listeners
    fn setup_network_listeners(&self) -> Result<(), CacheError> {
        let window = web_sys::window()
            .ok_or_else(|| CacheError::InitError("No window object available".into()))?;

        let offline_manager = Arc::clone(&self.offline_manager);
        
        // Online event
        let online_callback = Closure::wrap(Box::new(move || {
            let mut manager = offline_manager.write();
            manager.set_online_status(true);
        }) as Box<dyn FnMut()>);
        
        window.add_event_listener_with_callback("online", online_callback.as_ref().unchecked_ref())
            .map_err(|e| CacheError::InitError(format!("Failed to add online listener: {:?}", e)))?;
        
        online_callback.forget();

        let offline_manager = Arc::clone(&self.offline_manager);
        
        // Offline event
        let offline_callback = Closure::wrap(Box::new(move || {
            let mut manager = offline_manager.write();
            manager.set_online_status(false);
        }) as Box<dyn FnMut()>);
        
        window.add_event_listener_with_callback("offline", offline_callback.as_ref().unchecked_ref())
            .map_err(|e| CacheError::InitError(format!("Failed to add offline listener: {:?}", e)))?;
        
        offline_callback.forget();
        
        Ok(())
    }

    /// Queue operation for offline sync
    async fn queue_operation(&self, operation: OfflineOperation) -> Result<(), CacheError> {
        let manager = self.offline_manager.read();
        manager.queue_operation(operation);
        Ok(())
    }

    /// Process offline queue
    pub async fn process_offline_queue(&self) -> Result<(), CacheError> {
        let mut manager = self.offline_manager.write();
        manager.process_queue().await
    }
    
    /// Get a transaction for the cache store
    fn transaction(&self, mode: IdbTransactionMode) -> Result<IdbTransaction, CacheError> {
        self.db.transaction_with_str(STORE_NAME, mode)
            .map_err(|e| CacheError::StorageError(format!("Failed to start transaction: {:?}", e)))
    }
    
    /// Get the object store for the current transaction
    fn store(&self, tx: &IdbTransaction) -> Result<IdbObjectStore, CacheError> {
        tx.object_store(STORE_NAME)
            .map_err(|e| CacheError::StorageError(format!("Failed to get object store: {:?}", e)))
    }
    
    /// Update cache statistics
    async fn update_stats(&self) -> Result<(), CacheError> {
        let tx = self.transaction(IdbTransactionMode::Readonly)?;
        let store = self.store(&tx)?;
        
        let count_req = store.count()
            .map_err(|e| CacheError::StorageError(format!("Failed to count entries: {:?}", e)))?;
            
        let count = JsFuture::from(count_req)
            .await
            .map_err(|e| CacheError::StorageError(format!("Failed to get count: {:?}", e)))?
            .as_f64()
            .unwrap_or(0.0) as usize;
            
        let mut stats = self.stats.lock();
        stats.item_count = count;
        
        Ok(())
    }
    
    /// Check storage quota
    async fn check_quota(&self, new_size: usize) -> Result<(), CacheError> {
        let stats = self.stats.lock();
        let total_size_mb = (stats.total_size_bytes + new_size) as f64 / 1024.0 / 1024.0;
        
        if total_size_mb > self.config.max_storage_mb as f64 {
            Err(CacheError::QuotaExceeded)
        } else {
            Ok(())
        }
    }
    
    /// Remove expired entries
    async fn remove_expired(&self) -> Result<usize, CacheError> {
        let tx = self.transaction(IdbTransactionMode::Readwrite)?;
        let store = self.store(&tx)?;
        let now = js_sys::Date::now();
        
        let req = store.get_all()
            .map_err(|e| CacheError::StorageError(format!("Failed to get entries: {:?}", e)))?;
            
        let entries = JsFuture::from(req)
            .await
            .map_err(|e| CacheError::StorageError(format!("Failed to get entries: {:?}", e)))?;
            
        let entries: Vec<CacheEntry> = serde_wasm_bindgen::from_value(entries)
            .map_err(|e| CacheError::StorageError(format!("Failed to deserialize entries: {:?}", e)))?;
            
        let mut removed = 0;
        for entry in entries {
            if entry.metadata.is_expired() {
                store.delete(&JsValue::from_str(&entry.key))
                    .map_err(|e| CacheError::StorageError(format!("Failed to delete entry: {:?}", e)))?;
                removed += 1;
            }
        }
        
        if removed > 0 {
            let mut stats = self.stats.lock();
            stats.expirations += removed as u64;
        }
        
        Ok(removed)
    }
}

#[async_trait(?Send)]
impl CacheBackend for IndexedDBCache {
    fn backend_type(&self) -> CacheType {
        CacheType::IndexedDB
    }
    
    fn capabilities(&self) -> CacheCapabilities {
        CacheCapabilities {
            persistent: true,
            streaming: true,
            compression: true,
            encryption: true,
            transactions: true,
            pubsub: false,
        }
    }
    
    async fn initialize(&mut self, config: StorageConfig) -> Result<(), CacheError> {
        self.config = config;
        self.update_stats().await
    }
    
    async fn get(&self, key: &str) -> Result<CacheEntry, CacheError> {
        let tx = self.transaction(IdbTransactionMode::Readonly)?;
        let store = self.store(&tx)?;
        
        let req = store.get(&JsValue::from_str(key))
            .map_err(|e| CacheError::StorageError(format!("Failed to get entry: {:?}", e)))?;
            
        let value = JsFuture::from(req)
            .await
            .map_err(|e| CacheError::StorageError(format!("Failed to get entry: {:?}", e)))?;
            
        if value.is_undefined() {
            return Err(CacheError::NotFound);
        }
        
        let mut entry: CacheEntry = serde_wasm_bindgen::from_value(value)
            .map_err(|e| CacheError::StorageError(format!("Failed to deserialize entry: {:?}", e)))?;
            
        if entry.metadata.is_expired() {
            return Err(CacheError::Expired);
        }
        
        entry.metadata.access_count += 1;
        entry.metadata.last_accessed = std::time::SystemTime::now();
        
        let tx = self.transaction(IdbTransactionMode::Readwrite)?;
        let store = self.store(&tx)?;
        
        let value = serde_wasm_bindgen::to_value(&entry)
            .map_err(|e| CacheError::StorageError(format!("Failed to serialize entry: {:?}", e)))?;
            
        store.put_with_key(&value, &JsValue::from_str(key))
            .map_err(|e| CacheError::StorageError(format!("Failed to update entry: {:?}", e)))?;
            
        let mut stats = self.stats.lock();
        stats.hits += 1;
        
        Ok(entry)
    }
    
    async fn set(&self, entry: CacheEntry) -> Result<(), CacheError> {
        // Check quota
        self.check_quota(entry.metadata.size_bytes).await?;
        
        let tx = self.transaction(IdbTransactionMode::Readwrite)?;
        let store = self.store(&tx)?;
        
        // Add version and last_modified for offline sync
        let mut entry = entry;
        entry.metadata.extra.insert("version".to_string(), self.get_next_version().to_string());
        entry.metadata.extra.insert("last_modified".to_string(), js_sys::Date::now().to_string());
        
        let value = serde_wasm_bindgen::to_value(&entry)
            .map_err(|e| CacheError::StorageError(format!("Failed to serialize entry: {:?}", e)))?;
            
        store.put_with_key(&value, &JsValue::from_str(&entry.key))
            .map_err(|e| CacheError::StorageError(format!("Failed to store entry: {:?}", e)))?;
            
        // Queue operation for offline sync
        self.queue_operation(OfflineOperation::Set(entry)).await?;
        
        self.update_stats().await?;
        
        Ok(())
    }
    
    async fn delete(&self, key: &str) -> Result<(), CacheError> {
        let tx = self.transaction(IdbTransactionMode::Readwrite)?;
        let store = self.store(&tx)?;
        
        store.delete(&JsValue::from_str(key))
            .map_err(|e| CacheError::StorageError(format!("Failed to delete entry: {:?}", e)))?;
            
        // Queue operation for offline sync
        self.queue_operation(OfflineOperation::Delete(key.to_string())).await?;
        
        self.update_stats().await?;
        
        Ok(())
    }
    
    async fn exists(&self, key: &str) -> Result<bool, CacheError> {
        let tx = self.transaction(IdbTransactionMode::Readonly)?;
        let store = self.store(&tx)?;
        
        let req = store.count_with_key(&JsValue::from_str(key))
            .map_err(|e| CacheError::StorageError(format!("Failed to check entry: {:?}", e)))?;
            
        let count = JsFuture::from(req)
            .await
            .map_err(|e| CacheError::StorageError(format!("Failed to check entry: {:?}", e)))?
            .as_f64()
            .unwrap_or(0.0);
            
        Ok(count > 0.0)
    }
    
    async fn get_many(&self, keys: &[String]) -> Result<Vec<Option<CacheEntry>>, CacheError> {
        let tx = self.transaction(IdbTransactionMode::Readonly)?;
        let store = self.store(&tx)?;
        let mut results = Vec::with_capacity(keys.len());
        let mut stats = self.stats.lock();
        
        for key in keys {
            let req = store.get(&JsValue::from_str(key))
                .map_err(|e| CacheError::StorageError(format!("Failed to get entry: {:?}", e)))?;
                
            let value = JsFuture::from(req)
                .await
                .map_err(|e| CacheError::StorageError(format!("Failed to get entry: {:?}", e)))?;
                
            if value.is_undefined() {
                stats.misses += 1;
                results.push(None);
                continue;
            }
            
            let mut entry: CacheEntry = serde_wasm_bindgen::from_value(value)
                .map_err(|e| CacheError::StorageError(format!("Failed to deserialize entry: {:?}", e)))?;
                
            if entry.metadata.is_expired() {
                stats.misses += 1;
                results.push(None);
                continue;
            }
            
            entry.metadata.access_count += 1;
            entry.metadata.last_accessed = std::time::SystemTime::now();
            stats.hits += 1;
            results.push(Some(entry));
        }
        
        Ok(results)
    }
    
    async fn set_many(&self, entries: Vec<CacheEntry>) -> Result<(), CacheError> {
        let total_size: usize = entries.iter()
            .map(|e| e.metadata.size_bytes)
            .sum();
            
        self.check_quota(total_size).await?;
        
        let tx = self.transaction(IdbTransactionMode::Readwrite)?;
        let store = self.store(&tx)?;
        
        for entry in entries {
            let value = serde_wasm_bindgen::to_value(&entry)
                .map_err(|e| CacheError::StorageError(format!("Failed to serialize entry: {:?}", e)))?;
                
            store.put_with_key(&value, &JsValue::from_str(&entry.key))
                .map_err(|e| CacheError::StorageError(format!("Failed to store entry: {:?}", e)))?;
        }
        
        self.update_stats().await?;
        
        Ok(())
    }
    
    async fn delete_many(&self, keys: &[String]) -> Result<(), CacheError> {
        let tx = self.transaction(IdbTransactionMode::Readwrite)?;
        let store = self.store(&tx)?;
        
        for key in keys {
            store.delete(&JsValue::from_str(key))
                .map_err(|e| CacheError::StorageError(format!("Failed to delete entry: {:?}", e)))?;
        }
        
        self.update_stats().await?;
        
        Ok(())
    }
    
    async fn clear(&self) -> Result<(), CacheError> {
        let tx = self.transaction(IdbTransactionMode::Readwrite)?;
        let store = self.store(&tx)?;
        
        store.clear()
            .map_err(|e| CacheError::StorageError(format!("Failed to clear cache: {:?}", e)))?;
            
        // Queue operation for offline sync
        self.queue_operation(OfflineOperation::Clear).await?;
        
        let mut stats = self.stats.lock();
        *stats = CacheStats::default();
        
        Ok(())
    }
    
    async fn stats(&self) -> Result<CacheStats, CacheError> {
        Ok(self.stats.lock().clone())
    }
    
    async fn cleanup(&self) -> Result<(), CacheError> {
        let removed = self.remove_expired().await?;
        if removed > 0 {
            self.update_stats().await?;
        }
        Ok(())
    }
    
    async fn optimize(&self) -> Result<(), CacheError> {
        // IndexedDB handles optimization internally
        Ok(())
    }
    
    async fn backup(&self, _path: &str) -> Result<(), CacheError> {
        // Browser storage doesn't support direct backup
        Err(CacheError::Unavailable("Backup not supported in browser environment".into()))
    }
    
    async fn restore(&self, _path: &str) -> Result<(), CacheError> {
        // Browser storage doesn't support direct restore
        Err(CacheError::Unavailable("Restore not supported in browser environment".into()))
    }
    
    async fn health_check(&self) -> Result<(), CacheError> {
        // Simple health check - verify we can read/write
        let key = "health_check";
        let entry = CacheEntry::new(
            key.to_string(),
            CacheValue::Response(crate::llm::LLMResponse {
                text: "test".to_string(),
                tokens_used: 1,
                model: "test".to_string(),
                cached: false,
                context: None,
                metadata: std::collections::HashMap::new(),
            }),
            Some(std::time::Duration::from_secs(60)),
            None,
        );
        
        self.set(entry).await?;
        self.delete(key).await?;
        
        Ok(())
    }

    /// Get next version number
    fn get_next_version(&self) -> u64 {
        static VERSION: AtomicU64 = AtomicU64::new(0);
        VERSION.fetch_add(1, Ordering::SeqCst)
    }

    /// Resolve conflicts between local and remote versions
    async fn resolve_conflicts(&self, key: &str, local: &CacheEntry, remote: &CacheEntry) -> Result<CacheEntry, CacheError> {
        let local_version = local.metadata.extra.get("version")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
            
        let remote_version = remote.metadata.extra.get("version")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0);
            
        let local_modified = local.metadata.extra.get("last_modified")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0);
            
        let remote_modified = remote.metadata.extra.get("last_modified")
            .and_then(|v| v.parse::<f64>().ok())
            .unwrap_or(0.0);
            
        // Simple last-write-wins strategy
        if remote_version > local_version || (remote_version == local_version && remote_modified > local_modified) {
            Ok(remote.clone())
        } else {
            Ok(local.clone())
        }
    }
}

#[async_trait]
impl CompressibleCache for IndexedDBCache {
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
impl EncryptableCache for IndexedDBCache {
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