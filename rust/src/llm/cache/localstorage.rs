use std::sync::Arc;
use parking_lot::Mutex;
use async_trait::async_trait;
use wasm_bindgen::prelude::*;
use web_sys::Storage;
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

const PREFIX: &str = "lightrag_cache:";
const MAX_KEY_LENGTH: usize = 64;
const MAX_VALUE_SIZE: usize = 5 * 1024 * 1024; // 5MB limit per item

/// LocalStorage cache implementation for browser environments
pub struct LocalStorageCache {
    /// LocalStorage instance
    storage: Storage,
    
    /// Cache configuration
    config: StorageConfig,
    
    /// Cache statistics
    stats: Arc<Mutex<CacheStats>>,
}

impl LocalStorageCache {
    /// Create a new LocalStorage cache
    pub fn new(config: StorageConfig) -> Result<Self, CacheError> {
        let window = web_sys::window()
            .ok_or_else(|| CacheError::InitError("No window object available".into()))?;
            
        let storage = window.local_storage()
            .map_err(|e| CacheError::InitError(format!("Failed to get localStorage: {:?}", e)))?
            .ok_or_else(|| CacheError::InitError("localStorage not available".into()))?;
            
        Ok(Self {
            storage,
            config,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        })
    }
    
    /// Build storage key with prefix
    fn build_key(&self, key: &str) -> String {
        format!("{}{}", PREFIX, key)
    }
    
    /// Update cache statistics
    fn update_stats(&self) -> Result<(), CacheError> {
        let mut stats = self.stats.lock();
        let mut total_size = 0;
        let mut count = 0;
        
        for i in 0..self.storage.length().unwrap_or(0) {
            if let Some(key) = self.storage.key(i).unwrap_or(None) {
                if key.starts_with(PREFIX) {
                    if let Some(value) = self.storage.get_item(&key).unwrap_or(None) {
                        total_size += value.len();
                        count += 1;
                    }
                }
            }
        }
        
        stats.item_count = count as usize;
        stats.total_size_bytes = total_size;
        
        Ok(())
    }
    
    /// Check storage quota
    fn check_quota(&self, new_size: usize) -> Result<(), CacheError> {
        let stats = self.stats.lock();
        let total_size_mb = (stats.total_size_bytes + new_size) as f64 / 1024.0 / 1024.0;
        
        if total_size_mb > self.config.max_storage_mb as f64 {
            Err(CacheError::QuotaExceeded)
        } else {
            Ok(())
        }
    }
    
    /// Remove expired entries
    fn remove_expired(&self) -> Result<usize, CacheError> {
        let mut removed = 0;
        let now = js_sys::Date::now();
        
        for i in 0..self.storage.length().unwrap_or(0) {
            if let Some(key) = self.storage.key(i).unwrap_or(None) {
                if key.starts_with(PREFIX) {
                    if let Some(value) = self.storage.get_item(&key).unwrap_or(None) {
                        if let Ok(entry) = serde_json::from_str::<CacheEntry>(&value) {
                            if entry.metadata.is_expired() {
                                self.storage.remove_item(&key)
                                    .map_err(|e| CacheError::StorageError(format!("Failed to remove item: {:?}", e)))?;
                                removed += 1;
                            }
                        }
                    }
                }
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
impl CacheBackend for LocalStorageCache {
    fn backend_type(&self) -> CacheType {
        CacheType::LocalStorage
    }
    
    fn capabilities(&self) -> CacheCapabilities {
        CacheCapabilities {
            persistent: true,
            streaming: false,
            compression: true,
            encryption: true,
            transactions: false,
            pubsub: false,
        }
    }
    
    async fn initialize(&mut self, config: StorageConfig) -> Result<(), CacheError> {
        self.config = config;
        self.update_stats()
    }
    
    async fn get(&self, key: &str) -> Result<CacheEntry, CacheError> {
        let storage_key = self.build_key(key);
        
        let value = self.storage.get_item(&storage_key)
            .map_err(|e| CacheError::StorageError(format!("Failed to get item: {:?}", e)))?
            .ok_or(CacheError::NotFound)?;
            
        let mut entry: CacheEntry = serde_json::from_str(&value)
            .map_err(|e| CacheError::StorageError(format!("Failed to deserialize entry: {:?}", e)))?;
            
        if entry.metadata.is_expired() {
            self.storage.remove_item(&storage_key)
                .map_err(|e| CacheError::StorageError(format!("Failed to remove expired item: {:?}", e)))?;
            return Err(CacheError::Expired);
        }
        
        entry.metadata.access_count += 1;
        entry.metadata.last_accessed = std::time::SystemTime::now();
        
        let value = serde_json::to_string(&entry)
            .map_err(|e| CacheError::StorageError(format!("Failed to serialize entry: {:?}", e)))?;
            
        self.storage.set_item(&storage_key, &value)
            .map_err(|e| CacheError::StorageError(format!("Failed to update entry: {:?}", e)))?;
            
        let mut stats = self.stats.lock();
        stats.hits += 1;
        
        Ok(entry)
    }
    
    async fn set(&self, entry: CacheEntry) -> Result<(), CacheError> {
        let storage_key = self.build_key(&entry.key);
        
        if storage_key.len() > MAX_KEY_LENGTH {
            return Err(CacheError::StorageError("Key too long".into()));
        }
        
        let value = serde_json::to_string(&entry)
            .map_err(|e| CacheError::StorageError(format!("Failed to serialize entry: {:?}", e)))?;
            
        if value.len() > MAX_VALUE_SIZE {
            return Err(CacheError::StorageError("Value too large".into()));
        }
        
        self.check_quota(value.len())?;
        
        self.storage.set_item(&storage_key, &value)
            .map_err(|e| CacheError::StorageError(format!("Failed to store entry: {:?}", e)))?;
            
        self.update_stats()?;
        
        Ok(())
    }
    
    async fn delete(&self, key: &str) -> Result<(), CacheError> {
        let storage_key = self.build_key(key);
        
        if self.storage.get_item(&storage_key).unwrap_or(None).is_none() {
            return Err(CacheError::NotFound);
        }
        
        self.storage.remove_item(&storage_key)
            .map_err(|e| CacheError::StorageError(format!("Failed to delete entry: {:?}", e)))?;
            
        self.update_stats()?;
        
        Ok(())
    }
    
    async fn exists(&self, key: &str) -> Result<bool, CacheError> {
        let storage_key = self.build_key(key);
        Ok(self.storage.get_item(&storage_key).unwrap_or(None).is_some())
    }
    
    async fn get_many(&self, keys: &[String]) -> Result<Vec<Option<CacheEntry>>, CacheError> {
        let mut results = Vec::with_capacity(keys.len());
        let mut stats = self.stats.lock();
        
        for key in keys {
            let storage_key = self.build_key(key);
            
            if let Some(value) = self.storage.get_item(&storage_key).unwrap_or(None) {
                match serde_json::from_str::<CacheEntry>(&value) {
                    Ok(mut entry) => {
                        if entry.metadata.is_expired() {
                            self.storage.remove_item(&storage_key)
                                .map_err(|e| CacheError::StorageError(format!("Failed to remove expired item: {:?}", e)))?;
                            stats.misses += 1;
                            results.push(None);
                        } else {
                            entry.metadata.access_count += 1;
                            entry.metadata.last_accessed = std::time::SystemTime::now();
                            
                            let updated_value = serde_json::to_string(&entry)
                                .map_err(|e| CacheError::StorageError(format!("Failed to serialize entry: {:?}", e)))?;
                                
                            self.storage.set_item(&storage_key, &updated_value)
                                .map_err(|e| CacheError::StorageError(format!("Failed to update entry: {:?}", e)))?;
                                
                            stats.hits += 1;
                            results.push(Some(entry));
                        }
                    }
                    Err(_) => {
                        stats.misses += 1;
                        results.push(None);
                    }
                }
            } else {
                stats.misses += 1;
                results.push(None);
            }
        }
        
        Ok(results)
    }
    
    async fn set_many(&self, entries: Vec<CacheEntry>) -> Result<(), CacheError> {
        let total_size: usize = entries.iter()
            .map(|e| serde_json::to_string(e).map(|s| s.len()).unwrap_or(0))
            .sum();
            
        self.check_quota(total_size)?;
        
        for entry in entries {
            let storage_key = self.build_key(&entry.key);
            
            if storage_key.len() > MAX_KEY_LENGTH {
                return Err(CacheError::StorageError("Key too long".into()));
            }
            
            let value = serde_json::to_string(&entry)
                .map_err(|e| CacheError::StorageError(format!("Failed to serialize entry: {:?}", e)))?;
                
            if value.len() > MAX_VALUE_SIZE {
                return Err(CacheError::StorageError("Value too large".into()));
            }
            
            self.storage.set_item(&storage_key, &value)
                .map_err(|e| CacheError::StorageError(format!("Failed to store entry: {:?}", e)))?;
        }
        
        self.update_stats()?;
        
        Ok(())
    }
    
    async fn delete_many(&self, keys: &[String]) -> Result<(), CacheError> {
        for key in keys {
            let storage_key = self.build_key(key);
            self.storage.remove_item(&storage_key)
                .map_err(|e| CacheError::StorageError(format!("Failed to delete entry: {:?}", e)))?;
        }
        
        self.update_stats()?;
        
        Ok(())
    }
    
    async fn clear(&self) -> Result<(), CacheError> {
        let mut keys_to_remove = Vec::new();
        
        for i in 0..self.storage.length().unwrap_or(0) {
            if let Some(key) = self.storage.key(i).unwrap_or(None) {
                if key.starts_with(PREFIX) {
                    keys_to_remove.push(key);
                }
            }
        }
        
        for key in keys_to_remove {
            self.storage.remove_item(&key)
                .map_err(|e| CacheError::StorageError(format!("Failed to remove item: {:?}", e)))?;
        }
        
        let mut stats = self.stats.lock();
        *stats = CacheStats::default();
        
        Ok(())
    }
    
    async fn stats(&self) -> Result<CacheStats, CacheError> {
        Ok(self.stats.lock().clone())
    }
    
    async fn cleanup(&self) -> Result<(), CacheError> {
        let removed = self.remove_expired()?;
        if removed > 0 {
            self.update_stats()?;
        }
        Ok(())
    }
    
    async fn optimize(&self) -> Result<(), CacheError> {
        // LocalStorage doesn't support optimization
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
}

#[async_trait]
impl CompressibleCache for LocalStorageCache {
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
impl EncryptableCache for LocalStorageCache {
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