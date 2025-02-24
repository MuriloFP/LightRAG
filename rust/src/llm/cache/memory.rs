use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, Duration};
use parking_lot::RwLock;
use async_trait::async_trait;
use super::backend::{CacheBackend, CompressibleCache, EncryptableCache, CacheError, CacheCapabilities, EncryptionAlgorithm};
use super::types::{CacheEntry, CacheStats, CacheType, StorageConfig, CacheValue};
use crate::llm::LLMResponse;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
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
use std::any::Any;

/// In-memory cache implementation
pub struct MemoryCache {
    data: Arc<RwLock<HashMap<String, CacheEntry>>>,
    config: StorageConfig,
    stats: Arc<RwLock<CacheStats>>,
}

impl MemoryCache {
    /// Create a new memory cache
    pub fn new() -> Self {
        Self {
            data: Arc::new(RwLock::new(HashMap::new())),
            config: StorageConfig::default(),
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }
    
    /// Update cache statistics
    async fn update_stats(&self) {
        let storage = self.data.read();
        let mut stats = self.stats.write();
        
        stats.item_count = storage.len();
        stats.total_size_bytes = storage
            .values()
            .map(|entry| entry.metadata.size_bytes)
            .sum();
    }
    
    /// Check storage quota
    async fn check_quota(&self, new_size: usize) -> Result<(), CacheError> {
        let stats = self.stats.read();
        let total_size_mb = (stats.total_size_bytes + new_size) as f64 / 1024.0 / 1024.0;
        
        if total_size_mb > self.config.max_storage_mb as f64 {
            Err(CacheError::QuotaExceeded)
        } else {
            Ok(())
        }
    }
    
    /// Remove expired entries
    async fn remove_expired(&self) -> usize {
        let mut storage = self.data.write();
        let before_len = storage.len();
        
        storage.retain(|_, entry| !self.is_expired(entry));
        
        let removed = before_len - storage.len();
        if removed > 0 {
            let mut stats = self.stats.write();
            stats.expirations += removed as u64;
        }
        
        removed
    }

    /// Check if an entry is expired
    fn is_expired(&self, entry: &CacheEntry) -> bool {
        entry.metadata.is_expired()
    }

    /// Get a batch of entries starting from an offset
    pub async fn get_entries_batch(&self, offset: usize, limit: usize) -> Result<Vec<CacheEntry>, CacheError> {
        let entries = self.data.read();
        let mut result = Vec::new();
        
        for (_, entry) in entries.iter().skip(offset).take(limit) {
            if !self.is_expired(entry) {
                result.push(entry.clone());
            }
        }
        
        Ok(result)
    }

    /// Get high priority entries above a threshold
    pub async fn get_high_priority_entries(&self, min_priority: u32, limit: usize) -> Result<Vec<CacheEntry>, CacheError> {
        let entries = self.data.read();
        let mut result = Vec::new();
        
        for (_, entry) in entries.iter() {
            if !self.is_expired(entry) && entry.priority as u32 >= min_priority {
                result.push(entry.clone());
                if result.len() >= limit {
                    break;
                }
            }
        }
        
        Ok(result)
    }
}

#[async_trait]
impl CacheBackend for MemoryCache {
    fn backend_type(&self) -> CacheType {
        CacheType::Memory
    }
    
    fn capabilities(&self) -> CacheCapabilities {
        CacheCapabilities {
            persistent: false,
            streaming: true,
            compression: true,
            encryption: false,
            transactions: false,
            pubsub: false,
        }
    }
    
    async fn initialize(&mut self, config: StorageConfig) -> Result<(), CacheError> {
        self.config = config;
        Ok(())
    }

    async fn get(&self, key: &str) -> Result<CacheEntry, CacheError> {
        let entries = self.data.read();
        if let Some(entry) = entries.get(key) {
            if !self.is_expired(entry) {
                // Increment hit counter
                {
                    let mut stats = self.stats.write();
                    stats.hits += 1;
                }
                Ok(entry.clone())
            } else {
                // Increment miss counter
                {
                    let mut stats = self.stats.write();
                    stats.misses += 1;
                }
                Err(CacheError::NotFound)
            }
        } else {
            // Increment miss counter
            {
                let mut stats = self.stats.write();
                stats.misses += 1;
            }
            Err(CacheError::NotFound)
        }
    }
    
    async fn set(&self, entry: CacheEntry) -> Result<(), CacheError> {
        self.check_quota(entry.metadata.size_bytes).await?;
        {
            let mut entries = self.data.write();
            entries.insert(entry.key.clone(), entry);
        }
        self.update_stats().await;
        Ok(())
    }
    
    async fn delete(&self, key: &str) -> Result<(), CacheError> {
        let removed = {
            let mut entries = self.data.write();
            entries.remove(key)
        };
        if removed.is_some() {
            self.update_stats().await;
            Ok(())
        } else {
            Err(CacheError::NotFound)
        }
    }

    async fn exists(&self, key: &str) -> Result<bool, CacheError> {
        let entries = self.data.read();
        Ok(entries.contains_key(key))
    }
    
    async fn get_many(&self, keys: &[String]) -> Result<Vec<Option<CacheEntry>>, CacheError> {
        let entries = self.data.read();
        let mut results = Vec::with_capacity(keys.len());
        let mut hits = 0;
        let mut misses = 0;
        
        for key in keys {
            if let Some(entry) = entries.get(key) {
                if !self.is_expired(entry) {
                    results.push(Some(entry.clone()));
                    hits += 1;
                    continue;
                }
            }
            results.push(None);
            misses += 1;
        }
        
        // Update stats
        if hits > 0 || misses > 0 {
            let mut stats = self.stats.write();
            stats.hits += hits;
            stats.misses += misses;
        }
        
        Ok(results)
    }
    
    async fn set_many(&self, entries: Vec<CacheEntry>) -> Result<(), CacheError> {
        let total_new_size: usize = entries.iter().map(|entry| entry.metadata.size_bytes).sum();
        self.check_quota(total_new_size).await?;
        {
            let mut cache_entries = self.data.write();
            for entry in entries {
                cache_entries.insert(entry.key.clone(), entry);
            }
        }
        self.update_stats().await;
        Ok(())
    }
    
    async fn delete_many(&self, keys: &[String]) -> Result<(), CacheError> {
        {
            let mut entries = self.data.write();
            for key in keys {
                entries.remove(key);
            }
        }
        self.update_stats().await;
        Ok(())
    }

    async fn clear(&self) -> Result<(), CacheError> {
        {
            let mut entries = self.data.write();
            entries.clear();
        }
        self.update_stats().await;
        Ok(())
    }

    async fn stats(&self) -> Result<CacheStats, CacheError> {
        Ok(self.stats.read().clone())
    }
    
    async fn cleanup(&self) -> Result<(), CacheError> {
        let removed = self.remove_expired().await;
        if removed > 0 {
            self.update_stats().await;
        }
        Ok(())
    }
    
    async fn optimize(&self) -> Result<(), CacheError> {
        // In-memory cache doesn't need optimization
        Ok(())
    }
    
    async fn backup(&self, _path: &str) -> Result<(), CacheError> {
        // In-memory cache doesn't support backup, but should return Ok as a no-op
        // instead of an error to support tiered cache patterns
        Ok(())
    }
    
    async fn restore(&self, _path: &str) -> Result<(), CacheError> {
        // In-memory cache doesn't support restore, but should return Ok as a no-op
        // instead of an error to support tiered cache patterns
        Ok(())
    }
    
    async fn health_check(&self) -> Result<(), CacheError> {
        // Simple health check - just verify we can read/write
        let key = "health_check".to_string();
        let value = CacheValue::Response(LLMResponse {
            text: "test".to_string(),
            tokens_used: 1,
            model: "test".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        });
        let entry = CacheEntry::new(
            key.clone(),
            value,
            None,
            None,
        );
        
        self.set(entry).await?;
        self.delete(&key).await?;
        
        Ok(())
    }

    async fn get_entries_batch(&self, offset: usize, limit: usize) -> Result<Vec<CacheEntry>, CacheError> {
        let entries = self.data.read();
        let mut result = Vec::new();
        
        for (_, entry) in entries.iter().skip(offset).take(limit) {
            if !self.is_expired(entry) {
                result.push(entry.clone());
            }
        }
        
        Ok(result)
    }

    fn as_compressible(&self) -> Option<&dyn CompressibleCache> {
        Some(self)
    }

    fn as_mut_compressible(&mut self) -> Option<&mut dyn CompressibleCache> {
        Some(self)
    }

    fn as_encryptable(&self) -> Option<&dyn EncryptableCache> {
        Some(self)
    }

    fn as_mut_encryptable(&mut self) -> Option<&mut dyn EncryptableCache> {
        Some(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl CompressibleCache for MemoryCache {
    fn compression_level(&self) -> u32 {
        4 // Default compression level
    }

    fn set_compression_level(&mut self, _level: u32) {
        // Memory cache doesn't actually compress data
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CacheError> {
        Ok(data.to_vec()) // Memory cache doesn't actually compress
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CacheError> {
        Ok(data.to_vec()) // Memory cache doesn't actually decompress
    }
}

#[async_trait]
impl EncryptableCache for MemoryCache {
    async fn encrypt(&self, data: Vec<u8>, algorithm: EncryptionAlgorithm) -> Result<Vec<u8>, CacheError> {
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
                cipher.encrypt(nonce, data.as_ref())
                    .map_err(|e| CacheError::EncryptionError(e.to_string()))?
            }
            EncryptionAlgorithm::ChaCha20Poly1305 => {
                let cipher = ChaCha20Poly1305::new_from_slice(key)
                    .map_err(|e| CacheError::EncryptionError(e.to_string()))?;
                
                let nonce = ChaChaPolNonce::from_slice(&nonce);
                cipher.encrypt(nonce, data.as_ref())
                    .map_err(|e| CacheError::EncryptionError(e.to_string()))?
            }
        };

        let mut result = nonce;
        result.extend(encrypted);
        Ok(result)
    }

    async fn decrypt(&self, data: Vec<u8>, algorithm: EncryptionAlgorithm) -> Result<Vec<u8>, CacheError> {
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