use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use async_trait::async_trait;
use super::backend::{CacheBackend, CacheError, CacheCapabilities};
use super::types::{CacheEntry, CacheStats, CacheType, StorageConfig, CacheValue, CachePriority};
use crate::llm::LLMResponse;

/// In-memory cache implementation
pub struct MemoryCache {
    /// Cache storage
    storage: Arc<RwLock<HashMap<String, CacheEntry>>>,
    
    /// Cache configuration
    config: StorageConfig,
    
    /// Cache statistics
    stats: Arc<RwLock<CacheStats>>,
}

impl MemoryCache {
    /// Create a new memory cache
    pub fn new(config: StorageConfig) -> Self {
        Self {
            storage: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }
    
    /// Update cache statistics
    async fn update_stats(&self) {
        let storage = self.storage.read().await;
        let mut stats = self.stats.write().await;
        
        stats.item_count = storage.len();
        stats.total_size_bytes = storage
            .values()
            .map(|entry| entry.metadata.size_bytes)
            .sum();
    }
    
    /// Check storage quota
    async fn check_quota(&self, new_size: usize) -> Result<(), CacheError> {
        let storage = self.storage.read().await;
        let current_size: usize = storage
            .values()
            .map(|entry| entry.metadata.size_bytes)
            .sum();
        
        let total_size_mb = (current_size + new_size) as f64 / 1024.0 / 1024.0;
        
        if total_size_mb > self.config.max_memory_mb as f64 {
            Err(CacheError::QuotaExceeded)
        } else {
            Ok(())
        }
    }
    
    /// Remove expired entries
    async fn remove_expired(&self) -> usize {
        let mut storage = self.storage.write().await;
        let before_len = storage.len();
        
        storage.retain(|_, entry| !entry.metadata.is_expired());
        
        let removed = before_len - storage.len();
        if removed > 0 {
            let mut stats = self.stats.write().await;
            stats.expirations += removed as u64;
        }
        
        removed
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
            compression: false,
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
        let storage = self.storage.read().await;
        
        if let Some(entry) = storage.get(key) {
            if entry.metadata.is_expired() {
                return Err(CacheError::Expired);
            }
            
            let mut entry = entry.clone();
            entry.metadata.access_count += 1;
            entry.metadata.last_accessed = std::time::SystemTime::now();
            
            let mut stats = self.stats.write().await;
            stats.hits += 1;
            
            Ok(entry)
        } else {
            let mut stats = self.stats.write().await;
            stats.misses += 1;
            Err(CacheError::NotFound)
        }
    }
    
    async fn set(&self, entry: CacheEntry) -> Result<(), CacheError> {
        // Check quota before inserting
        self.check_quota(entry.metadata.size_bytes).await?;
        
        let mut storage = self.storage.write().await;
        storage.insert(entry.key.clone(), entry);
        
        // Update stats while holding the write lock
        let mut stats = self.stats.write().await;
        stats.item_count = storage.len();
        stats.total_size_bytes = storage
            .values()
            .map(|entry| entry.metadata.size_bytes)
            .sum();
        
        Ok(())
    }
    
    async fn delete(&self, key: &str) -> Result<(), CacheError> {
        let mut storage = self.storage.write().await;
        
        if storage.remove(key).is_some() {
            self.update_stats().await;
            Ok(())
        } else {
            Err(CacheError::NotFound)
        }
    }
    
    async fn exists(&self, key: &str) -> Result<bool, CacheError> {
        let storage = self.storage.read().await;
        Ok(storage.contains_key(key))
    }
    
    async fn get_many(&self, keys: &[String]) -> Result<Vec<Option<CacheEntry>>, CacheError> {
        let storage = self.storage.read().await;
        let mut results = Vec::with_capacity(keys.len());
        
        for key in keys {
            let entry = storage.get(key).cloned();
            if let Some(mut entry) = entry {
                if !entry.metadata.is_expired() {
                    entry.metadata.access_count += 1;
                    entry.metadata.last_accessed = std::time::SystemTime::now();
                    results.push(Some(entry));
                    continue;
                }
            }
            results.push(None);
        }
        
        Ok(results)
    }
    
    async fn get_entries_batch(&self, offset: usize, limit: usize) -> Result<Vec<CacheEntry>, CacheError> {
        let storage = self.storage.read().await;
        
        let entries: Vec<CacheEntry> = storage
            .values()
            .skip(offset)
            .take(limit)
            .filter(|entry| !entry.metadata.is_expired())
            .cloned()
            .collect();
            
        Ok(entries)
    }
    
    async fn set_many(&self, entries: Vec<CacheEntry>) -> Result<(), CacheError> {
        // Calculate total size
        let total_new_size: usize = entries.iter()
            .map(|e| e.metadata.size_bytes)
            .sum();
            
        // Check quota
        self.check_quota(total_new_size).await?;
        
        // Store entries
        let mut storage = self.storage.write().await;
        for entry in entries {
            storage.insert(entry.key.clone(), entry);
        }
        
        // Update stats
        self.update_stats().await;
        
        Ok(())
    }
    
    async fn delete_many(&self, keys: &[String]) -> Result<(), CacheError> {
        let mut storage = self.storage.write().await;
        
        for key in keys {
            storage.remove(key);
        }
        
        self.update_stats().await;
        Ok(())
    }
    
    async fn clear(&self) -> Result<(), CacheError> {
        let mut storage = self.storage.write().await;
        storage.clear();
        
        let mut stats = self.stats.write().await;
        *stats = CacheStats::default();
        
        Ok(())
    }
    
    async fn stats(&self) -> Result<CacheStats, CacheError> {
        let stats = self.stats.read().await;
        Ok(stats.clone())
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
        // In-memory cache doesn't support backup
        Err(CacheError::Unavailable("Backup not supported for in-memory cache".into()))
    }
    
    async fn restore(&self, _path: &str) -> Result<(), CacheError> {
        // In-memory cache doesn't support restore
        Err(CacheError::Unavailable("Restore not supported for in-memory cache".into()))
    }
    
    async fn health_check(&self) -> Result<(), CacheError> {
        // Simple health check - just verify we can read/write
        let key = "health_check".to_string();
        let entry = CacheEntry::new(
            key.clone(),
            CacheValue::Response(LLMResponse {
                text: "test".to_string(),
                tokens_used: 1,
                model: "test".to_string(),
                cached: false,
                context: None,
                metadata: HashMap::new(),
            }),
            Some(std::time::Duration::from_secs(60)), // 1 minute TTL
            Some(CachePriority::Low),
        );
        
        self.set(entry).await?;
        self.delete(&key).await?;
        
        Ok(())
    }
} 