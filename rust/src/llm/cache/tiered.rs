use std::any::Any;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::pin::Pin;

use super::backend::{CacheBackend, CacheError, CacheCapabilities, EncryptableCache, CompressibleCache, EncryptionAlgorithm};
use super::types::{CacheEntry, CacheStats, CacheType, StorageConfig};
use super::memory::MemoryCache;

/// Configuration for tiered cache
#[derive(Debug, Clone)]
pub struct TieredCacheConfig {
    /// Memory cache configuration
    pub memory_config: StorageConfig,
    /// Persistent storage configuration
    pub storage_config: StorageConfig,
    /// Sync interval in seconds
    pub sync_interval_secs: u64,
    /// Whether to write-through to persistent storage
    pub write_through: bool,
    /// Maximum items to sync in one batch
    pub max_sync_batch_size: usize,
    /// Cache warming configuration
    pub warming_config: Option<CacheWarmingConfig>,
}

impl Default for TieredCacheConfig {
    fn default() -> Self {
        Self {
            memory_config: StorageConfig::default(),
            storage_config: StorageConfig::default(),
            sync_interval_secs: 60,
            write_through: true,
            max_sync_batch_size: 100,
            warming_config: None,
        }
    }
}

/// Cache warming configuration
#[derive(Debug, Clone)]
pub struct CacheWarmingConfig {
    /// Whether to enable cache warming
    pub enabled: bool,
    /// Maximum number of items to prefetch
    pub max_prefetch_items: usize,
    /// Prefetch batch size
    pub prefetch_batch_size: usize,
    /// Background loading interval in seconds
    pub background_load_interval_secs: u64,
    /// Priority threshold for warming (0-100)
    pub priority_threshold: u32,
}

impl Default for CacheWarmingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_prefetch_items: 1000,
            prefetch_batch_size: 100,
            background_load_interval_secs: 300, // 5 minutes
            priority_threshold: 50,
        }
    }
}

impl TieredCacheConfig {
    /// Add cache warming configuration
    pub fn with_warming(mut self, warming_config: CacheWarmingConfig) -> Self {
        self.warming_config = Some(warming_config);
        self
    }
}

/// Two-tier cache implementation with memory and persistent storage
pub struct TieredCache {
    /// L1 cache (memory)
    l1_cache: Arc<RwLock<MemoryCache>>,
    
    /// L2 cache (persistent storage)
    l2_cache: Arc<RwLock<Box<dyn CacheBackend>>>,
    
    /// Cache configuration
    config: TieredCacheConfig,
    
    /// Background sync task handle
    sync_task: Option<tokio::task::JoinHandle<()>>,

    /// Cache warming task handle
    warming_task: Option<tokio::task::JoinHandle<()>>,
}

impl TieredCache {
    /// Create a new tiered cache with the given configuration
    pub async fn new(config: TieredCacheConfig, l2_backend: Box<dyn CacheBackend>) -> Result<Self, CacheError> {
        let l1_cache = Arc::new(RwLock::new(MemoryCache::new()));
        let l2_cache = Arc::new(RwLock::new(l2_backend));
        
        // Initialize both caches
        l1_cache.write().await.initialize(config.memory_config.clone()).await?;
        l2_cache.write().await.initialize(config.storage_config.clone()).await?;
        
        let mut cache = Self {
            l1_cache,
            l2_cache,
            config: config.clone(),
            sync_task: None,
            warming_task: None,
        };
        
        // Start background sync if interval > 0
        if config.sync_interval_secs > 0 {
            cache.start_background_sync().await?;
        }

        // Start cache warming if enabled
        if let Some(warming_config) = &config.warming_config {
            if warming_config.enabled {
                cache.start_cache_warming().await?;
            }
        }
        
        Ok(cache)
    }
    
    /// Start background synchronization task
    async fn start_background_sync(&self) -> Result<(), CacheError> {
        let l1_cache = Arc::clone(&self.l1_cache);
        let l2_cache = Arc::clone(&self.l2_cache);
        let config = self.config.clone();
        
        let handle = tokio::spawn(async move {
            let interval = Duration::from_secs(config.sync_interval_secs);
            
            loop {
                tokio::time::sleep(interval).await;
                
                // Sync L1 to L2
                if let Err(e) = Self::sync_caches(&l1_cache, &l2_cache, &config).await {
                    tracing::error!("Background sync failed: {}", e);
                }
                
                // Cleanup expired entries
                if let Err(e) = l1_cache.write().await.cleanup().await {
                    tracing::error!("L1 cache cleanup failed: {}", e);
                }
                if let Err(e) = l2_cache.write().await.cleanup().await {
                    tracing::error!("L2 cache cleanup failed: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Synchronize caches (L1 -> L2)
    async fn sync_caches(
        l1: &Arc<RwLock<MemoryCache>>,
        l2: &Arc<RwLock<Box<dyn CacheBackend>>>,
        config: &TieredCacheConfig,
    ) -> Result<(), CacheError> {
        let l1_stats = l1.read().await.stats().await?;
        let mut synced = 0;
        
        // Get all entries from L1 using the trait method
        let l1_entries = l1.read().await.get_entries_batch(0, l1_stats.item_count).await?;
        
        // Sync in batches
        for chunk in l1_entries.chunks(config.max_sync_batch_size) {
            let mut l2 = l2.write().await;
            l2.set_many(chunk.to_vec()).await?;
            synced += chunk.len();
            
            if synced >= l1_stats.item_count {
                break;
            }
        }
        
        Ok(())
    }

    /// Start cache warming task
    async fn start_cache_warming(&mut self) -> Result<(), CacheError> {
        if let Some(warming_config) = &self.config.warming_config {
            if !warming_config.enabled {
                return Ok(());
            }

            let l1_cache = Arc::clone(&self.l1_cache);
            let l2_cache = Arc::clone(&self.l2_cache);
            let config = warming_config.clone();
            
            let handle = tokio::spawn(async move {
                let interval = Duration::from_secs(config.background_load_interval_secs);
                
                loop {
                    tokio::time::sleep(interval).await;
                    
                    // Warm cache with high-priority items
                    if let Err(e) = Self::warm_cache(&l1_cache, &l2_cache, &config).await {
                        tracing::error!("Cache warming failed: {}", e);
                    }
                }
            });

            self.warming_task = Some(handle);
        }
        
        Ok(())
    }

    /// Warm cache by pre-fetching high-priority items
    async fn warm_cache(
        l1: &Arc<RwLock<MemoryCache>>,
        l2: &Arc<RwLock<Box<dyn CacheBackend>>>,
        config: &CacheWarmingConfig,
    ) -> Result<(), CacheError> {
        // Get high-priority items from L2 that aren't in L1
        let l2_entries = l2.read().await.get_high_priority_entries(
            config.priority_threshold,
            config.max_prefetch_items
        ).await?;

        let mut warmed = 0;
        
        // Pre-fetch in batches
        for chunk in l2_entries.chunks(config.prefetch_batch_size) {
            let mut l1 = l1.write().await;
            l1.set_many(chunk.to_vec()).await?;
            warmed += chunk.len();
            
            if warmed >= config.max_prefetch_items {
                break;
            }
        }

        tracing::debug!("Warmed {} cache entries", warmed);
        Ok(())
    }

    /// Pre-fetch specific keys into L1 cache
    pub async fn prefetch(&self, keys: &[String]) -> Result<(), CacheError> {
        if let Some(config) = &self.config.warming_config {
            if !config.enabled {
                return Ok(());
            }

            let mut warmed = 0;
            
            // Pre-fetch in batches
            for chunk in keys.chunks(config.prefetch_batch_size) {
                let l2_entries = self.l2_cache.read().await.get_many(chunk).await?;
                let valid_entries: Vec<_> = l2_entries.into_iter()
                    .filter_map(|e| e)
                    .collect();
                
                if !valid_entries.is_empty() {
                    self.l1_cache.write().await.set_many(valid_entries).await?;
                    warmed += chunk.len();
                }
                
                if warmed >= config.max_prefetch_items {
                    break;
                }
            }
        }
        
        Ok(())
    }

    /// Pre-fetch entries by priority threshold
    pub async fn prefetch_by_priority(&self, min_priority: u32) -> Result<(), CacheError> {
        if let Some(config) = &self.config.warming_config {
            if !config.enabled || min_priority < config.priority_threshold {
                return Ok(());
            }

            Self::warm_cache(
                &self.l1_cache,
                &self.l2_cache,
                config
            ).await?;
        }
        
        Ok(())
    }

    /// Get all entries from the cache
    async fn get_all_entries(&self) -> Result<Vec<CacheEntry>, CacheError> {
        let mut entries = Vec::new();
        let mut cursor = 0;
        let batch_size = 100;

        loop {
            let batch = self.l1_cache.read().await.get_entries_batch(cursor, batch_size).await?;
            if batch.is_empty() {
                break;
            }
            
            entries.extend(batch);
            cursor += batch_size;
        }

        Ok(entries)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[async_trait]
impl CacheBackend for TieredCache {
    fn backend_type(&self) -> CacheType {
        CacheType::Tiered
    }
    
    fn capabilities(&self) -> CacheCapabilities {
        CacheCapabilities {
            persistent: true,
            streaming: true,
            compression: true,
            encryption: true,
            transactions: false,
            pubsub: false,
        }
    }
    
    async fn initialize(&mut self, config: StorageConfig) -> Result<(), CacheError> {
        let mut l1 = self.l1_cache.write().await;
        let mut l2 = self.l2_cache.write().await;
        
        l1.initialize(config.clone()).await?;
        l2.initialize(config).await?;
        
        Ok(())
    }
    
    async fn get(&self, key: &str) -> Result<CacheEntry, CacheError> {
        // Try L1 first
        match self.l1_cache.read().await.get(key).await {
            Ok(entry) => Ok(entry),
            Err(_) => {
                // On L1 miss, try L2
                match self.l2_cache.read().await.get(key).await {
                    Ok(entry) => {
                        // Cache in L1
                        self.l1_cache.write().await.set(entry.clone()).await?;
                        Ok(entry)
                    }
                    Err(e) => Err(e)
                }
            }
        }
    }
    
    async fn set(&self, entry: CacheEntry) -> Result<(), CacheError> {
        // Always set in L1
        self.l1_cache.write().await.set(entry.clone()).await?;
        
        // Write-through to L2 if enabled
        if self.config.write_through {
            self.l2_cache.write().await.set(entry).await?;
        }
        
        Ok(())
    }
    
    async fn delete(&self, key: &str) -> Result<(), CacheError> {
        // Delete from both caches
        let l1_result = self.l1_cache.write().await.delete(key).await;
        let l2_result = self.l2_cache.write().await.delete(key).await;
        
        // Return error if either delete failed
        l1_result.and(l2_result)
    }
    
    async fn exists(&self, key: &str) -> Result<bool, CacheError> {
        // Check L1 first
        if self.l1_cache.read().await.exists(key).await? {
            return Ok(true);
        }
        
        // Then check L2
        self.l2_cache.read().await.exists(key).await
    }
    
    async fn get_many(&self, keys: &[String]) -> Result<Vec<Option<CacheEntry>>, CacheError> {
        let mut results = Vec::with_capacity(keys.len());
        let mut l2_keys = Vec::new();
        
        // Try L1 first
        let l1_results = self.l1_cache.read().await.get_many(keys).await?;
        
        // For each key
        for (i, key) in keys.iter().enumerate() {
            if let Some(entry) = l1_results[i].clone() {
                results.push(Some(entry));
            } else {
                results.push(None);
                l2_keys.push(key.clone());
            }
        }
        
        // Get missing entries from L2
        if !l2_keys.is_empty() {
            let l2_results = self.l2_cache.read().await.get_many(&l2_keys).await?;
            let mut l2_index = 0;
            
            // Update results and cache in L1
            for i in 0..results.len() {
                if results[i].is_none() {
                    if let Some(entry) = l2_results[l2_index].clone() {
                        // Cache in L1
                        self.l1_cache.write().await.set(entry.clone()).await?;
                        results[i] = Some(entry);
                    }
                    l2_index += 1;
                }
            }
        }
        
        Ok(results)
    }
    
    async fn set_many(&self, entries: Vec<CacheEntry>) -> Result<(), CacheError> {
        // Set in L1
        self.l1_cache.write().await.set_many(entries.clone()).await?;
        
        // Write-through to L2 if enabled
        if self.config.write_through {
            self.l2_cache.write().await.set_many(entries).await?;
        }
        
        Ok(())
    }
    
    async fn delete_many(&self, keys: &[String]) -> Result<(), CacheError> {
        // Delete from both caches
        let l1_result = self.l1_cache.write().await.delete_many(keys).await;
        let l2_result = self.l2_cache.write().await.delete_many(keys).await;
        
        // Return error if either delete failed
        l1_result.and(l2_result)
    }
    
    async fn clear(&self) -> Result<(), CacheError> {
        // Clear both caches
        let l1_result = self.l1_cache.write().await.clear().await;
        let l2_result = self.l2_cache.write().await.clear().await;
        
        // Return error if either clear failed
        l1_result.and(l2_result)
    }
    
    async fn stats(&self) -> Result<CacheStats, CacheError> {
        let l1_stats = self.l1_cache.read().await.stats().await?;
        let l2_stats = self.l2_cache.read().await.stats().await?;
        
        // Combine stats
        Ok(CacheStats {
            item_count: l1_stats.item_count + l2_stats.item_count,
            total_size_bytes: l1_stats.total_size_bytes + l2_stats.total_size_bytes,
            hits: l1_stats.hits + l2_stats.hits,
            misses: l1_stats.misses + l2_stats.misses,
            evictions: l1_stats.evictions + l2_stats.evictions,
            expirations: l1_stats.expirations + l2_stats.expirations,
        })
    }
    
    async fn cleanup(&self) -> Result<(), CacheError> {
        // Cleanup both caches
        let l1_result = self.l1_cache.write().await.cleanup().await;
        let l2_result = self.l2_cache.write().await.cleanup().await;
        
        // Return error if either cleanup failed
        l1_result.and(l2_result)
    }
    
    async fn optimize(&self) -> Result<(), CacheError> {
        // Optimize both caches
        let l1_result = self.l1_cache.write().await.optimize().await;
        let l2_result = self.l2_cache.write().await.optimize().await;
        
        // Return error if either optimize failed
        l1_result.and(l2_result)
    }
    
    async fn backup(&self, path: &str) -> Result<(), CacheError> {
        // Only backup L2 (persistent storage)
        self.l2_cache.write().await.backup(path).await
    }
    
    async fn restore(&self, path: &str) -> Result<(), CacheError> {
        // Restore L2 and sync to L1
        self.l2_cache.write().await.restore(path).await?;
        
        // Clear L1 and let it repopulate from L2
        self.l1_cache.write().await.clear().await?;
        
        Ok(())
    }
    
    async fn health_check(&self) -> Result<(), CacheError> {
        // Check both caches
        let l1_result = self.l1_cache.write().await.health_check().await;
        let l2_result = self.l2_cache.write().await.health_check().await;
        
        // Return error if either health check failed
        l1_result.and(l2_result)
    }

    async fn get_entries_batch(&self, offset: usize, limit: usize) -> Result<Vec<CacheEntry>, CacheError> {
        let entries = self.l2_cache.read().await.get_entries_batch(offset, limit).await?;
        Ok(entries)
    }
}

impl CompressibleCache for TieredCache {
    fn compression_level(&self) -> u32 {
        let guard = self.l2_cache.blocking_read();
        if let Some(backend) = guard.as_compressible() {
            backend.compression_level()
        } else {
            4 // Default compression level
        }
    }

    fn set_compression_level(&mut self, level: u32) {
        let mut guard = self.l2_cache.blocking_write();
        if let Some(backend) = guard.as_mut_compressible() {
            backend.set_compression_level(level);
        }
    }

    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CacheError> {
        let guard = self.l2_cache.blocking_read();
        if let Some(backend) = guard.as_compressible() {
            backend.compress(data)
        } else {
            Ok(data.to_vec())
        }
    }

    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CacheError> {
        let guard = self.l2_cache.blocking_read();
        if let Some(backend) = guard.as_compressible() {
            backend.decompress(data)
        } else {
            Ok(data.to_vec())
        }
    }
}

#[async_trait]
impl EncryptableCache for TieredCache {
    async fn encrypt(&self, data: Vec<u8>, algorithm: EncryptionAlgorithm) -> Result<Vec<u8>, CacheError> {
        let guard = self.l2_cache.blocking_read();
        if let Some(backend) = guard.as_encryptable() {
            backend.encrypt(data, algorithm).await
        } else {
            Err(CacheError::UnsupportedOperation("Encryption not supported".to_string()))
        }
    }

    async fn decrypt(&self, data: Vec<u8>, algorithm: EncryptionAlgorithm) -> Result<Vec<u8>, CacheError> {
        let guard = self.l2_cache.blocking_read();
        if let Some(backend) = guard.as_encryptable() {
            backend.decrypt(data, algorithm).await
        } else {
            Err(CacheError::UnsupportedOperation("Decryption not supported".to_string()))
        }
    }

    fn is_encryption_ready(&self) -> bool {
        let guard = self.l2_cache.blocking_read();
        guard.as_encryptable()
            .map(|backend| backend.is_encryption_ready())
            .unwrap_or(false)
    }

    fn supports_encryption(&self) -> bool {
        let guard = self.l2_cache.blocking_read();
        guard.as_encryptable()
            .map(|backend| backend.supports_encryption())
            .unwrap_or(false)
    }

    fn supported_algorithms(&self) -> Vec<EncryptionAlgorithm> {
        let guard = self.l2_cache.blocking_read();
        guard.as_encryptable()
            .map(|backend| backend.supported_algorithms())
            .unwrap_or_default()
    }
} 