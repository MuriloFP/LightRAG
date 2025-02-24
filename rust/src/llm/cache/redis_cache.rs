#[cfg(feature = "redis")]
use std::sync::Arc;
#[cfg(feature = "redis")]
use std::time::Duration;
#[cfg(feature = "redis")]
use tokio::sync::Mutex;
#[cfg(feature = "redis")]
use redis::AsyncCommands;
#[cfg(feature = "redis")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "redis")]
use crate::llm::cache::{
    backend::{CacheBackend, CacheError, CacheCapabilities},
    types::{CacheEntry, CacheStats, StorageConfig, CacheType},
};
#[cfg(feature = "redis")]
use async_trait::async_trait;

#[cfg(feature = "redis")]
pub struct RedisCache {
    client: redis::Client,
    prefix: String,
    stats: Arc<Mutex<CacheStats>>,
}

#[cfg(feature = "redis")]
impl RedisCache {
    pub fn new(client: redis::Client, prefix: String) -> Self {
        RedisCache { 
            client, 
            prefix,
            stats: Arc::new(Mutex::new(CacheStats::default())),
        }
    }

    fn build_key(&self, key: &str) -> String {
        format!("{}:{}", self.prefix, key)
    }

    async fn update_stats(&self) -> Result<(), CacheError> {
        let mut conn = self.client.get_async_connection().await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;

        let pattern = format!("{}:*", self.prefix);
        let keys: Vec<String> = conn.keys(&pattern).await
            .map_err(|e| CacheError::StorageError(e.to_string()))?;

        let mut stats = self.stats.lock().await;
        stats.item_count = keys.len();

        Ok(())
    }
}

#[cfg(feature = "redis")]
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

    async fn initialize(&mut self, _config: StorageConfig) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            // Redis client is already initialized in new()
            self.update_stats().await
        }
    }

    async fn get(&self, key: &str) -> Result<CacheEntry, CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let key = self.build_key(key);
            let data: Option<Vec<u8>> = conn.get(&key).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            match data {
                Some(data) => {
                    let mut entry: CacheEntry = bincode::deserialize(&data)
                        .map_err(|e| CacheError::StorageError(e.to_string()))?;
                    
                    if entry.metadata.is_expired() {
                        conn.del(&key).await
                            .map_err(|e| CacheError::StorageError(e.to_string()))?;
                        return Err(CacheError::Expired);
                    }

                    entry.metadata.access_count += 1;
                    entry.metadata.last_accessed = std::time::SystemTime::now();

                    let mut stats = self.stats.lock().await;
                    stats.hits += 1;

                    Ok(entry)
                }
                None => {
                    let mut stats = self.stats.lock().await;
                    stats.misses += 1;
                    Err(CacheError::NotFound)
                }
            }
        }
    }

    async fn set(&self, entry: CacheEntry) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let key = self.build_key(&entry.key);
            let data = bincode::serialize(&entry)
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            conn.set(&key, data).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            self.update_stats().await?;
            Ok(())
        }
    }

    async fn delete(&self, key: &str) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let key = self.build_key(key);
            conn.del(&key).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            self.update_stats().await?;
            Ok(())
        }
    }

    async fn exists(&self, key: &str) -> Result<bool, CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let key = self.build_key(key);
            let exists: bool = conn.exists(&key).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            Ok(exists)
        }
    }

    async fn get_many(&self, keys: &[String]) -> Result<Vec<Option<CacheEntry>>, CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let redis_keys: Vec<String> = keys.iter()
                .map(|k| self.build_key(k))
                .collect();

            let data: Vec<Option<Vec<u8>>> = conn.get(&redis_keys[..]).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let mut results = Vec::with_capacity(keys.len());
            let mut stats = self.stats.lock().await;

            for data in data {
                match data {
                    Some(bytes) => {
                        match bincode::deserialize(&bytes) {
                            Ok(mut entry) => {
                                if entry.metadata.is_expired() {
                                    stats.misses += 1;
                                    results.push(None);
                                } else {
                                    entry.metadata.access_count += 1;
                                    entry.metadata.last_accessed = std::time::SystemTime::now();
                                    stats.hits += 1;
                                    results.push(Some(entry));
                                }
                            }
                            Err(_) => {
                                stats.misses += 1;
                                results.push(None);
                            }
                        }
                    }
                    None => {
                        stats.misses += 1;
                        results.push(None);
                    }
                }
            }

            Ok(results)
        }
    }

    async fn set_many(&self, entries: Vec<CacheEntry>) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let mut pipeline = redis::pipe();
            for entry in entries {
                let key = self.build_key(&entry.key);
                let data = bincode::serialize(&entry)
                    .map_err(|e| CacheError::StorageError(e.to_string()))?;
                pipeline.set(&key, data);
            }

            pipeline.query_async(&mut conn).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            self.update_stats().await?;
            Ok(())
        }
    }

    async fn delete_many(&self, keys: &[String]) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let redis_keys: Vec<String> = keys.iter()
                .map(|k| self.build_key(k))
                .collect();

            conn.del(&redis_keys[..]).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            self.update_stats().await?;
            Ok(())
        }
    }

    async fn clear(&self) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let pattern = format!("{}:*", self.prefix);
            let keys: Vec<String> = conn.keys(&pattern).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            if !keys.is_empty() {
                conn.del(&keys[..]).await
                    .map_err(|e| CacheError::StorageError(e.to_string()))?;
            }

            let mut stats = self.stats.lock().await;
            *stats = CacheStats::default();

            Ok(())
        }
    }

    async fn stats(&self) -> Result<CacheStats, CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let stats = self.stats.lock().await;
            Ok(stats.clone())
        }
    }

    async fn cleanup(&self) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let pattern = format!("{}:*", self.prefix);
            let keys: Vec<String> = conn.keys(&pattern).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            for key in keys {
                let data: Option<Vec<u8>> = conn.get(&key).await
                    .map_err(|e| CacheError::StorageError(e.to_string()))?;

                if let Some(data) = data {
                    if let Ok(entry) = bincode::deserialize::<CacheEntry>(&data) {
                        if entry.metadata.is_expired() {
                            conn.del(&key).await
                                .map_err(|e| CacheError::StorageError(e.to_string()))?;
                        }
                    }
                }
            }

            self.update_stats().await?;
            Ok(())
        }
    }

    async fn optimize(&self) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            // Redis handles optimization internally
            Ok(())
        }
    }

    async fn backup(&self, _path: &str) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            // Redis handles backups through its own mechanisms
            Err(CacheError::UnsupportedOperation("Redis backup should be handled through Redis's native backup mechanisms".into()))
        }
    }

    async fn restore(&self, _path: &str) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            // Redis handles restores through its own mechanisms
            Err(CacheError::UnsupportedOperation("Redis restore should be handled through Redis's native restore mechanisms".into()))
        }
    }

    async fn health_check(&self) -> Result<(), CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let _: String = conn.ping().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            Ok(())
        }
    }

    async fn get_entries_batch(&self, offset: usize, limit: usize) -> Result<Vec<CacheEntry>, CacheError> {
        #[cfg(not(feature = "redis"))]
        return Err(CacheError::UnsupportedOperation("Redis feature not enabled".to_string()));

        #[cfg(feature = "redis")]
        {
            let mut conn = self.client.get_async_connection().await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let pattern = format!("{}:*", self.prefix);
            let keys: Vec<String> = conn.keys(&pattern).await
                .map_err(|e| CacheError::StorageError(e.to_string()))?;

            let start = offset.min(keys.len());
            let end = (offset + limit).min(keys.len());
            let batch_keys = &keys[start..end];

            let mut entries = Vec::new();
            for key in batch_keys {
                let data: Option<Vec<u8>> = conn.get(key).await
                    .map_err(|e| CacheError::StorageError(e.to_string()))?;

                if let Some(data) = data {
                    if let Ok(entry) = bincode::deserialize::<CacheEntry>(&data) {
                        if !entry.metadata.is_expired() {
                            entries.push(entry);
                        }
                    }
                }
            }

            Ok(entries)
        }
    }
} 