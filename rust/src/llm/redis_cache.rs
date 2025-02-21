use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, Duration};
use redis::{Client, AsyncCommands, RedisError};
use r2d2;
use serde_json;
use async_trait::async_trait;
use std::sync::atomic::Ordering;
use futures::{Stream, stream};
use std::pin::Pin;
use crate::types::llm::{StreamingResponse, LLMClient, LLMError, LLMResponse};
use super::cache::{
    config::{CacheBackend, CacheConfig},
    entry::CacheEntry,
    metrics::CacheMetrics,
    ResponseCache,
};

impl From<RedisError> for LLMError {
    fn from(err: RedisError) -> Self {
        LLMError::CacheError(err.to_string())
    }
}

impl From<serde_json::Error> for LLMError {
    fn from(err: serde_json::Error) -> Self {
        LLMError::CacheError(format!("Serialization error: {}", err))
    }
}

/// Redis-based implementation of ResponseCache
pub struct RedisCache {
    /// Cache configuration
    config: CacheConfig,
    
    /// Redis client
    client: Client,
    
    /// Redis connection pool
    pool: Arc<r2d2::Pool<Client>>,

    /// Cache metrics
    metrics: Arc<CacheMetrics>,

    /// LLM client for verification
    llm_client: Option<Arc<dyn LLMClient>>,
}

impl RedisCache {
    /// Create a new Redis cache
    pub async fn new(config: CacheConfig) -> Result<Self, LLMError> {
        if let CacheBackend::Redis(redis_config) = &config.backend {
            let client = Client::open(redis_config.url.as_str())
                .map_err(|e| LLMError::CacheError(format!("Failed to create Redis client: {}", e)))?;

            let pool_config = r2d2::Pool::builder()
                .max_size(redis_config.pool_size as u32);

            let pool = pool_config.build(client.clone())
                .map_err(|e| LLMError::CacheError(format!("Failed to create connection pool: {}", e)))?;

            Ok(Self {
                config,
                client,
                pool: Arc::new(pool),
                metrics: Arc::new(CacheMetrics::default()),
                llm_client: None,
            })
        } else {
            Err(LLMError::CacheError("Invalid cache backend configuration".to_string()))
        }
    }

    /// Set the LLM client for verification
    pub fn set_llm_client(&mut self, client: Arc<dyn LLMClient>) {
        self.llm_client = Some(client);
    }

    /// Get a connection from the pool
    async fn get_conn(&self) -> Result<r2d2::PooledConnection<Client>, LLMError> {
        self.pool.get()
            .map_err(|e| LLMError::CacheError(format!("Failed to get Redis connection: {}", e)))
    }

    /// Build cache key
    fn build_key(&self, key: &str) -> String {
        format!("llm:cache:{}", key)
    }

    /// Calculate cosine similarity between two vectors
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }

    /// Get embedding for a prompt
    async fn get_embedding(&self, prompt: &str) -> Option<Vec<f32>> {
        let mut conn = self.get_conn().await.ok()?;
        let key = format!("{}:embedding", self.build_key(prompt));
        
        let embedding_data: Option<String> = redis::cmd("GET")
            .arg(&key)
            .query(&mut *conn)
            .ok()?;

        if let Some(data) = embedding_data {
            serde_json::from_str(&data).ok()
        } else {
            None
        }
    }

    /// Store embedding for a prompt
    async fn store_embedding(&self, prompt: &str, embedding: &[f32]) -> Result<(), LLMError> {
        let mut conn = self.get_conn().await?;
        let key = format!("{}:embedding", self.build_key(prompt));
        
        let embedding_json = serde_json::to_string(embedding)
            .map_err(|e| LLMError::CacheError(format!("Failed to serialize embedding: {}", e)))?;

        let _: () = redis::cmd("SET")
            .arg(&key)
            .arg(embedding_json)
            .query(&mut *conn)
            .map_err(|e| LLMError::CacheError(format!("Failed to store embedding: {}", e)))?;

        Ok(())
    }

    /// Put an entry with embedding for similarity search
    pub async fn put_with_embedding(&self, prompt: &str, response: LLMResponse, embedding: Vec<f32>) -> Result<(), LLMError> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut conn = self.client.get_async_connection().await?;
        let key = format!("{}:{}", self.config.prefix, prompt);
        let entry = CacheEntry {
            response,
            created_at: SystemTime::now(),
            expires_at: self.config.ttl.map(|ttl| SystemTime::now() + ttl),
            embedding: Some(embedding),
            access_count: 0,
            last_accessed: SystemTime::now(),
            llm_verified: false,
            checksum: None,
            compressed_data: None,
            original_size: None,
            metadata: HashMap::new(),
            chunks: None,
            total_duration: None,
            total_tokens: None,
            is_streaming: false,
        };
        let data = serde_json::to_string(&entry)?;

        if let Some(ttl) = self.config.ttl {
            conn.set_ex(&key, data, ttl.as_secs() as usize).await?;
        } else {
            conn.set(&key, data).await?;
        }

        self.metrics.size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    /// Find similar entry using vector similarity
    pub async fn find_similar_entry(&self, prompt_embedding: &[f32]) -> Option<LLMResponse> {
        let mut conn = self.get_conn().await.ok()?;
        let mut best_match = None;
        let mut best_similarity = self.config.similarity_threshold;

        // Get all cache keys
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg("llm:cache:*")
            .query(&mut *conn)
            .ok()?;

        for key in keys {
            // Skip embedding keys
            if key.ends_with(":embedding") {
                continue;
            }

            // Get the embedding for this key
            let prompt = key.trim_start_matches("llm:cache:");
            if let Some(embedding) = self.get_embedding(prompt).await {
                let similarity = Self::cosine_similarity(prompt_embedding, &embedding);
                if similarity > best_similarity {
                    // Get the cache entry
                    let entry_data: Option<String> = redis::cmd("GET")
                        .arg(&key)
                        .query(&mut *conn)
                        .ok()?;

                    if let Some(data) = entry_data {
                        if let Ok(entry) = serde_json::from_str::<CacheEntry>(&data) {
                            if !entry.is_expired() {
                                best_similarity = similarity;
                                best_match = Some(entry.response);
                            }
                        }
                    }
                }
            }
        }

        if best_match.is_some() {
            self.metrics.similarity_matches.fetch_add(1, Ordering::Relaxed);
        }

        best_match
    }

    /// Get cache metrics
    pub fn get_metrics(&self) -> &CacheMetrics {
        &self.metrics
    }

    fn create_entry(&self, response: LLMResponse, ttl: Option<Duration>) -> CacheEntry {
        CacheEntry {
            response,
            created_at: SystemTime::now(),
            expires_at: ttl.map(|ttl| SystemTime::now() + ttl),
            embedding: None,
            access_count: 0,
            last_accessed: SystemTime::now(),
            llm_verified: false,
            checksum: None,
            compressed_data: None,
            original_size: None,
            metadata: HashMap::new(),
            chunks: None,
            total_duration: None,
            total_tokens: None,
            is_streaming: false,
        }
    }

    fn create_streaming_entry(&self, chunks: Vec<StreamingResponse>, ttl: Option<Duration>) -> CacheEntry {
        let total_text: String = chunks.iter().map(|c| c.text.clone()).collect();
        let total_tokens = chunks.iter().map(|c| c.chunk_tokens).sum();
        let total_duration = chunks.last()
            .and_then(|c| c.timing.as_ref())
            .map(|t| t.total_duration);

        let response = LLMResponse {
            text: total_text,
            tokens_used: total_tokens,
            model: chunks.last()
                .map(|c| c.metadata.get("model").cloned().unwrap_or_default())
                .unwrap_or_default(),
            cached: true,
            context: None,
            metadata: chunks.last()
                .map(|c| c.metadata.clone())
                .unwrap_or_default(),
        };

        CacheEntry {
            response,
            created_at: SystemTime::now(),
            expires_at: ttl.map(|ttl| SystemTime::now() + ttl),
            embedding: None,
            access_count: 0,
            last_accessed: SystemTime::now(),
            llm_verified: false,
            checksum: None,
            compressed_data: None,
            original_size: None,
            metadata: HashMap::new(),
            chunks: Some(chunks),
            total_duration,
            total_tokens: Some(total_tokens),
            is_streaming: true,
        }
    }

    fn is_expired(&self, entry: &CacheEntry) -> bool {
        if let Some(expires_at) = entry.expires_at {
            SystemTime::now() > expires_at
        } else {
            false
        }
    }
}

#[async_trait]
impl ResponseCache for RedisCache {
    async fn get(&self, prompt: &str) -> Option<LLMResponse> {
        if !self.config.enabled {
            return None;
        }

        let mut conn = self.get_conn().await.ok()?;
        let key = self.build_key(prompt);

        // Try exact match first
        let entry_data: Option<String> = redis::cmd("GET")
            .arg(&key)
            .query(&mut *conn)
            .ok()?;

        if let Some(data) = entry_data {
            if let Ok(mut entry) = serde_json::from_str::<CacheEntry>(&data) {
                if entry.is_expired() {
                    self.metrics.misses.fetch_add(1, Ordering::Relaxed);
                    return None;
                }

                entry.record_access();
                self.metrics.hits.fetch_add(1, Ordering::Relaxed);

                // Update access metrics in Redis
                if let Ok(entry_json) = serde_json::to_string(&entry) {
                    let _: Result<(), _> = redis::cmd("SET")
                        .arg(&key)
                        .arg(entry_json)
                        .query(&mut *conn);
                }

                Some(entry.response)
            } else {
                None
            }
        } else if self.config.use_fuzzy_match {
            // Try similarity matching
            if let Some(embedding) = self.get_embedding(prompt).await {
                self.find_similar_entry(&embedding).await
            } else {
                None
            }
        } else {
            self.metrics.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    async fn put(&self, prompt: &str, response: LLMResponse) -> Result<(), LLMError> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut conn = self.client.get_async_connection().await?;
        let key = format!("{}:{}", self.config.prefix, prompt);
        let entry = CacheEntry {
            response,
            created_at: SystemTime::now(),
            expires_at: self.config.ttl.map(|ttl| SystemTime::now() + ttl),
            embedding: None,
            access_count: 0,
            last_accessed: SystemTime::now(),
            llm_verified: false,
            checksum: None,
            compressed_data: None,
            original_size: None,
            metadata: HashMap::new(),
            chunks: None,
            total_duration: None,
            total_tokens: None,
            is_streaming: false,
        };
        let data = serde_json::to_string(&entry)?;

        if let Some(ttl) = self.config.ttl {
            conn.set_ex(&key, data, ttl.as_secs() as usize).await?;
        } else {
            conn.set(&key, data).await?;
        }

        self.metrics.size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    async fn cleanup(&self) -> Result<(), LLMError> {
        let mut conn = self.get_conn().await?;
        
        // Get all keys
        let keys: Vec<String> = redis::cmd("KEYS")
            .arg("llm:cache:*")
            .query(&mut *conn)
            .map_err(|e| LLMError::CacheError(format!("Failed to get keys: {}", e)))?;

        let mut removed = 0;
        for key in keys {
            let entry_data: Option<String> = redis::cmd("GET")
                .arg(&key)
                .query(&mut *conn)
                .map_err(|e| LLMError::CacheError(format!("Failed to get entry: {}", e)))?;

            if let Some(data) = entry_data {
                if let Ok(entry) = serde_json::from_str::<CacheEntry>(&data) {
                    if entry.is_expired() {
                        let _: () = redis::cmd("DEL")
                            .arg(&key)
                            .query(&mut *conn)
                            .map_err(|e| LLMError::CacheError(format!("Failed to delete key: {}", e)))?;
                        removed += 1;
                    }
                }
            }
        }

        if removed > 0 {
            self.metrics.evictions.fetch_add(removed, Ordering::Relaxed);
            self.metrics.size.fetch_sub(removed, Ordering::Relaxed);
        }

        Ok(())
    }

    async fn clear(&self) -> Result<(), LLMError> {
        let mut conn = self.get_conn().await?;
        
        let _: () = redis::cmd("DEL")
            .arg("llm:cache:*")
            .query(&mut *conn)
            .map_err(|e| LLMError::CacheError(format!("Failed to clear cache: {}", e)))?;

        self.metrics.size.store(0, Ordering::Relaxed);
        Ok(())
    }

    fn get_config(&self) -> &CacheConfig {
        &self.config
    }

    fn update_config(&mut self, config: CacheConfig) -> Result<(), LLMError> {
        self.config = config;
        Ok(())
    }

    async fn get_stream(&self, prompt: &str) -> Option<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>> {
        if !self.config.enabled || !self.config.stream_cache_enabled {
            return None;
        }

        let mut conn = self.client.get_async_connection().await.ok()?;
        let key = format!("{}:{}", self.config.prefix, prompt);
        let data: Option<String> = conn.get(&key).await.ok()?;

        if let Some(data) = data {
            if let Ok(entry) = serde_json::from_str::<CacheEntry>(&data) {
                if !self.is_expired(&entry) && entry.is_streaming() {
                    if let Some(chunks) = entry.get_chunks() {
                        self.metrics.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let chunks = chunks.to_vec();
                        return Some(Box::pin(stream::iter(chunks.into_iter().map(Ok))));
                    }
                }
            }
        }

        self.metrics.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }

    async fn put_stream(&self, prompt: &str, chunks: Vec<StreamingResponse>) -> Result<(), LLMError> {
        if !self.config.enabled || !self.config.stream_cache_enabled {
            return Ok(());
        }

        let mut conn = self.client.get_async_connection().await?;
        let key = format!("{}:{}", self.config.prefix, prompt);
        let entry = self.create_streaming_entry(chunks, self.config.stream_ttl);
        let data = serde_json::to_string(&entry)?;

        if let Some(ttl) = self.config.stream_ttl {
            conn.set_ex(&key, data, ttl.as_secs() as usize).await?;
        } else {
            conn.set(&key, data).await?;
        }

        self.metrics.size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
} 