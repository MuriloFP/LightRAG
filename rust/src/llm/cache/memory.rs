use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use async_trait::async_trait;
use futures::{Stream, stream};
use std::pin::Pin;
use tokio::sync::RwLock;
use crate::llm::{LLMError, LLMResponse};
use crate::types::llm::StreamingResponse;
use super::{ResponseCache, CacheConfig, CacheEntry, CacheMetrics};
use super::types::CacheType;

/// In-memory implementation of ResponseCache
pub struct InMemoryCache {
    /// Cache entries
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    
    /// Cache configuration
    config: CacheConfig,
    
    /// Cache metrics
    metrics: Arc<CacheMetrics>,
}

impl Default for InMemoryCache {
    fn default() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config: CacheConfig::default(),
            metrics: Arc::new(CacheMetrics::default()),
        }
    }
}

impl InMemoryCache {
    /// Create a new in-memory cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: Arc::new(CacheMetrics::default()),
        }
    }

    /// Check if an entry is expired
    pub fn is_expired(&self, entry: &CacheEntry) -> bool {
        if let Some(expires_at) = entry.expires_at {
            SystemTime::now() > expires_at
        } else {
            false
        }
    }

    /// Find similar entry using vector similarity
    pub async fn find_similar(&self, embedding: Vec<f32>, threshold: f32) -> Option<LLMResponse> {
        if !self.config.similarity_enabled {
            return None;
        }

        let entries = self.entries.read().await;
        let mut best_match = None;
        let mut best_similarity = threshold;

        for entry in entries.values() {
            if let Some(entry_embedding) = &entry.embedding {
                let similarity = cosine_similarity(&embedding, entry_embedding);
                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_match = Some(entry.response.clone());
                }
            }
        }

        best_match
    }

    /// Store an entry with embedding
    pub async fn put_with_embedding(&self, prompt: &str, response: LLMResponse, embedding: Vec<f32>) -> Result<(), LLMError> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut entries = self.entries.write().await;
        let mut entry = CacheEntry::new(response, self.config.ttl, Some(self.config.cache_type.clone()));
        entry.embedding = Some(embedding);
        entries.insert(prompt.to_string(), entry);
        self.metrics.size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub fn build_key(&self, prompt: &str) -> String {
        format!("{}:{}:{}",
            self.config.prefix,
            self.config.cache_type.as_str(),
            prompt
        )
    }

    pub async fn put(&self, prompt: &str, response: LLMResponse) -> Result<(), LLMError> {
        if !self.config.enabled {
            return Ok(());
        }

        let key = self.build_key(prompt);
        let mut entry = CacheEntry::new(response, self.config.ttl, Some(self.config.cache_type.clone()));
        
        if self.config.use_compression {
            entry.compress()?;
        }
        
        if self.config.validate_integrity {
            entry.set_checksum()?;
        }

        let mut entries = self.entries.write().await;
        entries.insert(key, entry);
        self.metrics.size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub async fn get(&self, prompt: &str) -> Option<LLMResponse> {
        if !self.config.enabled {
            return None;
        }

        let key = self.build_key(prompt);
        let entries = self.entries.read().await;
        if let Some(entry) = entries.get(&key) {
            if !self.is_expired(entry) {
                self.metrics.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Some(entry.response.clone());
            }
        }
        self.metrics.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }

    pub async fn put_stream(&self, prompt: &str, chunks: Vec<StreamingResponse>) -> Result<(), LLMError> {
        if !self.config.enabled || !self.config.stream_cache_enabled {
            return Ok(());
        }

        let key = self.build_key(prompt);
        let entry = CacheEntry::new_streaming(chunks, self.config.stream_ttl, Some(self.config.cache_type.clone()));
        let mut entries = self.entries.write().await;
        entries.insert(key, entry);
        self.metrics.size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub async fn get_stream(&self, prompt: &str) -> Option<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>> {
        if !self.config.enabled || !self.config.stream_cache_enabled {
            return None;
        }

        let key = self.build_key(prompt);
        let entries = self.entries.read().await;
        if let Some(entry) = entries.get(&key) {
            if !self.is_expired(entry) && entry.is_streaming() {
                if let Some(chunks) = entry.get_chunks() {
                    self.metrics.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let chunks = chunks.to_vec();
                    return Some(Box::pin(stream::iter(chunks.into_iter().map(Ok))));
                }
            }
        }
        self.metrics.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        None
    }

    pub async fn cleanup(&self) -> Result<(), LLMError> {
        let mut entries = self.entries.write().await;
        let before_len = entries.len();
        entries.retain(|_, entry| !self.is_expired(entry));
        let removed = before_len - entries.len();
        
        if removed > 0 {
            self.metrics.size.fetch_sub(removed, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(())
    }

    pub async fn clear(&self) -> Result<(), LLMError> {
        let mut entries = self.entries.write().await;
        entries.clear();
        self.metrics.size.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    pub fn get_config(&self) -> &CacheConfig {
        &self.config
    }

    pub fn update_config(&mut self, config: CacheConfig) -> Result<(), LLMError> {
        self.config = config;
        Ok(())
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use futures::StreamExt;

    #[tokio::test]
    async fn test_cache_types() {
        let mut config = CacheConfig {
            enabled: true,
            ttl: Some(Duration::from_secs(60)),
            ..Default::default()
        };
        let mut cache = InMemoryCache::new(config.clone());

        // Test Query type (default)
        let response1 = LLMResponse {
            text: "Query response".to_string(),
            tokens_used: 2,
            model: "test".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        };
        cache.put("test_prompt", response1.clone()).await.unwrap();
        
        // Should find with same type
        let result = cache.get("test_prompt").await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "Query response");

        // Test Extract type
        config.cache_type = CacheType::Extract;
        cache.update_config(config.clone()).unwrap();
        
        let response2 = LLMResponse {
            text: "Extract response".to_string(),
            tokens_used: 2,
            model: "test".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        };
        cache.put("test_prompt", response2.clone()).await.unwrap();

        // Should not find Query type response
        let result = cache.get("test_prompt").await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "Extract response");

        // Test Keywords type
        config.cache_type = CacheType::Keywords;
        cache.update_config(config.clone()).unwrap();
        
        let response3 = LLMResponse {
            text: "Keywords response".to_string(),
            tokens_used: 2,
            model: "test".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        };
        cache.put("test_prompt", response3.clone()).await.unwrap();

        // Should find Keywords type response
        let result = cache.get("test_prompt").await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "Keywords response");

        // Test Custom type
        config.cache_type = CacheType::Custom("test_type".to_string());
        cache.update_config(config).unwrap();
        
        let response4 = LLMResponse {
            text: "Custom response".to_string(),
            tokens_used: 2,
            model: "test".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        };
        cache.put("test_prompt", response4.clone()).await.unwrap();

        // Should find Custom type response
        let result = cache.get("test_prompt").await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "Custom response");
    }

    #[tokio::test]
    async fn test_cache_type_streaming() {
        let mut config = CacheConfig {
            enabled: true,
            stream_cache_enabled: true,
            ttl: Some(Duration::from_secs(60)),
            ..Default::default()
        };
        let mut cache = InMemoryCache::new(config.clone());

        // Create test chunks
        let chunks = vec![
            StreamingResponse {
                text: "Hello ".to_string(),
                chunk_tokens: 1,
                total_tokens: 2,
                metadata: HashMap::new(),
                timing: None,
                done: false,
            },
            StreamingResponse {
                text: "world!".to_string(),
                chunk_tokens: 1,
                total_tokens: 2,
                metadata: HashMap::new(),
                timing: None,
                done: true,
            },
        ];

        // Test with Query type (default)
        cache.put_stream("test_prompt", chunks.clone()).await.unwrap();
        let stream = cache.get_stream("test_prompt").await.unwrap();
        let received_chunks: Vec<_> = stream.collect().await;
        assert_eq!(received_chunks.len(), 2);
        assert_eq!(received_chunks[0].as_ref().unwrap().text, "Hello ");

        // Test with Extract type
        config.cache_type = CacheType::Extract;
        cache.update_config(config.clone()).unwrap();
        cache.put_stream("test_prompt", chunks.clone()).await.unwrap();
        let stream = cache.get_stream("test_prompt").await.unwrap();
        let received_chunks: Vec<_> = stream.collect().await;
        assert_eq!(received_chunks.len(), 2);
        assert_eq!(received_chunks[1].as_ref().unwrap().text, "world!");

        // Test with Custom type
        config.cache_type = CacheType::Custom("streaming_test".to_string());
        cache.update_config(config).unwrap();
        cache.put_stream("test_prompt", chunks.clone()).await.unwrap();
        let stream = cache.get_stream("test_prompt").await.unwrap();
        let received_chunks: Vec<_> = stream.collect().await;
        assert_eq!(received_chunks.len(), 2);
    }

    #[tokio::test]
    async fn test_streaming_cache() {
        let config = CacheConfig {
            enabled: true,
            stream_cache_enabled: true,
            ttl: Some(Duration::from_secs(60)),
            ..Default::default()
        };
        let cache = InMemoryCache::new(config);

        // Create test chunks
        let chunks = vec![
            StreamingResponse {
                text: "Hello ".to_string(),
                chunk_tokens: 1,
                total_tokens: 2,
                metadata: HashMap::new(),
                timing: None,
                done: false,
            },
            StreamingResponse {
                text: "world!".to_string(),
                chunk_tokens: 1,
                total_tokens: 2,
                metadata: HashMap::new(),
                timing: None,
                done: true,
            },
        ];

        // Test putting streaming response
        cache.put_stream("test_prompt", chunks.clone()).await.unwrap();

        // Test getting streaming response
        let stream = cache.get_stream("test_prompt").await.unwrap();
        let received_chunks: Vec<_> = stream.collect().await;
        assert_eq!(received_chunks.len(), 2);
        assert_eq!(received_chunks[0].as_ref().unwrap().text, "Hello ");
        assert_eq!(received_chunks[1].as_ref().unwrap().text, "world!");
    }

    #[tokio::test]
    async fn test_streaming_cache_expiration() {
        let config = CacheConfig {
            enabled: true,
            stream_cache_enabled: true,
            ttl: Some(Duration::from_millis(100)), // Very short TTL for normal entries
            stream_ttl: Some(Duration::from_millis(100)), // Very short TTL for streaming entries
            ..Default::default()
        };
        let cache = InMemoryCache::new(config);

        // Create test chunk
        let chunks = vec![StreamingResponse {
            text: "test".to_string(),
            chunk_tokens: 1,
            total_tokens: 1,
            metadata: HashMap::new(),
            timing: None,
            done: true,
        }];

        // Put streaming response
        cache.put_stream("test_prompt", chunks).await.unwrap();

        // Wait for expiration
        tokio::time::sleep(Duration::from_millis(200)).await;

        // Should return None for expired entry
        assert!(cache.get_stream("test_prompt").await.is_none());
    }

    #[tokio::test]
    async fn test_streaming_cache_disabled() {
        let config = CacheConfig {
            enabled: true,
            stream_cache_enabled: false, // Explicitly disable streaming cache
            ..Default::default()
        };
        let cache = InMemoryCache::new(config);

        let chunks = vec![StreamingResponse {
            text: "test".to_string(),
            chunk_tokens: 1,
            total_tokens: 1,
            metadata: HashMap::new(),
            timing: None,
            done: true,
        }];

        // Put should succeed but not actually store
        cache.put_stream("test_prompt", chunks).await.unwrap();

        // Should return None when streaming is disabled
        assert!(cache.get_stream("test_prompt").await.is_none());
    }
} 