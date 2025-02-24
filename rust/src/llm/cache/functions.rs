use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::llm::{LLMError, LLMResponse};
use crate::types::llm::StreamingResponse;
use super::{CacheConfig, CacheEntry, CacheMetrics};
use futures::{Stream, stream};
use std::pin::Pin;

// Global cache state
static CACHE: tokio::sync::OnceCell<Arc<RwLock<HashMap<String, CacheEntry>>>> = tokio::sync::OnceCell::const_new();
static METRICS: tokio::sync::OnceCell<Arc<CacheMetrics>> = tokio::sync::OnceCell::const_new();

/// Initialize the cache with configuration
pub async fn init_cache(_config: CacheConfig) -> Result<(), LLMError> {
    CACHE.get_or_init(|| async { Arc::new(RwLock::new(HashMap::new())) }).await;
    METRICS.get_or_init(|| async { Arc::new(CacheMetrics::default()) }).await;
    Ok(())
}

/// Get a cached response
pub async fn get_cached_response(prompt: &str, config: &CacheConfig) -> Option<LLMResponse> {
    if !config.enabled {
        return None;
    }

    let cache = CACHE.get().unwrap();
    let entries = cache.read().await;
    let key = build_cache_key(prompt, config);
    
    if let Some(entry) = entries.get(&key) {
        if !is_entry_expired(entry) {
            METRICS.get().unwrap().hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Some(entry.response.clone());
        }
    }
    
    METRICS.get().unwrap().misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    None
}

/// Store a response in the cache
pub async fn store_response(prompt: &str, response: LLMResponse, config: &CacheConfig) -> Result<(), LLMError> {
    if !config.enabled {
        return Ok(());
    }

    let cache = CACHE.get().unwrap();
    let mut entries = cache.write().await;
    let key = build_cache_key(prompt, config);
    
    let entry = CacheEntry::new(
        response,
        config.ttl,
        None
    );
    
    entries.insert(key, entry);
    METRICS.get().unwrap().size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Get a cached streaming response
pub async fn get_cached_stream(
    prompt: &str,
    config: &CacheConfig
) -> Option<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>> {
    if !config.enabled {
        return None;
    }

    let cache = CACHE.get().unwrap();
    let entries = cache.read().await;
    let key = build_cache_key(prompt, config);
    
    if let Some(entry) = entries.get(&key) {
        if !is_entry_expired(entry) && entry.is_streaming() {
            if let Some(chunks) = entry.get_chunks() {
                METRICS.get().unwrap().hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let chunks = chunks.to_vec();
                return Some(Box::pin(stream::iter(chunks.into_iter().map(Ok))));
            }
        }
    }
    
    METRICS.get().unwrap().misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    None
}

/// Store a streaming response in the cache
pub async fn store_stream(
    prompt: &str,
    chunks: Vec<StreamingResponse>,
    config: &CacheConfig
) -> Result<(), LLMError> {
    if !config.enabled {
        return Ok(());
    }

    let cache = CACHE.get().unwrap();
    let mut entries = cache.write().await;
    let key = build_cache_key(prompt, config);
    
    let entry = CacheEntry::new_streaming(
        chunks,
        config.ttl,
        None
    );
    
    entries.insert(key, entry);
    METRICS.get().unwrap().size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// Clean up expired entries
pub async fn cleanup_cache(config: &CacheConfig) -> Result<(), LLMError> {
    if !config.enabled {
        return Ok(());
    }

    let cache = CACHE.get().unwrap();
    let mut entries = cache.write().await;
    let before_len = entries.len();
    
    entries.retain(|_, entry| !is_entry_expired(entry));
    
    let removed = before_len - entries.len();
    if removed > 0 {
        METRICS.get().unwrap().size.fetch_sub(removed, std::sync::atomic::Ordering::Relaxed);
    }
    
    Ok(())
}

/// Clear all entries from the cache
pub async fn clear_cache() -> Result<(), LLMError> {
    let cache = CACHE.get().unwrap();
    let mut entries = cache.write().await;
    let size = entries.len();
    
    entries.clear();
    METRICS.get().unwrap().size.fetch_sub(size, std::sync::atomic::Ordering::Relaxed);
    
    Ok(())
}

/// Get cache metrics
pub fn get_metrics() -> Arc<CacheMetrics> {
    METRICS.get().unwrap().clone()
}

// Helper functions
fn build_cache_key(prompt: &str, config: &CacheConfig) -> String {
    format!("{:?}:{}", config.cache_type, prompt)
}

fn is_entry_expired(entry: &CacheEntry) -> bool {
    if let Some(expires_at) = entry.expires_at {
        std::time::SystemTime::now() > expires_at
    } else {
        false
    }
} 