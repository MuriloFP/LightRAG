pub mod config;
pub mod entry;
pub mod metrics;
pub mod types;
pub mod backend;
pub mod functions;
pub mod memory;
pub mod memory_backend;
pub mod tiered;
#[cfg(not(target_arch = "wasm32"))]
pub mod sqlite;
#[cfg(feature = "redis")]
pub mod redis;
#[cfg(target_arch = "wasm32")]
pub mod indexeddb;
#[cfg(target_arch = "wasm32")]
pub mod localstorage;
#[cfg(target_arch = "wasm32")]
pub mod offline;

pub use config::CacheConfig;
pub use entry::CacheEntry;
pub use metrics::CacheMetrics;
pub use types::{CacheType, CacheStats, StorageConfig, CacheValue};
pub use backend::{CacheBackend, CacheError, CacheCapabilities};
pub use functions::{init_cache, get_cached_response, store_response, get_cached_stream, store_stream, cleanup_cache, clear_cache, get_metrics};
pub use memory::MemoryCache;
#[cfg(feature = "redis")]
pub use redis::RedisCache;
#[cfg(not(target_arch = "wasm32"))]
pub use sqlite::SQLiteCache;
pub use tiered::TieredCache;
#[cfg(target_arch = "wasm32")]
pub use indexeddb::IndexedDBCache;
#[cfg(target_arch = "wasm32")]
pub use localstorage::LocalStorageCache;
#[cfg(target_arch = "wasm32")]
pub use offline::{OfflineManager, OfflineOperation};

/// Trait for implementing a response cache
#[async_trait::async_trait]
pub trait ResponseCache {
    /// Get a cached response for a prompt
    async fn get(&self, prompt: &str) -> Option<crate::llm::LLMResponse>;
    
    /// Store a response in the cache
    async fn put(&self, prompt: &str, response: crate::llm::LLMResponse) -> Result<(), crate::llm::LLMError>;
    
    /// Get a cached streaming response
    async fn get_stream(&self, prompt: &str) -> Option<std::pin::Pin<Box<dyn futures::Stream<Item = Result<crate::types::llm::StreamingResponse, crate::llm::LLMError>> + Send>>>;
    
    /// Store a streaming response in the cache
    async fn put_stream(&self, prompt: &str, chunks: Vec<crate::types::llm::StreamingResponse>) -> Result<(), crate::llm::LLMError>;
    
    /// Clean up expired entries
    async fn cleanup(&self) -> Result<(), crate::llm::LLMError>;
    
    /// Clear all entries
    async fn clear(&self) -> Result<(), crate::llm::LLMError>;

    /// Get cache configuration
    fn get_config(&self) -> &CacheConfig;
    
    /// Update cache configuration
    fn update_config(&mut self, config: CacheConfig) -> Result<(), crate::llm::LLMError>;
}