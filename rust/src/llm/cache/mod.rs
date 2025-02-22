pub mod config;
pub mod entry;
pub mod metrics;
pub mod memory;
pub mod types;

pub use config::*;
pub use entry::*;
pub use metrics::*;
pub use memory::*;
pub use types::CacheType;

use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use crate::llm::{LLMError, LLMResponse};
use crate::types::llm::StreamingResponse;

/// Trait for cache implementations
#[async_trait]
pub trait ResponseCache: Send + Sync {
    /// Get a cached response for a prompt
    async fn get(&self, prompt: &str) -> Option<LLMResponse>;
    
    /// Store a response in the cache
    async fn put(&self, prompt: &str, response: LLMResponse) -> Result<(), LLMError>;

    /// Get a cached streaming response
    async fn get_stream(&self, prompt: &str) -> Option<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>>;

    /// Store a streaming response in the cache
    async fn put_stream(&self, prompt: &str, chunks: Vec<StreamingResponse>) -> Result<(), LLMError>;
    
    /// Remove expired entries
    async fn cleanup(&self) -> Result<(), LLMError>;
    
    /// Clear all entries
    async fn clear(&self) -> Result<(), LLMError>;
    
    /// Get the current configuration
    fn get_config(&self) -> &CacheConfig;
    
    /// Update the configuration
    fn update_config(&mut self, config: CacheConfig) -> Result<(), LLMError>;
} 