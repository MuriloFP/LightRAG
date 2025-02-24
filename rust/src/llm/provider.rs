use std::pin::Pin;
use futures::Stream;
use async_trait::async_trait;
use crate::types::llm::{LLMParams, LLMResponse, LLMError, StreamingResponse};
use super::ProviderConfig;

/// Trait for LLM provider implementations
#[async_trait]
pub trait Provider: Send + Sync {
    /// Initialize the provider
    async fn initialize(&mut self) -> Result<(), LLMError>;
    
    /// Generate text completion
    async fn complete(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError>;
    
    /// Generate text completion with streaming
    async fn complete_stream(&self, prompt: &str, params: &LLMParams) 
        -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError>;
    
    /// Generate text completion in batch
    async fn complete_batch(&self, prompts: &[String], params: &LLMParams) -> Result<Vec<LLMResponse>, LLMError>;
    
    /// Get provider configuration
    fn get_config(&self) -> &ProviderConfig;
    
    /// Update provider configuration
    fn update_config(&mut self, config: ProviderConfig) -> Result<(), LLMError>;
} 