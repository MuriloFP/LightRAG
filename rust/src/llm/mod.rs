use std::pin::Pin;
use futures::Stream;
use async_trait::async_trait;
use std::collections::HashMap;

// Re-export common types from types module
pub use crate::types::llm::{
    LLMParams,
    LLMResponse,
    LLMError,
    StreamingResponse,
    StreamingTiming,
};

/// Module providing caching functionality for LLM responses.
/// 
/// This module implements:
/// - In-memory caching with TTL support
/// - Cache configuration and management
/// - Thread-safe cache operations
pub mod cache;

/// Module for rate limiting functionality
pub mod rate_limiter;

/// Module for streaming support
pub mod streaming;

/// Module for Redis cache implementation
pub mod redis_cache;

/// Module for multi-model support
pub mod multi_model;

/// Module for configuration
pub mod config;

/// Module for error handling
pub mod error;

/// Module for provider trait definition
pub mod provider;

/// Module containing function-based implementations for different LLM providers.
/// 
/// Supported providers:
/// - OpenAI: For GPT models via openai_complete()
/// - Ollama: For local LLM deployment via ollama_complete()
/// - Anthropic: For Claude models via anthropic_complete()
pub mod providers;

// Re-export key components
pub use providers::*;
pub use cache::*;
pub use rate_limiter::*;
pub use redis_cache::RedisCache;
pub use multi_model::*;
pub use provider::Provider;

/// Configuration for LLM providers
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub api_key: Option<String>,
    pub api_endpoint: Option<String>,
    pub model: String,
    pub timeout_secs: u64,
    pub max_retries: u32,
    pub org_id: Option<String>,
    pub extra_config: std::collections::HashMap<String, String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_endpoint: None,
            model: "gpt-4".to_string(),
            timeout_secs: 30,
            max_retries: 3,
            org_id: None,
            extra_config: std::collections::HashMap::new(),
        }
    }
}

/// Factory function to create a provider instance
pub fn create_provider(provider_type: &str, config: ProviderConfig) -> Result<Box<dyn Provider>, LLMError> {
    match provider_type {
        "openai" => Ok(Box::new(providers::openai::OpenAIProvider::new(config)?)),
        "anthropic" => Ok(Box::new(providers::anthropic::AnthropicProvider::new(config)?)),
        "ollama" => Ok(Box::new(providers::ollama::OllamaProvider::new(config)?)),
        _ => Err(LLMError::ConfigError(format!("Unknown provider type: {}", provider_type))),
    }
} 