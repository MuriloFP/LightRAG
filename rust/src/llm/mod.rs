// Re-export common types from types module
pub use crate::types::llm::{
    LLMClient,
    LLMConfig,
    LLMParams,
    LLMResponse,
    LLMError,
};

/// Module providing caching functionality for LLM responses.
/// 
/// This module implements:
/// - In-memory caching with TTL support
/// - Cache configuration and management
/// - Thread-safe cache operations
pub mod cache;

/// Module containing implementations for different LLM providers.
/// 
/// Supported providers:
/// - OpenAI: For GPT models
/// - Ollama: For local LLM deployment
pub mod providers;

pub mod rate_limiter;

pub use providers::*;
pub use cache::*;
pub use rate_limiter::*;

pub mod redis_cache;

pub use redis_cache::RedisCache; 