use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use async_trait::async_trait;

/// Errors that can occur during embedding operations
#[derive(Error, Debug)]
pub enum EmbeddingError {
    /// API request failed
    #[error("API request failed: {0}")]
    RequestFailed(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid response from provider
    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    /// Token limit exceeded
    #[error("Token limit exceeded: {0}")]
    TokenLimitExceeded(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Cache error
    #[error("Cache error: {0}")]
    CacheError(String),

    /// Failed to initialize provider
    #[error("Failed to initialize provider: {0}")]
    InitializationError(String),

    /// Failed to parse response
    #[error("Failed to parse response: {0}")]
    ParseError(String),

    /// Failed to quantize embedding
    #[error("Failed to quantize embedding: {0}")]
    QuantizationError(String),

    /// Failed to compress/decompress data
    #[error("Failed to compress/decompress data: {0}")]
    CompressionError(String),
}

/// Configuration for embedding operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// Model identifier/name
    pub model: String,

    /// API endpoint (if applicable)
    pub api_endpoint: Option<String>,

    /// API key (if required)
    pub api_key: Option<String>,

    /// Organization ID (if applicable)
    pub org_id: Option<String>,

    /// Timeout in seconds
    pub timeout_secs: u64,

    /// Maximum retries
    pub max_retries: u32,

    /// Whether to use caching
    pub use_cache: bool,

    /// Batch size for embedding operations
    pub batch_size: usize,

    /// Maximum concurrent requests
    pub max_concurrent_requests: usize,

    /// Cache configuration
    pub cache_config: Option<CacheConfig>,

    /// Rate limiting configuration
    pub rate_limit_config: Option<RateLimitConfig>,

    /// Additional configuration parameters
    pub extra_config: HashMap<String, String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: String::from("text-embedding-ada-002"),
            api_endpoint: None,
            api_key: None,
            org_id: None,
            timeout_secs: 30,
            max_retries: 3,
            use_cache: true,
            batch_size: 32,
            max_concurrent_requests: 8,
            cache_config: Some(CacheConfig::default()),
            rate_limit_config: Some(RateLimitConfig::default()),
            extra_config: HashMap::new(),
        }
    }
}

/// Configuration for embedding cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether caching is enabled
    pub enabled: bool,

    /// Similarity threshold for cache hits (0.0 to 1.0)
    pub similarity_threshold: f32,

    /// Whether to use LLM verification for cache hits
    pub use_llm_verification: bool,

    /// Time-to-live in seconds (None for no expiry)
    pub ttl_seconds: Option<u64>,

    /// Maximum cache size (None for unlimited)
    pub max_size: Option<usize>,

    /// Whether to use quantization for embeddings
    pub use_quantization: bool,

    /// Number of bits for quantization (default: 8)
    pub quantization_bits: u8,

    /// Whether to use compression for embeddings
    pub use_compression: bool,

    /// Maximum size of compressed data in bytes (None for unlimited)
    pub max_compressed_size: Option<usize>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            similarity_threshold: 0.95,
            use_llm_verification: false,
            ttl_seconds: Some(3600 * 24), // 24 hours
            max_size: Some(10000),
            use_quantization: true,
            quantization_bits: 8,
            use_compression: true,
            max_compressed_size: Some(1024 * 1024), // 1MB
        }
    }
}

/// Configuration for rate limiting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests per minute
    pub requests_per_minute: u32,
    
    /// Maximum tokens per minute
    pub tokens_per_minute: u32,
    
    /// Maximum concurrent requests
    pub max_concurrent: u32,
    
    /// Whether to use token bucket algorithm
    pub use_token_bucket: bool,
    
    /// Burst size for token bucket
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_minute: 150_000, // Higher limit for embeddings
            max_concurrent: 10,
            use_token_bucket: true,
            burst_size: 5,
        }
    }
}

/// Response from embedding generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The generated embedding vector
    pub embedding: Vec<f32>,

    /// Number of tokens used
    pub tokens_used: usize,

    /// Model used for generation
    pub model: String,

    /// Whether the response was cached
    pub cached: bool,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Trait for embedding providers
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Initialize the provider
    async fn initialize(&mut self) -> Result<(), EmbeddingError>;

    /// Generate embeddings for a single text
    async fn embed(&self, text: &str) -> Result<EmbeddingResponse, EmbeddingError>;

    /// Generate embeddings for multiple texts
    async fn batch_embed(
        &self,
        texts: &[String],
    ) -> Result<Vec<EmbeddingResponse>, EmbeddingError> {
        let mut responses = Vec::with_capacity(texts.len());
        for text in texts {
            responses.push(self.embed(text).await?);
        }
        Ok(responses)
    }

    /// Get the current configuration
    fn get_config(&self) -> &EmbeddingConfig;

    /// Update the configuration
    fn update_config(&mut self, config: EmbeddingConfig) -> Result<(), EmbeddingError>;
} 