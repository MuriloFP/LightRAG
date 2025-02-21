use std::collections::HashMap;
use thiserror::Error;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use crate::processing::keywords::ConversationTurn;
use std::sync::Arc;
use tokio::sync::RwLock;
use crate::llm::rate_limiter::{RateLimiter, RateLimit, RateLimitConfig};

/// Query mode for RAG operations
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QueryMode {
    /// Local context from nearby chunks
    Local,
    /// Global context from knowledge graph
    Global,
    /// Hybrid approach combining local and global context
    Hybrid,
    /// Naive approach using direct similarity search
    Naive,
    /// Mix mode combining graph and vector retrieval with weighted scoring
    Mix,
}

impl Default for QueryMode {
    fn default() -> Self {
        Self::Hybrid
    }
}

/// Parameters for RAG queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    /// Query mode to use
    pub mode: QueryMode,

    /// Whether to stream the response
    pub stream: bool,

    /// Whether to only return context without LLM processing
    pub only_need_context: bool,

    /// Whether to only return prompt without LLM processing
    pub only_need_prompt: bool,

    /// Response type (e.g. "Multiple Paragraphs", "Single Paragraph", "Bullet Points")
    pub response_type: String,

    /// Number of top results to consider
    pub top_k: usize,

    /// Minimum similarity threshold for results
    pub similarity_threshold: f32,

    /// Maximum tokens for context window
    pub max_context_tokens: usize,

    /// Maximum tokens for text chunks
    pub max_token_for_text_unit: usize,

    /// Maximum tokens for global context
    pub max_token_for_global_context: usize,

    /// Maximum tokens for local context
    pub max_token_for_local_context: usize,

    /// High-level keywords for retrieval
    pub hl_keywords: Option<Vec<String>>,

    /// Low-level keywords for retrieval
    pub ll_keywords: Option<Vec<String>>,

    /// Conversation history
    pub conversation_history: Option<Vec<ConversationTurn>>,

    /// Number of conversation history turns to include
    pub history_turns: Option<usize>,

    /// Additional parameters
    pub extra_params: HashMap<String, String>,
}

impl Default for QueryParams {
    fn default() -> Self {
        Self {
            mode: QueryMode::default(),
            stream: false,
            only_need_context: false,
            only_need_prompt: false,
            response_type: "Multiple Paragraphs".to_string(),
            top_k: 5,
            similarity_threshold: 0.7,
            max_context_tokens: 3000,
            max_token_for_text_unit: 4000,
            max_token_for_global_context: 4000,
            max_token_for_local_context: 4000,
            hl_keywords: None,
            ll_keywords: None,
            conversation_history: None,
            history_turns: Some(3),
            extra_params: HashMap::new(),
        }
    }
}

/// Errors that can occur during LLM operations
#[derive(Error, Debug)]
pub enum LLMError {
    /// API request failed
    #[error("API request failed: {0}")]
    RequestFailed(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    /// Invalid response format
    #[error("Invalid response format: {0}")]
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
}

/// Parameters for LLM requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMParams {
    /// Maximum number of tokens to generate
    pub max_tokens: usize,

    /// Temperature for generation (0.0 to 1.0)
    pub temperature: f32,

    /// Top-p sampling parameter
    pub top_p: f32,

    /// Whether to use streaming response
    pub stream: bool,

    /// System prompt (if supported by model)
    pub system_prompt: Option<String>,

    /// Query parameters for RAG
    pub query_params: Option<QueryParams>,

    /// Additional model-specific parameters
    pub extra_params: HashMap<String, String>,
}

impl Default for LLMParams {
    fn default() -> Self {
        Self {
            max_tokens: 1000,
            temperature: 0.7,
            top_p: 1.0,
            stream: false,
            system_prompt: None,
            query_params: None,
            extra_params: HashMap::new(),
        }
    }
}

/// Configuration for LLM client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
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

    /// Rate limiting configuration
    pub rate_limit_config: Option<RateLimitConfig>,

    /// Additional configuration parameters
    pub extra_config: HashMap<String, String>,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            model: String::from("gpt-3.5-turbo"),
            api_endpoint: None,
            api_key: None,
            org_id: None,
            timeout_secs: 30,
            max_retries: 3,
            use_cache: true,
            rate_limit_config: Some(RateLimitConfig::default()),
            extra_config: HashMap::new(),
        }
    }
}

/// Response from LLM generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMResponse {
    /// Generated text
    pub text: String,

    /// Number of tokens used
    pub tokens_used: usize,

    /// Model used for generation
    pub model: String,

    /// Whether the response was cached
    pub cached: bool,

    /// Context used for generation (if any)
    pub context: Option<String>,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Streaming response chunk from LLM generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingResponse {
    /// Generated text chunk
    pub text: String,

    /// Whether this is the final chunk
    pub done: bool,

    /// Timing information for this chunk
    pub timing: Option<StreamingTiming>,

    /// Number of tokens in this chunk
    pub chunk_tokens: usize,

    /// Total tokens used so far
    pub total_tokens: usize,

    /// Additional metadata for this chunk
    pub metadata: HashMap<String, String>,
}

/// Timing information for streaming responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingTiming {
    /// Time taken to generate this chunk in microseconds
    pub chunk_duration: u64,

    /// Total duration since start in microseconds
    pub total_duration: u64,

    /// Time taken for prompt evaluation in microseconds
    pub prompt_eval_duration: Option<u64>,

    /// Time taken for token generation in microseconds
    pub token_gen_duration: Option<u64>,
}

/// Trait for LLM clients
#[async_trait]
pub trait LLMClient: Send + Sync {
    /// Initialize the client
    async fn initialize(&mut self) -> Result<(), LLMError>;

    /// Generate text from a prompt
    async fn generate(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError>;

    /// Generate text with conversation history
    async fn generate_with_history(
        &self,
        prompt: &str,
        _history: &[ConversationTurn],
        params: &LLMParams
    ) -> Result<LLMResponse, LLMError> {
        // Default implementation falls back to regular generate
        self.generate(prompt, params).await
    }

    /// Generate text with streaming response
    async fn generate_stream(
        &self,
        prompt: &str,
        params: &LLMParams
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        // Default implementation returns error if streaming not supported
        Err(LLMError::ConfigError("Streaming not supported by this client".to_string()))
    }

    /// Generate text with context building
    async fn generate_with_context(
        &self,
        prompt: &str,
        context: &str,
        params: &LLMParams
    ) -> Result<LLMResponse, LLMError> {
        // Default implementation combines context with prompt
        let full_prompt = format!("Context:\n{}\n\nQuery:\n{}", context, prompt);
        self.generate(&full_prompt, params).await
    }

    /// Generate text for multiple prompts in batch
    async fn batch_generate(
        &self,
        prompts: &[String],
        params: &LLMParams,
    ) -> Result<Vec<LLMResponse>, LLMError>;

    /// Get the current configuration
    fn get_config(&self) -> &LLMConfig;

    /// Update the configuration
    fn update_config(&mut self, config: LLMConfig) -> Result<(), LLMError>;

    /// Check rate limits and acquire permission
    async fn check_rate_limit(&self, tokens: u32) -> Result<RateLimit, LLMError> {
        if let Some(rate_limit_config) = &self.get_config().rate_limit_config {
            let rate_limiter = RateLimiter::new(rate_limit_config.clone());
            rate_limiter.try_acquire(tokens).await.map_err(|e| LLMError::RateLimitExceeded(e.to_string()))
        } else {
            Ok(RateLimit::new(Arc::new(RwLock::new(0))))
        }
    }
} 