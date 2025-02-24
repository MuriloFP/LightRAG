use std::collections::HashMap;
use thiserror::Error;
use serde::{Deserialize, Serialize};
use crate::processing::keywords::ConversationTurn;
use crate::llm::rate_limiter::RateLimitError;

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

    /// Whether to only need prompt without LLM processing
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

    /// Number of documents to retrieve
    pub num_docs: usize,

    /// Whether to include metadata
    pub include_metadata: bool,

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
            top_k: 3,
            similarity_threshold: 0.7,
            max_context_tokens: 2048,
            max_token_for_text_unit: 512,
            max_token_for_global_context: 1024,
            max_token_for_local_context: 1024,
            hl_keywords: None,
            ll_keywords: None,
            conversation_history: None,
            history_turns: None,
            num_docs: 3,
            include_metadata: false,
            extra_params: HashMap::new(),
        }
    }
}

/// Parameters for LLM requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMParams {
    /// Model identifier (e.g. "openai/gpt-4", "anthropic/claude-2")
    pub model: String,

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
            model: "openai/gpt-4".to_string(),
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

/// Error types for LLM operations
#[derive(Debug, Error)]
pub enum LLMError {
    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("Request failed: {0}")]
    RequestFailed(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Rate limit exceeded: {0}")]
    RateLimitExceeded(String),

    #[error("Token limit exceeded: {0}")]
    TokenLimitExceeded(String),

    #[error("Cache error: {0}")]
    CacheError(String),

    #[error("Streaming error: {0}")]
    StreamingError(String),

    #[error("Other error: {0}")]
    Other(String),
}

impl From<RateLimitError> for LLMError {
    fn from(err: RateLimitError) -> Self {
        match err {
            RateLimitError::LimitExceeded(msg) => LLMError::RateLimitExceeded(msg),
            RateLimitError::ConfigError(msg) => LLMError::ConfigError(msg),
        }
    }
} 