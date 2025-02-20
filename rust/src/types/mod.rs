//! Core types and configuration for SuperLightRAG

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;

/// Custom error type for SuperLightRAG operations
#[derive(Error, Debug)]
pub enum Error {
    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Storage errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// API errors
    #[error("API error: {0}")]
    Api(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),
}

/// Result type for SuperLightRAG operations
pub type Result<T> = std::result::Result<T, Error>;

/// Configuration for SuperLightRAG
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Working directory for storage
    pub working_dir: PathBuf,

    /// Maximum memory usage in bytes
    pub max_memory: usize,

    /// Vector dimension for embeddings
    pub vector_dim: usize,

    /// Maximum number of nodes in graph
    pub max_nodes: usize,

    /// Maximum number of edges in graph
    pub max_edges: usize,

    /// Chunk size for text processing
    pub chunk_size: usize,

    /// Overlap size for text chunks
    pub chunk_overlap: usize,

    /// API configuration
    pub api_config: ApiConfig,
}

/// API provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// OpenAI API configuration
    pub openai: Option<OpenAIConfig>,

    /// Anthropic API configuration
    pub anthropic: Option<AnthropicConfig>,

    /// Custom API configuration
    pub custom: Option<CustomApiConfig>,
}

/// OpenAI API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAIConfig {
    /// API key
    pub api_key: String,

    /// Organization ID (optional)
    pub org_id: Option<String>,

    /// Model name for completions
    pub model: String,

    /// Model name for embeddings
    pub embedding_model: String,
}

/// Anthropic API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicConfig {
    /// API key
    pub api_key: String,

    /// Model name
    pub model: String,
}

/// Custom API configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomApiConfig {
    /// API endpoint
    pub endpoint: String,

    /// API key
    pub api_key: String,

    /// Additional headers
    pub headers: std::collections::HashMap<String, String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            working_dir: PathBuf::from("./super_lightrag_data"),
            max_memory: if cfg!(feature = "mobile") { 200 * 1024 * 1024 } else { 512 * 1024 * 1024 }, // mobile: 200MB, default: 512MB
            vector_dim: 1536,              // OpenAI's default
            max_nodes: 100_000,
            max_edges: 500_000,
            chunk_size: 1000,
            chunk_overlap: 100,
            api_config: ApiConfig {
                openai: None,
                anthropic: None,
                custom: None,
            },
        }
    }
} 