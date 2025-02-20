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

    /// Vector storage specific errors
    #[error("Vector storage error: {0}")]
    VectorStorage(String),

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

    /// Additional configuration values
    #[serde(default)]
    pub extra_config: std::collections::HashMap<String, serde_json::Value>,
}

impl Config {
    /// Gets a float value from the extra configuration
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.extra_config.get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
    }

    /// Gets a usize value from the extra configuration
    pub fn get_usize(&self, key: &str) -> Option<usize> {
        self.extra_config.get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Ensures the working directory exists
    pub fn ensure_working_dir(&self) -> Result<()> {
        if !self.working_dir.exists() {
            std::fs::create_dir_all(&self.working_dir)
                .map_err(|e| Error::Config(format!("Failed to create working directory: {}", e)))?;
        }
        Ok(())
    }
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
        let config = Self {
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
            extra_config: std::collections::HashMap::new(),
        };
        
        // Ensure working directory exists
        if let Err(e) = config.ensure_working_dir() {
            eprintln!("Warning: Failed to create working directory: {}", e);
        }
        
        config
    }
} 