//! Core types and configuration for SuperLightRAG

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use thiserror::Error;
use std::collections::HashMap;

// Submodules
/// Language model (LLM) types and interfaces.
/// 
/// This module provides:
/// - Common types for LLM interactions
/// - Client trait definitions
/// - Configuration structures
/// - Error types specific to LLM operations
pub mod llm;

/// Embedding types and interfaces.
/// 
/// This module provides:
/// - Common types for embedding operations
/// - Provider trait definitions
/// - Configuration structures
/// - Error types specific to embedding operations
pub mod embeddings;

pub mod error;

// Re-exports
pub use error::{Error, Result};
pub use llm::{LLMParams, LLMResponse, LLMError};
pub use embeddings::{EmbeddingProvider, EmbeddingConfig, EmbeddingError, EmbeddingResponse};

/// A node in the knowledge graph with its properties and labels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphNode {
    /// Unique identifier for the node.
    pub id: String,
    /// List of labels/types associated with the node.
    pub labels: Vec<String>,
    /// Map of property key-value pairs associated with the node.
    pub properties: HashMap<String, serde_json::Value>,
}

/// An edge in the knowledge graph with its properties.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphEdge {
    /// Unique identifier for the edge.
    pub id: String,
    /// Type/label of the edge relationship.
    pub edge_type: Option<String>,
    /// ID of the source node.
    pub source: String,
    /// ID of the target node.
    pub target: String,
    /// Map of property key-value pairs associated with the edge.
    pub properties: HashMap<String, serde_json::Value>,
}

/// A knowledge graph containing nodes and their relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// List of nodes in the graph.
    pub nodes: Vec<KnowledgeGraphNode>,
    /// List of edges connecting the nodes.
    pub edges: Vec<KnowledgeGraphEdge>,
}

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

    /// Gets a boolean value from the config
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.extra_config.get(key).and_then(|v| v.as_bool())
    }

    /// Gets a u8 value from the config
    pub fn get_u8(&self, key: &str) -> Option<u8> {
        self.extra_config.get(key).and_then(|v| v.as_u64()).map(|v| v as u8)
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