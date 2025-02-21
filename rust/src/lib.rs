//! SuperLightRAG - A lightweight, cross-platform implementation of LightRAG
//! 
//! This library provides a streamlined version of the LightRAG system,
//! optimized for mobile and desktop platforms.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

use std::sync::Arc;
use tokio::sync::RwLock;
// Module declarations
/// Storage module providing implementations for key-value, vector, and graph storage.
/// 
/// This module contains the core storage components used by SuperLightRAG:
/// - KV Storage: For storing and retrieving document metadata and content
/// - Vector Storage: For managing embeddings and similarity search
/// - Graph Storage: For maintaining relationships between documents and entities
pub mod storage;

/// Processing module for document and text manipulation.
/// 
/// Provides utilities for:
/// - Text chunking and tokenization
/// - Document parsing and cleaning
/// - Metadata extraction
/// - Content normalization
pub mod processing;

/// API module for external service integrations.
/// 
/// Handles communication with:
/// - Language models (LLMs)
/// - Embedding services
/// - External APIs
/// - Authentication and rate limiting
pub mod api;

/// LLM module for language model operations.
/// 
/// Provides:
/// - LLM client interfaces
/// - Response caching
/// - Provider implementations (OpenAI, Ollama)
/// - Error handling and retries
pub mod llm;

/// Common types and configuration structures.
/// 
/// Contains:
/// - Error types and Result aliases
/// - Configuration structures
/// - Common data types used across modules
/// - Type conversion traits
pub mod types;

/// Utility functions and helper traits.
/// 
/// Provides:
/// - Common helper functions
/// - Extension traits
/// - Logging utilities
/// - Testing helpers
pub mod utils;

/// NanoVectorDB module providing a lightweight vector database implementation.
/// 
/// This module contains a simplified version of a vector database optimized for:
/// - Low memory footprint
/// - Fast similarity search
/// - Cross-platform compatibility
/// - Persistence capabilities
pub mod nano_vectordb;

/// Embeddings module for handling embeddings and similarity search.
/// 
/// Provides:
/// - Embedding generation
/// - Similarity search
/// - Embedding storage
pub mod embeddings;

// Re-exports
pub use crate::types::{Error, Result};

/// Main SuperLightRAG struct that coordinates all operations
#[derive(Clone)]
pub struct SuperLightRAG {
    // Storage components
    kv_storage: Arc<RwLock<Box<dyn storage::kv::KVStorage>>>,
    vector_storage: Arc<RwLock<Box<dyn storage::vector::VectorStorage>>>,
    graph_storage: Arc<RwLock<Box<dyn storage::graph::GraphStorage>>>,
    
    // Configuration
    config: Arc<types::Config>,
}

impl SuperLightRAG {
    /// Creates a new instance with default configuration
    pub async fn new() -> Result<Self> {
        Self::with_config(types::Config::default()).await
    }

    /// Creates a new instance with custom configuration
    pub async fn with_config(config: types::Config) -> Result<Self> {
        let config = Arc::new(config);
        
        // Create KV storage with appropriate namespaces
        let kv_storage = Arc::new(RwLock::new(Box::new(storage::kv::JsonKVStorage::new(&config, "kv_store_full_docs")?) as Box<dyn storage::kv::KVStorage>));
        let vector_storage = Arc::new(RwLock::new(Box::new(storage::vector::NanoVectorStorage::new(&config)?) as Box<dyn storage::vector::VectorStorage>));
        let graph_storage = Arc::new(RwLock::new(Box::new(storage::graph::PetgraphStorage::new(&config)?) as Box<dyn storage::graph::GraphStorage>));

        Ok(Self {
            kv_storage,
            vector_storage,
            graph_storage,
            config,
        })
    }

    /// Initializes all storage components
    pub async fn initialize(&self) -> Result<()> {
        // Ensure working directory exists
        self.config.ensure_working_dir()?;

        let mut kv = self.kv_storage.write().await;
        let mut vector = self.vector_storage.write().await;
        let mut graph = self.graph_storage.write().await;

        kv.initialize().await?;
        vector.initialize().await?;
        graph.initialize().await?;

        Ok(())
    }

    /// Finalizes all storage components
    pub async fn finalize(&self) -> Result<()> {
        let mut kv = self.kv_storage.write().await;
        let mut vector = self.vector_storage.write().await;
        let mut graph = self.graph_storage.write().await;

        kv.finalize().await?;
        vector.finalize().await?;
        graph.finalize().await?;

        Ok(())
    }

    /// Creates a new KV storage instance with the given namespace
    pub async fn create_kv_storage(&self, namespace: &str) -> Result<Arc<RwLock<Box<dyn storage::kv::KVStorage>>>> {
        Ok(Arc::new(RwLock::new(Box::new(storage::kv::JsonKVStorage::new(&self.config, namespace)?) as Box<dyn storage::kv::KVStorage>)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::HashMap;
    use tempfile::TempDir;
    use crate::types::Config;

    #[tokio::test]
    async fn test_create_default() {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
        let rag = SuperLightRAG::with_config(config).await.unwrap();
        assert!(rag.initialize().await.is_ok());
        assert!(rag.finalize().await.is_ok());
    }

    #[tokio::test]
    async fn test_kv_storage_with_namespace() -> Result<()> {
        let rag = SuperLightRAG::new().await?;
        let kv_storage = rag.create_kv_storage("test_namespace").await?;
        
        // Test basic operations
        let mut storage = kv_storage.write().await;
        storage.initialize().await?;
        
        let mut data = HashMap::new();
        data.insert("test_key".to_string(), {
            let mut inner_data = HashMap::new();
            inner_data.insert("field".to_string(), json!("test_value"));
            inner_data
        });
        storage.upsert(data).await?;
        
        let value = storage.get_by_id("test_key").await?;
        assert_eq!(value, Some({
            let mut inner_data = HashMap::new();
            inner_data.insert("field".to_string(), json!("test_value"));
            inner_data
        }));
        
        storage.finalize().await?;
        Ok(())
    }
} 