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
pub mod storage;
pub mod processing;
pub mod api;
pub mod types;
pub mod utils;
pub mod nano_vectordb;

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
        
        Ok(Self {
            kv_storage: Arc::new(RwLock::new(Box::new(storage::kv::JsonKVStorage::new(&config)?))),
            vector_storage: Arc::new(RwLock::new(Box::new(storage::vector::NanoVectorStorage::new(&config)?))),
            graph_storage: Arc::new(RwLock::new(Box::new(storage::graph::PetgraphStorage::new(&config)?))),
            config,
        })
    }

    /// Initializes all storage components
    pub async fn initialize(&self) -> Result<()> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_create_default() {
        let rag = SuperLightRAG::new().await.unwrap();
        assert!(rag.initialize().await.is_ok());
        assert!(rag.finalize().await.is_ok());
    }
} 