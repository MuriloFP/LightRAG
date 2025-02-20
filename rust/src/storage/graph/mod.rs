use crate::types::Result;
use async_trait::async_trait;

/// Trait for graph storage operations
#[async_trait]
pub trait GraphStorage: Send + Sync {
    async fn initialize(&mut self) -> Result<()>;
    async fn finalize(&mut self) -> Result<()>;
}

/// Module for graph storage implementation using petgraph.
pub mod petgraph_storage;

// Re-export items from petgraph_storage for easier access
pub use petgraph_storage::{EdgeData, NodeData, PetgraphStorage};