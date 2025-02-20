use crate::types::Result;
use async_trait::async_trait;

/// Trait for graph storage operations
#[async_trait]
pub trait GraphStorage: Send + Sync {
    /// Initializes the graph storage system.
    /// 
    /// This method should be called before any other operations to:
    /// - Load persisted data from disk
    /// - Set up any required data structures
    /// - Initialize connections or resources
    /// 
    /// # Returns
    /// A Result indicating success or failure of the initialization
    async fn initialize(&mut self) -> Result<()>;

    /// Finalizes the graph storage system.
    /// 
    /// This method should be called when shutting down to:
    /// - Persist data to disk
    /// - Clean up resources
    /// - Close any open connections
    /// 
    /// # Returns
    /// A Result indicating success or failure of the finalization
    async fn finalize(&mut self) -> Result<()>;
}

/// Module for graph storage implementation using petgraph.
pub mod petgraph_storage;

// Re-export items from petgraph_storage for easier access
pub use petgraph_storage::{EdgeData, NodeData, PetgraphStorage};