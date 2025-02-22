use crate::types::Result;
use async_trait::async_trait;
use std::collections::HashMap;
use crate::storage::graph::embeddings::EmbeddingAlgorithm;
use serde_json::Value;
use crate::types::KnowledgeGraph;

/// Trait for graph storage operations
#[async_trait]
pub trait GraphStorage: Send + Sync {
    /// Query the graph using keywords
    async fn query_with_keywords(&self, keywords: &[String]) -> Result<String>;

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

    /// Checks if a node exists in the graph
    async fn has_node(&self, node_id: &str) -> bool;

    /// Checks if an edge exists between two nodes
    async fn has_edge(&self, source_id: &str, target_id: &str) -> bool;

    /// Gets node data by ID
    async fn get_node(&self, node_id: &str) -> Option<NodeData>;

    /// Gets edge data between two nodes
    async fn get_edge(&self, source_id: &str, target_id: &str) -> Option<EdgeData>;

    /// Gets all edges connected to a node
    async fn get_node_edges(&self, node_id: &str) -> Option<Vec<(String, String, EdgeData)>>;

    /// Gets the total degree (in + out) of a node
    async fn node_degree(&self, node_id: &str) -> Result<usize>;

    /// Gets the total degree of an edge (sum of degrees of its nodes)
    async fn edge_degree(&self, src_id: &str, tgt_id: &str) -> Result<usize>;

    /// Inserts or updates a node
    async fn upsert_node(&mut self, node_id: &str, attributes: HashMap<String, Value>) -> Result<()>;

    /// Inserts or updates an edge
    async fn upsert_edge(&mut self, source_id: &str, target_id: &str, data: EdgeData) -> Result<()>;

    /// Deletes a node and its edges
    async fn delete_node(&mut self, node_id: &str) -> Result<()>;

    /// Deletes an edge
    async fn delete_edge(&mut self, source_id: &str, target_id: &str) -> Result<()>;

    /// Batch node insertion/update
    async fn upsert_nodes(&mut self, nodes: Vec<(String, HashMap<String, Value>)>) -> Result<()>;

    /// Batch edge insertion/update
    async fn upsert_edges(&mut self, edges: Vec<(String, String, EdgeData)>) -> Result<()>;

    /// Batch node deletion
    async fn remove_nodes(&mut self, node_ids: Vec<String>) -> Result<()>;

    /// Batch edge deletion
    async fn remove_edges(&mut self, edges: Vec<(String, String)>) -> Result<()>;

    /// Generates embeddings for all nodes
    async fn embed_nodes(&self, algorithm: EmbeddingAlgorithm) -> Result<(Vec<f32>, Vec<String>)>;

    /// Gets all labels in the graph
    async fn get_all_labels(&self) -> Result<Vec<String>>;

    /// Gets a knowledge graph starting from a node
    async fn get_knowledge_graph(&self, node_label: &str, max_depth: i32) -> Result<KnowledgeGraph>;
}

/// Module for graph storage implementation using petgraph.
pub mod petgraph_storage;

// Re-export items from petgraph_storage for easier access
pub use petgraph_storage::{EdgeData, NodeData, PetgraphStorage};

/// Module for graph node embedding algorithms and configurations.
/// 
/// This module provides:
/// - Node2Vec embedding implementation
/// - Embedding algorithm configurations
/// - Embedding generation utilities
/// - Graph embedding traits and types
pub mod embeddings;

/// Module for GraphML format handling.
pub mod graphml;