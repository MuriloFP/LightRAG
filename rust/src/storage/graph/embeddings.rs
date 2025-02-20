use crate::types::{Result, Error};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::visit::EdgeRef;
use node2vec::node2vec::embed;
use std::collections::HashMap;
use petgraph::Directed;
use std::str::FromStr;

/// Supported node embedding algorithms
#[derive(Debug, Clone, Copy)]
pub enum EmbeddingAlgorithm {
    /// Node2Vec embedding algorithm
    Node2Vec,
    // Add more algorithms here as needed
}

impl FromStr for EmbeddingAlgorithm {
    type Err = Error;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "node2vec" => Ok(EmbeddingAlgorithm::Node2Vec),
            _ => Err(Error::InvalidInput(format!("Unsupported embedding algorithm: {}", s))),
        }
    }
}

/// Configuration for Node2Vec embedding
#[derive(Debug, Clone)]
pub struct Node2VecConfig {
    /// Number of dimensions in the output embedding
    pub dimensions: usize,
    /// Length of the random walks
    pub walk_length: usize,
    /// Number of random walks per node
    pub num_walks: usize,
    /// Return parameter (p) controls likelihood of returning to previous node
    pub p: f32,
    /// In-out parameter (q) controls search strategy
    pub q: f32,
    /// Window size for skip-gram
    pub window_size: usize,
    /// Number of iterations
    pub iter: usize,
}

impl Default for Node2VecConfig {
    fn default() -> Self {
        Self {
            dimensions: 128,
            walk_length: 80,
            num_walks: 10,
            p: 1.0,
            q: 1.0,
            window_size: 10,
            iter: 1,
        }
    }
}

/// Generates node embeddings using the Node2Vec algorithm
pub fn generate_node2vec_embeddings<N, E>(
    graph: &Graph<N, E>,
    config: &Node2VecConfig,
) -> Result<(Vec<f32>, Vec<NodeIndex>)> {
    // Create a new graph with the required types
    let mut converted_graph: Graph<usize, f32, Directed> = Graph::new();
    let mut node_map = HashMap::new();
    let mut reverse_node_map = HashMap::new();
    
    // Add nodes
    for idx in graph.node_indices() {
        let new_idx = converted_graph.add_node(idx.index());
        node_map.insert(idx, new_idx);
        reverse_node_map.insert(new_idx, idx);
    }
    
    // Add edges
    for edge in graph.edge_references() {
        let source = node_map[&edge.source()];
        let target = node_map[&edge.target()];
        converted_graph.add_edge(source, target, 1.0);
    }
    
    // Generate embeddings using node2vec
    if converted_graph.edge_count() == 0 {
        let num_nodes = converted_graph.node_count();
        let embeddings = vec![0.0; num_nodes * config.dimensions];
        let mut sorted_indices: Vec<_> = reverse_node_map.keys().cloned().collect();
        sorted_indices.sort();
        let mut node_indices = Vec::new();
        for idx in sorted_indices {
            // Safe to unwrap because idx comes from reverse_node_map keys
            node_indices.push(*reverse_node_map.get(&idx).unwrap());
        }
        return Ok((embeddings, node_indices));
    }

    let embeddings = embed(
        &converted_graph,
        config.num_walks,
        config.walk_length,
        Some(config.dimensions)
    ).map_err(|e| Error::Storage(format!("Node2Vec embedding error: {}", e)))?;
    
    // Convert embeddings to the format we need
    let mut result_embeddings = Vec::new();
    let mut node_indices = Vec::new();

    // Iterate over all nodes in sorted order based on the new node indices
    let mut all_new_indices: Vec<_> = reverse_node_map.keys().cloned().collect();
    all_new_indices.sort();

    for new_idx in all_new_indices {
        // If an embedding exists for this node, use it; otherwise, use zeros
        if let Some(embedding) = embeddings.get(&new_idx) {
            result_embeddings.extend(embedding);
        } else {
            result_embeddings.extend(vec![0.0; config.dimensions]);
        }
        // Map back to the original node index (safe to unwrap because new_idx is from reverse_node_map keys)
        node_indices.push(*reverse_node_map.get(&new_idx).unwrap());
    }

    Ok((result_embeddings, node_indices))
}

/// Trait for graph embedding algorithms
pub trait GraphEmbedding {
    /// Generate embeddings for the graph nodes
    fn embed(&self, dimensions: usize) -> Result<(Vec<f32>, Vec<String>)>;
} 