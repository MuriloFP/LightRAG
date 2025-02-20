use std::collections::HashMap;

/// A node in the HNSW index.
#[derive(Debug, Clone)]
pub struct HNSWNode {
    /// Unique identifier of the node.
    pub id: String,
    /// The vector representation of the node.
    pub vector: Vec<f32>,
    /// Connections represented as: layer -> Vec<node indices>
    pub connections: HashMap<usize, Vec<usize>>,
}

impl HNSWNode {
    /// Creates a new HNSW node with the given id and vector.
    pub fn new(id: &str, vector: Vec<f32>) -> Self {
        HNSWNode {
            id: id.to_string(),
            vector,
            connections: HashMap::new(),
        }
    }
}

/// HNSW Index structure for approximate nearest neighbor search.
#[derive(Debug)]
pub struct HNSWIndex {
    /// The collection of nodes in the index.
    pub nodes: Vec<HNSWNode>,
    /// The entry point index for search.
    pub entry_point: Option<usize>,
    /// Maximum number of layers in the index.
    pub max_layers: usize,
    /// The ef construction parameter: number of candidate neighbors during construction.
    pub ef_construction: usize,
    /// The maximum number of connections (M) per node at each layer.
    pub m: usize,
}

impl HNSWIndex {
    /// Creates a new HNSW index with the given parameters.
    pub fn new(max_layers: usize, ef_construction: usize, m: usize) -> Self {
        HNSWIndex {
            nodes: Vec::new(),
            entry_point: None,
            max_layers,
            ef_construction,
            m,
        }
    }

    /// Adds a new node to the HNSW index and returns its index.
    pub fn add_node(&mut self, node: HNSWNode) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(node);
        // Set the first node added as the entry point.
        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
        }
        idx
    }

    /// Computes the cosine similarity between two vectors.
    pub fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
        let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
        let norm1 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
        let norm2 = v2.iter().map(|a| a * a).sum::<f32>().sqrt();
        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot / (norm1 * norm2)
        }
    }

    /// Queries the HNSW index with the given query vector.
    /// This is a simple placeholder implementation that performs a linear scan over the nodes.
    /// In a full HNSW implementation, the search would navigate through layers for efficiency.
    pub fn query(&self, query: &[f32], top_k: usize, _search_ef: usize) -> Vec<(String, f32)> {
        let mut results: Vec<(String, f32)> = self.nodes
            .iter()
            .map(|node| {
                let sim = Self::cosine_similarity(query, &node.vector);
                (node.id.clone(), sim)
            })
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.into_iter().take(top_k).collect()
    }

    /// Deletes nodes from the HNSW index whose ids are in the provided list.
    pub fn delete_nodes(&mut self, delete_ids: &[String]) {
        self.nodes.retain(|node| !delete_ids.contains(&node.id));
        // Update entry point: if index becomes empty, set to None; otherwise, use the first node.
        if self.nodes.is_empty() {
            self.entry_point = None;
        } else {
            self.entry_point = Some(0);
        }
    }
} 