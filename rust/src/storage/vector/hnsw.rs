use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::fs;
use crate::types::{Result, Error};

/// A node in the HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWNode {
    /// Unique identifier of the node.
    pub id: String,
    /// The vector representation of the node.
    pub vector: Vec<f32>,
    /// Connections represented as: layer -> Vec<node indices>
    #[serde(default)]
    pub connections: HashMap<usize, Vec<usize>>,
    /// The maximum layer this node appears in
    #[serde(default)]
    pub max_layer: usize,
}

impl HNSWNode {
    /// Creates a new HNSW node with the given id and vector.
    pub fn new(id: &str, vector: Vec<f32>) -> Self {
        let mut node = HNSWNode {
            id: id.to_string(),
            vector,
            connections: HashMap::new(),
            max_layer: 0,
        };
        // Initialize connections for layer 0
        node.connections.insert(0, Vec::new());
        node
    }
}

/// A candidate node during search with its distance
#[derive(Debug, Clone)]
struct Candidate {
    node_idx: usize,
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance.eq(&other.distance)
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
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
    /// Batch size for index updates
    pub batch_size: usize,
    /// Number of updates before forcing index synchronization
    pub sync_threshold: usize,
    /// Number of updates since last sync
    updates_since_sync: usize,
    /// File path for persistence
    file_path: Option<PathBuf>,
    /// Vector dimension
    dimension: Option<usize>,
}

/// Serialization format for HNSW index
#[derive(Serialize, Deserialize)]
struct HNSWIndexSerialized {
    nodes: Vec<HNSWNode>,
    entry_point: Option<usize>,
    max_layers: usize,
    ef_construction: usize,
    m: usize,
    batch_size: usize,
    sync_threshold: usize,
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
            batch_size: 100,
            sync_threshold: 1000,
            updates_since_sync: 0,
            file_path: None,
            dimension: None,
        }
    }

    /// Set the file path for persistence
    pub fn set_persistence_path(&mut self, path: PathBuf) {
        self.file_path = Some(path);
    }

    /// Load the index from file
    pub fn load(&mut self) -> Result<()> {
        if let Some(path) = &self.file_path {
            if path.exists() {
                let content = fs::read_to_string(path)?;
                if !content.trim().is_empty() {
                    let loaded: Vec<HNSWNode> = serde_json::from_str(&content)?;
                    self.nodes = loaded;
                    self.entry_point = if self.nodes.is_empty() { None } else { Some(0) };
                    self.updates_since_sync = 0;
                }
            }
        }
        Ok(())
    }

    /// Save the index to file
    pub fn save(&self) -> Result<()> {
        if let Some(path) = &self.file_path {
            // Ensure parent directory exists
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let content = serde_json::to_string_pretty(&self.nodes)?;
            fs::write(path, content)?;
        }
        Ok(())
    }

    /// Adds a new node to the HNSW index and returns its index.
    pub fn add_node(&mut self, mut node: HNSWNode) -> Result<usize> {
        // Validate vector dimension
        let vec_dim = node.vector.len();
        if let Some(dim) = self.dimension {
            if vec_dim != dim {
                return Err(Error::VectorStorage(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    dim, vec_dim
                )));
            }
        } else {
            self.dimension = Some(vec_dim);
        }

        let idx = self.nodes.len();
        
        // Assign a random maximum layer level
        let mut rng = rand::thread_rng();
        node.max_layer = self.get_random_level(&mut rng);

        // Initialize connections for each layer
        for level in 0..=node.max_layer {
            node.connections.insert(level, Vec::new());
        }

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.nodes.push(node);
            return Ok(idx);
        }

        // Connect the new node to existing nodes
        let mut curr_ep = self.entry_point.unwrap();
        
        // Start from the highest layer and work down
        for level in (0..=node.max_layer).rev() {
            // Find nearest neighbors at this level
            let neighbors = self.search_layer(node.vector.as_slice(), curr_ep, level, self.ef_construction);
            
            // Updated reverse connections block in add_node method
            for neighbor in &neighbors {
                let neighbor_idx = neighbor.node_idx;
                let (exceeded, n_vector, current_connections) = {
                    if let Some(n) = self.nodes.get_mut(neighbor_idx) {
                        let conns = n.connections.entry(level).or_insert_with(Vec::new);
                        conns.push(idx);
                        if conns.len() > self.m {
                            let n_vector = n.vector.clone();
                            let current_connections = std::mem::take(conns);
                            (true, n_vector, current_connections)
                        } else {
                            (false, Vec::new(), Vec::new())
                        }
                    } else {
                        (false, Vec::new(), Vec::new())
                    }
                };
                if exceeded {
                    let nodes_clone: Vec<_> = self.nodes.iter().map(|node| node.vector.clone()).collect();
                    let conn_vectors: Vec<(usize, Vec<f32>)> = current_connections.into_iter()
                        .map(|c| (c, nodes_clone[c].clone()))
                        .collect();
                    let mut conn_with_dist: Vec<(usize, f32)> = conn_vectors.into_iter()
                        .map(|(c, vec)| (c, Self::cosine_similarity(&n_vector, &vec)))
                        .collect();
                    conn_with_dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let new_connections: Vec<usize> = conn_with_dist.into_iter()
                        .take(self.m)
                        .map(|(c, _)| c)
                        .collect();
                    if let Some(n) = self.nodes.get_mut(neighbor_idx) {
                        n.connections.insert(level, new_connections);
                    }
                }
            }

            // Update entry point for next layer if we have neighbors
            if !neighbors.is_empty() {
                curr_ep = neighbors[0].node_idx;
            }
        }

        self.nodes.push(node);
        self.updates_since_sync += 1;

        // Check if we need to sync
        if self.updates_since_sync >= self.sync_threshold {
            self.sync();
        }

        Ok(idx)
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
    pub fn query(&self, query: &[f32], top_k: usize, search_ef: usize) -> Result<Vec<(String, f32)>> {
        // Validate query dimension
        if let Some(dim) = self.dimension {
            if query.len() != dim {
                return Err(Error::VectorStorage(format!(
                    "Query dimension mismatch: expected {}, got {}",
                    dim, query.len()
                )));
            }
        }

        if let Some(ep) = self.entry_point {
            let mut curr_ep = ep;
            let mut curr_layer = self.nodes[ep].max_layer;

            // Traverse layers from top to bottom
            while curr_layer > 0 {
                let neighbors = self.search_layer(query, curr_ep, curr_layer, 1);
                if !neighbors.is_empty() {
                    curr_ep = neighbors[0].node_idx;
                }
                curr_layer -= 1;
            }

            // For the bottom layer, use search_ef to control exploration
            let candidates = self.search_layer(query, curr_ep, 0, search_ef.max(top_k));
            
            // Map candidates to (id, distance) pairs
            Ok(candidates.into_iter()
                .map(|c| (self.nodes[c.node_idx].id.clone(), c.distance))
                .collect())
        } else {
            Ok(Vec::new())
        }
    }

    /// Searches for nearest neighbors at a specific layer
    fn search_layer(&self, query: &[f32], ep: usize, level: usize, ef: usize) -> Vec<Candidate> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates = BinaryHeap::new();
        let mut results = BinaryHeap::new();

        let dist = Self::cosine_similarity(query, &self.nodes[ep].vector);
        candidates.push(Candidate { node_idx: ep, distance: dist });
        results.push(Candidate { node_idx: ep, distance: dist });
        visited.insert(ep);

        while let Some(current) = candidates.pop() {
            let worst_dist = results.peek().map_or(f32::NEG_INFINITY, |r| r.distance);
            if current.distance < worst_dist && results.len() >= ef {
                break;
            }

            if let Some(node) = self.nodes.get(current.node_idx) {
                if let Some(neighbors) = node.connections.get(&level) {
                    for &neighbor_idx in neighbors {
                        if !visited.insert(neighbor_idx) {
                            continue;
                        }
                        let neighbor_dist = Self::cosine_similarity(query, &self.nodes[neighbor_idx].vector);
                        let candidate = Candidate { node_idx: neighbor_idx, distance: neighbor_dist };
                        if neighbor_dist > worst_dist || results.len() < ef {
                            candidates.push(candidate.clone());
                            results.push(candidate);
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // If we haven't found enough candidates, explore more nodes
        if results.len() < ef {
            for (idx, node) in self.nodes.iter().enumerate() {
                if !visited.contains(&idx) {
                    let d = Self::cosine_similarity(query, &node.vector);
                    let candidate = Candidate { node_idx: idx, distance: d };
                    results.push(candidate);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        results.into_sorted_vec()
    }

    /// Deletes nodes from the HNSW index whose ids are in the provided list.
    pub fn delete_nodes(&mut self, delete_ids: &[String]) {
        let id_set: std::collections::HashSet<_> = delete_ids.iter().collect();
        let mut indices_to_delete = Vec::new();

        // First, collect indices of nodes to delete
        for (i, node) in self.nodes.iter().enumerate() {
            if id_set.contains(&node.id) {
                indices_to_delete.push(i);
            }
        }

        // Sort in reverse order to safely remove nodes
        indices_to_delete.sort_unstable_by(|a, b| b.cmp(a));

        // Remove nodes and update connections
        for &idx in &indices_to_delete {
            // Remove the node
            self.nodes.remove(idx);

            // Update indices in all connections
            for node in self.nodes.iter_mut() {
                for connections in node.connections.values_mut() {
                    // Remove connections to deleted node
                    connections.retain(|&x| x != idx);
                    // Update indices for nodes after the deleted one
                    for conn_idx in connections.iter_mut() {
                        if *conn_idx > idx {
                            *conn_idx -= 1;
                        }
                    }
                }
            }
        }

        // If we have no nodes left, clear entry point
        if self.nodes.is_empty() {
            self.entry_point = None;
            return;
        }

        // Find new entry point (node with highest layer)
        let mut max_layer = 0;
        let mut max_layer_idx = 0;
        for (idx, node) in self.nodes.iter().enumerate() {
            if node.max_layer > max_layer {
                max_layer = node.max_layer;
                max_layer_idx = idx;
            }
        }
        self.entry_point = Some(max_layer_idx);

        // Rebuild connections for affected nodes
        for layer in (0..=max_layer).rev() {
            // First, collect all nodes at this layer and their vectors
            let layer_nodes: Vec<_> = self.nodes.iter().enumerate()
                .filter(|(_, node)| node.max_layer >= layer)
                .map(|(i, node)| (i, node.vector.clone()))
                .collect();

            // For each node at this layer
            for &(i, ref vec_i) in &layer_nodes {
                // Find nearest neighbors
                let mut neighbors = Vec::new();
                for &(j, ref vec_j) in &layer_nodes {
                    if i != j {
                        let dist = Self::cosine_similarity(vec_i, vec_j);
                        neighbors.push((j, dist));
                    }
                }

                // Sort by similarity and take top M
                neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top_neighbors: Vec<_> = neighbors.into_iter()
                    .take(self.m)
                    .map(|(idx, _)| idx)
                    .collect();

                // Update connections for this node
                if let Some(node) = self.nodes.get_mut(i) {
                    node.connections.insert(layer, top_neighbors);
                }
            }
        }

        self.updates_since_sync += 1;
        if self.updates_since_sync >= self.sync_threshold {
            self.sync();
        }
    }

    /// Get a random level for a new node using the same distribution as HNSW paper
    fn get_random_level(&self, rng: &mut impl rand::Rng) -> usize {
        let mut level = 0;
        while rng.gen::<f32>() < 0.5 && level < self.max_layers - 1 {
            level += 1;
        }
        level
    }

    /// Synchronize the index by saving to disk
    fn sync(&mut self) {
        if let Err(e) = self.save() {
            eprintln!("Failed to sync HNSW index: {}", e);
        }
        self.updates_since_sync = 0;
    }
} 