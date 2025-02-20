use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::fs;
use crate::types::Result;

/// A node in the HNSW index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HNSWNode {
    /// Unique identifier of the node.
    pub id: String,
    /// The vector representation of the node.
    pub vector: Vec<f32>,
    /// Connections represented as: layer -> Vec<node indices>
    pub connections: HashMap<usize, Vec<usize>>,
    /// The maximum layer this node appears in
    pub max_layer: usize,
}

impl HNSWNode {
    /// Creates a new HNSW node with the given id and vector.
    pub fn new(id: &str, vector: Vec<f32>) -> Self {
        HNSWNode {
            id: id.to_string(),
            vector,
            connections: HashMap::new(),
            max_layer: 0,
        }
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
#[derive(Debug, Serialize, Deserialize)]
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
    #[serde(skip)]
    file_path: Option<PathBuf>,
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
                    let loaded: HNSWIndex = serde_json::from_str(&content)?;
                    self.nodes = loaded.nodes;
                    self.entry_point = loaded.entry_point;
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
            let content = serde_json::to_string_pretty(self)?;
            fs::write(path, content)?;
        }
        Ok(())
    }

    /// Adds a new node to the HNSW index and returns its index.
    pub fn add_node(&mut self, mut node: HNSWNode) -> usize {
        let idx = self.nodes.len();
        
        // Assign a random maximum layer level
        let mut rng = rand::thread_rng();
        node.max_layer = self.get_random_level(&mut rng);

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(idx);
            self.nodes.push(node);
            return idx;
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
    pub fn query(&self, query: &[f32], top_k: usize, search_ef: usize) -> Vec<(String, f32)> {
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

            // Search the bottom layer with higher ef
            let candidates = self.search_layer(query, curr_ep, 0, search_ef);
            
            // Convert candidates to results
            let mut results: Vec<(String, f32)> = candidates.into_iter()
                .map(|c| (self.nodes[c.node_idx].id.clone(), c.distance))
                .collect();
            results.truncate(top_k);
            results
        } else {
            Vec::new()
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

        results.into_sorted_vec()
    }

    /// Deletes nodes from the HNSW index whose ids are in the provided list.
    pub fn delete_nodes(&mut self, delete_ids: &[String]) {
        let id_set: std::collections::HashSet<_> = delete_ids.iter().collect();

        // Iterate in reverse order to safely remove nodes
        for i in (0..self.nodes.len()).rev() {
            if id_set.contains(&self.nodes[i].id) {
                // Remove connections to this node from other nodes
                let idx = i;
                for node in self.nodes.iter_mut() {
                    for connections in node.connections.values_mut() {
                        connections.retain(|&x| x != idx);
                    }
                }

                // Remove the node
                self.nodes.remove(i);

                // Update indices in all connections for nodes with higher index
                for node in self.nodes.iter_mut() {
                    for connections in node.connections.values_mut() {
                        for conn_idx in connections.iter_mut() {
                            if *conn_idx > i {
                                *conn_idx -= 1;
                            }
                        }
                    }
                }
            }
        }

        // Update entry point
        if self.nodes.is_empty() {
            self.entry_point = None;
        } else if let Some(ep) = self.entry_point {
            if ep >= self.nodes.len() {
                self.entry_point = Some(0);
            }
        }

        // Prune any connection indices that are now out of bounds
        let current_len = self.nodes.len();
        for node in self.nodes.iter_mut() {
            for connections in node.connections.values_mut() {
                connections.retain(|&conn| conn < current_len);
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