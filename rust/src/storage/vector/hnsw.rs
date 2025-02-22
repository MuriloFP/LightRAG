use std::collections::HashMap;
use std::collections::BinaryHeap;
use std::cmp::Ordering;
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::fs;
use crate::types::{Result, Error};
use tracing;

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
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Natural ordering: higher score is better
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance.partial_cmp(&other.distance).unwrap_or(std::cmp::Ordering::Equal)
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

    /// Set the dimension for this index
    pub fn set_dimension(&mut self, dim: usize) {
        self.dimension = Some(dim);
    }

    /// Load the index from file
    pub fn load(&mut self) -> Result<()> {
        if let Some(path) = &self.file_path {
            if path.exists() {
                let content = fs::read_to_string(path)?;
                if !content.trim().is_empty() {
                    let loaded: Vec<HNSWNode> = serde_json::from_str(&content)?;
                    // Set dimension based on first node if not already set
                    if self.dimension.is_none() && !loaded.is_empty() {
                        self.dimension = Some(loaded[0].vector.len());
                    }
                    // Validate dimensions of all nodes
                    if let Some(dim) = self.dimension {
                        for node in &loaded {
                            if node.vector.len() != dim {
                                return Err(Error::VectorStorage(format!(
                                    "Vector dimension mismatch in loaded data: expected {}, got {}",
                                    dim, node.vector.len()
                                )));
                            }
                        }
                    }
                    self.nodes = loaded;
                    if !self.nodes.is_empty() {
                        self.entry_point = Some(0);
                    } else {
                        self.entry_point = None;
                    }
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
        // Set dimension if not set
        if self.dimension.is_none() {
            self.dimension = Some(node.vector.len());
            tracing::debug!("Setting HNSW dimension to {}", node.vector.len());
        }

        // Validate vector dimension
        if let Some(dim) = self.dimension {
            if node.vector.len() != dim {
                tracing::error!("Vector dimension mismatch: expected {}, got {}", dim, node.vector.len());
                return Err(Error::VectorStorage(format!(
                    "Vector dimension mismatch: expected {}, got {}",
                    dim, node.vector.len()
                )));
            }
        }

        let idx = self.nodes.len();
        tracing::debug!("Adding node {} at index {}", node.id, idx);
        
        // Assign a random maximum layer level
        let mut rng = rand::thread_rng();
        node.max_layer = self.get_random_level(&mut rng);
        tracing::debug!("Assigned max_layer {} to node {}", node.max_layer, node.id);

        // Initialize connections for each layer
        for level in 0..=node.max_layer {
            node.connections.insert(level, Vec::new());
        }

        // If this is the first node, make it the entry point
        if self.entry_point.is_none() {
            tracing::debug!("First node in index, setting as entry point");
            self.entry_point = Some(idx);
            self.nodes.push(node);
            return Ok(idx);
        }

        // Connect the new node to existing nodes
        let mut curr_ep = self.entry_point.unwrap();
        tracing::debug!("Starting node connection from entry point {}", curr_ep);
        
        // Start from the highest layer and work down
        for level in (0..=node.max_layer).rev() {
            tracing::debug!("Processing layer {} for node {}", level, node.id);
            // Find nearest neighbors at this level
            let neighbors = self.search_layer(node.vector.as_slice(), curr_ep, level, self.ef_construction);
            tracing::debug!("Found {} neighbors at layer {}", neighbors.len(), level);
            
            // Add connections
            for neighbor in &neighbors {
                let neighbor_idx = neighbor.node_idx;
                if let Some(n) = self.nodes.get_mut(neighbor_idx) {
                    let conns = n.connections.entry(level).or_insert_with(Vec::new);
                    if !conns.contains(&idx) {
                        conns.push(idx);
                        tracing::debug!("Added connection from node {} to new node {} at layer {}", neighbor_idx, idx, level);
                    }
                }
            }

            // Update entry point for next layer if we have neighbors
            if !neighbors.is_empty() {
                curr_ep = neighbors[0].node_idx;
                tracing::debug!("Updated entry point to {} for next layer", curr_ep);
            }
        }

        self.nodes.push(node);
        self.updates_since_sync += 1;

        // Check if we need to sync
        if self.updates_since_sync >= self.sync_threshold {
            tracing::debug!("Sync threshold reached, saving index");
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
        // If index is empty, return empty results
        if self.nodes.is_empty() {
            tracing::debug!("Query on empty index, returning empty results");
            return Ok(Vec::new());
        }

        // Validate query dimension
        if let Some(dim) = self.dimension {
            if query.len() != dim {
                tracing::error!("Query dimension mismatch: expected {}, got {}", dim, query.len());
                return Err(Error::VectorStorage(format!(
                    "Query dimension mismatch: expected {}, got {}",
                    dim, query.len()
                )));
            }
        }

        let ep = if let Some(ep) = self.entry_point {
            ep
        } else {
            0
        };
        tracing::debug!("Starting query from entry point {}", ep);

        let mut curr_ep = ep;
        let mut curr_layer = self.nodes[ep].max_layer;
        tracing::debug!("Initial layer for query: {}", curr_layer);

        // Traverse layers from top to bottom
        while curr_layer > 0 {
            let neighbors = self.search_layer(query, curr_ep, curr_layer, 1);
            if !neighbors.is_empty() {
                curr_ep = neighbors[0].node_idx;
                tracing::debug!("Layer {}: moved to entry point {}", curr_layer, curr_ep);
            }
            curr_layer -= 1;
        }

        // For the bottom layer, use search_ef to control exploration
        let candidates = self.search_layer(query, curr_ep, 0, search_ef.max(top_k));
        tracing::debug!("Bottom layer search found {} candidates", candidates.len());
        
        // Return only available results, up to top_k
        let results: Vec<(String, f32)> = candidates.into_iter()
            .take(top_k)
            .map(|c| (self.nodes[c.node_idx].id.clone(), c.distance))
            .collect();
        tracing::debug!("Returning {} results", results.len());
        Ok(results)
    }

    /// Searches for nearest neighbors at a specific layer
    fn search_layer(&self, query: &[f32], ep: usize, level: usize, ef: usize) -> Vec<Candidate> {
        tracing::debug!("Searching layer {} from entry point {} with ef {}", level, ep, ef);
        use std::cmp::Reverse;
        let mut visited = std::collections::HashSet::new();
        let mut candidates = std::collections::BinaryHeap::new();
        let mut results: std::collections::BinaryHeap<Reverse<Candidate>> = std::collections::BinaryHeap::new();

        // Compute score so that a perfect match (cosine=1.0) gives 0.0
        let score = Self::cosine_similarity(query, &self.nodes[ep].vector);
        let entry_candidate = Candidate { node_idx: ep, distance: score };
        candidates.push(entry_candidate.clone());
        results.push(Reverse(entry_candidate));
        visited.insert(ep);
        tracing::debug!("Initial entry point {} has score {}", ep, score);

        let mut explored = 0;
        // In our min-heap 'results', the worst (lowest) score is at the top
        while let Some(current) = candidates.pop() {
            explored += 1;
            // Get the worst score in current results (if any), default to -infinity
            let worst_score = results.peek().map_or(f32::NEG_INFINITY, |r| r.0.distance);
            // If current candidate is worse than the worst in results and we have enough answers, stop
            if current.distance < worst_score && results.len() >= ef {
                tracing::debug!("Search stopping at {} explored nodes: score {} < worst {}", explored, current.distance, worst_score);
                break;
            }
            
            if let Some(node) = self.nodes.get(current.node_idx) {
                if let Some(neighbors) = node.connections.get(&level) {
                    tracing::debug!("Exploring {} neighbors of node {}", neighbors.len(), current.node_idx);
                    for &neighbor_idx in neighbors {
                        if !visited.insert(neighbor_idx) {
                            continue;
                        }
                        let neighbor_score = Self::cosine_similarity(query, &self.nodes[neighbor_idx].vector);
                        let candidate = Candidate { node_idx: neighbor_idx, distance: neighbor_score };
                        tracing::debug!("Neighbor {} has score {}", neighbor_idx, neighbor_score);
                        // If we haven't filled ef or this candidate improves the worst score in results
                        if results.len() < ef || candidate.distance > worst_score {
                            candidates.push(candidate.clone());
                            results.push(Reverse(candidate));
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }
        // Extract candidates from results and sort them in descending order (best score first)
        let mut final_results: Vec<Candidate> = results.into_iter().map(|r| r.0).collect();
        final_results.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap_or(std::cmp::Ordering::Equal));
        tracing::debug!("Layer {} search complete, found {} results", level, final_results.len());
        final_results
    }

    /// Deletes nodes from the HNSW index whose ids are in the provided list.
    pub fn delete_nodes(&mut self, delete_ids: &[String]) -> () {
        // Remove nodes that match the delete_ids
        self.nodes.retain(|node| !delete_ids.contains(&node.id));

        // If no nodes remain, clear entry point and reset dimension
        if self.nodes.is_empty() {
            self.entry_point = None;
            self.dimension = None;
            return;
        }

        // For every layer from 0 to self.max_layers - 1
        for layer in 0..self.max_layers {
            // Collect nodes that are active in this layer (node.max_layer >= layer)
            let layer_nodes: Vec<(usize, Vec<f32>)> = self.nodes.iter().enumerate()
                .filter(|(_, node)| node.max_layer >= layer)
                .map(|(i, node)| (i, node.vector.clone()))
                .collect();

            // For each node that is active in this layer, rebuild its connections
            for &(i, ref vec_i) in &layer_nodes {
                let mut neighbors = Vec::new();
                for &(j, ref vec_j) in &layer_nodes {
                    if i != j {
                        let dist = Self::cosine_similarity(vec_i, vec_j);
                        neighbors.push((j, dist));
                    }
                }
                // Sort neighbors by descending similarity and take top 'm'
                neighbors.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top_neighbors: Vec<usize> = neighbors.into_iter()
                    .take(self.m)
                    .map(|(j, _)| j)
                    .collect();
                if let Some(node) = self.nodes.get_mut(i) {
                    node.connections.insert(layer, top_neighbors);
                }
            }
            // For nodes not active in this layer, clear connections
            for (i, node) in self.nodes.iter_mut().enumerate() {
                if node.max_layer < layer {
                    node.connections.insert(layer, Vec::new());
                }
            }
        }

        // Update entry point: choose the node with the highest max_layer among current nodes
        self.entry_point = Some(self.nodes.iter().enumerate()
            .max_by_key(|(_, n)| n.max_layer)
            .map(|(i, _)| i)
            .unwrap());

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