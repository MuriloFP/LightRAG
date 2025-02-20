use async_trait::async_trait;
use crate::types::{Result, Config};
use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use std::fs;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use petgraph::algo::astar;
use std::collections::{HashSet, VecDeque};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NodeData {
    pub id: String,
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EdgeData {
    pub weight: f64,
    pub description: Option<String>,
    pub keywords: Option<Vec<String>>,
}

// Struct for persistence serialization
#[derive(Serialize, Deserialize)]
struct GraphPersistence {
    nodes: Vec<(String, NodeData)>, // (id, NodeData)
    edges: Vec<(String, String, EdgeData)>, // (source id, target id, EdgeData)
}

pub struct PetgraphStorage {
    pub graph: Graph<NodeData, EdgeData>,
    pub node_map: HashMap<String, NodeIndex>,
    pub file_path: PathBuf,
}

/// A simple structure defining a graph pattern for matching.
#[derive(Clone, Debug)]
pub struct GraphPattern {
    pub keyword: Option<String>,
    pub node_attribute: Option<String>,
}

/// A structure representing a match result in the graph.
#[derive(Clone, Debug)]
pub struct PatternMatch {
    pub node_ids: Vec<String>,
}

impl PetgraphStorage {
    pub fn new(config: &Config) -> Result<Self> {
        let file_path = config.working_dir.join("graph_storage.json");
        Ok(PetgraphStorage {
            graph: Graph::<NodeData, EdgeData>::new(),
            node_map: HashMap::new(),
            file_path,
        })
    }

    pub fn save_graph(&self) -> Result<()> {
        let mut nodes_vec: Vec<(String, NodeData)> = Vec::new();
        for (id, &node_idx) in &self.node_map {
            if let Some(node_data) = self.graph.node_weight(node_idx) {
                nodes_vec.push((id.clone(), node_data.clone()));
            }
        }
        let mut edges_vec: Vec<(String, String, EdgeData)> = Vec::new();
        for edge in self.graph.edge_references() {
            let src = edge.source();
            let tgt = edge.target();
            if let (Some(src_data), Some(tgt_data)) = (self.graph.node_weight(src), self.graph.node_weight(tgt)) {
                edges_vec.push((src_data.id.clone(), tgt_data.id.clone(), edge.weight().clone()));
            }
        }
        let persistence = GraphPersistence {
            nodes: nodes_vec,
            edges: edges_vec,
        };
        let content = serde_json::to_string_pretty(&persistence)?;
        fs::write(&self.file_path, content)?;
        Ok(())
    }

    pub fn load_graph(&mut self) -> Result<()> {
        if self.file_path.exists() {
            let content = fs::read_to_string(&self.file_path)?;
            let persistence: GraphPersistence = serde_json::from_str(&content)?;
            self.graph = Graph::<NodeData, EdgeData>::new();
            self.node_map.clear();
            for (_, node_data) in persistence.nodes {
                let idx = self.graph.add_node(node_data.clone());
                self.node_map.insert(node_data.id.clone(), idx);
            }
            for (src_id, tgt_id, edge_data) in persistence.edges {
                if let (Some(&src_idx), Some(&tgt_idx)) = (self.node_map.get(&src_id), self.node_map.get(&tgt_id)) {
                    self.graph.add_edge(src_idx, tgt_idx, edge_data);
                }
            }
        }
        Ok(())
    }

    pub fn upsert_node(&mut self, node_id: &str, attributes: HashMap<String, serde_json::Value>) -> Result<()> {
        if let Some(&node_idx) = self.node_map.get(node_id) {
            if let Some(node_data) = self.graph.node_weight_mut(node_idx) {
                for (k, v) in attributes {
                    node_data.attributes.insert(k, v);
                }
            }
        } else {
            let node_data = NodeData {
                id: node_id.to_string(),
                attributes,
            };
            let idx = self.graph.add_node(node_data.clone());
            self.node_map.insert(node_id.to_string(), idx);
        }
        Ok(())
    }

    pub fn upsert_edge(&mut self, source_id: &str, target_id: &str, data: EdgeData) -> Result<()> {
        // Ensure both nodes exist
        if !self.node_map.contains_key(source_id) {
            self.upsert_node(source_id, HashMap::new())?;
        }
        if !self.node_map.contains_key(target_id) {
            self.upsert_node(target_id, HashMap::new())?;
        }
        let src_idx = *self.node_map.get(source_id).unwrap();
        let tgt_idx = *self.node_map.get(target_id).unwrap();
        if let Some(edge_idx) = self.graph.find_edge(src_idx, tgt_idx) {
            if let Some(weight) = self.graph.edge_weight_mut(edge_idx) {
                *weight = data.clone();
            }
        } else {
            self.graph.add_edge(src_idx, tgt_idx, data);
        }
        Ok(())
    }

    pub fn delete_node(&mut self, node_id: &str) -> Result<()> {
        if let Some(&node_idx) = self.node_map.get(node_id) {
            self.graph.remove_node(node_idx);
            self.node_map.remove(node_id);
        }
        Ok(())
    }

    pub fn delete_edge(&mut self, source_id: &str, target_id: &str) -> Result<()> {
        if let (Some(&src_idx), Some(&tgt_idx)) = (self.node_map.get(source_id), self.node_map.get(target_id)) {
            if let Some(edge_idx) = self.graph.find_edge(src_idx, tgt_idx) {
                self.graph.remove_edge(edge_idx);
            }
        }
        Ok(())
    }

    pub fn has_node(&self, node_id: &str) -> bool {
        self.node_map.contains_key(node_id)
    }

    pub fn has_edge(&self, source_id: &str, target_id: &str) -> bool {
        if let (Some(&src_idx), Some(&tgt_idx)) = (self.node_map.get(source_id), self.node_map.get(target_id)) {
            self.graph.find_edge(src_idx, tgt_idx).is_some()
        } else {
            false
        }
    }

    pub fn node_degree(&self, node_id: &str) -> Result<usize> {
        if let Some(&idx) = self.node_map.get(node_id) {
            let out_deg = self.graph.edges_directed(idx, Direction::Outgoing).count();
            let in_deg = self.graph.edges_directed(idx, Direction::Incoming).count();
            Ok(out_deg + in_deg)
        } else {
            Ok(0)
        }
    }

    /// Finds the shortest path from 'from_id' to 'to_id' using A* search with unit weights.
    pub async fn find_path(&self, from_id: &str, to_id: &str) -> crate::types::Result<Vec<String>> {
        let start_idx = self.node_map.get(from_id)
            .ok_or_else(|| crate::types::Error::Storage(format!("Node {} not found", from_id)))?;
        let goal_idx = self.node_map.get(to_id)
            .ok_or_else(|| crate::types::Error::Storage(format!("Node {} not found", to_id)))?;
        
        // Use A* algorithm with cost 1 for every edge
        let result = astar(&self.graph, *start_idx, |finish| finish == *goal_idx, |_| 1, |_| 0);
        
        if let Some((_cost, path)) = result {
            // Convert NodeIndex path to node IDs
            let path_ids = path.iter().map(|idx| self.graph[*idx].id.clone()).collect();
            Ok(path_ids)
        } else {
            Err(crate::types::Error::Storage(format!("No path found from {} to {}", from_id, to_id)))
        }
    }

    /// Returns the neighborhood (all nodes within the given depth) of a node using BFS.
    pub async fn get_neighborhood(&self, node_id: &str, depth: u32) -> crate::types::Result<Vec<NodeData>> {
        let start_idx = self.node_map.get(node_id)
            .ok_or_else(|| crate::types::Error::Storage(format!("Node {} not found", node_id)))?;
        
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut distances = std::collections::HashMap::new();
        
        visited.insert(*start_idx);
        queue.push_back(*start_idx);
        distances.insert(*start_idx, 0u32);
        
        while let Some(current) = queue.pop_front() {
            let current_distance = *distances.get(&current).unwrap_or(&0);
            if current_distance < depth {
                for neighbor in self.graph.neighbors(current) {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        distances.insert(neighbor, current_distance + 1);
                        queue.push_back(neighbor);
                    }
                }
            }
        }
        
        let result: Vec<NodeData> = visited.into_iter()
            .filter_map(|idx| self.graph.node_weight(idx).cloned())
            .collect();
        Ok(result)
    }

    /// Extracts an induced subgraph containing only the nodes with IDs in 'node_ids'.
    pub async fn extract_subgraph(&self, node_ids: &[String]) -> crate::types::Result<PetgraphStorage> {
        let node_set: HashSet<String> = node_ids.iter().cloned().collect();
        let mut new_graph = petgraph::Graph::<NodeData, EdgeData>::new();
        let mut new_node_map = HashMap::new();
        let mut index_mapping: HashMap<petgraph::graph::NodeIndex, petgraph::graph::NodeIndex> = HashMap::new();
        
        // Add nodes that are in the specified set
        for idx in self.graph.node_indices() {
            let node = &self.graph[idx];
            if node_set.contains(&node.id) {
                let new_idx = new_graph.add_node(node.clone());
                new_node_map.insert(node.id.clone(), new_idx);
                index_mapping.insert(idx, new_idx);
            }
        }
        
        // Add edges if both endpoints are in the subgraph
        for edge in self.graph.edge_references() {
            let source = edge.source();
            let target = edge.target();
            if index_mapping.contains_key(&source) && index_mapping.contains_key(&target) {
                new_graph.add_edge(index_mapping[&source], index_mapping[&target], edge.weight().clone());
            }
        }
        
        let new_storage = PetgraphStorage {
            graph: new_graph,
            node_map: new_node_map,
            file_path: self.file_path.clone(),
        };
        Ok(new_storage)
    }

    /// Matches nodes in the graph based on the provided pattern.
    pub async fn match_pattern(&self, pattern: GraphPattern) -> crate::types::Result<Vec<PatternMatch>> {
        let mut matches = Vec::new();
        // A simple implementation: filter nodes where any attribute contains the keyword
        if let Some(ref keyword) = pattern.keyword {
            for node in self.graph.node_weights() {
                for (_key, value) in &node.attributes {
                    if let Some(text) = value.as_str() {
                        if text.contains(keyword) {
                            matches.push(PatternMatch { node_ids: vec![node.id.clone()] });
                            break;
                        }
                    }
                }
            }
        }
        Ok(matches)
    }

    /// Returns a clone of the node data for the node with the given id, if it exists.
    pub fn get_node(&self, node_id: &str) -> Option<NodeData> {
        self.node_map.get(node_id).and_then(|&idx| self.graph.node_weight(idx).cloned())
    }

    /// Returns a clone of the edge data between source_id and target_id, if such an edge exists.
    pub fn get_edge(&self, source_id: &str, target_id: &str) -> Option<EdgeData> {
        if let (Some(&src_idx), Some(&tgt_idx)) = (self.node_map.get(source_id), self.node_map.get(target_id)) {
            self.graph.find_edge(src_idx, tgt_idx).and_then(|edge_idx| self.graph.edge_weight(edge_idx).cloned())
        } else {
            None
        }
    }

    /// Returns all edges incident on the node with the given id as a vector of (source_id, target_id, EdgeData) tuples.
    pub fn get_node_edges(&self, node_id: &str) -> Option<Vec<(String, String, EdgeData)>> {
        if let Some(&idx) = self.node_map.get(node_id) {
            let mut result = Vec::new();
            for edge in self.graph.edges(idx) {
                let src = self.graph[edge.source()].id.clone();
                let tgt = self.graph[edge.target()].id.clone();
                result.push((src, tgt, edge.weight().clone()));
            }
            Some(result)
        } else {
            None
        }
    }

    // ---------------------------------------------
    // Filtering Methods
    // ---------------------------------------------

    /// Returns all nodes for which the given attribute's string value contains the specified substring.
    pub fn filter_nodes_by_attribute(&self, attribute: &str, value: &str) -> Vec<NodeData> {
        self.graph.node_weights()
            .filter(|node| {
                if let Some(attr) = node.attributes.get(attribute) {
                    if let Some(text) = attr.as_str() {
                        return text.contains(value);
                    }
                }
                false
            })
            .cloned()
            .collect()
    }

    /// Returns all edges (as (source_id, target_id, EdgeData)) for which either the 'description' or any of the 'keywords'
    /// contains the provided substring.
    pub fn filter_edges_by_attribute(&self, value: &str) -> Vec<(String, String, EdgeData)> {
        self.graph.edge_references()
            .filter(|edge| {
                let mut found = false;
                if let Some(desc) = &edge.weight().description {
                    if desc.contains(value) {
                        found = true;
                    }
                }
                if !found {
                    if let Some(keywords) = &edge.weight().keywords {
                        for kw in keywords {
                            if kw.contains(value) {
                                found = true;
                                break;
                            }
                        }
                    }
                }
                found
            })
            .map(|edge| {
                let src = self.graph[edge.source()].id.clone();
                let tgt = self.graph[edge.target()].id.clone();
                (src, tgt, edge.weight().clone())
            })
            .collect()
    }

    /// Returns the combined degree of an edge by summing the degrees of its endpoints.
    /// This mimics LightRAG's edge_degree computation (i.e., degree(src) + degree(tgt)).
    pub fn edge_degree(&self, source_id: &str, target_id: &str) -> crate::types::Result<usize> {
        if let (Some(&src_idx), Some(&tgt_idx)) = (self.node_map.get(source_id), self.node_map.get(target_id)) {
            let deg_src = self.graph.edges(src_idx).count();
            let deg_tgt = self.graph.edges(tgt_idx).count();
            Ok(deg_src + deg_tgt)
        } else {
            Err(crate::types::Error::Storage(format!("Edge from {} to {} not found", source_id, target_id)))
        }
    }

    // ---------------------------------------------
    // Batch Operations and Cascading Deletion
    // ---------------------------------------------

    /// Batch upsert nodes from a vector of (node_id, attributes).
    pub fn upsert_nodes(&mut self, nodes: Vec<(String, HashMap<String, serde_json::Value>)>) -> crate::types::Result<()> {
        for (node_id, attrs) in nodes {
            self.upsert_node(&node_id, attrs)?;
        }
        Ok(())
    }

    /// Batch upsert edges from a vector of (source_id, target_id, EdgeData).
    pub fn upsert_edges(&mut self, edges: Vec<(String, String, EdgeData)>) -> crate::types::Result<()> {
        for (src, tgt, data) in edges {
            self.upsert_edge(&src, &tgt, data)?;
        }
        Ok(())
    }

    /// Batch delete nodes given a vector of node IDs.
    pub fn delete_nodes_batch(&mut self, node_ids: Vec<String>) -> crate::types::Result<()> {
        for node_id in node_ids {
            self.delete_node(&node_id)?;
        }
        Ok(())
    }

    /// Batch delete edges given a vector of (source_id, target_id) tuples.
    pub fn delete_edges_batch(&mut self, edges: Vec<(String, String)>) -> crate::types::Result<()> {
        for (src, tgt) in edges {
            self.delete_edge(&src, &tgt)?;
        }
        Ok(())
    }

    /// Cascading deletion of a node. Deletes the node and then updates any other node that references this node in its "source_id" attribute.
    pub fn cascading_delete_node(&mut self, node_id: &str) -> crate::types::Result<()> {
        // Delete the node (incident edges are removed automatically by remove_node)
        self.delete_node(node_id)?;
        
        // For every remaining node, if the "source_id" attribute equals the deleted node_id, remove it.
        for idx in self.graph.node_indices() {
            if let Some(node_data) = self.graph.node_weight_mut(idx) {
                if let Some(value) = node_data.attributes.get_mut("source_id") {
                    if let Some(str_val) = value.as_str() {
                        if str_val == node_id {
                            node_data.attributes.remove("source_id");
                        }
                    }
                }
            }
        }
        Ok(())
    }

    // ---------------------------------------------
    // Helper Methods and Default Values
    // ---------------------------------------------

    /// Returns default edge properties when an edge is not found
    pub fn default_edge_properties() -> EdgeData {
        EdgeData {
            weight: 0.0,
            description: None,
            keywords: None,
        }
    }

    /// Encodes a node label to handle special characters
    /// This is a simple implementation that could be extended based on needs
    pub fn encode_node_label(label: &str) -> String {
        // For now, just trim quotes and return as is
        // Could be extended to handle special characters if needed
        label.trim_matches('"').to_string()
    }

    /// Decodes an encoded node label
    /// This is a simple implementation that could be extended based on needs
    pub fn decode_node_label(encoded_label: &str) -> String {
        // For now, just return as is
        // Could be extended to handle special characters if needed
        encoded_label.to_string()
    }

    /// Called after indexing operations are complete
    /// This is where we ensure the graph is persisted
    pub async fn index_done_callback(&mut self) -> crate::types::Result<()> {
        self.save_graph()?;
        Ok(())
    }

    /// Returns a clone of the edge data between source_id and target_id.
    /// If no edge exists, returns default edge properties.
    pub fn get_edge_with_default(&self, source_id: &str, target_id: &str) -> EdgeData {
        self.get_edge(source_id, target_id)
            .unwrap_or_else(|| Self::default_edge_properties())
    }

    /// Reorders the graph nodes and edges to provide a stable ordering for consistent reads.
    pub fn stabilize_graph(&mut self) -> crate::types::Result<()> {
        let mut new_graph = Graph::<NodeData, EdgeData>::new();
        let mut new_node_map = HashMap::new();
        
        // Collect and sort nodes by their id for a consistent ordering
        let mut nodes: Vec<&NodeData> = self.graph.node_weights().collect();
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        
        // Add nodes to the new graph in sorted order and build new node_map
        for node in nodes {
            let new_idx = new_graph.add_node(node.clone());
            new_node_map.insert(node.id.clone(), new_idx);
        }
        
        // Build a mapping from old NodeIndex to new NodeIndex
        let mut index_mapping: HashMap<petgraph::graph::NodeIndex, petgraph::graph::NodeIndex> = HashMap::new();
        for (id, old_idx) in &self.node_map {
            if let Some(new_idx) = new_node_map.get(id) {
                index_mapping.insert(*old_idx, *new_idx);
            }
        }
        
        // Collect and sort edges by (source id, target id) for consistent ordering
        let mut edges: Vec<_> = self.graph.edge_references().collect();
        edges.sort_by(|a, b| {
            let src_a = self.graph[a.source()].id.clone();
            let tgt_a = self.graph[a.target()].id.clone();
            let src_b = self.graph[b.source()].id.clone();
            let tgt_b = self.graph[b.target()].id.clone();
            src_a.cmp(&src_b).then(tgt_a.cmp(&tgt_b))
        });
        
        // Add edges to the new graph only if both endpoints exist in the new mapping
        for edge in edges {
            if let (Some(&new_src), Some(&new_tgt)) = (index_mapping.get(&edge.source()), index_mapping.get(&edge.target())) {
                new_graph.add_edge(new_src, new_tgt, edge.weight().clone());
            }
        }
        
        self.graph = new_graph;
        self.node_map = new_node_map;
        Ok(())
    }

    /// Generates a simple embedding for each node using a dummy node2vec-like approach.
    /// This method creates an embedding vector for each node based on a simple hash of its id, for demonstration purposes.
    pub async fn embed_nodes(&self, dimension: usize) -> crate::types::Result<(Vec<f32>, Vec<String>)> {
        // Collect node ids and sort them for consistency
        let mut node_ids: Vec<String> = self.graph.node_weights().map(|node| node.id.clone()).collect();
        node_ids.sort();
        
        let mut embeddings = Vec::new();
        // For each node, compute a dummy embedding vector
        for node_id in &node_ids {
            // Simple hash: sum of byte values mod 100 scaled to [0,1]
            let sum: u32 = node_id.bytes().map(|b| b as u32).sum();
            let val = (sum % 100) as f32 / 100.0;
            let emb = vec![val; dimension];
            embeddings.extend(emb);
        }
        Ok((embeddings, node_ids))
    }
}

#[async_trait]
impl crate::storage::graph::GraphStorage for PetgraphStorage {
    async fn initialize(&mut self) -> Result<()> {
        self.load_graph()?;
        Ok(())
    }

    async fn finalize(&mut self) -> Result<()> {
        self.save_graph()?;
        Ok(())
    }
} 