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

/// Data structure representing a node in the graph with its attributes.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NodeData {
    /// Unique identifier for the node.
    pub id: String,
    /// Map of attribute key-value pairs associated with the node.
    pub attributes: HashMap<String, serde_json::Value>,
}

/// Data structure representing an edge in the graph with its properties.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct EdgeData {
    /// Weight or strength of the connection between nodes.
    pub weight: f64,
    /// Optional textual description of the relationship.
    pub description: Option<String>,
    /// Optional list of keywords characterizing the relationship.
    pub keywords: Option<Vec<String>>,
}

/// Serializable structure for persisting graph data to disk.
#[derive(Serialize, Deserialize)]
struct GraphPersistence {
    /// List of node tuples containing (id, NodeData).
    nodes: Vec<(String, NodeData)>,
    /// List of edge tuples containing (source id, target id, EdgeData).
    edges: Vec<(String, String, EdgeData)>,
}

/// Main graph storage implementation using petgraph as the underlying graph structure.
pub struct PetgraphStorage {
    /// The core graph structure containing nodes and edges.
    pub graph: Graph<NodeData, EdgeData>,
    /// Mapping from node IDs to their corresponding indices in the graph.
    pub node_map: HashMap<String, NodeIndex>,
    /// Path to the file where the graph data is persisted.
    pub file_path: PathBuf,
}

/// A pattern definition for matching subgraphs within the main graph.
#[derive(Clone, Debug)]
pub struct GraphPattern {
    /// Optional keyword to match in edge properties.
    pub keyword: Option<String>,
    /// Optional attribute name to match in node properties.
    pub node_attribute: Option<String>,
}

/// Result structure containing matched nodes from a pattern search.
#[derive(Clone, Debug)]
pub struct PatternMatch {
    /// List of node IDs that match the pattern criteria.
    pub node_ids: Vec<String>,
}

impl PetgraphStorage {
    /// Creates a new PetgraphStorage instance.
    /// 
    /// # Arguments
    /// * `config` - Configuration object containing working directory information
    /// 
    /// # Returns
    /// A Result containing the new PetgraphStorage instance or an error
    pub fn new(config: &Config) -> Result<Self> {
        let file_path = config.working_dir.join("graph_storage.json");
        Ok(PetgraphStorage {
            graph: Graph::<NodeData, EdgeData>::new(),
            node_map: HashMap::new(),
            file_path,
        })
    }

    /// Persists the current state of the graph to disk.
    /// 
    /// # Returns
    /// A Result indicating success or failure of the save operation
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

    /// Loads the graph state from disk.
    /// 
    /// # Returns
    /// A Result indicating success or failure of the load operation
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

    /// Inserts or updates a node in the graph with the given attributes.
    /// 
    /// # Arguments
    /// * `node_id` - Unique identifier for the node
    /// * `attributes` - Map of attribute key-value pairs to associate with the node
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
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

    /// Inserts or updates an edge between two nodes with the given properties.
    /// 
    /// # Arguments
    /// * `source_id` - ID of the source node
    /// * `target_id` - ID of the target node
    /// * `data` - Edge properties to associate with the connection
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
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

    /// Removes a node and all its incident edges from the graph.
    /// 
    /// # Arguments
    /// * `node_id` - ID of the node to delete
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
    pub fn delete_node(&mut self, node_id: &str) -> Result<()> {
        if let Some(&node_idx) = self.node_map.get(node_id) {
            self.graph.remove_node(node_idx);
            self.node_map.remove(node_id);
        }
        Ok(())
    }

    /// Removes an edge between two nodes from the graph.
    /// 
    /// # Arguments
    /// * `source_id` - ID of the source node
    /// * `target_id` - ID of the target node
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
    pub fn delete_edge(&mut self, source_id: &str, target_id: &str) -> Result<()> {
        if let (Some(&src_idx), Some(&tgt_idx)) = (self.node_map.get(source_id), self.node_map.get(target_id)) {
            if let Some(edge_idx) = self.graph.find_edge(src_idx, tgt_idx) {
                self.graph.remove_edge(edge_idx);
            }
        }
        Ok(())
    }

    /// Checks if a node with the given ID exists in the graph.
    /// 
    /// # Arguments
    /// * `node_id` - ID of the node to check
    /// 
    /// # Returns
    /// true if the node exists, false otherwise
    pub fn has_node(&self, node_id: &str) -> bool {
        self.node_map.contains_key(node_id)
    }

    /// Checks if an edge exists between two nodes.
    /// 
    /// # Arguments
    /// * `source_id` - ID of the source node
    /// * `target_id` - ID of the target node
    /// 
    /// # Returns
    /// true if the edge exists, false otherwise
    pub fn has_edge(&self, source_id: &str, target_id: &str) -> bool {
        if let (Some(&src_idx), Some(&tgt_idx)) = (self.node_map.get(source_id), self.node_map.get(target_id)) {
            self.graph.find_edge(src_idx, tgt_idx).is_some()
        } else {
            false
        }
    }

    /// Calculates the total degree (in + out) of a node.
    /// 
    /// # Arguments
    /// * `node_id` - ID of the node
    /// 
    /// # Returns
    /// A Result containing the node's degree or an error
    pub fn node_degree(&self, node_id: &str) -> Result<usize> {
        if let Some(&idx) = self.node_map.get(node_id) {
            let out_deg = self.graph.edges_directed(idx, Direction::Outgoing).count();
            let in_deg = self.graph.edges_directed(idx, Direction::Incoming).count();
            Ok(out_deg + in_deg)
        } else {
            Ok(0)
        }
    }

    /// Finds the shortest path between two nodes using A* search algorithm.
    /// 
    /// # Arguments
    /// * `from_id` - ID of the starting node
    /// * `to_id` - ID of the target node
    /// 
    /// # Returns
    /// A Result containing a vector of node IDs representing the path, or an error if no path exists
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

    /// Retrieves all nodes within a specified distance from a given node.
    /// 
    /// # Arguments
    /// * `node_id` - ID of the starting node
    /// * `depth` - Maximum distance to traverse from the starting node
    /// 
    /// # Returns
    /// A Result containing a vector of NodeData for all nodes within the specified depth
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

    /// Creates a new graph containing only the specified nodes and their interconnecting edges.
    /// 
    /// # Arguments
    /// * `node_ids` - List of node IDs to include in the subgraph
    /// 
    /// # Returns
    /// A Result containing a new PetgraphStorage instance representing the subgraph
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

    /// Searches for nodes in the graph that match a specified pattern.
    /// 
    /// # Arguments
    /// * `pattern` - Pattern criteria to match against nodes and edges
    /// 
    /// # Returns
    /// A Result containing a vector of PatternMatch objects representing the matches found
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

    /// Retrieves the data associated with a specific node.
    /// 
    /// # Arguments
    /// * `node_id` - ID of the node to retrieve
    /// 
    /// # Returns
    /// Option containing the NodeData if the node exists, None otherwise
    pub fn get_node(&self, node_id: &str) -> Option<NodeData> {
        self.node_map.get(node_id).and_then(|&idx| self.graph.node_weight(idx).cloned())
    }

    /// Retrieves the data associated with an edge between two nodes.
    /// 
    /// # Arguments
    /// * `source_id` - ID of the source node
    /// * `target_id` - ID of the target node
    /// 
    /// # Returns
    /// Option containing the EdgeData if the edge exists, None otherwise
    pub fn get_edge(&self, source_id: &str, target_id: &str) -> Option<EdgeData> {
        if let (Some(&src_idx), Some(&tgt_idx)) = (self.node_map.get(source_id), self.node_map.get(target_id)) {
            self.graph.find_edge(src_idx, tgt_idx).and_then(|edge_idx| self.graph.edge_weight(edge_idx).cloned())
        } else {
            None
        }
    }

    /// Retrieves all edges connected to a specific node.
    /// 
    /// # Arguments
    /// * `node_id` - ID of the node
    /// 
    /// # Returns
    /// Option containing a vector of (source_id, target_id, EdgeData) tuples for all incident edges
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

    /// Finds nodes that have a specific attribute containing a given value.
    /// 
    /// # Arguments
    /// * `attribute` - Name of the attribute to search
    /// * `value` - Value to search for within the attribute
    /// 
    /// # Returns
    /// Vector of NodeData for all nodes matching the search criteria
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

    /// Finds edges where either the description or keywords contain a specific value.
    /// 
    /// # Arguments
    /// * `value` - Value to search for in edge descriptions and keywords
    /// 
    /// # Returns
    /// Vector of (source_id, target_id, EdgeData) tuples for all matching edges
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

    /// Calculates the combined degree of an edge by summing the degrees of its endpoints.
    /// 
    /// # Arguments
    /// * `source_id` - ID of the source node
    /// * `target_id` - ID of the target node
    /// 
    /// # Returns
    /// A Result containing the combined degree or an error if the edge doesn't exist
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

    /// Performs batch insertion or update of multiple nodes.
    /// 
    /// # Arguments
    /// * `nodes` - Vector of (node_id, attributes) pairs to upsert
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
    pub fn upsert_nodes(&mut self, nodes: Vec<(String, HashMap<String, serde_json::Value>)>) -> crate::types::Result<()> {
        for (node_id, attrs) in nodes {
            self.upsert_node(&node_id, attrs)?;
        }
        Ok(())
    }

    /// Performs batch insertion or update of multiple edges.
    /// 
    /// # Arguments
    /// * `edges` - Vector of (source_id, target_id, EdgeData) tuples to upsert
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
    pub fn upsert_edges(&mut self, edges: Vec<(String, String, EdgeData)>) -> crate::types::Result<()> {
        for (src, tgt, data) in edges {
            self.upsert_edge(&src, &tgt, data)?;
        }
        Ok(())
    }

    /// Deletes multiple nodes in a single operation.
    /// 
    /// # Arguments
    /// * `node_ids` - Vector of node IDs to delete
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
    pub fn delete_nodes_batch(&mut self, node_ids: Vec<String>) -> crate::types::Result<()> {
        for node_id in node_ids {
            self.delete_node(&node_id)?;
        }
        Ok(())
    }

    /// Deletes multiple edges in a single operation.
    /// 
    /// # Arguments
    /// * `edges` - Vector of (source_id, target_id) pairs representing edges to delete
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
    pub fn delete_edges_batch(&mut self, edges: Vec<(String, String)>) -> crate::types::Result<()> {
        for (src, tgt) in edges {
            self.delete_edge(&src, &tgt)?;
        }
        Ok(())
    }

    /// Deletes a node and updates any nodes that reference it in their source_id attribute.
    /// 
    /// # Arguments
    /// * `node_id` - ID of the node to delete
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
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

    /// Returns default edge properties for when an edge is not found.
    /// 
    /// # Returns
    /// A new EdgeData instance with default values
    pub fn default_edge_properties() -> EdgeData {
        EdgeData {
            weight: 0.0,
            description: None,
            keywords: None,
        }
    }

    /// Encodes a node label to handle special characters.
    /// 
    /// # Arguments
    /// * `label` - Label to encode
    /// 
    /// # Returns
    /// The encoded label as a String
    pub fn encode_node_label(label: &str) -> String {
        // For now, just trim quotes and return as is
        // Could be extended to handle special characters if needed
        label.trim_matches('"').to_string()
    }

    /// Decodes an encoded node label.
    /// 
    /// # Arguments
    /// * `encoded_label` - Previously encoded label to decode
    /// 
    /// # Returns
    /// The decoded label as a String
    pub fn decode_node_label(encoded_label: &str) -> String {
        // For now, just return as is
        // Could be extended to handle special characters if needed
        encoded_label.to_string()
    }

    /// Callback function executed after indexing operations are complete.
    /// 
    /// # Returns
    /// A Result indicating success or failure of post-indexing operations
    pub async fn index_done_callback(&mut self) -> crate::types::Result<()> {
        self.save_graph()?;
        Ok(())
    }

    /// Retrieves edge data between two nodes, returning default properties if no edge exists.
    /// 
    /// # Arguments
    /// * `source_id` - ID of the source node
    /// * `target_id` - ID of the target node
    /// 
    /// # Returns
    /// EdgeData for the edge if it exists, or default edge properties
    pub fn get_edge_with_default(&self, source_id: &str, target_id: &str) -> EdgeData {
        self.get_edge(source_id, target_id)
            .unwrap_or_else(|| Self::default_edge_properties())
    }

    /// Reorders graph nodes and edges to provide a stable ordering for consistent reads.
    /// 
    /// # Returns
    /// A Result indicating success or failure of the stabilization operation
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

    /// Generates simple embeddings for nodes using a basic node2vec-like approach.
    /// 
    /// # Arguments
    /// * `dimension` - Dimensionality of the embeddings to generate
    /// 
    /// # Returns
    /// A Result containing a tuple of (embeddings vector, node IDs vector)
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
    /// Initializes the graph storage by loading data from disk.
    /// 
    /// # Returns
    /// A Result indicating success or failure of initialization
    async fn initialize(&mut self) -> Result<()> {
        self.load_graph()?;
        Ok(())
    }

    /// Finalizes the graph storage by saving data to disk.
    /// 
    /// # Returns
    /// A Result indicating success or failure of finalization
    async fn finalize(&mut self) -> Result<()> {
        self.save_graph()?;
        Ok(())
    }
} 