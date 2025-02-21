use async_trait::async_trait;
use crate::types::{Result, Config, Error};
use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Direction;
use petgraph::visit::EdgeRef;
use petgraph::algo::astar;
use std::collections::{HashSet, VecDeque};
use super::embeddings::{EmbeddingAlgorithm, Node2VecConfig, generate_node2vec_embeddings};
use crate::storage::graph::GraphStorage;
use tokio::task::JoinError;
use tracing::info;
use std::fmt;
use super::graphml::GraphMlHandler;

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
    embedding_handlers: HashMap<&'static str, fn(&petgraph::graph::Graph<crate::storage::graph::NodeData, crate::storage::graph::EdgeData>) -> Result<(Vec<f32>, Vec<petgraph::graph::NodeIndex>)>>,
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

impl From<JoinError> for crate::types::Error {
    fn from(err: JoinError) -> Self {
        crate::types::Error::Storage(format!("Task join error: {}", err))
    }
}

impl fmt::Display for NodeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap_or_default())
    }
}

impl fmt::Display for EdgeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self).unwrap_or_default())
    }
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
        let file_path = config.working_dir.join(format!("graph_storage.graphml"));
        let mut embedding_handlers: HashMap<&'static str, fn(&petgraph::graph::Graph<crate::storage::graph::NodeData, crate::storage::graph::EdgeData>) -> Result<(Vec<f32>, Vec<petgraph::graph::NodeIndex>)>> = HashMap::new();
        embedding_handlers.insert("node2vec", node2vec_handler);
        Ok(PetgraphStorage {
            graph: Graph::<NodeData, EdgeData>::new(),
            node_map: HashMap::new(),
            file_path,
            embedding_handlers,
        })
    }

    /// Stabilizes the graph to ensure consistent ordering.
    /// This matches LightRAG's stabilization functionality.
    pub fn stabilize_graph(&mut self) -> Result<()> {
        // Log the state before stabilization (if not already logged earlier)
        tracing::info!("Stabilizing graph: {} nodes, {} edges", self.graph.node_count(), self.graph.edge_count());

        // Create a new graph with stable ordering
        let mut new_graph = Graph::<NodeData, EdgeData>::new();
        let mut new_node_map = HashMap::new();

        // Add nodes in sorted order
        let mut sorted_nodes: Vec<_> = self.node_map.iter().collect();
        sorted_nodes.sort_by(|(id1, _), (id2, _)| id1.cmp(id2));
        // Debug assertion for node order
        for window in sorted_nodes.windows(2) {
            debug_assert!(window[0].0 <= window[1].0, "Nodes not sorted: {} > {}", window[0].0, window[1].0);
        }

        for (id, &old_idx) in sorted_nodes {
            if let Some(node_data) = self.graph.node_weight(old_idx) {
                let new_idx = new_graph.add_node(node_data.clone());
                new_node_map.insert(id.clone(), new_idx);
            }
        }

        // Add edges in sorted order
        let mut edges = Vec::new();
        for edge in self.graph.edge_references() {
            let source = &self.graph[edge.source()].id;
            let target = &self.graph[edge.target()].id;
            let edge_data = edge.weight().clone();
            
            // For undirected graphs, ensure consistent source/target ordering
            let (src, tgt, data) = if !self.graph.is_directed() && source > target {
                (target, source, edge_data)
            } else {
                (source, target, edge_data)
            };
            
            edges.push((src.clone(), tgt.clone(), data));
        }

        // Sort edges by source->target key
        edges.sort_by(|(src1, tgt1, _), (src2, tgt2, _)| {
            let key1 = format!("{} -> {}", src1, tgt1);
            let key2 = format!("{} -> {}", src2, tgt2);
            key1.cmp(&key2)
        });
        // Debug assertion for edge order
        for window in edges.windows(2) {
            let key1 = format!("{} -> {}", window[0].0, window[0].1);
            let key2 = format!("{} -> {}", window[1].0, window[1].1);
            debug_assert!(key1 <= key2, "Edges not sorted: {} > {}", key1, key2);
        }

        // Add sorted edges to new graph
        for (src, tgt, data) in edges {
            if let (Some(&src_idx), Some(&tgt_idx)) = (new_node_map.get(&src), new_node_map.get(&tgt)) {
                new_graph.add_edge(src_idx, tgt_idx, data);
            }
        }

        // Replace old graph with stabilized version
        self.graph = new_graph;
        self.node_map = new_node_map;

        // Log the state after stabilization
        tracing::info!("Graph stabilized: {} nodes, {} edges", self.graph.node_count(), self.graph.edge_count());

        // Collect node IDs from the graph
        let node_ids: Vec<String> = self.graph.node_weights().map(|n| n.id.clone()).collect();
        // Create a sorted copy
        let mut sorted_ids = node_ids.clone();
        sorted_ids.sort();
        // Debug assertion to ensure the nodes are in sorted order
        debug_assert_eq!(node_ids, sorted_ids, "Node IDs are not sorted after stabilization");

        Ok(())
    }

    /// Saves the graph to disk in GraphML format.
    pub fn save_graph(&self) -> Result<()> {
        // Create a GraphMl instance with pretty printing enabled
        let handler = GraphMlHandler::new(self.graph.clone()).pretty_print(true);
        handler.write_graphml(&self.file_path)?;
        info!("Saved graph to {}", self.file_path.display());
        Ok(())
    }

    /// Loads the graph from disk in GraphML format.
    pub fn load_graph(&mut self) -> Result<()> {
        if self.file_path.exists() {
            // Load graph from GraphML file
            let graph = GraphMlHandler::read_graphml(&self.file_path)
                .map_err(|e| Error::Storage(format!("Failed to parse GraphML: {}", e)))?;
            
            // Update internal state
            self.graph = graph;
            
            // Rebuild node map
            self.node_map.clear();
            for node_idx in self.graph.node_indices() {
                if let Some(node_data) = self.graph.node_weight(node_idx) {
                    self.node_map.insert(node_data.id.clone(), node_idx);
                }
            }
            
            info!("Loaded graph from {} with {} nodes and {} edges", 
                  self.file_path.display(),
                  self.graph.node_count(),
                  self.graph.edge_count());
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
    pub async fn upsert_node_impl(&mut self, node_id: &str, attributes: HashMap<String, serde_json::Value>) -> Result<()> {
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
    pub async fn upsert_edge_impl(&mut self, source_id: &str, target_id: &str, data: EdgeData) -> Result<()> {
        if !self.has_node_impl(source_id).await {
            self.upsert_node_impl(source_id, HashMap::new()).await?;
        }
        if !self.has_node_impl(target_id).await {
            self.upsert_node_impl(target_id, HashMap::new()).await?;
        }
        let src_idx = *self.node_map.get(source_id).unwrap();
        let tgt_idx = *self.node_map.get(target_id).unwrap();
        if let Some(edge_idx) = self.graph.find_edge(src_idx, tgt_idx) {
            if let Some(weight) = self.graph.edge_weight_mut(edge_idx) {
                *weight = data;
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
    pub async fn delete_node_impl(&mut self, node_id: &str) -> Result<()> {
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
    pub async fn delete_edge_impl(&mut self, source_id: &str, target_id: &str) -> Result<()> {
        let source_id = source_id.to_string();
        let target_id = target_id.to_string();
        let node_map = self.node_map.clone();
        let mut graph = self.graph.clone();
        
        let result = tokio::task::spawn_blocking(move || {
            if let (Some(&src_idx), Some(&tgt_idx)) = (node_map.get(&source_id), node_map.get(&target_id)) {
                if let Some(edge_idx) = graph.find_edge(src_idx, tgt_idx) {
                    graph.remove_edge(edge_idx);
                }
            }
            graph
        }).await?;

        // Update the storage state with the modified graph
        self.graph = result;
        Ok(())
    }

    /// Checks if a node with the given ID exists in the graph.
    /// 
    /// # Arguments
    /// * `node_id` - ID of the node to check
    /// 
    /// # Returns
    /// true if the node exists, false otherwise
    pub async fn has_node_impl(&self, node_id: &str) -> bool {
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
    pub async fn has_edge_impl(&self, source_id: &str, target_id: &str) -> bool {
        let source_id = source_id.to_string();
        let target_id = target_id.to_string();
        let node_map = self.node_map.clone();
        let graph = self.graph.clone();
        tokio::task::spawn_blocking(move || {
            if let (Some(&src_idx), Some(&tgt_idx)) = (node_map.get(&source_id), node_map.get(&target_id)) {
                graph.find_edge(src_idx, tgt_idx).is_some()
        } else {
            false
        }
        }).await.unwrap_or(false)
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
            embedding_handlers: self.embedding_handlers.clone(),
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
            self.graph.find_edge(src_idx, tgt_idx)
                .and_then(|edge_idx| self.graph.edge_weight(edge_idx).cloned())
        } else {
            // Return default edge properties when no edge found
            Some(EdgeData {
                weight: 0.0,
                description: None,
                keywords: None,
            })
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
    pub async fn upsert_nodes_impl(&mut self, nodes: Vec<(String, HashMap<String, serde_json::Value>)>) -> crate::types::Result<()> {
        let mut graph = self.graph.clone();
        let mut node_map = self.node_map.clone();

        let result = tokio::task::spawn_blocking(move || {
        for (node_id, attrs) in nodes {
                if let Some(&node_idx) = node_map.get(&node_id) {
                    if let Some(node_data) = graph.node_weight_mut(node_idx) {
                        for (k, v) in attrs {
                            node_data.attributes.insert(k, v);
                        }
                    }
                } else {
                    let node_data = NodeData {
                        id: node_id.clone(),
                        attributes: attrs,
                    };
                    let idx = graph.add_node(node_data);
                    node_map.insert(node_id, idx);
                }
            }
            (graph, node_map)
        }).await?;

        self.graph = result.0;
        self.node_map = result.1;
        Ok(())
    }

    /// Performs batch insertion or update of multiple edges.
    /// 
    /// # Arguments
    /// * `edges` - Vector of (source_id, target_id, EdgeData) tuples to upsert
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
    pub async fn upsert_edges_impl(&mut self, edges: Vec<(String, String, EdgeData)>) -> crate::types::Result<()> {
        let mut graph = self.graph.clone();
        let node_map = self.node_map.clone();

        let result = tokio::task::spawn_blocking(move || {
        for (src, tgt, data) in edges {
                if let (Some(&src_idx), Some(&tgt_idx)) = (node_map.get(&src), node_map.get(&tgt)) {
                    if let Some(edge_idx) = graph.find_edge(src_idx, tgt_idx) {
                        if let Some(weight) = graph.edge_weight_mut(edge_idx) {
                            *weight = data;
                        }
                    } else {
                        graph.add_edge(src_idx, tgt_idx, data);
                    }
                }
            }
            graph
        }).await?;

        self.graph = result;
        Ok(())
    }

    /// Deletes multiple nodes in a single operation.
    /// 
    /// # Arguments
    /// * `node_ids` - Vector of node IDs to delete
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
    pub async fn remove_nodes_impl(&mut self, node_ids: Vec<String>) -> crate::types::Result<()> {
        let mut graph = self.graph.clone();
        let mut node_map = self.node_map.clone();

        let result = tokio::task::spawn_blocking(move || {
        for node_id in node_ids {
                if let Some(&idx) = node_map.get(&node_id) {
                    graph.remove_node(idx);
                    node_map.remove(&node_id);
                }
        }
            (graph, node_map)
        }).await?;

        self.graph = result.0;
        self.node_map = result.1;
        Ok(())
    }

    /// Deletes multiple edges in a single operation.
    /// 
    /// # Arguments
    /// * `edges` - Vector of (source_id, target_id) pairs representing edges to delete
    /// 
    /// # Returns
    /// A Result indicating success or failure of the operation
    pub async fn remove_edges_impl(&mut self, edges: Vec<(String, String)>) -> crate::types::Result<()> {
        for (src, tgt) in edges {
            self.delete_edge_impl(&src, &tgt).await?;
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
    pub async fn cascading_delete_node(&mut self, node_id: &str) -> Result<()> {
        // First, delete the target node normally
        self.delete_node_impl(node_id).await?;

        let node_id_str = node_id.to_string();
        let mut graph = self.graph.clone();

        let result = tokio::task::spawn_blocking(move || {
            for node_idx in graph.node_indices() {
                if let Some(node_data) = graph.node_weight_mut(node_idx) {
                    if let Some(val) = node_data.attributes.get("source_id") {
                        if *val == serde_json::Value::String(node_id_str.clone()) {
                            node_data.attributes.remove("source_id");
                        }
                    }
                }
            }
            graph
        }).await?;

        self.graph = result;
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
    pub async fn get_edge_with_default(&self, source_id: &str, target_id: &str) -> EdgeData {
        <Self as GraphStorage>::get_edge(self, source_id, target_id)
            .await
            .unwrap_or_else(|| Self::default_edge_properties())
    }

    /// Generate embeddings for all nodes in the graph using the specified algorithm
    pub async fn embed_nodes(&self, algorithm: EmbeddingAlgorithm) -> Result<(Vec<f32>, Vec<String>)> {
        // Map the EmbeddingAlgorithm enum to a dictionary key string
        let algorithm_key = match algorithm {
            EmbeddingAlgorithm::Node2Vec => "node2vec",
            // Future algorithms can be added here
        };
        
        // Get the handler from the dictionary
        let handler = *self.embedding_handlers.get(algorithm_key).ok_or_else(|| crate::types::Error::Storage(format!("Embedding algorithm not supported: {}", algorithm_key)))?;
        let graph = self.graph.clone();
        
        tokio::task::spawn_blocking(move || {
            let (embeddings, indices) = handler(&graph)?;
            // Convert node indices to node IDs
            let node_ids = indices.into_iter()
                .filter_map(|idx| graph.node_weight(idx).map(|node| node.id.clone()))
                .collect();
            Ok((embeddings, node_ids))
        }).await?
    }

    /// Generate embeddings with custom configuration
    pub fn embed_nodes_with_config(
        &self,
        algorithm: EmbeddingAlgorithm,
        config: Node2VecConfig,
    ) -> Result<(Vec<f32>, Vec<String>)> {
        match algorithm {
            EmbeddingAlgorithm::Node2Vec => {
                let (embeddings, indices) = generate_node2vec_embeddings(&self.graph, &config)?;
                
                // Convert node indices to node IDs
                let node_ids = indices.into_iter()
                    .filter_map(|idx| self.graph.node_weight(idx).map(|node| node.id.clone()))
                    .collect();
                
        Ok((embeddings, node_ids))
            }
        }
    }

    async fn query_with_keywords(&self, keywords: &[String]) -> Result<String> {
        let mut context = String::new();
        
        for keyword in keywords {
            // Search for nodes containing the keyword
            for node_index in self.graph.node_indices() {
                if let Some(node_data) = self.graph.node_weight(node_index) {
                    if let Some(content) = node_data.attributes.get("content").and_then(|v| v.as_str()) {
                        if content.to_lowercase().contains(&keyword.to_lowercase()) {
                            // Add node content to context
                            context.push_str(content);
                            context.push_str("\n");
                            
                            // Get connected nodes (both incoming and outgoing edges)
                            for neighbor in self.graph.neighbors(node_index) {
                                if let Some(neighbor_data) = self.graph.node_weight(neighbor) {
                                    if let Some(neighbor_content) = neighbor_data.attributes.get("content").and_then(|v| v.as_str()) {
                                        context.push_str(neighbor_content);
                                        context.push_str("\n");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(context)
    }
}

#[async_trait]
impl GraphStorage for PetgraphStorage {
    async fn finalize(&mut self) -> Result<()> {
        // Create handler and write with pretty printing
        let handler = GraphMlHandler::new(self.graph.clone()).pretty_print(true);
        let file_path = self.file_path.clone();
        
        info!("Saving graph with {} nodes and {} edges to {}", 
              self.graph.node_count(), 
              self.graph.edge_count(),
              file_path.display());

        // Write graph to file
        tokio::task::spawn_blocking(move || {
            handler.write_graphml(&file_path)
        }).await??;

        // Verify file was written
        let check_path = self.file_path.clone();
        if !check_path.exists() {
            return Err(Error::Storage("Failed to write graph file".to_string()));
        }

        info!("Successfully saved graph to {}", self.file_path.display());
        Ok(())
    }

    async fn initialize(&mut self) -> Result<()> {
        info!("Initializing graph storage from {}", self.file_path.display());
        
        // If file doesn't exist, start with empty graph
        if !self.file_path.exists() {
            info!("No existing graph file found, starting with empty graph");
            self.graph = Graph::new();
            self.node_map = HashMap::new();
            return Ok(());
        }

        // Load graph from GraphML file
        let file_path = self.file_path.clone();
        let result = tokio::task::spawn_blocking(move || {
            GraphMlHandler::read_graphml(&file_path)
        }).await??;

        // Update internal state
        self.graph = result;
        
        // Rebuild node map
        self.node_map.clear();
        for node_idx in self.graph.node_indices() {
            if let Some(node_data) = self.graph.node_weight(node_idx) {
                self.node_map.insert(node_data.id.clone(), node_idx);
            }
        }

        info!("Successfully loaded graph with {} nodes and {} edges", 
              self.graph.node_count(), 
              self.graph.edge_count());

        Ok(())
    }

    async fn has_node(&self, node_id: &str) -> bool {
        let node_id = node_id.to_string();
        let node_map = self.node_map.clone();
        tokio::task::spawn_blocking(move || {
            node_map.contains_key(&node_id)
        }).await.unwrap_or(false)
    }

    async fn has_edge(&self, source_id: &str, target_id: &str) -> bool {
        self.has_edge_impl(source_id, target_id).await
    }

    async fn get_node(&self, node_id: &str) -> Option<NodeData> {
        let node_id = node_id.to_string();
        let node_map = self.node_map.clone();
        let graph = self.graph.clone();
        tokio::task::spawn_blocking(move || {
            node_map.get(&node_id).and_then(|&idx| graph.node_weight(idx).cloned())
        }).await.unwrap_or(None)
    }

    async fn get_edge(&self, source_id: &str, target_id: &str) -> Option<EdgeData> {
        let source_id = source_id.to_string();
        let target_id = target_id.to_string();
        let node_map = self.node_map.clone();
        let graph = self.graph.clone();
        tokio::task::spawn_blocking(move || {
            if let (Some(&src_idx), Some(&tgt_idx)) = (node_map.get(&source_id), node_map.get(&target_id)) {
                graph.find_edge(src_idx, tgt_idx)
                    .and_then(|edge_idx| graph.edge_weight(edge_idx).cloned())
            } else {
                None
            }
        }).await.unwrap_or(None)
    }

    async fn get_node_edges(&self, node_id: &str) -> Option<Vec<(String, String, EdgeData)>> {
        let node_id = node_id.to_string();
        let node_map = self.node_map.clone();
        let graph = self.graph.clone();
        tokio::task::spawn_blocking(move || {
            node_map.get(&node_id).map(|&node_idx| {
                let mut edges = Vec::new();
                for edge in graph.edges(node_idx) {
                    let src = graph[edge.source()].id.clone();
                    let tgt = graph[edge.target()].id.clone();
                    edges.push((
                        src,
                        tgt,
                        edge.weight().clone()
                    ));
                }
                edges
            })
        }).await.unwrap_or(None)
    }

    async fn node_degree(&self, node_id: &str) -> Result<usize> {
        let node_id = node_id.to_string();
        let node_map = self.node_map.clone();
        let graph = self.graph.clone();
        tokio::task::spawn_blocking(move || {
            if let Some(&idx) = node_map.get(&node_id) {
                let out_deg = graph.edges_directed(idx, Direction::Outgoing).count();
                let in_deg = graph.edges_directed(idx, Direction::Incoming).count();
                Ok(out_deg + in_deg)
            } else {
                Ok(0)
            }
        }).await?
    }

    async fn edge_degree(&self, src_id: &str, tgt_id: &str) -> Result<usize> {
        let src_id = src_id.to_string();
        let tgt_id = tgt_id.to_string();
        let node_map = self.node_map.clone();
        let graph = self.graph.clone();
        tokio::task::spawn_blocking(move || {
            let mut total = 0;
            if let Some(&src_idx) = node_map.get(&src_id) {
                total += graph.edges_directed(src_idx, Direction::Outgoing).count();
                total += graph.edges_directed(src_idx, Direction::Incoming).count();
            }
            if let Some(&tgt_idx) = node_map.get(&tgt_id) {
                total += graph.edges_directed(tgt_idx, Direction::Outgoing).count();
                total += graph.edges_directed(tgt_idx, Direction::Incoming).count();
            }
            Ok(total)
        }).await?
    }

    async fn upsert_node(&mut self, node_id: &str, attributes: HashMap<String, serde_json::Value>) -> Result<()> {
        let node_id = node_id.to_string();
        let attributes = attributes.clone();
        let mut graph = self.graph.clone();
        let mut node_map = self.node_map.clone();

        tracing::info!("Upserting node {} with {} attributes", node_id, attributes.len());

        let result = tokio::task::spawn_blocking(move || {
            if let Some(&node_idx) = node_map.get(&node_id) {
                if let Some(node_data) = graph.node_weight_mut(node_idx) {
                    for (k, v) in attributes {
                        node_data.attributes.insert(k, v);
                    }
                    tracing::debug!("Updated existing node {}", node_id);
                }
            } else {
                let node_data = NodeData {
                    id: node_id.clone(),
                    attributes,
                };
                let idx = graph.add_node(node_data);
                node_map.insert(node_id.clone(), idx);
                tracing::debug!("Created new node {}", node_id);
            }
            (graph, node_map)
        }).await?;

        self.graph = result.0;
        self.node_map = result.1;
        Ok(())
    }

    async fn upsert_edge(&mut self, source_id: &str, target_id: &str, data: EdgeData) -> Result<()> {
        let source_id = source_id.to_string();
        let target_id = target_id.to_string();
        let data = data.clone();
        
        tracing::info!("Upserting edge {} -> {} with weight {}", source_id, target_id, data.weight);

        // First ensure nodes exist
        if !<Self as crate::storage::graph::GraphStorage>::has_node(self, &source_id).await {
            tracing::debug!("Creating missing source node {}", source_id);
            <Self as crate::storage::graph::GraphStorage>::upsert_node(self, &source_id, std::collections::HashMap::new()).await?;
        }
        if !<Self as crate::storage::graph::GraphStorage>::has_node(self, &target_id).await {
            tracing::debug!("Creating missing target node {}", target_id);
            <Self as crate::storage::graph::GraphStorage>::upsert_node(self, &target_id, std::collections::HashMap::new()).await?;
        }

        // Get the indices after ensuring nodes exist
        let src_idx = match self.node_map.get(&source_id) {
            Some(&idx) => idx,
            None => return Err(Error::Storage(format!("Source node {} not found", source_id))),
        };
        let tgt_idx = match self.node_map.get(&target_id) {
            Some(&idx) => idx,
            None => return Err(Error::Storage(format!("Target node {} not found", target_id))),
        };

        let mut graph = self.graph.clone();
        
        let result = tokio::task::spawn_blocking(move || {
            if let Some(edge_idx) = graph.find_edge(src_idx, tgt_idx) {
                if let Some(weight) = graph.edge_weight_mut(edge_idx) {
                    *weight = data.clone();
                    tracing::debug!("Updated existing edge {} -> {}", source_id, target_id);
                }
            } else {
                graph.add_edge(src_idx, tgt_idx, data.clone());
                tracing::debug!("Created new edge {} -> {}", source_id, target_id);
            }
            graph
        }).await?;

        self.graph = result;
        Ok(())
    }

    async fn delete_node(&mut self, node_id: &str) -> Result<()> {
        let node_id_str = node_id.to_string();
        let mut graph = self.graph.clone();
        let mut node_map = self.node_map.clone();

        let result = tokio::task::spawn_blocking(move || {
            if let Some(&node_idx) = node_map.get(&node_id_str) {
                graph.remove_node(node_idx);
                node_map.remove(&node_id_str);
            }
            (graph, node_map)
        }).await?;

        self.graph = result.0;
        self.node_map = result.1;
        Ok(())
    }

    async fn delete_edge(&mut self, source_id: &str, target_id: &str) -> Result<()> {
        let source_id = source_id.to_string();
        let target_id = target_id.to_string();
        let node_map = self.node_map.clone();
        let mut graph = self.graph.clone();
        
        let result = tokio::task::spawn_blocking(move || {
            if let (Some(&src_idx), Some(&tgt_idx)) = (node_map.get(&source_id), node_map.get(&target_id)) {
                if let Some(edge_idx) = graph.find_edge(src_idx, tgt_idx) {
                    graph.remove_edge(edge_idx);
                }
            }
            graph
        }).await?;

        // Update the storage state with the modified graph
        self.graph = result;
        Ok(())
    }

    async fn upsert_nodes(&mut self, nodes: Vec<(String, HashMap<String, serde_json::Value>)>) -> Result<()> {
        for (node_id, attrs) in nodes {
            <Self as crate::storage::graph::GraphStorage>::upsert_node(self, &node_id, attrs).await?;
        }
        Ok(())
    }

    async fn upsert_edges(&mut self, edges: Vec<(String, String, EdgeData)>) -> Result<()> {
        for (src, tgt, data) in edges {
            self.upsert_edge(&src, &tgt, data).await?;
        }
        Ok(())
    }

    async fn remove_nodes(&mut self, node_ids: Vec<String>) -> Result<()> {
        let mut graph = self.graph.clone();
        let mut node_map = self.node_map.clone();

        let result = tokio::task::spawn_blocking(move || {
        for node_id in node_ids {
                if let Some(&idx) = node_map.get(&node_id) {
                    graph.remove_node(idx);
                    node_map.remove(&node_id);
                }
        }
            (graph, node_map)
        }).await?;

        self.graph = result.0;
        self.node_map = result.1;
        Ok(())
    }

    async fn remove_edges(&mut self, edges: Vec<(String, String)>) -> Result<()> {
        for (src, tgt) in edges {
            self.delete_edge_impl(&src, &tgt).await?;
        }
        Ok(())
    }

    async fn embed_nodes(&self, algorithm: EmbeddingAlgorithm) -> Result<(Vec<f32>, Vec<String>)> {
        let graph = self.graph.clone();
        tokio::task::spawn_blocking(move || {
            match algorithm {
                EmbeddingAlgorithm::Node2Vec => {
                    let config = Node2VecConfig::default();
                    let (embeddings, indices) = generate_node2vec_embeddings(&graph, &config)?;
                    
                    // Convert node indices to node IDs
                    let node_ids = indices.into_iter()
                        .filter_map(|idx| graph.node_weight(idx).map(|node| node.id.clone()))
                        .collect();
                    
                    Ok((embeddings, node_ids))
                }
            }
        }).await?
    }

    async fn query_with_keywords(&self, keywords: &[String]) -> Result<String> {
        let mut context = String::new();
        
        for keyword in keywords {
            // Search for nodes containing the keyword
            for node_index in self.graph.node_indices() {
                if let Some(node_data) = self.graph.node_weight(node_index) {
                    if let Some(content) = node_data.attributes.get("content").and_then(|v| v.as_str()) {
                        if content.to_lowercase().contains(&keyword.to_lowercase()) {
                            // Add node content to context
                            context.push_str(content);
                            context.push_str("\n");
                            
                            // Get connected nodes (both incoming and outgoing edges)
                            for neighbor in self.graph.neighbors(node_index) {
                                if let Some(neighbor_data) = self.graph.node_weight(neighbor) {
                                    if let Some(neighbor_content) = neighbor_data.attributes.get("content").and_then(|v| v.as_str()) {
                                        context.push_str(neighbor_content);
                                        context.push_str("\n");
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(context)
    }
}

// Helper function for Node2Vec embedding
fn node2vec_handler(graph: &petgraph::graph::Graph<crate::storage::graph::NodeData, crate::storage::graph::EdgeData>) -> Result<(Vec<f32>, Vec<NodeIndex>)> {
    let config = Node2VecConfig::default();
    generate_node2vec_embeddings(graph, &config)
} 