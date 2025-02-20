use async_trait::async_trait;
use super_lightrag::storage::graph::{PetgraphStorage, EdgeData, NodeData, GraphStorage};
use super_lightrag::types::{Config, Result};
use serde_json::json;
use std::collections::HashMap;
use tempfile::TempDir;

#[tokio::test]
async fn test_upsert_and_query_nodes_and_edges() -> Result<()> {
    // Create a temporary working directory
    let temp_dir = TempDir::new().expect("Failed to create temp dir");

    // Create a custom config overriding the working_dir
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();

    // Create a new instance of PetgraphStorage
    let mut storage = PetgraphStorage::new(&config)?;
    // Import the GraphStorage trait to bring initialize and finalize into scope
    use super_lightrag::storage::graph::GraphStorage;
    storage.initialize().await?; // load graph; should be empty initially

    // Upsert nodes: "node1" and "node2"
    let mut node1_attributes = HashMap::new();
    node1_attributes.insert("label".to_string(), json!("Node 1"));
    storage.upsert_node("node1", node1_attributes)?;

    let mut node2_attributes = HashMap::new();
    node2_attributes.insert("label".to_string(), json!("Node 2"));
    storage.upsert_node("node2", node2_attributes)?;

    assert!(storage.has_node("node1"));
    assert!(storage.has_node("node2"));

    // Upsert an edge between node1 and node2
    let edge_data = EdgeData {
        weight: 1.5,
        description: Some("Edge from 1 to 2".to_string()),
        keywords: Some(vec!["test".to_string()]),
    };
    storage.upsert_edge("node1", "node2", edge_data.clone())?;

    assert!(storage.has_edge("node1", "node2"));

    // Check node_degree of node1, should be 1 (only one edge exists)
    let degree_node1 = storage.node_degree("node1")?;
    assert_eq!(degree_node1, 1);

    // Delete the edge between node1 and node2
    storage.delete_edge("node1", "node2")?;
    assert!(!storage.has_edge("node1", "node2"));

    // Upsert the edge again for persistence testing
    storage.upsert_edge("node1", "node2", edge_data.clone())?;

    // Delete node2, which should remove associated edges
    storage.delete_node("node2")?;
    assert!(!storage.has_node("node2"));
    assert!(!storage.has_edge("node1", "node2"));

    // Finalize storage (save graph)
    storage.finalize().await?;

    // Create a new storage instance to test loading from file
    let mut storage_new = PetgraphStorage::new(&config)?;
    storage_new.initialize().await?;

    // Check that node1 exists and node2 does not
    assert!(storage_new.has_node("node1"));
    assert!(!storage_new.has_node("node2"));

    // TempDir will clean up automatically
    Ok(())
} 