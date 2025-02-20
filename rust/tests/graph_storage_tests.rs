use std::collections::HashMap;
use tempfile::TempDir;
use serde_json::Value;

use super_lightrag::storage::graph::petgraph_storage::{PetgraphStorage, EdgeData};
use super_lightrag::storage::graph::GraphStorage;
use super_lightrag::types::{Config, Result};

#[tokio::test]
async fn test_graph_storage() -> Result<()> {
    // Create a temporary working directory
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    
    // Create a custom config overriding the working_dir
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    
    // Create a new storage instance
    let mut storage = PetgraphStorage::new(&config)?;
    storage.initialize().await?;

    // Test node operations
    let mut node_attrs = HashMap::new();
    node_attrs.insert("key1".to_string(), Value::String("value1".to_string()));
    
    storage.upsert_node("node1", node_attrs.clone()).await?;
    let has_node = storage.has_node("node1").await;
    assert!(has_node);

    // Test edge operations
    let edge_data = EdgeData {
        weight: 1.0,
        description: Some("test edge".to_string()),
        keywords: Some(vec!["keyword1".to_string()]),
    };

    storage.upsert_edge("node1", "node2", edge_data.clone()).await?;
    let has_edge = storage.has_edge("node1", "node2").await;
    assert!(has_edge);

    // Test edge deletion
    storage.delete_edge("node1", "node2").await?;
    let has_edge = storage.has_edge("node1", "node2").await;
    assert!(!has_edge);

    // Upsert the edge again for persistence testing
    storage.upsert_edge("node1", "node2", edge_data.clone()).await?;

    // Delete node2, which should remove associated edges
    storage.delete_node("node2").await?;
    let has_node2 = storage.has_node("node2").await;
    let has_edge = storage.has_edge("node1", "node2").await;
    assert!(!has_node2);
    assert!(!has_edge);

    // Finalize storage (save graph)
    storage.finalize().await?;

    // Create a new storage instance to test loading from file
    let mut storage_new = PetgraphStorage::new(&config)?;
    storage_new.initialize().await?;

    // Check that node1 exists and node2 does not
    let has_node1 = storage_new.has_node("node1").await;
    let has_node2 = storage_new.has_node("node2").await;
    assert!(has_node1);
    assert!(!has_node2);

    Ok(())
} 