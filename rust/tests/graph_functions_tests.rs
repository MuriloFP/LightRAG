use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use tempfile::TempDir;

use super_lightrag::storage::graph::petgraph_storage::{PetgraphStorage, GraphPattern, PatternMatch, EdgeData, NodeData};
use super_lightrag::storage::graph::GraphStorage;
use super_lightrag::types::{Config, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_crud_operations() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        let mut storage = PetgraphStorage::new(&config)?;
        storage.initialize().await?;
        
        // Upsert nodes
        let mut attrs1 = HashMap::new();
        attrs1.insert("name".into(), json!("Alice"));
        storage.upsert_node("node1", attrs1.clone())?;

        let mut attrs2 = HashMap::new();
        attrs2.insert("name".into(), json!("Bob"));
        storage.upsert_node("node2", attrs2.clone())?;
        
        assert!(storage.has_node("node1"));
        assert!(storage.has_node("node2"));
        
        // Upsert edge
        let edge_data = EdgeData {
            weight: 1.0,
            description: Some("friendship".to_string()),
            keywords: Some(vec!["social".to_string()]),
        };
        storage.upsert_edge("node1", "node2", edge_data.clone())?;
        assert!(storage.has_edge("node1", "node2"));
        
        // Delete edge
        storage.delete_edge("node1", "node2")?;
        assert!(!storage.has_edge("node1", "node2"));
        
        // Test get_edge_with_default: should return default edge data
        let default_edge = storage.get_edge_with_default("node1", "node2");
        assert_eq!(default_edge.weight, 0.0);
        
        // Delete node
        storage.delete_node("node2")?;
        assert!(!storage.has_node("node2"));
        
        storage.finalize().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_degree_and_path() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        let mut storage = PetgraphStorage::new(&config)?;
        storage.initialize().await?;
        
        // Build a chain: node1 -> node2 -> node3
        storage.upsert_node("node1", HashMap::new())?;
        storage.upsert_node("node2", HashMap::new())?;
        storage.upsert_node("node3", HashMap::new())?;
        
        let edge = EdgeData {
            weight: 1.0,
            description: None,
            keywords: None,
        };
        storage.upsert_edge("node1", "node2", edge.clone())?;
        storage.upsert_edge("node2", "node3", edge.clone())?;
        
        let degree = storage.node_degree("node2")?;
        // node2 should have in-degree 1 and out-degree 1 = total 2
        assert_eq!(degree, 2);
        
        let path = storage.find_path("node1", "node3").await?;
        assert_eq!(path, vec!["node1".to_string(), "node2".to_string(), "node3".to_string()]);
        
        storage.finalize().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_get_neighborhood() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        let mut storage = PetgraphStorage::new(&config)?;
        storage.initialize().await?;
        
        // Create a star graph: center and three peripheral nodes
        storage.upsert_node("center", HashMap::new())?;
        for i in 2..=4 {
            let node_id = format!("node{}", i);
            storage.upsert_node(&node_id, HashMap::new())?;
            storage.upsert_edge("center", &node_id, EdgeData { weight: 1.0, description: None, keywords: None })?;
        }
        
        let neighborhood = storage.get_neighborhood("center", 1).await?;
        // Should include center plus three neighbors
        assert_eq!(neighborhood.len(), 4);
        
        storage.finalize().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_extract_subgraph() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
        let mut storage = PetgraphStorage::new(&config)?;
        storage.initialize().await?;
        
        // Create nodes: A, B, C, D
        for id in &["A", "B", "C", "D"] {
            storage.upsert_node(id, HashMap::new())?;
        }
        // Create edges: A-B, B-C, C-D, A-D
        let edge = EdgeData { weight: 1.0, description: Some("link".to_string()), keywords: None };
        storage.upsert_edge("A", "B", edge.clone())?;
        storage.upsert_edge("B", "C", edge.clone())?;
        storage.upsert_edge("C", "D", edge.clone())?;
        storage.upsert_edge("A", "D", edge.clone())?;
        
        // Extract subgraph with nodes A, B, C
        let sub_storage = storage.extract_subgraph(&vec!["A".to_string(), "B".to_string(), "C".to_string()]).await?;
        assert!(sub_storage.has_node("A"));
        assert!(sub_storage.has_node("B"));
        assert!(sub_storage.has_node("C"));
        assert!(!sub_storage.has_node("D"));
        assert!(sub_storage.has_edge("A", "B"));
        assert!(sub_storage.has_edge("B", "C"));
        assert!(!sub_storage.has_edge("A", "D"));
        
        storage.finalize().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_match_and_filter() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
        let mut storage = PetgraphStorage::new(&config)?;
        storage.initialize().await?;
        
        // Upsert nodes with an attribute 'label'
        let mut attrs_a = HashMap::new();
        attrs_a.insert("label".into(), json!("TypeA"));
        storage.upsert_node("A", attrs_a.clone())?;
        
        let mut attrs_b = HashMap::new();
        attrs_b.insert("label".into(), json!("TypeB"));
        storage.upsert_node("B", attrs_b.clone())?;
        
        // Upsert edge with description and keywords
        let edge_data = EdgeData { weight: 1.0, description: Some("connects A and B".to_string()), keywords: Some(vec!["relation".to_string()]) };
        storage.upsert_edge("A", "B", edge_data.clone())?;
        
        // Test match_pattern
        let pattern = GraphPattern { keyword: Some("Type".to_string()), node_attribute: None };
        let matches = storage.match_pattern(pattern).await?;
        assert!(!matches.is_empty());
        
        // Test filter_nodes_by_attribute
        let filtered_nodes = storage.filter_nodes_by_attribute("label", "TypeA");
        assert_eq!(filtered_nodes.len(), 1);
        assert_eq!(filtered_nodes[0].id, "A");
        
        // Test filter_edges_by_attribute
        let filtered_edges = storage.filter_edges_by_attribute("relation");
        assert_eq!(filtered_edges.len(), 1);
        let (src, tgt, _) = &filtered_edges[0];
        assert_eq!(src, "A");
        assert_eq!(tgt, "B");
        
        storage.finalize().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_batch_operations_and_cascading() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
        let mut storage = PetgraphStorage::new(&config)?;
        storage.initialize().await?;
        
        // Batch upsert nodes
        let nodes_data = vec![
            ("N1".to_string(), { let mut m = HashMap::new(); m.insert("attr".to_string(), json!("val1")); m }),
            ("N2".to_string(), { let mut m = HashMap::new(); m.insert("attr".to_string(), json!("val2")); m }),
        ];
        storage.upsert_nodes(nodes_data)?;
        assert!(storage.has_node("N1"));
        assert!(storage.has_node("N2"));
        
        // Batch upsert edges
        let edge_data = EdgeData { weight: 2.0, description: Some("edge batch".to_string()), keywords: None };
        let edges_data = vec![
            ("N1".to_string(), "N2".to_string(), edge_data.clone()),
            ("N2".to_string(), "N1".to_string(), edge_data.clone()),
        ];
        storage.upsert_edges(edges_data)?;
        assert!(storage.has_edge("N1", "N2"));
        
        // Batch delete edges
        storage.delete_edges_batch(vec![("N2".to_string(), "N1".to_string())])?;
        assert!(!storage.has_edge("N2", "N1"));
        
        // Batch delete nodes
        storage.delete_nodes_batch(vec!["N2".to_string()])?;
        assert!(!storage.has_node("N2"));
        
        // Test cascading deletion
        storage.upsert_node("N3", { let mut m = HashMap::new(); m.insert("source_id".to_string(), json!("N1")); m })?;
        storage.cascading_delete_node("N1")?;
        assert!(!storage.has_node("N1"));
        if let Some(n3) = storage.get_node("N3") {
            assert!(n3.attributes.get("source_id").is_none());
        }
        
        storage.finalize().await?;
        Ok(())
    }

    #[tokio::test]
    async fn test_stabilize_and_embedding() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
        let mut storage = PetgraphStorage::new(&config)?;
        storage.initialize().await?;
        
        // Upsert nodes in unsorted order
        storage.upsert_node("B", HashMap::new())?;
        storage.upsert_node("A", HashMap::new())?;
        storage.upsert_node("C", HashMap::new())?;
        
        storage.stabilize_graph()?; // Should reorder nodes in sorted order
        
        // Check that sorted order is A, B, C
        let mut node_ids: Vec<String> = storage.graph.node_weights().map(|n| n.id.clone()).collect();
        node_ids.sort();
        assert_eq!(node_ids, vec!["A".to_string(), "B".to_string(), "C".to_string()]);
        
        // Test embedding generation
        let (embeddings, ids) = storage.embed_nodes(4).await?;
        assert_eq!(embeddings.len(), ids.len() * 4);
        
        storage.finalize().await?;
        Ok(())
    }
} 