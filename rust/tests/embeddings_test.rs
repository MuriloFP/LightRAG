use super_lightrag::storage::graph::embeddings::{EmbeddingAlgorithm, Node2VecConfig, generate_node2vec_embeddings};
use super_lightrag::storage::graph::{PetgraphStorage, GraphStorage};
use super_lightrag::types::{Config, Result};
use petgraph::graph::Graph;
use std::collections::HashMap;
use std::str::FromStr;
use tempfile::TempDir;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node2vec_basic() {
        // Create a simple test graph
        let mut graph = Graph::<(), ()>::new();
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        
        graph.add_edge(n1, n2, ());
        graph.add_edge(n2, n3, ());
        graph.add_edge(n3, n1, ()); // Create a cycle

        let config = Node2VecConfig {
            dimensions: 64,  // Smaller dimension for testing
            walk_length: 10,
            num_walks: 5,
            ..Default::default()
        };

        let result = generate_node2vec_embeddings(&graph, &config);
        assert!(result.is_ok());

        let (embeddings, indices) = result.unwrap();
        
        // Check basic properties
        assert_eq!(indices.len(), 3); // Should have 3 nodes
        assert_eq!(embeddings.len(), 64 * 3); // 64-dim embeddings for 3 nodes
    }

    #[test]
    fn test_embedding_algorithm_from_str() {
        assert!(EmbeddingAlgorithm::from_str("node2vec").is_ok());
        assert!(EmbeddingAlgorithm::from_str("NODE2VEC").is_ok());
        assert!(EmbeddingAlgorithm::from_str("invalid").is_err());
    }

    #[test]
    fn test_node2vec_config() {
        let config = Node2VecConfig::default();
        assert_eq!(config.dimensions, 128);
        assert_eq!(config.walk_length, 80);
        assert_eq!(config.num_walks, 10);
        assert_eq!(config.p, 1.0);
        assert_eq!(config.q, 1.0);
        assert_eq!(config.window_size, 10);
        assert_eq!(config.iter, 1);
    }

    #[test]
    fn test_node2vec_empty_graph() {
        let graph = Graph::<(), ()>::new();
        let config = Node2VecConfig::default();
        
        let result = generate_node2vec_embeddings(&graph, &config);
        assert!(result.is_ok());
        
        let (embeddings, indices) = result.unwrap();
        assert!(embeddings.is_empty());
        assert!(indices.is_empty());
    }

    #[tokio::test]
    async fn test_petgraph_storage_embeddings() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
        let mut storage = PetgraphStorage::new(&config)?;
        storage.initialize().await?;
        
        // Create a simple triangle graph
        storage.upsert_node("A", HashMap::new()).await?;
        storage.upsert_node("B", HashMap::new()).await?;
        storage.upsert_node("C", HashMap::new()).await?;
        
        let edge_data = super_lightrag::storage::graph::EdgeData {
            weight: 1.0,
            description: Some("friendship".to_string()),
            keywords: Some(vec!["social".to_string()]),
        };
        
        storage.upsert_edge("A", "B", edge_data.clone()).await?;
        storage.upsert_edge("B", "C", edge_data.clone()).await?;
        storage.upsert_edge("C", "A", edge_data.clone()).await?;

        // Test default embeddings
        let (embeddings, node_ids) = storage.embed_nodes(EmbeddingAlgorithm::Node2Vec)?;
        assert_eq!(node_ids.len(), 3);
        assert_eq!(embeddings.len(), 128 * 3); // Default dimension is 128

        // Test custom config embeddings
        let custom_config = super_lightrag::storage::graph::embeddings::Node2VecConfig {
            dimensions: 32,
            walk_length: 5,
            num_walks: 3,
            ..Default::default()
        };
        let (embeddings, node_ids) = storage.embed_nodes_with_config(EmbeddingAlgorithm::Node2Vec, custom_config)?;
        assert_eq!(node_ids.len(), 3);
        assert_eq!(embeddings.len(), 32 * 3);

        storage.finalize().await?;
        Ok(())
    }
} 