use serde_json::json;
use std::collections::HashMap;
use tempfile::TempDir;

use super_lightrag::storage::vector::{NanoVectorStorage, VectorData, VectorStorage};
use super_lightrag::types::{Config, Result};

#[tokio::test]
async fn test_hnsw_basic_operations() -> Result<()> {
    // Create a temporary directory for testing
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    
    let mut storage = NanoVectorStorage::new(&config)?;
    storage.initialize().await?;

    // Create test vectors with normalized values
    let vec1 = VectorData {
        id: "vec1".to_string(),
        vector: vec![1.0, 0.0, 0.0],
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("label".to_string(), json!("first"));
            meta
        },
    };

    let vec2 = VectorData {
        id: "vec2".to_string(),
        vector: vec![0.0, 1.0, 0.0],
        metadata: {
            let mut meta = HashMap::new();
            meta.insert("label".to_string(), json!("second"));
            meta
        },
    };

    // Test batch upsert
    let response = storage.upsert(vec![vec1.clone(), vec2.clone()]).await?;
    assert_eq!(response.inserted.len(), 2);
    assert_eq!(response.updated.len(), 0);

    // Test cosine similarity search
    let query_vec = vec![0.9, 0.1, 0.0]; // More similar to vec1
    let results = storage.query(query_vec.clone(), 2).await?;
    assert!(!results.is_empty());
    assert_eq!(results[0].id, "vec1");
    assert!(results[0].distance > 0.8); // High similarity to vec1

    // Test persistence
    storage.finalize().await?;

    // Create new instance and verify data survived
    let mut new_storage = NanoVectorStorage::new(&config)?;
    new_storage.initialize().await?;
    let results = new_storage.query(vec![1.0, 0.0, 0.0], 1).await?;
    assert_eq!(results[0].id, "vec1");

    Ok(())
}

#[tokio::test]
async fn test_hnsw_updates_and_deletions() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    
    let mut storage = NanoVectorStorage::new(&config)?;
    storage.initialize().await?;

    // Insert initial vectors
    let vec1 = VectorData {
        id: "vec1".to_string(),
        vector: vec![1.0, 0.0, 0.0],
        metadata: HashMap::new(),
    };

    let vec2 = VectorData {
        id: "vec2".to_string(),
        vector: vec![0.0, 1.0, 0.0],
        metadata: HashMap::new(),
    };

    storage.upsert(vec![vec1, vec2]).await?;

    // Update vec2 with new vector
    let vec2_updated = VectorData {
        id: "vec2".to_string(),
        vector: vec![0.8, 0.2, 0.0], // More similar to vec1 now
        metadata: HashMap::new(),
    };

    let response = storage.upsert(vec![vec2_updated]).await?;
    assert_eq!(response.updated.len(), 1);
    assert_eq!(response.updated[0], "vec2");

    // Query to verify update
    let query_vec = vec![1.0, 0.0, 0.0];
    let results = storage.query(query_vec.clone(), 2).await?;
    assert!(results[1].distance > 0.7); // vec2 should now be more similar to query

    // Test deletion
    storage.delete(vec!["vec2".to_string()]).await?;
    let results = storage.query(query_vec, 2).await?;
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].id, "vec1");

    Ok(())
}

#[tokio::test]
async fn test_hnsw_batch_operations() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    
    let mut storage = NanoVectorStorage::new(&config)?;
    storage.initialize().await?;

    // Create a batch of test vectors
    let mut vectors = Vec::new();
    for i in 0..10 {
        let theta = (i as f32) * std::f32::consts::PI / 5.0;
        vectors.push(VectorData {
            id: format!("vec{}", i),
            vector: vec![theta.cos(), theta.sin(), 0.0],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("index".to_string(), json!(i));
                meta
            },
        });
    }

    // Test batch insert
    let response = storage.upsert(vectors).await?;
    assert_eq!(response.inserted.len(), 10);

    // Test batch query
    let query_vec = vec![1.0, 0.0, 0.0]; // Should be closest to vec0
    let results = storage.query(query_vec.clone(), 3).await?;
    assert_eq!(results.len(), 3);
    assert_eq!(results[0].id, "vec0");

    // Test batch delete
    let delete_ids: Vec<String> = (0..5).map(|i| format!("vec{}", i)).collect();
    storage.delete(delete_ids).await?;

    // Verify deletion
    let results = storage.query(query_vec, 10).await?;
    assert_eq!(results.len(), 5);
    assert!(results.iter().all(|r| r.id.trim_start_matches("vec").parse::<i32>().unwrap() >= 5));

    Ok(())
}

#[tokio::test]
async fn test_hnsw_persistence_and_reload() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    
    // First storage instance
    let mut storage1 = NanoVectorStorage::new(&config)?;
    storage1.initialize().await?;

    // Insert some vectors
    let vectors: Vec<VectorData> = (0..5).map(|i| {
        let theta = (i as f32) * std::f32::consts::PI / 2.5;
        VectorData {
            id: format!("vec{}", i),
            vector: vec![theta.cos(), theta.sin(), 0.0],
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("index".to_string(), json!(i));
                meta
            },
        }
    }).collect();

    storage1.upsert(vectors).await?;
    storage1.finalize().await?;

    // Create second instance and verify data
    let mut storage2 = NanoVectorStorage::new(&config)?;
    storage2.initialize().await?;

    let query_vec = vec![1.0, 0.0, 0.0];
    let results = storage2.query(query_vec, 2).await?;
    assert_eq!(results[0].id, "vec0"); // Should find the same nearest neighbor

    // Verify metadata persisted
    assert_eq!(
        results[0].metadata.get("index").and_then(|v| v.as_i64()),
        Some(0)
    );

    Ok(())
} 