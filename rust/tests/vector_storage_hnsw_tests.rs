use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;

use super_lightrag::storage::vector::{NanoVectorStorage, VectorData, VectorStorage, UpsertResponse, SearchResult};
use super_lightrag::types::{Config, Result};

#[tokio::test]
async fn test_hnsw_upsert_query_delete() -> Result<()> {
    // Create default configuration and a NanoVectorStorage instance
    let config = Config::default();
    let mut storage = NanoVectorStorage::new(&config)?;
    storage.initialize().await?;

    // Create two test vectors
    let vec1 = VectorData {
        id: "vec1".to_string(),
        vector: vec![1.0, 0.0, 0.0],
        metadata: HashMap::new(),
    };

    let mut meta = HashMap::new();
    meta.insert("info".to_string(), json!("second"));
    let vec2 = VectorData {
        id: "vec2".to_string(),
        vector: vec![0.0, 1.0, 0.0],
        metadata: meta,
    };

    // Upsert both vectors
    let upsert_response: UpsertResponse = storage.upsert(vec![vec1.clone(), vec2.clone()]).await?;
    assert_eq!(upsert_response.inserted.len(), 2, "Both vectors should be inserted");

    // Query with a vector close to vec1 (should get vec1 as top result)
    let query_vector = vec![1.0, 0.0, 0.0];
    let results = storage.query(query_vector.clone(), 2).await?;
    assert!(!results.is_empty(), "Query results should not be empty");
    assert_eq!(results[0].id, "vec1", "vec1 should be the top result");

    // Delete vec1
    storage.delete(vec!["vec1".to_string()]).await?;

    // Query again; vec1 should no longer be in the results
    let results_after_delete = storage.query(query_vector, 2).await?;
    for res in results_after_delete {
        assert_ne!(res.id, "vec1", "vec1 should not appear after deletion");
    }

    storage.finalize().await?;
    Ok(())
} 