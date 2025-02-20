use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;

use super_lightrag::storage::vector::{NanoVectorStorage, VectorData};
use super_lightrag::storage::vector::{SearchResult, UpsertResponse, VectorStorage};
use super_lightrag::types::Config;
use super_lightrag::types::Result;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_upsert_and_query() -> Result<()> {
        // Create a default config
        let config = Config::default();
        let mut storage = NanoVectorStorage::new(&config)?;
        storage.initialize().await?;

        // Create test vectors
        let vec1 = VectorData {
            id: "vec1".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
        };

        let mut meta2 = HashMap::new();
        meta2.insert("info".to_string(), json!("second"));
        let vec2 = VectorData {
            id: "vec2".to_string(),
            vector: vec![0.0, 1.0, 0.0],
            metadata: meta2.clone(),
        };

        // Upsert both vectors
        let response: UpsertResponse = storage.upsert(vec![vec1.clone(), vec2.clone()]).await?;
        assert_eq!(response.inserted.len(), 2);
        assert_eq!(response.updated.len(), 0);

        // Query with a vector similar to vec1
        let query_vec = vec![1.0, 0.0, 0.0];
        let results = storage.query(query_vec.clone(), 2).await?;
        assert!(!results.is_empty(), "Query results should not be empty");
        // Expect vec1 to have high similarity
        assert_eq!(results[0].id, "vec1");
        assert!(results[0].distance >= 0.9, "Expected high similarity for vec1");

        // Update vec2 with a vector closer to vec1
        let vec2_updated = VectorData {
            id: "vec2".to_string(),
            vector: vec![0.9, 0.1, 0.0],
            metadata: meta2.clone(),
        };
        let response2: UpsertResponse = storage.upsert(vec![vec2_updated.clone()]).await?;
        assert_eq!(response2.updated.len(), 1);
        assert_eq!(response2.inserted.len(), 0);

        // Now query again; expect both vec1 and updated vec2 in the results
        let results2 = storage.query(query_vec, 5).await?;
        let ids: Vec<String> = results2.iter().map(|r| r.id.clone()).collect();
        assert!(ids.contains(&"vec1".to_string()));
        assert!(ids.contains(&"vec2".to_string()));

        // Delete vec1
        storage.delete(vec!["vec1".to_string()]).await?;
        let results3 = storage.query(vec![1.0, 0.0, 0.0], 5).await?;
        let ids_after_delete: Vec<String> = results3.iter().map(|r| r.id.clone()).collect();
        assert!(!ids_after_delete.contains(&"vec1".to_string()), "vec1 should be deleted");

        storage.finalize().await?;
        Ok(())
    }
} 