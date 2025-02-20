use serde_json::json;
use std::collections::HashMap;
use tempfile::TempDir;

use super_lightrag::storage::vector::{NanoVectorStorage, VectorData, VectorStorage};
use super_lightrag::types::{Config, Result};

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_upsert_and_query() -> Result<()> {
        // Create a temporary directory for testing
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
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
        let response = storage.upsert(vec![vec1.clone(), vec2.clone()]).await?;
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

        let response = storage.upsert(vec![vec2_updated]).await?;
        assert_eq!(response.inserted.len(), 0);
        assert_eq!(response.updated.len(), 1);

        // Query again with the same vector
        let results = storage.query(query_vec.clone(), 2).await?;
        assert!(!results.is_empty(), "Query results should not be empty after reload");
        assert_eq!(results[0].id, "vec1");
        assert!(results[0].distance >= 0.9, "Expected high similarity for vec1 after reload");

        // Test deletion
        storage.delete(vec!["vec2".to_string()]).await?;
        let results = storage.query(query_vec, 2).await?;
        assert_eq!(results.len(), 1, "Should only have one result after deletion");
        assert_eq!(results[0].id, "vec1");

        Ok(())
    }

    #[tokio::test]
    async fn test_batch_operations() -> Result<()> {
        // Create a temporary directory for testing
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
        let mut storage = NanoVectorStorage::new(&config)?;
        storage.initialize().await?;

        // Create a batch of test vectors
        let mut vectors = Vec::new();
        for i in 0..10 {
            let mut metadata = HashMap::new();
            metadata.insert("index".to_string(), json!(i));
            vectors.push(VectorData {
                id: format!("vec{}", i),
                vector: vec![i as f32, (i + 1) as f32, (i + 2) as f32],
                metadata,
            });
        }

        // Insert batch
        let response = storage.upsert(vectors).await?;
        assert_eq!(response.inserted.len(), 10);
        assert_eq!(response.updated.len(), 0);

        // Query for nearest neighbors to vec5
        let query_vec = vec![5.0, 6.0, 7.0];
        let results = storage.query(query_vec.clone(), 3).await?;
        assert_eq!(results.len(), 3, "Should get 3 nearest neighbors");
        assert_eq!(results[0].id, "vec5", "First result should be vec5");

        // Test batch deletion
        let delete_ids = (0..5).map(|i| format!("vec{}", i)).collect();
        storage.delete(delete_ids).await?;
        
        let results = storage.query(query_vec, 10).await?;
        assert_eq!(results.len(), 5, "Should have 5 vectors remaining");
        assert!(results.iter().all(|r| r.id.trim_start_matches("vec").parse::<i32>().unwrap() >= 5), 
                "All remaining vectors should have id >= 5");

        Ok(())
    }
} 