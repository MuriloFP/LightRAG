use serde_json::json;
use std::collections::HashMap;
use tempfile::TempDir;
use std::time::SystemTime;

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
            created_at: SystemTime::now(),
            optimized: None,
        };

        let mut meta2 = HashMap::new();
        meta2.insert("info".to_string(), json!("second"));
        let vec2 = VectorData {
            id: "vec2".to_string(),
            vector: vec![0.0, 1.0, 0.0],
            metadata: meta2.clone(),
            created_at: SystemTime::now(),
            optimized: None,
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
            created_at: SystemTime::now(),
            optimized: None,
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
                created_at: SystemTime::now(),
                optimized: None,
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

    #[tokio::test]
    async fn test_get_with_fields() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
        let mut storage = NanoVectorStorage::new(&config)?;
        storage.initialize().await?;

        // Create test vector with multiple metadata fields
        let mut metadata = HashMap::new();
        metadata.insert("field1".to_string(), json!("value1"));
        metadata.insert("field2".to_string(), json!("value2"));
        metadata.insert("field3".to_string(), json!("value3"));

        let vec1 = VectorData {
            id: "vec1".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            metadata,
            created_at: SystemTime::now(),
            optimized: None,
        };

        // Insert vector
        storage.upsert(vec![vec1]).await?;

        // Test getting all fields
        let result = storage.get_with_fields(&["vec1".to_string()], None).await?;
        assert_eq!(result[0].metadata.len(), 3);

        // Test getting specific fields
        let fields = vec!["field1".to_string(), "field3".to_string()];
        let result = storage.get_with_fields(&["vec1".to_string()], Some(&fields)).await?;
        assert_eq!(result[0].metadata.len(), 2);
        assert!(result[0].metadata.contains_key("field1"));
        assert!(result[0].metadata.contains_key("field3"));
        assert!(!result[0].metadata.contains_key("field2"));

        Ok(())
    }

    #[tokio::test]
    async fn test_configuration_handling() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
        // Set custom configuration values
        config.extra_config.insert(
            "vector_db_storage.cosine_threshold".to_string(),
            json!(0.5)
        );
        config.extra_config.insert(
            "vector_db_storage.batch_size".to_string(),
            json!(64)
        );

        let mut storage = NanoVectorStorage::new(&config)?;
        storage.initialize().await?;

        // Insert test vectors
        let vec1 = VectorData {
            id: "vec1".to_string(),
            vector: vec![1.0, 0.0, 0.0],
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
            optimized: None,
        };

        let vec2 = VectorData {
            id: "vec2".to_string(),
            vector: vec![0.0, 1.0, 0.0],
            metadata: HashMap::new(),
            created_at: SystemTime::now(),
            optimized: None,
        };

        storage.upsert(vec![vec1, vec2]).await?;

        // Query with a vector that has 0.7 similarity to vec1
        // With threshold 0.5, this should return the result
        let query_vec = vec![0.7, 0.7, 0.0];
        let results = storage.query(query_vec.clone(), 2).await?;
        assert!(!results.is_empty(), "Should get results with similarity > 0.5");

        // Create a new storage with higher threshold
        config.extra_config.insert(
            "vector_db_storage.cosine_threshold".to_string(),
            json!(0.9)
        );
        let mut storage_strict = NanoVectorStorage::new(&config)?;
        storage_strict.initialize().await?;
        storage_strict.upsert(vec![
            VectorData {
                id: "vec1".to_string(),
                vector: vec![1.0, 0.0, 0.0],
                metadata: HashMap::new(),
                created_at: SystemTime::now(),
                optimized: None,
            },
            VectorData {
                id: "vec2".to_string(),
                vector: vec![0.0, 1.0, 0.0],
                metadata: HashMap::new(),
                created_at: SystemTime::now(),
                optimized: None,
            },
        ]).await?;

        // Same query should now return no results due to higher threshold
        let results = storage_strict.query(query_vec, 2).await?;
        assert!(results.is_empty(), "Should get no results with higher threshold");

        Ok(())
    }

    #[tokio::test]
    async fn test_entity_operations() -> Result<()> {
        let temp_dir = TempDir::new().expect("Failed to create temp dir");
        let mut config = Config::default();
        config.working_dir = temp_dir.path().to_path_buf();
        
        let mut storage = NanoVectorStorage::new(&config)?;
        storage.initialize().await?;

        // Create an entity vector
        let mut metadata = HashMap::new();
        metadata.insert("type".to_string(), json!("entity"));
        let entity_name = "test_entity";
        let entity_id = super_lightrag::utils::compute_mdhash_id(entity_name, "ent-");
        
        let entity_vec = VectorData {
            id: entity_id.clone(),
            vector: vec![1.0, 0.0, 0.0],
            metadata,
            created_at: SystemTime::now(),
            optimized: None,
        };

        // Create relation vectors
        let mut rel1_metadata = HashMap::new();
        rel1_metadata.insert("src_id".to_string(), json!(entity_name));
        let rel1 = VectorData {
            id: "rel1".to_string(),
            vector: vec![0.0, 1.0, 0.0],
            metadata: rel1_metadata,
            created_at: SystemTime::now(),
            optimized: None,
        };

        let mut rel2_metadata = HashMap::new();
        rel2_metadata.insert("tgt_id".to_string(), json!(entity_name));
        let rel2 = VectorData {
            id: "rel2".to_string(),
            vector: vec![0.0, 0.0, 1.0],
            metadata: rel2_metadata,
            created_at: SystemTime::now(),
            optimized: None,
        };

        // Insert vectors
        storage.upsert(vec![entity_vec, rel1, rel2]).await?;

        // Test entity deletion
        storage.delete_entity(entity_name).await?;
        let result = storage.get_with_fields(&[entity_id], None).await?;
        assert!(result.is_empty(), "Entity should be deleted");

        // Test relation deletion
        storage.delete_entity_relation(entity_name).await?;
        let results = storage.get_with_fields(&["rel1".to_string(), "rel2".to_string()], None).await?;
        assert!(results.is_empty(), "Relations should be deleted");

        Ok(())
    }
} 