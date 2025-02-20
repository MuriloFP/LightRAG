#![cfg(test)]

use super_lightrag::nano_vectordb::*;
use ndarray::Array2;
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use anyhow::Result;

// Helper to generate a temporary file path in the OS temp directory
fn create_temp_file_path(filename: &str) -> String {
    let mut temp_dir = std::env::temp_dir();
    temp_dir.push(filename);
    temp_dir.to_str().unwrap().to_string()
}

#[test]
fn test_init_and_save() -> Result<()> {
    let embedding_dim = 8;
    let storage_file = create_temp_file_path("test_nano_vectordb.json");
    // Remove file if it exists
    let _ = fs::remove_file(&storage_file);

    // Create a new NanoVectorDB instance
    let mut nvdb = NanoVectorDB::new(embedding_dim, "cosine", &storage_file)?;
    assert_eq!(nvdb.storage.data.len(), 0);

    // Create fake data: a single record
    let mut data_record = Data {
        __id__: "0".to_string(),
        __vector__: vec![0.5; embedding_dim],
        __created_at__: 0.0,
        metadata: HashMap::new(),
    };
    data_record.metadata.insert("test_key".to_string(), json!("test_value"));

    let array = Array2::from_shape_vec((1, embedding_dim), vec![0.5; embedding_dim])?;

    // Upsert data
    let report = nvdb.upsert(vec![data_record.clone()], vec![array]);
    // Check that report contains an insert for id "0"
    assert!(report["insert"].as_array().unwrap().contains(&json!("0")));

    // Save the NVDB
    nvdb.save()?;

    // Create a new instance loading from the file
    let nvdb_loaded = NanoVectorDB::new(embedding_dim, "cosine", &storage_file)?;
    assert_eq!(nvdb_loaded.storage.data.len(), 1);
    assert_eq!(nvdb_loaded.storage.data[0].metadata.get("test_key"), Some(&json!("test_value")));

    // Clean up test file
    let _ = fs::remove_file(&storage_file);
    Ok(())
}

#[test]
fn test_cosine_similarity_search() -> Result<()> {
    let embedding_dim = 2;
    let storage_file = create_temp_file_path("test_similarity.json");
    let mut nvdb = NanoVectorDB::new(embedding_dim, "cosine", &storage_file)?;

    // Create test vectors
    let mut data1 = Data {
        __id__: "vec1".to_string(),
        __vector__: vec![1.0, 0.0],
        __created_at__: 0.0,
        metadata: HashMap::new(),
    };
    let mut data2 = Data {
        __id__: "vec2".to_string(),
        __vector__: vec![0.0, 1.0], // Orthogonal to vec1
        __created_at__: 0.0,
        metadata: HashMap::new(),
    };
    data1.metadata.insert("label".to_string(), json!("first"));
    data2.metadata.insert("label".to_string(), json!("second"));

    let arrays = vec![
        Array2::from_shape_vec((1, 2), vec![1.0, 0.0])?,
        Array2::from_shape_vec((1, 2), vec![0.0, 1.0])?,
    ];

    nvdb.upsert(vec![data1, data2], arrays);

    // Query with a vector similar to vec1
    let results = nvdb.query(&[1.0, 0.1], 2);
    assert_eq!(results.len(), 1); // Only vec1 should be above threshold
    assert_eq!(results[0].__id__, "vec1");
    assert_eq!(results[0].metadata.get("label"), Some(&json!("first")));

    // Clean up
    let _ = fs::remove_file(&storage_file);
    Ok(())
}

#[test]
fn test_entity_operations() -> Result<()> {
    let embedding_dim = 2;
    let storage_file = create_temp_file_path("test_entities.json");
    let mut nvdb = NanoVectorDB::new(embedding_dim, "cosine", &storage_file)?;

    // Create an entity
    let entity_id = format!("ent-{:x}", md5::compute("test_entity".as_bytes()));
    let mut entity_data = Data {
        __id__: entity_id.clone(),
        __vector__: vec![1.0, 0.0],
        __created_at__: 0.0,
        metadata: HashMap::new(),
    };
    entity_data.metadata.insert("name".to_string(), json!("test_entity"));

    let array = Array2::from_shape_vec((1, 2), vec![1.0, 0.0])?;
    nvdb.upsert(vec![entity_data], vec![array]);

    // Create a relation
    let mut relation_data = Data {
        __id__: "rel1".to_string(),
        __vector__: vec![0.0, 1.0],
        __created_at__: 0.0,
        metadata: HashMap::new(),
    };
    relation_data.metadata.insert("src_id".to_string(), json!("test_entity"));
    relation_data.metadata.insert("tgt_id".to_string(), json!("other_entity"));

    let array = Array2::from_shape_vec((1, 2), vec![0.0, 1.0])?;
    nvdb.upsert(vec![relation_data], vec![array]);

    // Test delete_entity
    nvdb.delete_entity("test_entity")?;
    assert!(!nvdb.storage.data.iter().any(|d| d.__id__ == entity_id));

    // Test delete_entity_relation
    nvdb.delete_entity_relation("test_entity")?;
    assert!(nvdb.storage.data.is_empty());

    // Clean up
    let _ = fs::remove_file(&storage_file);
    Ok(())
}

#[test]
fn test_multi_tenant() -> Result<()> {
    let embedding_dim = 8;
    let storage_dir = std::env::temp_dir().join("nano_multi_tenant_test");
    let storage_dir_str = storage_dir.to_str().unwrap().to_string();
    // Remove directory if it exists
    if storage_dir.exists() {
        let _ = fs::remove_dir_all(&storage_dir);
    }

    // Create MultiTenantNanoVDB
    let mut mt = MultiTenantNanoVDB::new(embedding_dim, "cosine", 2, &storage_dir_str);
    let tenant_id = mt.create_tenant();
    // Retrieve tenant
    let tenant = mt.get_tenant(&tenant_id).expect("Tenant should exist");
    
    // Upsert data into tenant
    let mut data_record = Data {
        __id__: "1".to_string(),
        __vector__: vec![1.0; embedding_dim],
        __created_at__: 0.0,
        metadata: HashMap::new(),
    };
    data_record.metadata.insert("test_key".to_string(), json!("test_value"));

    let array = Array2::from_shape_vec((1, embedding_dim), vec![1.0; embedding_dim])?;
    let report = tenant.upsert(vec![data_record.clone()], vec![array]);
    assert!(report["insert"].as_array().unwrap().contains(&json!("1")));
    
    // Save tenants
    mt.save();
    // Check that tenant file exists
    let tenant_file = storage_dir.join(MultiTenantNanoVDB::jsonfile_from_id(&tenant_id));
    assert!(tenant_file.exists());
    
    // Delete tenant
    mt.delete_tenant(&tenant_id);
    assert!(!tenant_file.exists());
    
    // Clean up storage directory if exists
    if storage_dir.exists() {
        let _ = fs::remove_dir_all(&storage_dir);
    }
    Ok(())
} 