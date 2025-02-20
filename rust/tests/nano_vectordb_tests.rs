#![cfg(test)]

use super_lightrag::nano_vectordb::*;
use ndarray::Array2;
use rand::Rng;
use std::fs;
use std::path::Path;

// Helper to generate a temporary file path in the OS temp directory
fn create_temp_file_path(filename: &str) -> String {
    let mut temp_dir = std::env::temp_dir();
    temp_dir.push(filename);
    temp_dir.to_str().unwrap().to_string()
}

#[test]
fn test_init_and_save() {
    let embedding_dim = 8;
    let storage_file = create_temp_file_path("test_nano_vectordb.json");
    // Remove file if it exists
    let _ = fs::remove_file(&storage_file);

    // Create a new NanoVectorDB instance
    let mut nvdb = NanoVectorDB::new(embedding_dim, "cosine", &storage_file).expect("Failed to create NanoVectorDB");
    assert_eq!(nvdb.storage.data.len(), 0);

    // Create fake data: a single record
    let data_record = Data {
        __id__: "0".to_string(),
        __vector__: vec![0.5; embedding_dim],
    };

    let array = Array2::from_shape_vec((1, embedding_dim), vec![0.5; embedding_dim]).unwrap();

    // Upsert data
    let report = nvdb.upsert(vec![data_record.clone()], vec![array]);
    // Check that report contains an insert for id "0"
    assert!(report["insert"].as_array().unwrap().contains(&serde_json::json!("0")));

    // Save the NVDB
    nvdb.save().expect("Failed to save NanoVectorDB");

    // Create a new instance loading from the file
    let nvdb_loaded = NanoVectorDB::new(embedding_dim, "cosine", &storage_file).expect("Failed to load NanoVectorDB");
    assert_eq!(nvdb_loaded.storage.data.len(), 1);

    // Clean up test file
    let _ = fs::remove_file(&storage_file);
}

#[test]
fn test_multi_tenant() {
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
    let data_record = Data {
        __id__: "1".to_string(),
        __vector__: vec![1.0; embedding_dim],
    };
    let array = Array2::from_shape_vec((1, embedding_dim), vec![1.0; embedding_dim]).unwrap();
    let report = tenant.upsert(vec![data_record.clone()], vec![array]);
    assert!(report["insert"].as_array().unwrap().contains(&serde_json::json!("1")));
    
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
} 