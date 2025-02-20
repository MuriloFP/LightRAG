use super_lightrag::storage::kv::doc_status::JsonDocStatusStorage;
use super_lightrag::types::{Config, Result};
use serde_json::json;
use std::collections::HashMap;
use tempfile::TempDir;

#[tokio::test]
async fn test_doc_status_upsert_and_get_by_id() -> Result<()> {
    // Create temporary directory and config
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();

    // Create instance for doc status storage with namespace "doc_status_test"
    let mut storage = JsonDocStatusStorage::new(&config, "doc_status_test")?;
    storage.initialize().await?;

    // Prepare test document data with a 'status' field
    let mut doc = HashMap::new();
    doc.insert("status".to_string(), json!("Pending"));
    doc.insert("content".to_string(), json!("Document 1 content"));

    let mut data = HashMap::new();
    data.insert("doc1".to_string(), doc);

    // Upsert the document
    storage.upsert(data).await?;

    // Retrieve document by id
    let retrieved = storage.get_by_id("doc1").await?;
    assert!(retrieved.is_some(), "Document should exist");
    let retrieved_doc = retrieved.unwrap();
    assert_eq!(retrieved_doc.get("status").unwrap(), "Pending");

    storage.finalize().await?;
    Ok(())
}

#[tokio::test]
async fn test_get_status_counts() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();

    let mut storage = JsonDocStatusStorage::new(&config, "doc_status_counts")?;
    storage.initialize().await?;

    // Upsert several documents with different statuses
    let mut data = HashMap::new();

    let mut doc1 = HashMap::new();
    doc1.insert("status".to_string(), json!("Pending"));
    data.insert("doc1".to_string(), doc1);

    let mut doc2 = HashMap::new();
    doc2.insert("status".to_string(), json!("Completed"));
    data.insert("doc2".to_string(), doc2);

    let mut doc3 = HashMap::new();
    doc3.insert("status".to_string(), json!("Pending"));
    data.insert("doc3".to_string(), doc3);

    storage.upsert(data).await?;

    let counts = storage.get_status_counts().await?;
    assert_eq!(counts.get("Pending"), Some(&2));
    assert_eq!(counts.get("Completed"), Some(&1));

    storage.finalize().await?;
    Ok(())
}

#[tokio::test]
async fn test_get_docs_by_status() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();

    let mut storage = JsonDocStatusStorage::new(&config, "doc_status_docs")?;
    storage.initialize().await?;

    // Upsert several documents
    let mut data = HashMap::new();

    let mut doc1 = HashMap::new();
    doc1.insert("status".to_string(), json!("Processing"));
    doc1.insert("info".to_string(), json!("Doc one info"));
    data.insert("doc1".to_string(), doc1);

    let mut doc2 = HashMap::new();
    doc2.insert("status".to_string(), json!("Processing"));
    doc2.insert("info".to_string(), json!("Doc two info"));
    data.insert("doc2".to_string(), doc2);

    let mut doc3 = HashMap::new();
    doc3.insert("status".to_string(), json!("Failed"));
    data.insert("doc3".to_string(), doc3);

    storage.upsert(data).await?;

    let docs = storage.get_docs_by_status("Processing").await?;
    assert_eq!(docs.len(), 2);
    assert!(docs.contains_key("doc1"));
    assert!(docs.contains_key("doc2"));
    
    storage.finalize().await?;
    Ok(())
}

#[tokio::test]
async fn test_delete_operation() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();

    let mut storage = JsonDocStatusStorage::new(&config, "doc_status_delete")?;
    storage.initialize().await?;

    // Upsert a document
    let mut data = HashMap::new();
    let mut doc = HashMap::new();
    doc.insert("status".to_string(), json!("Completed"));
    data.insert("doc_del".to_string(), doc);
    storage.upsert(data).await?;

    // Ensure document exists
    let retrieved = storage.get_by_id("doc_del").await?;
    assert!(retrieved.is_some());

    // Delete the document
    storage.delete(&["doc_del".to_string()]).await?;
    let retrieved_after = storage.get_by_id("doc_del").await?;
    assert!(retrieved_after.is_none());

    storage.finalize().await?;
    Ok(())
}

// Integration test: using both KV and doc status operations together
#[tokio::test]
async fn test_integration_kv_and_doc_status() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();

    // Create a basic JsonKVStorage instance using doc status namespace
    let mut doc_storage = JsonDocStatusStorage::new(&config, "integration_test")?;
    doc_storage.initialize().await?;

    // Upsert multiple documents with varying statuses
    let mut data = HashMap::new();
    let mut d1 = HashMap::new();
    d1.insert("status".to_string(), json!("Pending"));
    d1.insert("title".to_string(), json!("Doc 1"));
    data.insert("doc1".to_string(), d1);

    let mut d2 = HashMap::new();
    d2.insert("status".to_string(), json!("Completed"));
    d2.insert("title".to_string(), json!("Doc 2"));
    data.insert("doc2".to_string(), d2);

    let mut d3 = HashMap::new();
    d3.insert("status".to_string(), json!("Pending"));
    d3.insert("title".to_string(), json!("Doc 3"));
    data.insert("doc3".to_string(), d3);

    doc_storage.upsert(data).await?;

    // Retrieve by ids using KV method
    let docs_by_ids = doc_storage.get_by_ids(&["doc1".to_string(), "doc2".to_string(), "doc3".to_string()]).await?;
    assert_eq!(docs_by_ids.len(), 3);

    // Get status counts
    let counts = doc_storage.get_status_counts().await?;
    assert_eq!(counts.get("Pending"), Some(&2));
    assert_eq!(counts.get("Completed"), Some(&1));

    // Get docs by status
    let pending_docs = doc_storage.get_docs_by_status("Pending").await?;
    assert_eq!(pending_docs.len(), 2);
    
    // Delete one document and re-check counts
    doc_storage.delete(&["doc1".to_string()]).await?;
    let counts_after = doc_storage.get_status_counts().await?;
    assert_eq!(counts_after.get("Pending"), Some(&1));

    doc_storage.finalize().await?;
    Ok(())
} 