#![cfg(test)]

use super_lightrag::storage::kv::{KVStorage, JsonKVStorage};
use super_lightrag::types::{Config, Result};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use tempfile::TempDir;

#[tokio::test]
async fn test_upsert_and_get() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    let mut storage = JsonKVStorage::new(&config, "test")?;
    storage.initialize().await?;

    let mut inner_data = HashMap::new();
    inner_data.insert("field".to_string(), json!("test_value"));

    let mut data = HashMap::new();
    data.insert("test_key".to_string(), inner_data.clone());
    storage.upsert(data).await?;

    let retrieved = storage.get_by_id("test_key").await?;
    assert_eq!(retrieved, Some(inner_data));
    storage.finalize().await?;
    Ok(())
}

#[tokio::test]
async fn test_upsert_only_new_keys() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    let mut storage = JsonKVStorage::new(&config, "test")?;
    storage.initialize().await?;

    // First upsert
    let mut inner_data1 = HashMap::new();
    inner_data1.insert("field".to_string(), json!("value1"));
    let mut data = HashMap::new();
    data.insert("key1".to_string(), inner_data1.clone());
    storage.upsert(data).await?;

    // Second upsert with same key but different value
    let mut inner_data2 = HashMap::new();
    inner_data2.insert("field".to_string(), json!("new_value"));
    let mut inner_data3 = HashMap::new();
    inner_data3.insert("field".to_string(), json!("value2"));
    let mut data = HashMap::new();
    data.insert("key1".to_string(), inner_data2); // Should not update
    data.insert("key2".to_string(), inner_data3.clone());
    storage.upsert(data).await?;

    // key1 should retain original value, key2 should be added
    let key1 = storage.get_by_id("key1").await?;
    let key2 = storage.get_by_id("key2").await?;
    assert_eq!(key1, Some(inner_data1));
    assert_eq!(key2, Some(inner_data3));

    storage.finalize().await?;
    Ok(())
}

#[tokio::test]
async fn test_filter_keys() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    let mut storage = JsonKVStorage::new(&config, "test")?;
    storage.initialize().await?;

    // Insert some data
    let mut inner_data1 = HashMap::new();
    inner_data1.insert("field".to_string(), json!("value1"));
    let mut inner_data2 = HashMap::new();
    inner_data2.insert("field".to_string(), json!("value2"));
    let mut data = HashMap::new();
    data.insert("key1".to_string(), inner_data1);
    data.insert("key2".to_string(), inner_data2);
    storage.upsert(data).await?;

    // Create a set of keys to filter
    let mut keys = HashSet::new();
    keys.insert("key1".to_string());
    keys.insert("key3".to_string());
    keys.insert("key4".to_string());

    // Filter keys
    let missing = storage.filter_keys(&keys).await?;
    assert_eq!(missing.len(), 2);
    assert!(missing.contains("key3"));
    assert!(missing.contains("key4"));
    assert!(!missing.contains("key1"));

    storage.finalize().await?;
    Ok(())
}

#[tokio::test]
async fn test_get_by_ids() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    let mut storage = JsonKVStorage::new(&config, "test")?;
    storage.initialize().await?;

    // Insert test data
    let mut inner_data1 = HashMap::new();
    inner_data1.insert("field".to_string(), json!("value1"));
    let mut inner_data2 = HashMap::new();
    inner_data2.insert("field".to_string(), json!("value2"));
    let mut data = HashMap::new();
    data.insert("key1".to_string(), inner_data1.clone());
    data.insert("key2".to_string(), inner_data2.clone());
    storage.upsert(data).await?;

    // Test get_by_ids
    let ids = vec!["key1".to_string(), "key3".to_string(), "key2".to_string()];
    let values = storage.get_by_ids(&ids).await?;
    assert_eq!(values.len(), 2); // Only existing keys are returned
    assert_eq!(values[0], inner_data1);
    assert_eq!(values[1], inner_data2);

    storage.finalize().await?;
    Ok(())
}

#[tokio::test]
async fn test_persistence() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();

    // Create and populate first instance
    let mut storage1 = JsonKVStorage::new(&config, "test")?;
    storage1.initialize().await?;
    let mut inner_data = HashMap::new();
    inner_data.insert("field".to_string(), json!("value1"));
    let mut data = HashMap::new();
    data.insert("key1".to_string(), inner_data.clone());
    storage1.upsert(data).await?;
    storage1.finalize().await?;

    // Create second instance and verify data
    let mut storage2 = JsonKVStorage::new(&config, "test")?;
    storage2.initialize().await?;
    let value = storage2.get_by_id("key1").await?;
    assert_eq!(value, Some(inner_data));

    Ok(())
}

#[tokio::test]
async fn test_delete() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    let mut storage = JsonKVStorage::new(&config, "test")?;
    storage.initialize().await?;

    // Insert test data
    let mut data = HashMap::new();
    for i in 1..=3 {
        let mut inner_data = HashMap::new();
        inner_data.insert("field".to_string(), json!(format!("value{}", i)));
        data.insert(format!("key{}", i), inner_data);
    }
    storage.upsert(data).await?;

    // Delete some keys
    storage.delete(&["key1".to_string(), "key2".to_string()]).await?;

    // Verify deletions
    assert_eq!(storage.get_by_id("key1").await?, None);
    assert_eq!(storage.get_by_id("key2").await?, None);
    let key3_data = storage.get_by_id("key3").await?.unwrap();
    assert_eq!(key3_data.get("field").unwrap(), &json!("value3"));

    // Verify persistence after delete
    storage.finalize().await?;
    let mut storage2 = JsonKVStorage::new(&Config::default(), "test")?;
    storage2.initialize().await?;
    assert_eq!(storage2.get_by_id("key1").await?, None);
    let key3_data = storage2.get_by_id("key3").await?.unwrap();
    assert_eq!(key3_data.get("field").unwrap(), &json!("value3"));

    Ok(())
}

#[tokio::test]
async fn test_drop() -> Result<()> {
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    let mut storage = JsonKVStorage::new(&config, "test")?;
    storage.initialize().await?;

    // Insert test data
    let mut data = HashMap::new();
    for i in 1..=2 {
        let mut inner_data = HashMap::new();
        inner_data.insert("field".to_string(), json!(format!("value{}", i)));
        data.insert(format!("key{}", i), inner_data);
    }
    storage.upsert(data).await?;

    // Drop all data
    storage.drop().await?;

    // Verify all data is gone
    assert_eq!(storage.get_by_id("key1").await?, None);
    assert_eq!(storage.get_by_id("key2").await?, None);

    // Verify empty after reloading
    storage.finalize().await?;
    let mut storage2 = JsonKVStorage::new(&Config::default(), "test")?;
    storage2.initialize().await?;
    assert_eq!(storage2.get_by_id("key1").await?, None);
    assert_eq!(storage2.get_by_id("key2").await?, None);

    Ok(())
}

#[tokio::test]
async fn test_concurrent_access() -> Result<()> {
    use tokio::task;
    
    let temp_dir = TempDir::new()?;
    let mut config = Config::default();
    config.working_dir = temp_dir.path().to_path_buf();
    let mut storage = JsonKVStorage::new(&config, "test")?;
    storage.initialize().await?;
    let storage = std::sync::Arc::new(tokio::sync::Mutex::new(storage));

    // Spawn multiple tasks that try to access the storage concurrently
    let mut handles = vec![];
    for i in 0..5 {
        let storage = storage.clone();
        handles.push(task::spawn(async move {
            let mut inner_data = HashMap::new();
            inner_data.insert("field".to_string(), json!(format!("value{}", i)));
            let mut data = HashMap::new();
            data.insert(format!("key{}", i), inner_data);
            let mut storage = storage.lock().await;
            storage.upsert(data).await.unwrap();
        }));
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }

    // Verify all data was written correctly
    let storage = storage.lock().await;
    for i in 0..5 {
        let key = format!("key{}", i);
        let value = storage.get_by_id(&key).await?.unwrap();
        assert_eq!(value.get("field").unwrap(), &json!(format!("value{}", i)));
    }

    Ok(())
} 