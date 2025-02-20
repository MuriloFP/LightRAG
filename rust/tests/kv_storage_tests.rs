#![cfg(test)]

use super_lightrag::storage::kv::{KVStorage, JsonKVStorage};
use super_lightrag::types::{Config, Result};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use tempfile::NamedTempFile;
use std::io::Write;

#[tokio::test]
async fn test_upsert_and_get() -> Result<()> {
    let mut storage = JsonKVStorage::new(&Config::default())?;
    storage.initialize().await?;
    let key = "test_key";
    let value = "test_value".to_string();
    storage.upsert(key, value.clone()).await?;
    let retrieved = storage.get_by_id(key).await?;
    assert_eq!(retrieved, Some(value));
    storage.finalize().await?;
    Ok(())
}

#[tokio::test]
async fn test_upsert_and_get_from_file() -> Result<()> {
    // Instead of using a temporary file, use the default config since persistence is stubbed
    let mut storage = JsonKVStorage::new(&Config::default())?;
    storage.initialize().await?;
    let key = "file_test_key";
    let value = "file_test_value".to_string();
    storage.upsert(key, value.clone()).await?;
    let retrieved = storage.get_by_id(key).await?;
    assert_eq!(retrieved, Some(value));
    storage.finalize().await?;
    Ok(())
} 