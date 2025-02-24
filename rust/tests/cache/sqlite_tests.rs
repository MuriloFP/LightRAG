use std::collections::HashMap;
use std::time::Duration;
use super_lightrag::llm::cache::{
    backend::CacheBackend,
    sqlite::SQLiteCache,
    types::{CacheEntry, CacheValue, StorageConfig, CacheStats, CacheMetadata, CachePriority, CacheType},
};
use super_lightrag::llm::LLMResponse;
use super_lightrag::types::llm::StreamingResponse;
use tempfile::tempdir;

#[tokio::test]
async fn test_basic_crud_operations() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    let mut cache = SQLiteCache::new(StorageConfig {
        storage_path: Some(db_path.to_str().unwrap().to_string()),
        ..Default::default()
    }).await.unwrap();

    // Test set and get
    let key = "test_key".to_string();
    let value = CacheValue::Response(LLMResponse {
        text: "test response".to_string(),
        tokens_used: 10,
        model: "test_model".to_string(),
        cached: false,
        context: None,
        metadata: HashMap::new(),
    });
    let entry = CacheEntry {
        key: key.clone(),
        value,
        metadata: CacheMetadata::new(Some(Duration::from_secs(3600)), 100),
        priority: CachePriority::default(),
        is_encrypted: false,
    };
    
    cache.set(entry).await.unwrap();
    let retrieved = cache.get(&key).await.unwrap();
    
    match retrieved.value {
        CacheValue::Response(resp) => {
            assert_eq!(resp.text, "test response");
            assert_eq!(resp.tokens_used, 10);
            assert_eq!(resp.model, "test_model");
        },
        _ => panic!("Wrong value type retrieved"),
    }

    // Test exists
    assert!(cache.exists(&key).await.unwrap());
    assert!(!cache.exists("nonexistent").await.unwrap());

    // Test delete
    cache.delete(&key).await.unwrap();
    assert!(!cache.exists(&key).await.unwrap());
}

#[tokio::test]
async fn test_ttl_functionality() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_ttl.db");
    
    let mut cache = SQLiteCache::new(StorageConfig {
        storage_path: Some(db_path.to_str().unwrap().to_string()),
        ..Default::default()
    }).await.unwrap();

    let key = "ttl_test".to_string();
    let value = CacheValue::Response(LLMResponse {
        text: "expires soon".to_string(),
        tokens_used: 5,
        model: "test_model".to_string(),
        cached: false,
        context: None,
        metadata: HashMap::new(),
    });

    // Set entry with 1 second TTL
    let entry = CacheEntry {
        key: key.clone(),
        value,
        metadata: CacheMetadata::new(Some(Duration::from_secs(1)), 100),
        priority: CachePriority::default(),
        is_encrypted: false,
    };
    cache.set(entry).await.unwrap();

    // Should exist immediately
    assert!(cache.exists(&key).await.unwrap());

    // Wait for expiration
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Should be expired now
    let result = cache.get(&key).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_vacuum_operations() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_vacuum.db");
    
    let mut cache = SQLiteCache::new(StorageConfig {
        storage_path: Some(db_path.to_str().unwrap().to_string()),
        ..Default::default()
    }).await.unwrap();

    // Add and delete some entries to create free space
    for i in 0..10 {
        let key = format!("key_{}", i);
        let value = CacheValue::Response(LLMResponse {
            text: "test".repeat(100), // Create some sizeable data
            tokens_used: i,
            model: "test_model".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        });
        let entry = CacheEntry {
            key: key.clone(),
            value,
            metadata: CacheMetadata::new(None, 100),
            priority: CachePriority::default(),
            is_encrypted: false,
        };
        cache.set(entry).await.unwrap();
    }

    // Delete half the entries
    for i in 0..5 {
        let key = format!("key_{}", i);
        cache.delete(&key).await.unwrap();
    }

    // Configure auto-vacuum
    cache.configure_auto_vacuum(1, 10, 0).await.unwrap(); // Full vacuum mode
    
    // Trigger optimization which should vacuum
    cache.optimize().await.unwrap();
}

#[tokio::test]
async fn test_batch_operations() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_batch.db");
    
    let mut cache = SQLiteCache::new(StorageConfig {
        storage_path: Some(db_path.to_str().unwrap().to_string()),
        ..Default::default()
    }).await.unwrap();

    // Create multiple entries
    let mut entries = Vec::new();
    for i in 0..3 {
        let key = format!("key_{}", i);
        let value = CacheValue::Response(LLMResponse {
            text: format!("value_{}", i),
            tokens_used: i as usize,
            model: "test_model".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        });
        entries.push(CacheEntry {
            key: key.clone(),
            value,
            metadata: CacheMetadata::new(None, 100),
            priority: CachePriority::default(),
            is_encrypted: false,
        });
    }

    // Test set_many
    cache.set_many(entries.clone()).await.unwrap();

    // Test get_many
    let keys: Vec<String> = entries.iter().map(|e| e.key.clone()).collect();
    let retrieved = cache.get_many(&keys).await.unwrap();
    
    assert_eq!(retrieved.len(), 3);
    for (i, entry) in retrieved.iter().enumerate() {
        let entry = entry.as_ref().unwrap();
        match &entry.value {
            CacheValue::Response(resp) => {
                assert_eq!(resp.text, format!("value_{}", i));
            },
            _ => panic!("Wrong value type retrieved"),
        }
    }

    // Test delete_many
    cache.delete_many(&keys).await.unwrap();
    for key in keys {
        assert!(!cache.exists(&key).await.unwrap());
    }
}

#[tokio::test]
async fn test_compression() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_compression.db");
    
    let mut cache = SQLiteCache::new(StorageConfig {
        storage_path: Some(db_path.to_str().unwrap().to_string()),
        use_compression: true,
        compression_level: Some(6),
        ..Default::default()
    }).await.unwrap();

    // Add data that should be compressible
    let key = "test_key".to_string();
    let value = CacheValue::Response(LLMResponse {
        text: "test ".repeat(1000), // Highly compressible data
        tokens_used: 10,
        model: "test_model".to_string(),
        cached: false,
        context: None,
        metadata: HashMap::new(),
    });
    let entry = CacheEntry {
        key: key.clone(),
        value,
        metadata: CacheMetadata::new(None, 100),
        priority: CachePriority::default(),
        is_encrypted: false,
    };
    
    cache.set(entry).await.unwrap();
    
    // Verify we can read back the compressed data
    let retrieved = cache.get(&key).await.unwrap();
    match retrieved.value {
        CacheValue::Response(resp) => {
            assert_eq!(resp.text, "test ".repeat(1000));
        },
        _ => panic!("Wrong value type retrieved"),
    }
}

#[tokio::test]
async fn test_backend_type_and_capabilities() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test_caps.db");
    
    let cache = SQLiteCache::new(StorageConfig {
        storage_path: Some(db_path.to_str().unwrap().to_string()),
        ..Default::default()
    }).await.unwrap();
    
    assert_eq!(cache.backend_type(), CacheType::SQLite);
    
    let capabilities = cache.capabilities();
    assert!(capabilities.persistent);
    assert!(!capabilities.streaming);
    assert!(capabilities.compression);
    assert!(capabilities.encryption);
    assert!(capabilities.transactions);
    assert!(!capabilities.pubsub);
} 