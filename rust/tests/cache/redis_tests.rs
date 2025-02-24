use std::collections::HashMap;
use std::time::Duration;
use redis::Client;
use super_lightrag::llm::cache::{
    backend::{CacheBackend, CacheError, CacheCapabilities},
    redis::RedisCache,
    types::{CacheEntry, CacheValue, StorageConfig, CacheStats, CacheMetadata, CachePriority, CacheType},
};
use super_lightrag::llm::LLMResponse;
use super_lightrag::types::llm::StreamingResponse;

#[tokio::test]
async fn test_basic_crud_operations() {
    let config = StorageConfig {
        connection_string: Some("redis://localhost:6379".to_string()),
        ..Default::default()
    };
    
    let mut cache = RedisCache::new(config).await.unwrap();

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
    let config = StorageConfig {
        connection_string: Some("redis://localhost:6379".to_string()),
        ..Default::default()
    };
    
    let mut cache = RedisCache::new(config).await.unwrap();

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
async fn test_batch_operations() {
    let config = StorageConfig {
        connection_string: Some("redis://localhost:6379".to_string()),
        ..Default::default()
    };
    
    let mut cache = RedisCache::new(config).await.unwrap();

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
    let config = StorageConfig {
        connection_string: Some("redis://localhost:6379".to_string()),
        use_compression: true,
        compression_level: Some(6),
        ..Default::default()
    };
    
    let mut cache = RedisCache::new(config).await.unwrap();

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
async fn test_concurrent_access() {
    let config = StorageConfig {
        connection_string: Some("redis://localhost:6379".to_string()),
        ..Default::default()
    };
    
    let cache = RedisCache::new(config).await.unwrap();
    let cache = std::sync::Arc::new(cache);
    let mut handles = Vec::new();

    // Spawn multiple tasks to write and read concurrently
    for i in 0..10 {
        let cache_clone = cache.clone();
        let handle = tokio::spawn(async move {
            let key = format!("concurrent_key_{}", i);
            let value = CacheValue::Response(LLMResponse {
                text: format!("concurrent_value_{}", i),
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

            // Write
            cache_clone.set(entry).await.unwrap();

            // Read back
            let retrieved = cache_clone.get(&key).await.unwrap();
            match retrieved.value {
                CacheValue::Response(resp) => {
                    assert_eq!(resp.text, format!("concurrent_value_{}", i));
                },
                _ => panic!("Wrong value type retrieved"),
            }
        });
        handles.push(handle);
    }

    // Wait for all tasks to complete
    for handle in handles {
        handle.await.unwrap();
    }
}

#[tokio::test]
async fn test_backend_type_and_capabilities() {
    let config = StorageConfig {
        connection_string: Some("redis://localhost:6379".to_string()),
        ..Default::default()
    };
    
    let cache = RedisCache::new(config).await.unwrap();
    
    assert_eq!(cache.backend_type(), CacheType::Redis);
    
    let capabilities = cache.capabilities();
    assert!(capabilities.persistent);
    assert!(capabilities.streaming);
    assert!(capabilities.compression);
    assert!(capabilities.encryption);
    assert!(capabilities.transactions);
    assert!(capabilities.pubsub);
}

#[tokio::test]
async fn test_health_check() {
    let config = StorageConfig {
        connection_string: Some("redis://localhost:6379".to_string()),
        ..Default::default()
    };
    
    let cache = RedisCache::new(config).await.unwrap();
    
    // Should succeed if Redis is running
    cache.health_check().await.unwrap();
}

#[tokio::test]
async fn test_stats() {
    let config = StorageConfig {
        connection_string: Some("redis://localhost:6379".to_string()),
        ..Default::default()
    };
    
    let mut cache = RedisCache::new(config).await.unwrap();

    // Get initial stats
    let initial_stats = cache.stats().await.unwrap();
    assert_eq!(initial_stats.item_count, 0);

    // Add some entries
    let key = "stats_test".to_string();
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
        metadata: CacheMetadata::new(None, 100),
        priority: CachePriority::default(),
        is_encrypted: false,
    };
    
    cache.set(entry).await.unwrap();

    // Get updated stats
    let stats = cache.stats().await.unwrap();
    assert!(stats.item_count > 0);
    assert!(stats.total_size_bytes > 0);
} 