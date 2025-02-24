use std::collections::HashMap;
use std::time::Duration;
use super_lightrag::llm::cache::{
    backend::CacheBackend,
    memory::MemoryCache,
    types::{CacheEntry, CacheValue, StorageConfig, CacheStats, CacheMetadata, CachePriority, CacheType},
};
use super_lightrag::llm::LLMResponse;
use super_lightrag::types::llm::StreamingResponse;

#[tokio::test]
async fn test_basic_crud_operations() {
    let mut cache = MemoryCache::new();
    let config = StorageConfig::default();
    cache.initialize(config).await.unwrap();

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
    let mut cache = MemoryCache::new();
    let config = StorageConfig::default();
    cache.initialize(config).await.unwrap();

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
async fn test_memory_limits() {
    let mut cache = MemoryCache::new();
    let config = StorageConfig {
        max_memory_mb: 1, // Set very low memory limit
        max_storage_mb: 1,
        ..Default::default()
    };
    cache.initialize(config).await.unwrap();

    // Create a large value
    let large_text = "x".repeat(1024 * 1024 * 2); // 2MB of data
    let value = CacheValue::Response(LLMResponse {
        text: large_text,
        tokens_used: 1000,
        model: "test_model".to_string(),
        cached: false,
        context: None,
        metadata: HashMap::new(),
    });

    let entry = CacheEntry {
        key: "large_key".to_string(),
        value,
        metadata: CacheMetadata::new(None, 2 * 1024 * 1024),
        priority: CachePriority::default(),
        is_encrypted: false,
    };
    
    // Should fail due to quota
    let result = cache.set(entry).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_batch_operations() {
    let mut cache = MemoryCache::new();
    let config = StorageConfig::default();
    cache.initialize(config).await.unwrap();

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
async fn test_cache_stats() {
    let mut cache = MemoryCache::new();
    let config = StorageConfig::default();
    cache.initialize(config).await.unwrap();

    // Initial stats should be empty
    let initial_stats = cache.stats().await.unwrap();
    assert_eq!(initial_stats.item_count, 0);
    assert_eq!(initial_stats.total_size_bytes, 0);

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

    // Check updated stats
    let stats = cache.stats().await.unwrap();
    assert_eq!(stats.item_count, 1);
    assert!(stats.total_size_bytes > 0);

    // Test cache clear
    cache.clear().await.unwrap();
    let final_stats = cache.stats().await.unwrap();
    assert_eq!(final_stats.item_count, 0);
}

#[tokio::test]
async fn test_streaming_response_cache() {
    let mut cache = MemoryCache::new();
    let config = StorageConfig::default();
    cache.initialize(config).await.unwrap();

    let key = "stream_test".to_string();
    let stream_chunks = vec![
        StreamingResponse {
            text: "chunk1".to_string(),
            done: false,
            timing: Default::default(),
            chunk_tokens: 5,
            total_tokens: 5,
            metadata: HashMap::new(),
        },
        StreamingResponse {
            text: "chunk2".to_string(),
            done: true,
            timing: Default::default(),
            chunk_tokens: 5,
            total_tokens: 10,
            metadata: HashMap::new(),
        },
    ];

    let value = CacheValue::Stream(stream_chunks.clone());
    let entry = CacheEntry {
        key: key.clone(),
        value,
        metadata: CacheMetadata::new(None, 100),
        priority: CachePriority::default(),
        is_encrypted: false,
    };
    
    cache.set(entry).await.unwrap();
    
    let retrieved = cache.get(&key).await.unwrap();
    match retrieved.value {
        CacheValue::Stream(chunks) => {
            assert_eq!(chunks.len(), 2);
            assert_eq!(chunks[0].text, "chunk1");
            assert_eq!(chunks[1].text, "chunk2");
        },
        _ => panic!("Wrong value type retrieved"),
    }
}

#[tokio::test]
async fn test_backend_type_and_capabilities() {
    let cache = MemoryCache::new();
    
    assert_eq!(cache.backend_type(), CacheType::Memory);
    
    let capabilities = cache.capabilities();
    assert!(!capabilities.persistent);
    assert!(capabilities.streaming);
    assert!(capabilities.compression);
    assert!(!capabilities.encryption);
    assert!(!capabilities.transactions);
    assert!(!capabilities.pubsub);
} 