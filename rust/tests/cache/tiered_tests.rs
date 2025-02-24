use std::collections::HashMap;
use std::time::Duration;
use super_lightrag::llm::cache::{
    backend::{CacheBackend, CacheError, CacheCapabilities},
    memory::MemoryCache,
    sqlite::SQLiteCache,
    tiered::{TieredCache, TieredCacheConfig, CacheWarmingConfig},
    types::{CacheEntry, CacheValue, StorageConfig, CacheStats, CacheMetadata, CachePriority, CacheType},
};
use super_lightrag::llm::LLMResponse;
use super_lightrag::types::llm::StreamingResponse;
use tempfile::tempdir;

#[tokio::test]
async fn test_cache_hierarchy_operations() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    let l2_config = StorageConfig {
        storage_path: Some(db_path.to_string_lossy().to_string()),
        ..Default::default()
    };
    
    let config = TieredCacheConfig {
        memory_config: StorageConfig::default(),
        storage_config: l2_config.clone(),
        sync_interval_secs: 1,
        write_through: true,
        max_sync_batch_size: 100,
        warming_config: None,
    };

    let l2_backend = SQLiteCache::new(l2_config).await.unwrap();
    let mut cache = TieredCache::new(config, Box::new(l2_backend)).await.unwrap();

    // Test L1 cache hit
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
        value: value.clone(),
        metadata: CacheMetadata::new(Some(Duration::from_secs(3600)), 100),
        priority: CachePriority::default(),
        is_encrypted: false,
    };

    // Set in cache
    cache.set(entry.clone()).await.unwrap();

    // Should be in L1 (fast retrieval)
    let retrieved = cache.get(&key).await.unwrap();
    match retrieved.value {
        CacheValue::Response(resp) => {
            assert_eq!(resp.text, "test response");
            assert_eq!(resp.tokens_used, 10);
        },
        _ => panic!("Wrong value type retrieved"),
    }
}

#[tokio::test]
async fn test_fallback_behavior() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    let l2_config = StorageConfig {
        storage_path: Some(db_path.to_string_lossy().to_string()),
        ..Default::default()
    };
    
    let config = TieredCacheConfig {
        memory_config: StorageConfig::default(),
        storage_config: l2_config.clone(),
        sync_interval_secs: 1,
        write_through: true,
        max_sync_batch_size: 100,
        warming_config: None,
    };

    let l2_backend = SQLiteCache::new(l2_config).await.unwrap();
    let mut cache = TieredCache::new(config, Box::new(l2_backend)).await.unwrap();

    // Set entry directly in L2
    let key = "fallback_test".to_string();
    let value = CacheValue::Response(LLMResponse {
        text: "fallback response".to_string(),
        tokens_used: 5,
        model: "test_model".to_string(),
        cached: false,
        context: None,
        metadata: HashMap::new(),
    });
    let entry = CacheEntry {
        key: key.clone(),
        value: value.clone(),
        metadata: CacheMetadata::new(Some(Duration::from_secs(3600)), 100),
        priority: CachePriority::default(),
        is_encrypted: false,
    };

    // Set in L2
    cache.set(entry.clone()).await.unwrap();

    // Clear L1 to force fallback
    cache.clear().await.unwrap();

    // Should fallback to L2 and populate L1
    let retrieved = cache.get(&key).await.unwrap();
    match retrieved.value {
        CacheValue::Response(resp) => {
            assert_eq!(resp.text, "fallback response");
            assert_eq!(resp.tokens_used, 5);
        },
        _ => panic!("Wrong value type retrieved"),
    }

    // Second get should be from L1
    let retrieved = cache.get(&key).await.unwrap();
    match retrieved.value {
        CacheValue::Response(resp) => {
            assert_eq!(resp.text, "fallback response");
            assert_eq!(resp.tokens_used, 5);
        },
        _ => panic!("Wrong value type retrieved"),
    }
}

#[tokio::test]
async fn test_cache_synchronization() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    let l2_config = StorageConfig {
        storage_path: Some(db_path.to_string_lossy().to_string()),
        ..Default::default()
    };
    
    let config = TieredCacheConfig {
        memory_config: StorageConfig::default(),
        storage_config: l2_config.clone(),
        sync_interval_secs: 1, // Short interval for testing
        write_through: false, // Disable write-through to test sync
        max_sync_batch_size: 100,
        warming_config: None,
    };

    let l2_backend = SQLiteCache::new(l2_config).await.unwrap();
    let mut cache = TieredCache::new(config, Box::new(l2_backend)).await.unwrap();

    // Add entries to L1
    for i in 0..5 {
        let key = format!("sync_test_{}", i);
        let value = CacheValue::Response(LLMResponse {
            text: format!("sync response {}", i),
            tokens_used: i,
            model: "test_model".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        });
        let entry = CacheEntry {
            key: key.clone(),
            value: value.clone(),
            metadata: CacheMetadata::new(Some(Duration::from_secs(3600)), 100),
            priority: CachePriority::default(),
            is_encrypted: false,
        };
        cache.set(entry).await.unwrap();
    }

    // Wait for sync interval
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Clear L1 to verify L2 sync
    cache.clear().await.unwrap();

    // Verify entries exist in L2
    for i in 0..5 {
        let key = format!("sync_test_{}", i);
        let retrieved = cache.get(&key).await.unwrap();
        match retrieved.value {
            CacheValue::Response(resp) => {
                assert_eq!(resp.text, format!("sync response {}", i));
                assert_eq!(resp.tokens_used, i);
            },
            _ => panic!("Wrong value type retrieved"),
        }
    }
}

#[tokio::test]
async fn test_performance_optimization() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    let l2_config = StorageConfig {
        storage_path: Some(db_path.to_string_lossy().to_string()),
        ..Default::default()
    };
    
    let warming_config = CacheWarmingConfig {
        enabled: true,
        max_prefetch_items: 10,
        prefetch_batch_size: 5,
        background_load_interval_secs: 1,
        priority_threshold: 50,
    };

    let config = TieredCacheConfig {
        memory_config: StorageConfig::default(),
        storage_config: l2_config.clone(),
        sync_interval_secs: 1,
        write_through: true,
        max_sync_batch_size: 100,
        warming_config: Some(warming_config),
    };

    let l2_backend = SQLiteCache::new(l2_config).await.unwrap();
    let mut cache = TieredCache::new(config, Box::new(l2_backend)).await.unwrap();

    // Add high-priority entries to L2
    for i in 0..5 {
        let key = format!("high_priority_{}", i);
        let value = CacheValue::Response(LLMResponse {
            text: format!("high priority response {}", i),
            tokens_used: i,
            model: "test_model".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        });
        let entry = CacheEntry {
            key: key.clone(),
            value: value.clone(),
            metadata: CacheMetadata::new(Some(Duration::from_secs(3600)), 100),
            priority: CachePriority::High,
            is_encrypted: false,
        };
        cache.set(entry).await.unwrap();
    }

    // Clear L1
    cache.clear().await.unwrap();

    // Wait for cache warming
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Verify high-priority entries were pre-fetched to L1
    for i in 0..5 {
        let key = format!("high_priority_{}", i);
        let retrieved = cache.get(&key).await.unwrap();
        match retrieved.value {
            CacheValue::Response(resp) => {
                assert_eq!(resp.text, format!("high priority response {}", i));
                assert_eq!(resp.tokens_used, i);
            },
            _ => panic!("Wrong value type retrieved"),
        }
    }
}

#[tokio::test]
async fn test_write_through() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    let l2_config = StorageConfig {
        storage_path: Some(db_path.to_string_lossy().to_string()),
        ..Default::default()
    };
    
    let config = TieredCacheConfig {
        memory_config: StorageConfig::default(),
        storage_config: l2_config.clone(),
        sync_interval_secs: 60, // Long interval to ensure write-through
        write_through: true,
        max_sync_batch_size: 100,
        warming_config: None,
    };

    let l2_backend = SQLiteCache::new(l2_config).await.unwrap();
    let mut cache = TieredCache::new(config, Box::new(l2_backend)).await.unwrap();

    // Add entry with write-through
    let key = "write_through_test".to_string();
    let value = CacheValue::Response(LLMResponse {
        text: "write through response".to_string(),
        tokens_used: 10,
        model: "test_model".to_string(),
        cached: false,
        context: None,
        metadata: HashMap::new(),
    });
    let entry = CacheEntry {
        key: key.clone(),
        value: value.clone(),
        metadata: CacheMetadata::new(Some(Duration::from_secs(3600)), 100),
        priority: CachePriority::default(),
        is_encrypted: false,
    };

    cache.set(entry).await.unwrap();

    // Clear L1 immediately
    cache.clear().await.unwrap();

    // Should still be in L2 due to write-through
    let retrieved = cache.get(&key).await.unwrap();
    match retrieved.value {
        CacheValue::Response(resp) => {
            assert_eq!(resp.text, "write through response");
            assert_eq!(resp.tokens_used, 10);
        },
        _ => panic!("Wrong value type retrieved"),
    }
}

#[tokio::test]
async fn test_batch_operations() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    let l2_config = StorageConfig {
        storage_path: Some(db_path.to_string_lossy().to_string()),
        ..Default::default()
    };
    
    let config = TieredCacheConfig {
        memory_config: StorageConfig::default(),
        storage_config: l2_config.clone(),
        sync_interval_secs: 1,
        write_through: true,
        max_sync_batch_size: 100,
        warming_config: None,
    };

    let l2_backend = SQLiteCache::new(l2_config).await.unwrap();
    let mut cache = TieredCache::new(config, Box::new(l2_backend)).await.unwrap();

    // Create multiple entries
    let mut entries = Vec::new();
    let mut keys = Vec::new();
    for i in 0..5 {
        let key = format!("batch_test_{}", i);
        keys.push(key.clone());
        let value = CacheValue::Response(LLMResponse {
            text: format!("batch response {}", i),
            tokens_used: i,
            model: "test_model".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        });
        entries.push(CacheEntry {
            key: key.clone(),
            value: value.clone(),
            metadata: CacheMetadata::new(Some(Duration::from_secs(3600)), 100),
            priority: CachePriority::default(),
            is_encrypted: false,
        });
    }

    // Test batch set
    cache.set_many(entries).await.unwrap();

    // Test batch get
    let retrieved = cache.get_many(&keys).await.unwrap();
    assert_eq!(retrieved.len(), 5);
    for (i, entry) in retrieved.iter().enumerate() {
        let entry = entry.as_ref().unwrap();
        match &entry.value {
            CacheValue::Response(resp) => {
                assert_eq!(resp.text, format!("batch response {}", i));
                assert_eq!(resp.tokens_used, i);
            },
            _ => panic!("Wrong value type retrieved"),
        }
    }

    // Test batch delete
    cache.delete_many(&keys).await.unwrap();
    for key in keys {
        assert!(!cache.exists(&key).await.unwrap());
    }
}

#[tokio::test]
async fn test_cache_stats() {
    let temp_dir = tempdir().unwrap();
    let db_path = temp_dir.path().join("test.db");
    
    let l2_config = StorageConfig {
        storage_path: Some(db_path.to_string_lossy().to_string()),
        ..Default::default()
    };
    
    let config = TieredCacheConfig {
        memory_config: StorageConfig::default(),
        storage_config: l2_config.clone(),
        sync_interval_secs: 1,
        write_through: true,
        max_sync_batch_size: 100,
        warming_config: None,
    };

    let l2_backend = SQLiteCache::new(l2_config).await.unwrap();
    let mut cache = TieredCache::new(config, Box::new(l2_backend)).await.unwrap();

    // Get initial stats
    let initial_stats = cache.stats().await.unwrap();
    assert_eq!(initial_stats.item_count, 0);
    assert_eq!(initial_stats.hits, 0);
    assert_eq!(initial_stats.misses, 0);

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
        value: value.clone(),
        metadata: CacheMetadata::new(Some(Duration::from_secs(3600)), 100),
        priority: CachePriority::default(),
        is_encrypted: false,
    };

    cache.set(entry).await.unwrap();

    // Get updated stats
    let stats = cache.stats().await.unwrap();
    assert_eq!(stats.item_count, 1);
    assert!(stats.total_size_bytes > 0);

    // Test cache hit
    cache.get(&key).await.unwrap();
    let stats = cache.stats().await.unwrap();
    assert!(stats.hits > 0);

    // Test cache miss
    let _ = cache.get("nonexistent").await;
    let stats = cache.stats().await.unwrap();
    assert!(stats.misses > 0);
} 