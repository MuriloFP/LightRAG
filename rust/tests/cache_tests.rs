use std::collections::HashMap;
use super_lightrag::llm::{
    LLMResponse,
    redis_cache::RedisCache,
    cache::{CacheConfig, CacheBackend, RedisConfig, ResponseCache, types::CacheType},
};
use super_lightrag::types::llm::StreamingResponse;
use futures::StreamExt;

fn create_test_response(text: &str) -> LLMResponse {
    LLMResponse {
        text: text.to_string(),
        tokens_used: 10,
        model: "test-model".to_string(),
        cached: false,
        context: None,
        metadata: HashMap::new(),
    }
}

fn create_test_config() -> CacheConfig {
    CacheConfig {
        enabled: true,
        max_entries: Some(10000),
        ttl: Some(std::time::Duration::from_secs(3600)),
        similarity_enabled: true,
        similarity_threshold: 0.95,
        use_fuzzy_match: true,
        use_persistent: false,
        use_llm_verification: false,
        llm_verification_prompt: None,
        backend: CacheBackend::Redis(RedisConfig {
            url: "redis://localhost".to_string(),
            pool_size: 5,
        }),
        eviction_strategy: super_lightrag::llm::cache::EvictionStrategy::LRU,
        enable_metrics: true,
        use_compression: false,
        max_compressed_size: None,
        use_encryption: false,
        encryption_key: None,
        validate_integrity: true,
        enable_sync: false,
        sync_interval: None,
        stream_cache_enabled: true,
        max_stream_chunks: Some(1000),
        compress_streams: false,
        stream_ttl: Some(std::time::Duration::from_secs(1800)),
        prefix: "cache".to_string(),
        cache_type: CacheType::Query,
    }
}

#[tokio::test]
async fn test_redis_cache() {
    let config = CacheConfig {
        enabled: true,
        max_entries: Some(10000),
        ttl: Some(std::time::Duration::from_secs(3600)),
        similarity_enabled: true,
        use_fuzzy_match: true,
        similarity_threshold: 0.95,
        use_persistent: false,
        use_llm_verification: false,
        llm_verification_prompt: None,
        backend: CacheBackend::Redis(RedisConfig {
            url: "redis://localhost".to_string(),
            pool_size: 5,
        }),
        eviction_strategy: super_lightrag::llm::cache::EvictionStrategy::LRU,
        enable_metrics: true,
        use_compression: false,
        max_compressed_size: None,
        use_encryption: false,
        encryption_key: None,
        validate_integrity: true,
        enable_sync: false,
        sync_interval: None,
        stream_cache_enabled: true,
        max_stream_chunks: Some(1000),
        compress_streams: false,
        stream_ttl: Some(std::time::Duration::from_secs(1800)),
        prefix: "cache".to_string(),
        cache_type: CacheType::Query,
    };

    if let Ok(cache) = RedisCache::new(config).await {
        let response = create_test_response("test response");
        let embedding = vec![1.0, 0.0, 0.0];

        // Test put with embedding
        assert!(cache.put_with_embedding("test", response.clone(), embedding).await.is_ok());

        // Test get
        let cached = cache.get("test").await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().text, "test response");

        // Test similarity search
        let similar_embedding = vec![0.9, 0.1, 0.0];
        let result = cache.find_similar_entry(&similar_embedding).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap().text, "test response");

        // Test clear
        assert!(cache.clear().await.is_ok());
        assert!(cache.get("test").await.is_none());
    }
}

#[tokio::test]
async fn test_redis_cache_expiration() {
    let config = CacheConfig {
        enabled: true,
        max_entries: Some(10000),
        ttl: Some(std::time::Duration::from_secs(1)),
        similarity_enabled: true,
        use_fuzzy_match: true,
        similarity_threshold: 0.95,
        use_persistent: false,
        use_llm_verification: false,
        llm_verification_prompt: None,
        backend: CacheBackend::Redis(RedisConfig {
            url: "redis://localhost".to_string(),
            pool_size: 5,
        }),
        eviction_strategy: super_lightrag::llm::cache::EvictionStrategy::LRU,
        enable_metrics: true,
        use_compression: false,
        max_compressed_size: None,
        use_encryption: false,
        encryption_key: None,
        validate_integrity: true,
        enable_sync: false,
        sync_interval: None,
        stream_cache_enabled: true,
        max_stream_chunks: Some(1000),
        compress_streams: false,
        stream_ttl: Some(std::time::Duration::from_secs(1800)),
        prefix: "cache".to_string(),
        cache_type: CacheType::Query,
    };

    if let Ok(cache) = RedisCache::new(config).await {
        let response = create_test_response("test response");
        assert!(cache.put("test", response).await.is_ok());

        // Should be able to get immediately
        assert!(cache.get("test").await.is_some());

        // Wait for expiration
        tokio::time::sleep(std::time::Duration::from_secs(2)).await;

        // Should be expired now
        assert!(cache.get("test").await.is_none());
    }
}

#[tokio::test]
async fn test_redis_cache_similarity_search() {
    let config = CacheConfig {
        enabled: true,
        max_entries: Some(10000),
        ttl: Some(std::time::Duration::from_secs(3600)),
        similarity_enabled: true,
        use_fuzzy_match: true,
        similarity_threshold: 0.8,
        use_persistent: false,
        use_llm_verification: false,
        llm_verification_prompt: None,
        backend: CacheBackend::Redis(RedisConfig {
            url: "redis://localhost".to_string(),
            pool_size: 5,
        }),
        eviction_strategy: super_lightrag::llm::cache::EvictionStrategy::LRU,
        enable_metrics: true,
        use_compression: false,
        max_compressed_size: None,
        use_encryption: false,
        encryption_key: None,
        validate_integrity: true,
        enable_sync: false,
        sync_interval: None,
        stream_cache_enabled: true,
        max_stream_chunks: Some(1000),
        compress_streams: false,
        stream_ttl: Some(std::time::Duration::from_secs(1800)),
        prefix: "cache".to_string(),
        cache_type: CacheType::Query,
    };

    if let Ok(cache) = RedisCache::new(config).await {
        let response = create_test_response("test response");
        let embedding = vec![1.0, 0.0, 0.0];

        assert!(cache.put_with_embedding("test", response, embedding).await.is_ok());

        // Similar embedding should find the entry
        let similar_embedding = vec![0.9, 0.1, 0.0];
        let result = cache.find_similar_entry(&similar_embedding).await;
        assert!(result.is_some());

        // Very different embedding should not find the entry
        let different_embedding = vec![0.0, 1.0, 0.0];
        let result = cache.find_similar_entry(&different_embedding).await;
        assert!(result.is_none());
    }
}

#[tokio::test]
async fn test_redis_cache_types() {
    let config = CacheConfig {
        enabled: true,
        max_entries: Some(10000),
        ttl: Some(std::time::Duration::from_secs(3600)),
        similarity_enabled: true,
        use_fuzzy_match: true,
        similarity_threshold: 0.95,
        use_persistent: false,
        use_llm_verification: false,
        llm_verification_prompt: None,
        backend: CacheBackend::Redis(RedisConfig {
            url: "redis://localhost".to_string(),
            pool_size: 5,
        }),
        eviction_strategy: super_lightrag::llm::cache::EvictionStrategy::LRU,
        enable_metrics: true,
        use_compression: false,
        max_compressed_size: None,
        use_encryption: false,
        encryption_key: None,
        validate_integrity: true,
        enable_sync: false,
        sync_interval: None,
        stream_cache_enabled: true,
        max_stream_chunks: Some(1000),
        compress_streams: false,
        stream_ttl: Some(std::time::Duration::from_secs(1800)),
        prefix: "cache".to_string(),
        cache_type: CacheType::Query,
    };

    let cache = match RedisCache::new(config.clone()).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test_redis_cache_types: {}", e);
            return;
        }
    };

    let response = create_test_response("test response");
    let embedding = vec![1.0, 0.0, 0.0];

    // Test put with embedding
    assert!(cache.put_with_embedding("test", response.clone(), embedding).await.is_ok());

    // Test get
    let cached = cache.get("test").await;
    assert!(cached.is_some());
    assert_eq!(cached.unwrap().text, "test response");

    // Test similarity search
    let similar_embedding = vec![0.9, 0.1, 0.0];
    let result = cache.find_similar_entry(&similar_embedding).await;
    assert!(result.is_some());
    assert_eq!(result.unwrap().text, "test response");

    // Test clear
    assert!(cache.clear().await.is_ok());
    assert!(cache.get("test").await.is_none());
}

#[tokio::test]
async fn test_redis_cache_type_streaming() {
    let mut config = create_test_config();
    config.stream_cache_enabled = true;
    let mut cache = match RedisCache::new(config.clone()).await {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Skipping test_redis_cache_type_streaming: {}", e);
            return;
        }
    };

    // Create test chunks
    let chunks = vec![
        StreamingResponse {
            text: "Hello ".to_string(),
            chunk_tokens: 1,
            total_tokens: 2,
            metadata: HashMap::new(),
            timing: None,
            done: false,
        },
        StreamingResponse {
            text: "world!".to_string(),
            chunk_tokens: 1,
            total_tokens: 2,
            metadata: HashMap::new(),
            timing: None,
            done: true,
        },
    ];

    // Test with Query type (default)
    assert!(cache.put_stream("test_prompt", chunks.clone()).await.is_ok());
    let stream = cache.get_stream("test_prompt").await.unwrap();
    let received_chunks: Vec<_> = stream.collect().await;
    assert_eq!(received_chunks.len(), 2);
    assert_eq!(received_chunks[0].as_ref().unwrap().text, "Hello ");

    // Test with Extract type
    config.cache_type = CacheType::Extract;
    cache.update_config(config.clone()).unwrap();
    assert!(cache.put_stream("test_prompt", chunks.clone()).await.is_ok());
    let stream = cache.get_stream("test_prompt").await.unwrap();
    let received_chunks: Vec<_> = stream.collect().await;
    assert_eq!(received_chunks.len(), 2);
    assert_eq!(received_chunks[1].as_ref().unwrap().text, "world!");

    // Cleanup
    assert!(cache.clear().await.is_ok());
} 