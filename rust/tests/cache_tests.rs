use std::collections::HashMap;
use super_lightrag::llm::{
    LLMResponse,
    redis_cache::RedisCache,
    cache::{CacheConfig, CacheBackend, RedisConfig, ResponseCache},
};

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

#[tokio::test]
async fn test_redis_cache() {
    let config = CacheConfig {
        enabled: true,
        max_entries: 10000,
        ttl: Some(std::time::Duration::from_secs(3600)),
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
    let mut config = CacheConfig {
        enabled: true,
        max_entries: 10000,
        ttl: Some(std::time::Duration::from_secs(1)),
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
    let mut config = CacheConfig {
        enabled: true,
        max_entries: 10000,
        ttl: Some(std::time::Duration::from_secs(3600)),
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