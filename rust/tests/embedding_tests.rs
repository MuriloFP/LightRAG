use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Duration;

use super_lightrag::{
    types::embeddings::{
        EmbeddingProvider, EmbeddingConfig, EmbeddingError, EmbeddingResponse, CacheConfig,
    },
    embeddings::{
        OpenAIEmbeddingProvider, OllamaEmbeddingProvider, cache::EmbeddingCache,
    },
};

/// Mock embedding provider for testing
struct MockEmbeddingProvider {
    config: EmbeddingConfig,
    responses: HashMap<String, EmbeddingResponse>,
    error_texts: Vec<String>,
}

impl MockEmbeddingProvider {
    fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            responses: HashMap::new(),
            error_texts: Vec::new(),
        }
    }

    fn add_response(&mut self, text: &str, embedding: Vec<f32>) {
        self.responses.insert(text.to_string(), EmbeddingResponse {
            embedding,
            tokens_used: text.split_whitespace().count(),
            model: "mock-model".to_string(),
            cached: false,
            metadata: HashMap::new(),
        });
    }

    fn add_error_text(&mut self, text: &str) {
        self.error_texts.push(text.to_string());
    }
}

#[async_trait]
impl EmbeddingProvider for MockEmbeddingProvider {
    async fn initialize(&mut self) -> Result<(), EmbeddingError> {
        Ok(())
    }

    async fn embed(&self, text: &str) -> Result<EmbeddingResponse, EmbeddingError> {
        if self.error_texts.contains(&text.to_string()) {
            return Err(EmbeddingError::RequestFailed("Mock error".to_string()));
        }

        self.responses.get(text)
            .cloned()
            .ok_or_else(|| EmbeddingError::RequestFailed("No mock response".to_string()))
    }

    fn get_config(&self) -> &EmbeddingConfig {
        &self.config
    }

    fn update_config(&mut self, config: EmbeddingConfig) -> Result<(), EmbeddingError> {
        self.config = config;
        Ok(())
    }
}

#[tokio::test]
async fn test_mock_provider() {
    let mut provider = MockEmbeddingProvider::new(EmbeddingConfig::default());
    
    // Add test responses
    provider.add_response("hello world", vec![0.1, 0.2, 0.3]);
    provider.add_error_text("error text");

    // Test successful embedding
    let response = provider.embed("hello world").await.unwrap();
    assert_eq!(response.embedding, vec![0.1, 0.2, 0.3]);
    assert_eq!(response.tokens_used, 2);
    assert!(!response.cached);

    // Test error case
    let error = provider.embed("error text").await.unwrap_err();
    assert!(matches!(error, EmbeddingError::RequestFailed(_)));
}

#[tokio::test]
async fn test_openai_provider_config() {
    let config = EmbeddingConfig {
        model: "text-embedding-ada-002".to_string(),
        api_endpoint: Some("https://api.openai.com".to_string()),
        api_key: Some("test-key".to_string()),
        org_id: Some("test-org".to_string()),
        ..Default::default()
    };

    let provider = OpenAIEmbeddingProvider::new(config.clone()).unwrap();
    assert_eq!(provider.get_config().model, "text-embedding-ada-002");
    assert_eq!(provider.get_config().api_key.as_ref().unwrap(), "test-key");
    assert_eq!(provider.get_config().org_id.as_ref().unwrap(), "test-org");
}

#[tokio::test]
async fn test_ollama_provider_config() {
    let config = EmbeddingConfig {
        model: "llama2".to_string(),
        api_endpoint: Some("http://localhost:11434".to_string()),
        ..Default::default()
    };

    let provider = OllamaEmbeddingProvider::new(config.clone()).unwrap();
    assert_eq!(provider.get_config().model, "llama2");
    assert_eq!(
        provider.get_config().api_endpoint.as_ref().unwrap(),
        "http://localhost:11434"
    );
}

#[tokio::test]
async fn test_batch_embedding() {
    let mut provider = MockEmbeddingProvider::new(EmbeddingConfig {
        batch_size: 2,
        ..Default::default()
    });

    provider.add_response("text1", vec![0.1, 0.2]);
    provider.add_response("text2", vec![0.3, 0.4]);
    provider.add_response("text3", vec![0.5, 0.6]);

    let texts = vec![
        "text1".to_string(),
        "text2".to_string(),
        "text3".to_string(),
    ];

    let responses = provider.batch_embed(&texts).await.unwrap();
    assert_eq!(responses.len(), 3);
    assert_eq!(responses[0].embedding, vec![0.1, 0.2]);
    assert_eq!(responses[1].embedding, vec![0.3, 0.4]);
    assert_eq!(responses[2].embedding, vec![0.5, 0.6]);
}

#[tokio::test]
async fn test_embedding_cache() {
    let config = CacheConfig {
        enabled: true,
        similarity_threshold: 0.6,
        use_llm_verification: false,
        ttl_seconds: Some(3600),
        max_size: Some(2),
        use_quantization: true,
        quantization_bits: 8,
        use_compression: true,
        max_compressed_size: Some(1024 * 1024),
    };

    let mut cache = EmbeddingCache::new(config);

    // Test basic caching
    let response1 = EmbeddingResponse {
        embedding: vec![0.1, 0.2],
        tokens_used: 2,
        model: "test-model".to_string(),
        cached: false,
        metadata: HashMap::new(),
    };

    cache.put("test text".to_string(), response1.clone());
    let cached = cache.get("test text").unwrap();
    assert_eq!(cached.embedding, response1.embedding);

    // Test similarity-based lookup with similar embedding
    let similar_embedding = vec![0.15, 0.25]; // Similar to [0.1, 0.2]
    let similar = cache.get_similar(&similar_embedding, 0.6);
    assert!(similar.is_some(), "Should find similar embedding in cache");
    assert_eq!(similar.unwrap().embedding, response1.embedding);

    // Test cache size limit
    let response2 = EmbeddingResponse {
        embedding: vec![0.3, 0.4],
        tokens_used: 2,
        model: "test-model".to_string(),
        cached: false,
        metadata: HashMap::new(),
    };

    let response3 = EmbeddingResponse {
        embedding: vec![0.5, 0.6],
        tokens_used: 2,
        model: "test-model".to_string(),
        cached: false,
        metadata: HashMap::new(),
    };

    cache.put("text2".to_string(), response2);
    cache.put("text3".to_string(), response3);

    // First entry should be evicted due to max_size=2
    assert!(cache.get("test text").is_none());
}

#[tokio::test]
async fn test_cache_ttl() {
    let config = CacheConfig {
        enabled: true,
        similarity_threshold: 0.8,
        use_llm_verification: false,
        ttl_seconds: Some(1), // 1 second TTL
        max_size: None,
        use_quantization: true,
        quantization_bits: 8,
        use_compression: true,
        max_compressed_size: Some(1024 * 1024),
    };

    let mut cache = EmbeddingCache::new(config);

    let response = EmbeddingResponse {
        embedding: vec![0.1, 0.2],
        tokens_used: 2,
        model: "test-model".to_string(),
        cached: false,
        metadata: HashMap::new(),
    };

    cache.put("test text".to_string(), response);
    
    // Should be in cache immediately
    assert!(cache.get("test text").is_some());

    // Wait for TTL to expire
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Should be expired now
    assert!(cache.get("test text").is_none());
}

#[tokio::test]
async fn test_cache_cleanup() {
    let config = CacheConfig {
        enabled: true,
        similarity_threshold: 0.8,
        use_llm_verification: false,
        ttl_seconds: Some(1),
        max_size: None,
        use_quantization: true,
        quantization_bits: 8,
        use_compression: true,
        max_compressed_size: Some(1024 * 1024),
    };

    let mut cache = EmbeddingCache::new(config);

    let response = EmbeddingResponse {
        embedding: vec![0.1, 0.2],
        tokens_used: 2,
        model: "test-model".to_string(),
        cached: false,
        metadata: HashMap::new(),
    };

    cache.put("test text".to_string(), response);
    assert!(cache.get("test text").is_some());

    // Wait for TTL to expire
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Clear expired entries
    cache.clear();
    assert!(cache.get("test text").is_none());
}

#[tokio::test]
async fn test_provider_initialization() {
    // Test OpenAI provider initialization
    let openai_config = EmbeddingConfig {
        api_key: None,
        api_endpoint: None,
        ..Default::default()
    };
    let mut provider = OpenAIEmbeddingProvider::new(openai_config).unwrap();
    let err = provider.initialize().await.unwrap_err();
    assert!(matches!(err, EmbeddingError::ConfigError(_)));

    // Test Ollama provider initialization
    let ollama_config = EmbeddingConfig {
        api_endpoint: None,
        ..Default::default()
    };
    let mut provider = OllamaEmbeddingProvider::new(ollama_config).unwrap();
    let err = provider.initialize().await.unwrap_err();
    assert!(matches!(err, EmbeddingError::ConfigError(_)));
}

#[tokio::test]
async fn test_config_update() {
    let mut provider = MockEmbeddingProvider::new(EmbeddingConfig::default());
    
    let new_config = EmbeddingConfig {
        model: "new-model".to_string(),
        batch_size: 64,
        max_concurrent_requests: 16,
        ..Default::default()
    };

    provider.update_config(new_config.clone()).unwrap();
    assert_eq!(provider.get_config().model, "new-model");
    assert_eq!(provider.get_config().batch_size, 64);
    assert_eq!(provider.get_config().max_concurrent_requests, 16);
}

#[tokio::test]
async fn test_quantization_with_different_bits() {
    let config = CacheConfig {
        enabled: true,
        similarity_threshold: 0.95,
        use_llm_verification: false,
        ttl_seconds: Some(3600 * 24),
        max_size: Some(10000),
        use_quantization: true,
        quantization_bits: 4,
        use_compression: true,
        max_compressed_size: Some(1024 * 1024),
    };

    // Create cache with 4-bit quantization
    let mut cache_4bit = EmbeddingCache::new(config.clone());

    // Change to 8-bit quantization
    let mut config_8bit = config;
    config_8bit.quantization_bits = 8;
    let mut cache_8bit = EmbeddingCache::new(config_8bit);

    // Test with different bit depths
    let original_embedding = vec![
        0.123456789123456, 
        -0.987654321987654,
        0.555555555555555,
        -0.333333333333333,
        0.777777777777777,
        0.246813579135791,
        -0.135791357913579,
        0.864197530864198,
        -0.444444444444444,
        0.666666666666667,
        // Add more challenging values that require higher precision
        0.123456789012345,
        -0.987654321098765,
        0.111111111111111,
        -0.222222222222222,
        0.333333333333333,
        0.444444444444444,
        -0.555555555555555,
        0.666666666666666,
        -0.777777777777777,
        0.888888888888888,
        0.999999999999999,
        -0.111111111111111,
        0.222222222222222,
        -0.333333333333333,
        0.444444444444444,
        -0.555555555555555,
        0.666666666666666,
        -0.777777777777777,
        0.888888888888888,
        -0.999999999999999
    ];
    let response = EmbeddingResponse {
        embedding: original_embedding.clone(),
        tokens_used: 2,
        model: "test-model".to_string(),
        cached: false,
        metadata: HashMap::new(),
    };

    // Store with 4-bit quantization
    cache_4bit.put("test_4bit".to_string(), response.clone()).unwrap();
    let retrieved_4bit = cache_4bit.get("test_4bit").unwrap();
    let similarity_4bit = cosine_similarity(&original_embedding, &retrieved_4bit.embedding);
    println!("4-bit similarity: {:.10}", similarity_4bit);
    println!("4-bit values:");
    for (i, v) in retrieved_4bit.embedding.iter().enumerate() {
        println!("  {}: {:.10}", i, v);
    }
    let error_4: f32 = retrieved_4bit.metadata.get("quantization_error").unwrap().parse().unwrap();
    println!("4-bit error: {:.10}", error_4);
    assert!(similarity_4bit > 0.95, "4-bit quantization should maintain high similarity");

    // Store with 8-bit quantization
    cache_8bit.put("test_8bit".to_string(), response).unwrap();
    let retrieved_8bit = cache_8bit.get("test_8bit").unwrap();
    let similarity_8bit = cosine_similarity(&original_embedding, &retrieved_8bit.embedding);
    println!("8-bit similarity: {:.10}", similarity_8bit);
    println!("8-bit values:");
    for (i, v) in retrieved_8bit.embedding.iter().enumerate() {
        println!("  {}: {:.10}", i, v);
    }
    let error_8: f32 = retrieved_8bit.metadata.get("quantization_error").unwrap().parse().unwrap();
    println!("8-bit error: {:.10}", error_8);

    // Instead of comparing cosine similarities (which are nearly 1 for both), we now assert 8-bit quantization yields lower error
    assert!(error_8 < error_4, "8-bit quantization should have lower quantization error than 4-bit");
}

#[tokio::test]
async fn test_compression_with_different_sizes() {
    let config = CacheConfig {
        enabled: true,
        similarity_threshold: 0.8,
        use_llm_verification: false,
        ttl_seconds: None,
        max_size: None,
        use_quantization: true,
        quantization_bits: 8,
        use_compression: true,
        max_compressed_size: Some(100), // Small limit to test compression
    };

    let mut cache = EmbeddingCache::new(config);

    // Create a large embedding
    let large_embedding: Vec<f32> = (0..1000).map(|i| (i as f32) / 1000.0).collect();
    let large_response = EmbeddingResponse {
        embedding: large_embedding.clone(),
        tokens_used: 2,
        model: "test-model".to_string(),
        cached: false,
        metadata: HashMap::new(),
    };

    // Store large embedding
    cache.put("large_test".to_string(), large_response);
    
    // Verify we can retrieve and decompress correctly
    let retrieved = cache.get("large_test").unwrap();
    assert_eq!(retrieved.embedding.len(), large_embedding.len());
    
    let similarity = cosine_similarity(&large_embedding, &retrieved.embedding);
    assert!(similarity > 0.99, "Compression should maintain high similarity");

    // Test with embedding too large for compression limit
    let huge_embedding: Vec<f32> = (0..10000).map(|i| (i as f32) / 10000.0).collect();
    let huge_response = EmbeddingResponse {
        embedding: huge_embedding,
        tokens_used: 2,
        model: "test-model".to_string(),
        cached: false,
        metadata: HashMap::new(),
    };

    // This should still work but fall back to uncompressed storage
    cache.put("huge_test".to_string(), huge_response.clone());
    let retrieved_huge = cache.get("huge_test").unwrap();
    assert_eq!(retrieved_huge.embedding, huge_response.embedding);
}

// Helper function for similarity calculation
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
} 