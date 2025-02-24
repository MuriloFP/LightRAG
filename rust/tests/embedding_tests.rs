use async_trait::async_trait;
use std::collections::HashMap;

use super_lightrag::{
    types::embeddings::{
        EmbeddingProvider, EmbeddingConfig, EmbeddingError, EmbeddingResponse,
    },
    embeddings::{
        OpenAIEmbeddingProvider, OllamaEmbeddingProvider,
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
async fn test_provider_initialization() {
    let config = EmbeddingConfig {
        model: "test-model".to_string(),
        api_endpoint: Some("http://test-endpoint".to_string()),
        api_key: Some("test-key".to_string()),
        ..Default::default()
    };

    let mut provider = MockEmbeddingProvider::new(config);
    assert!(provider.initialize().await.is_ok());
}

#[tokio::test]
async fn test_config_update() {
    let initial_config = EmbeddingConfig {
        model: "initial-model".to_string(),
        ..Default::default()
    };

    let mut provider = MockEmbeddingProvider::new(initial_config);
    
    let new_config = EmbeddingConfig {
        model: "new-model".to_string(),
        ..Default::default()
    };

    assert!(provider.update_config(new_config.clone()).is_ok());
    assert_eq!(provider.get_config().model, "new-model");
} 