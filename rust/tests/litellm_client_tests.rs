use super_lightrag::llm::{
    LLMClient,
    LLMConfig, LLMParams, LLMError, LLMResponse,
    providers::litellm::{
        LiteLLMClient,
        LiteLLMConfig,
        ProviderAdapter,
        ProviderConfig,
        LLMProvider,
    },
};
use super_lightrag::types::llm::StreamingResponse;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;
use std::collections::HashMap;
use std::sync::Arc;
use tokio;

// Mock provider adapter for testing
struct MockProvider {
    name: String,
    should_fail: bool,
    config: Arc<LLMConfig>,
}

impl MockProvider {
    fn new(name: &str, should_fail: bool, config: LLMConfig) -> Self {
        Self {
            name: name.to_string(),
            should_fail,
            config: Arc::new(config),
        }
    }
}

#[async_trait]
impl ProviderAdapter for MockProvider {
    async fn complete(&self, prompt: &str, _params: &LLMParams) -> Result<LLMResponse, LLMError> {
        if self.should_fail {
            Err(LLMError::RequestFailed("Mock provider failure".to_string()))
        } else {
            Ok(LLMResponse {
                text: format!("Response from {}: {}", self.name, prompt),
                tokens_used: 10,
                model: "mock-model".to_string(),
                cached: false,
                context: None,
                metadata: HashMap::new(),
            })
        }
    }

    async fn complete_stream(
        &self,
        _prompt: &str,
        _params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<super_lightrag::types::llm::StreamingResponse, LLMError>> + Send>>, LLMError> {
        unimplemented!("Streaming not implemented in mock")
    }

    async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, LLMError> {
        unimplemented!("Embedding not implemented in mock")
    }

    fn supports_streaming(&self) -> bool {
        false
    }

    fn max_tokens(&self) -> usize {
        1000
    }

    fn provider_name(&self) -> &str {
        &self.name
    }

    fn get_config(&self) -> &LLMConfig {
        &self.config
    }
}

fn create_test_config() -> LiteLLMConfig {
    LiteLLMConfig {
        api_base: String::new(),
        api_key: "test_key".to_string(),
        provider_configs: HashMap::new(),
        default_parameters: LLMParams::default(),
        fallback_providers: vec!["fallback1".to_string(), "fallback2".to_string()],
        use_cache: false,
        cache_config: None,
        timeout_secs: 30,
        max_retries: 1,
    }
}

#[tokio::test]
async fn test_litellm_client_creation() {
    let config = create_test_config();
    let client = LiteLLMClient::new(config);
    assert!(client.is_ok());
}

#[tokio::test]
async fn test_provider_registration() {
    let config = create_test_config();
    let mut client = LiteLLMClient::new(config).unwrap();

    // Register a mock provider
    let mock_config = LLMConfig::default();
    let provider = Box::new(MockProvider::new("test_provider", false, mock_config));
    client.register_provider("test_provider", provider);

    // Verify provider is registered
    assert!(client.get_provider_adapter("test_provider").is_some());
    assert!(client.get_provider_adapter("nonexistent").is_none());
}

#[tokio::test]
async fn test_fallback_behavior() {
    let config = create_test_config();
    let mut client = LiteLLMClient::new(config).unwrap();

    // Register primary provider that fails
    let mock_config = LLMConfig::default();
    let primary = Box::new(MockProvider::new("primary", true, mock_config.clone()));
    client.register_provider("primary", primary);

    // Register fallback providers
    let fallback1 = Box::new(MockProvider::new("fallback1", true, mock_config.clone())); // Also fails
    let fallback2 = Box::new(MockProvider::new("fallback2", false, mock_config.clone())); // Succeeds
    client.register_provider("fallback1", fallback1);
    client.register_provider("fallback2", fallback2);

    // Test generation with fallbacks
    let params = LLMParams {
        model: "primary/model".to_string(),
        ..Default::default()
    };

    let result = client.generate("test prompt", &params).await;
    assert!(result.is_ok());
    assert!(result.unwrap().text.contains("fallback2"));
}

#[tokio::test]
async fn test_no_provider_found() {
    let config = create_test_config();
    let client = LiteLLMClient::new(config).unwrap();

    let params = LLMParams {
        model: "nonexistent/model".to_string(),
        ..Default::default()
    };

    let result = client.generate("test prompt", &params).await;
    assert!(matches!(result, Err(LLMError::ConfigError(_))));
}

#[tokio::test]
async fn test_batch_generate() {
    let config = create_test_config();
    let mut client = LiteLLMClient::new(config).unwrap();

    // Register a working provider
    let mock_config = LLMConfig::default();
    let provider = Box::new(MockProvider::new("test_provider", false, mock_config));
    client.register_provider("test_provider", provider);

    let params = LLMParams {
        model: "test_provider/model".to_string(),
        ..Default::default()
    };

    let prompts = vec![
        "prompt1".to_string(),
        "prompt2".to_string(),
        "prompt3".to_string(),
    ];

    let results = client.batch_generate(&prompts, &params).await;
    assert!(results.is_ok());
    let responses = results.unwrap();
    assert_eq!(responses.len(), prompts.len());
    for (i, response) in responses.iter().enumerate() {
        assert!(response.text.contains(&format!("prompt{}", i + 1)));
    }
} 