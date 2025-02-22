use super_lightrag::llm::{
    LLMClient,
    LLMConfig, LLMParams, LLMError,
    providers::{
        litellm::{OpenAIAdapter, AnthropicAdapter, ProviderAdapter, LLMProvider},
        openai::OpenAIClient,
        anthropic::AnthropicClient,
    },
};
use std::env;
use tokio;

// Helper function to create test config
fn create_test_config() -> LLMConfig {
    LLMConfig {
        model: "gpt-4".to_string(),
        api_key: Some(env::var("OPENAI_API_KEY").unwrap_or_else(|_| "test_key".to_string())),
        api_endpoint: None,
        org_id: None,
        timeout_secs: 30,
        max_retries: 1,
        use_cache: false,
        rate_limit_config: None,
        similarity_threshold: 0.8,
        extra_config: Default::default(),
    }
}

// Helper function to create test params
fn create_test_params() -> LLMParams {
    LLMParams {
        model: "openai/gpt-4".to_string(),
        max_tokens: 1000,
        temperature: 0.7,
        top_p: 1.0,
        stream: false,
        system_prompt: None,
        query_params: None,
        extra_params: Default::default(),
    }
}

// Helper function to create Anthropic test config
fn create_anthropic_test_config() -> LLMConfig {
    LLMConfig {
        model: "claude-2".to_string(),
        api_key: Some(env::var("ANTHROPIC_API_KEY").unwrap_or_else(|_| "test_key".to_string())),
        api_endpoint: Some("https://api.anthropic.com".to_string()),
        org_id: None,
        timeout_secs: 30,
        max_retries: 1,
        use_cache: false,
        rate_limit_config: None,
        similarity_threshold: 0.8,
        extra_config: Default::default(),
    }
}

// Helper function to create Anthropic test params
fn create_anthropic_test_params() -> LLMParams {
    LLMParams {
        model: "anthropic/claude-2".to_string(),
        max_tokens: 1000,
        temperature: 0.7,
        top_p: 1.0,
        stream: false,
        system_prompt: None,
        query_params: None,
        extra_params: Default::default(),
    }
}

#[tokio::test]
async fn test_openai_adapter_creation() {
    let config = create_test_config();
    let client = OpenAIClient::new(config.clone()).unwrap();
    let adapter = OpenAIAdapter::new(client, config);
    assert!(adapter.is_ok());
}

#[tokio::test]
async fn test_openai_adapter_creation_no_api_key() {
    let mut config = create_test_config();
    config.api_key = None;
    let client = OpenAIClient::new(config.clone()).unwrap();
    let adapter = OpenAIAdapter::new(client, config);
    assert!(matches!(adapter.err(), Some(LLMError::ConfigError(_))));
}

#[tokio::test]
async fn test_openai_adapter_validate_params() {
    let config = create_test_config();
    let client = OpenAIClient::new(config.clone()).unwrap();
    let adapter = OpenAIAdapter::new(client, config).unwrap();

    // Test valid params
    let params = create_test_params();
    assert!(adapter.validate_params(&params).is_ok());

    // Test invalid provider
    let mut invalid_params = params.clone();
    invalid_params.model = "anthropic/claude-2".to_string();
    assert!(matches!(
        adapter.validate_params(&invalid_params),
        Err(LLMError::ConfigError(_))
    ));

    // Test invalid temperature
    let mut invalid_params = params.clone();
    invalid_params.temperature = 2.5;
    assert!(matches!(
        adapter.validate_params(&invalid_params),
        Err(LLMError::ConfigError(_))
    ));

    // Test invalid top_p
    let mut invalid_params = params;
    invalid_params.top_p = 1.5;
    assert!(matches!(
        adapter.validate_params(&invalid_params),
        Err(LLMError::ConfigError(_))
    ));
}

#[tokio::test]
async fn test_openai_adapter_error_mapping() {
    let config = create_test_config();
    let client = OpenAIClient::new(config.clone()).unwrap();
    let adapter = OpenAIAdapter::new(client, config).unwrap();

    // Test rate limit error mapping
    let rate_limit_error = std::io::Error::new(
        std::io::ErrorKind::Other,
        "rate limit exceeded, please try again in 20s"
    );
    assert!(matches!(
        adapter.map_error(rate_limit_error),
        LLMError::RateLimitExceeded(_)
    ));

    // Test invalid request error mapping
    let invalid_request_error = std::io::Error::new(
        std::io::ErrorKind::Other,
        "invalid_request_error: Invalid model specified"
    );
    assert!(matches!(
        adapter.map_error(invalid_request_error),
        LLMError::ConfigError(_)
    ));

    // Test token limit error mapping
    let token_limit_error = std::io::Error::new(
        std::io::ErrorKind::Other,
        "token limit exceeded for model gpt-4"
    );
    assert!(matches!(
        adapter.map_error(token_limit_error),
        LLMError::TokenLimitExceeded(_)
    ));

    // Test generic error mapping
    let generic_error = std::io::Error::new(
        std::io::ErrorKind::Other,
        "unknown error occurred"
    );
    assert!(matches!(
        adapter.map_error(generic_error),
        LLMError::RequestFailed(_)
    ));
}

// Integration test that requires actual API key
#[tokio::test]
#[ignore]
async fn test_openai_adapter_integration() {
    // Only run if OPENAI_API_KEY is set
    if let Ok(api_key) = env::var("OPENAI_API_KEY") {
        let mut config = create_test_config();
        config.api_key = Some(api_key);
        
        let client = OpenAIClient::new(config.clone()).unwrap();
        let adapter = OpenAIAdapter::new(client, config).unwrap();
        let params = create_test_params();

        // Test completion
        let result = adapter.complete("Hello, how are you?", &params).await;
        assert!(result.is_ok());

        // Test streaming
        let stream_result = adapter.complete_stream("Hello, how are you?", &params).await;
        assert!(stream_result.is_ok());

        // Test embeddings
        let embed_result = adapter.embed(&["Hello, world!".to_string()]).await;
        assert!(embed_result.is_ok());
    }
}

#[tokio::test]
async fn test_anthropic_adapter_creation() {
    let config = create_anthropic_test_config();
    let client = AnthropicClient::new(config.clone()).unwrap();
    let adapter = AnthropicAdapter::new(client, config);
    assert!(adapter.is_ok());
}

#[tokio::test]
async fn test_anthropic_adapter_creation_no_api_key() {
    let mut config = create_anthropic_test_config();
    config.api_key = None;
    let client = AnthropicClient::new(config.clone()).unwrap();
    let adapter = AnthropicAdapter::new(client, config);
    assert!(matches!(adapter.err(), Some(LLMError::ConfigError(_))));
}

#[tokio::test]
async fn test_anthropic_adapter_validate_params() {
    let config = create_anthropic_test_config();
    let client = AnthropicClient::new(config.clone()).unwrap();
    let adapter = AnthropicAdapter::new(client, config).unwrap();

    // Test valid params
    let params = create_anthropic_test_params();
    assert!(adapter.validate_params(&params).is_ok());

    // Test invalid provider
    let mut invalid_params = params.clone();
    invalid_params.model = "openai/gpt-4".to_string();
    assert!(matches!(
        adapter.validate_params(&invalid_params),
        Err(LLMError::ConfigError(_))
    ));

    // Test invalid temperature
    let mut invalid_params = params.clone();
    invalid_params.temperature = 1.5;
    assert!(matches!(
        adapter.validate_params(&invalid_params),
        Err(LLMError::ConfigError(_))
    ));

    // Test invalid top_p
    let mut invalid_params = params;
    invalid_params.top_p = 1.5;
    assert!(matches!(
        adapter.validate_params(&invalid_params),
        Err(LLMError::ConfigError(_))
    ));
}

#[tokio::test]
async fn test_adapter_max_tokens() {
    // Test OpenAI adapter
    let config = create_test_config();
    let client = OpenAIClient::new(config.clone()).unwrap();
    let openai_adapter = OpenAIAdapter::new(client, config).unwrap();
    assert_eq!(openai_adapter.max_tokens(), usize::MAX);

    // Test Anthropic adapter
    let config = create_anthropic_test_config();
    let client = AnthropicClient::new(config.clone()).unwrap();
    let anthropic_adapter = AnthropicAdapter::new(client, config).unwrap();
    assert_eq!(anthropic_adapter.max_tokens(), usize::MAX);
}

#[tokio::test]
async fn test_anthropic_adapter_integration() {
    // Only run if ANTHROPIC_API_KEY is set
    if let Ok(api_key) = env::var("ANTHROPIC_API_KEY") {
        let mut config = create_anthropic_test_config();
        config.api_key = Some(api_key);
        
        let client = AnthropicClient::new(config.clone()).unwrap();
        let adapter = AnthropicAdapter::new(client, config).unwrap();
        let params = create_anthropic_test_params();

        // Test completion
        let result = adapter.complete("Hello, how are you?", &params).await;
        assert!(result.is_ok());

        // Test streaming
        let stream_result = adapter.complete_stream("Hello, how are you?", &params).await;
        assert!(stream_result.is_ok());

        // Test embeddings (should return error as not supported)
        let embed_result = adapter.embed(&["Hello, world!".to_string()]).await;
        assert!(matches!(embed_result, Err(LLMError::ConfigError(_))));
    }
} 