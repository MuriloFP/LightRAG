use std::env;
use tokio;
use super_lightrag::llm::{
    LLMParams,
    Provider,
    ProviderConfig,
    providers::anthropic::AnthropicProvider,
};

fn setup_test_provider() -> AnthropicProvider {
    let config = ProviderConfig {
        api_key: env::var("ANTHROPIC_API_KEY").ok(),
        api_endpoint: Some("https://api.anthropic.com".to_string()),
        model: "claude-2".to_string(),
        timeout_secs: 30,
        max_retries: 3,
        org_id: None,
        extra_config: Default::default(),
    };
    
    AnthropicProvider::new(config).expect("Failed to create Anthropic provider")
}

#[tokio::test]
async fn test_anthropic_provider_initialization() {
    let mut provider = setup_test_provider();
    let result = provider.initialize().await;
    assert!(result.is_ok(), "Provider initialization failed");
}

#[tokio::test]
async fn test_anthropic_provider_completion() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "claude-2".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 1.0,
        stream: false,
        system_prompt: None,
        extra_params: Default::default(),
        query_params: None,
    };

    let result = provider.complete("What is 2+2?", &params).await;
    assert!(result.is_ok(), "Completion request failed");
    
    let response = result.unwrap();
    assert!(!response.text.is_empty(), "Response text should not be empty");
    assert!(response.tokens_used > 0, "Response should have tokens used");
}

#[tokio::test]
async fn test_anthropic_provider_streaming() {
    use futures::StreamExt;
    
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "claude-2".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 1.0,
        stream: true,
        system_prompt: None,
        extra_params: Default::default(),
        query_params: None,
    };

    let result = provider.complete_stream("What is 2+2?", &params).await;
    assert!(result.is_ok(), "Streaming request failed");
    
    let mut stream = result.unwrap();
    let mut received_chunks = 0;
    let mut final_text = String::new();
    
    while let Some(chunk) = stream.next().await {
        assert!(chunk.is_ok(), "Stream chunk should be ok");
        let chunk = chunk.unwrap();
        received_chunks += 1;
        final_text.push_str(&chunk.text);
    }
    
    assert!(received_chunks > 0, "Should receive at least one chunk");
    assert!(!final_text.is_empty(), "Final text should not be empty");
}

#[tokio::test]
async fn test_anthropic_provider_batch() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "claude-2".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 1.0,
        stream: false,
        system_prompt: None,
        extra_params: Default::default(),
        query_params: None,
    };

    let prompts = vec![
        "What is 2+2?".to_string(),
        "What is 3+3?".to_string(),
    ];

    let result = provider.complete_batch(&prompts, &params).await;
    assert!(result.is_ok(), "Batch request failed");
    
    let responses = result.unwrap();
    assert_eq!(responses.len(), prompts.len(), "Should get response for each prompt");
    
    for response in responses {
        assert!(!response.text.is_empty(), "Response text should not be empty");
        assert!(response.tokens_used > 0, "Response should have tokens used");
    }
}

#[tokio::test]
async fn test_anthropic_provider_with_system_prompt() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "claude-2".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 1.0,
        stream: false,
        system_prompt: Some("You are a math tutor.".to_string()),
        extra_params: Default::default(),
        query_params: None,
    };

    let result = provider.complete("What is 2+2?", &params).await;
    assert!(result.is_ok(), "Completion request with system prompt failed");
    
    let response = result.unwrap();
    assert!(!response.text.is_empty(), "Response text should not be empty");
    assert!(response.tokens_used > 0, "Response should have tokens used");
}

#[tokio::test]
async fn test_anthropic_provider_error_handling() {
    let mut config = ProviderConfig {
        api_key: Some("invalid_key".to_string()),
        api_endpoint: Some("https://api.anthropic.com".to_string()),
        model: "claude-2".to_string(),
        timeout_secs: 30,
        max_retries: 1,
        org_id: None,
        extra_config: Default::default(),
    };
    
    let provider = AnthropicProvider::new(config.clone()).expect("Failed to create Anthropic provider");
    let params = LLMParams {
        model: "claude-2".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 1.0,
        stream: false,
        system_prompt: None,
        extra_params: Default::default(),
        query_params: None,
    };

    let result = provider.complete("What is 2+2?", &params).await;
    assert!(result.is_err(), "Request with invalid key should fail");

    // Test with invalid model
    config.api_key = env::var("ANTHROPIC_API_KEY").ok();
    config.model = "invalid-model".to_string();
    let provider = AnthropicProvider::new(config).expect("Failed to create Anthropic provider");
    
    let result = provider.complete("What is 2+2?", &params).await;
    assert!(result.is_err(), "Request with invalid model should fail");
}

#[tokio::test]
async fn test_anthropic_provider_config_update() {
    let mut provider = setup_test_provider();
    let new_config = ProviderConfig {
        api_key: env::var("ANTHROPIC_API_KEY").ok(),
        api_endpoint: Some("https://api.anthropic.com".to_string()),
        model: "claude-instant-1".to_string(),
        timeout_secs: 60,
        max_retries: 5,
        org_id: None,
        extra_config: Default::default(),
    };
    
    let result = provider.update_config(new_config.clone());
    assert!(result.is_ok(), "Config update failed");
    
    let current_config = provider.get_config();
    assert_eq!(current_config.model, "claude-instant-1", "Model should be updated");
    assert_eq!(current_config.timeout_secs, 60, "Timeout should be updated");
} 