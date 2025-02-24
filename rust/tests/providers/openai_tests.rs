use std::env;
use tokio;
use super_lightrag::llm::{
    LLMParams,
    Provider,
    ProviderConfig,
    providers::openai::OpenAIProvider,
    LLMError,
};

fn setup_test_provider() -> OpenAIProvider {
    let config = ProviderConfig {
        api_key: None,
        api_endpoint: None,
        model: "gpt-3.5-turbo".to_string(),
        timeout_secs: 30,
        max_retries: 3,
        org_id: None,
        extra_config: Default::default(),
    };
    
    OpenAIProvider::new(config).expect("Failed to create OpenAI provider")
}

fn setup_real_provider() -> OpenAIProvider {
    let config = ProviderConfig {
        api_key: Some("test-key".to_string()),
        api_endpoint: Some("https://api.openai.com".to_string()),
        model: "gpt-3.5-turbo".to_string(),
        timeout_secs: 30,
        max_retries: 3,
        org_id: None,
        extra_config: Default::default(),
    };
    
    OpenAIProvider::new(config).expect("Failed to create OpenAI provider")
}

#[tokio::test]
async fn test_openai_provider_initialization() {
    let mut provider = setup_test_provider();
    let result = provider.initialize().await;
    assert!(result.is_ok(), "Provider initialization failed");
}

#[tokio::test]
async fn test_openai_provider_completion() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "gpt-3.5-turbo".to_string(),
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
    assert_eq!(response.text, "4", "Response text should be '4'");
    assert_eq!(response.tokens_used, 5, "Response should have 5 tokens used");
}

#[tokio::test]
async fn test_openai_provider_streaming() {
    use futures::StreamExt;
    
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "gpt-3.5-turbo".to_string(),
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
    
    assert_eq!(received_chunks, 1, "Should receive exactly one chunk");
    assert_eq!(final_text, "4", "Final text should be '4'");
}

#[tokio::test]
async fn test_openai_provider_batch() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "gpt-3.5-turbo".to_string(),
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
    
    assert_eq!(responses[0].text, "4", "First response should be '4'");
    assert_eq!(responses[1].text, "6", "Second response should be '6'");
    for response in &responses {
        assert_eq!(response.tokens_used, 5, "Each response should have 5 tokens used");
    }
}

#[tokio::test]
async fn test_openai_provider_with_system_prompt() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "gpt-3.5-turbo".to_string(),
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
    assert_eq!(response.text, "4", "Response text should be '4'");
    assert_eq!(response.tokens_used, 5, "Response should have 5 tokens used");
}

#[tokio::test]
async fn test_openai_provider_error_handling() {
    // Test invalid model error
    let config = ProviderConfig {
        api_key: None,
        api_endpoint: None,
        model: "invalid-model".to_string(),
        timeout_secs: 5,
        max_retries: 1,
        org_id: None,
        extra_config: Default::default(),
    };
    
    let provider = OpenAIProvider::new(config).expect("Failed to create OpenAI provider");
    let params = LLMParams {
        model: "invalid-model".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 1.0,
        stream: false,
        system_prompt: None,
        extra_params: Default::default(),
        query_params: None,
    };

    let result = provider.complete("What is 2+2?", &params).await;
    assert!(result.is_err(), "Request with invalid model should fail");
    match result {
        Err(LLMError::RequestFailed(msg)) => {
            assert_eq!(msg, "Invalid model", "Should get invalid model error");
        }
        _ => panic!("Expected RequestFailed error with 'Invalid model' message"),
    }

    // Test streaming with invalid model
    let stream_result = provider.complete_stream("What is 2+2?", &params).await;
    assert!(stream_result.is_err(), "Streaming request with invalid model should fail");
    match stream_result {
        Err(LLMError::RequestFailed(msg)) => {
            assert_eq!(msg, "Invalid model", "Should get invalid model error for streaming");
        }
        _ => panic!("Expected RequestFailed error with 'Invalid model' message for streaming"),
    }

    // Test batch with invalid model
    let batch_result = provider.complete_batch(&["What is 2+2?".to_string()], &params).await;
    assert!(batch_result.is_err(), "Batch request with invalid model should fail");
    match batch_result {
        Err(LLMError::RequestFailed(msg)) => {
            assert_eq!(msg, "Invalid model", "Should get invalid model error for batch");
        }
        _ => panic!("Expected RequestFailed error with 'Invalid model' message for batch"),
    }
}

#[tokio::test]
async fn test_openai_provider_config_update() {
    let mut provider = setup_test_provider();
    let new_config = ProviderConfig {
        api_key: None,
        api_endpoint: None,
        model: "gpt-4".to_string(),
        timeout_secs: 60,
        max_retries: 5,
        org_id: None,
        extra_config: Default::default(),
    };
    
    let result = provider.update_config(new_config.clone());
    assert!(result.is_ok(), "Config update failed");
    
    let current_config = provider.get_config();
    assert_eq!(current_config.model, "gpt-4", "Model should be updated");
    assert_eq!(current_config.timeout_secs, 60, "Timeout should be updated");
}

#[tokio::test]
async fn test_openai_provider_metadata() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "gpt-3.5-turbo".to_string(),
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
    let metadata = response.metadata;
    
    // Verify specific metadata values in test mode
    assert_eq!(metadata.get("total_tokens").unwrap(), "5", "Incorrect total_tokens");
    assert_eq!(metadata.get("prompt_tokens").unwrap(), "2", "Incorrect prompt_tokens");
    assert_eq!(metadata.get("completion_tokens").unwrap(), "3", "Incorrect completion_tokens");
    assert_eq!(metadata.get("model").unwrap(), &params.model, "Incorrect model");

    // Test streaming metadata
    let stream_result = provider.complete_stream("What is 2+2?", &params).await;
    assert!(stream_result.is_ok(), "Streaming request failed");
    
    let mut stream = stream_result.unwrap();
    use futures::StreamExt;
    
    if let Some(Ok(chunk)) = stream.next().await {
        let stream_metadata = chunk.metadata;
        assert_eq!(stream_metadata.get("total_tokens").unwrap(), "5", "Incorrect streaming total_tokens");
        assert_eq!(stream_metadata.get("prompt_tokens").unwrap(), "2", "Incorrect streaming prompt_tokens");
        assert_eq!(stream_metadata.get("completion_tokens").unwrap(), "3", "Incorrect streaming completion_tokens");
        assert_eq!(stream_metadata.get("model").unwrap(), &params.model, "Incorrect streaming model");
    } else {
        panic!("No streaming response received");
    }
} 