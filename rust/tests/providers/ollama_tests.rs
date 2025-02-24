use tokio;
use super_lightrag::llm::{
    LLMParams,
    Provider,
    ProviderConfig,
    providers::ollama::OllamaProvider,
};

fn setup_test_provider() -> OllamaProvider {
    let config = ProviderConfig {
        api_key: None,
        api_endpoint: Some("http://localhost:11434".to_string()),
        model: "llama2".to_string(),
        timeout_secs: 30,
        max_retries: 3,
        org_id: None,
        extra_config: Default::default(),
    };
    
    OllamaProvider::new(config).expect("Failed to create Ollama provider")
}

#[tokio::test]
async fn test_ollama_provider_initialization() {
    let mut provider = setup_test_provider();
    let result = provider.initialize().await;
    assert!(result.is_ok(), "Provider initialization failed");
}

#[tokio::test]
async fn test_ollama_provider_completion() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "llama2".to_string(),
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
async fn test_ollama_provider_streaming() {
    use futures::StreamExt;
    
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "llama2".to_string(),
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
async fn test_ollama_provider_batch() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "llama2".to_string(),
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
async fn test_ollama_provider_with_system_prompt() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "llama2".to_string(),
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
async fn test_ollama_provider_error_handling() {
    let config = ProviderConfig {
        api_key: None,
        api_endpoint: Some("http://invalid-endpoint:11434".to_string()),
        model: "llama2".to_string(),
        timeout_secs: 5,
        max_retries: 1,
        org_id: None,
        extra_config: Default::default(),
    };
    
    let provider = OllamaProvider::new(config).expect("Failed to create Ollama provider");
    let params = LLMParams {
        model: "llama2".to_string(),
        max_tokens: 100,
        temperature: 0.7,
        top_p: 1.0,
        stream: false,
        system_prompt: None,
        extra_params: Default::default(),
        query_params: None,
    };

    let result = provider.complete("What is 2+2?", &params).await;
    assert!(result.is_err(), "Request with invalid endpoint should fail");
}

#[tokio::test]
async fn test_ollama_provider_config_update() {
    let mut provider = setup_test_provider();
    let new_config = ProviderConfig {
        api_key: None,
        api_endpoint: Some("http://localhost:11434".to_string()),
        model: "codellama".to_string(),
        timeout_secs: 60,
        max_retries: 5,
        org_id: None,
        extra_config: Default::default(),
    };
    
    let result = provider.update_config(new_config.clone());
    assert!(result.is_ok(), "Config update failed");
    
    let current_config = provider.get_config();
    assert_eq!(current_config.model, "codellama", "Model should be updated");
    assert_eq!(current_config.timeout_secs, 60, "Timeout should be updated");
}

#[tokio::test]
async fn test_ollama_provider_metadata() {
    let provider = setup_test_provider();
    let params = LLMParams {
        model: "llama2".to_string(),
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
    
    assert!(metadata.contains_key("total_duration"), "Should have total_duration in metadata");
    assert!(metadata.contains_key("load_duration"), "Should have load_duration in metadata");
    assert!(metadata.contains_key("eval_duration"), "Should have eval_duration in metadata");
    assert!(metadata.contains_key("prompt_eval_count"), "Should have prompt_eval_count in metadata");
    assert!(metadata.contains_key("eval_count"), "Should have eval_count in metadata");
} 