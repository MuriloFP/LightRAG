use std::env;
use tokio;
use super_lightrag::llm::{
    LLMParams,
    Provider,
    ProviderConfig,
    providers::{
        openai::OpenAIProvider,
        anthropic::AnthropicProvider,
        ollama::OllamaProvider,
    },
    multi_model::{MultiModel, MultiModelBuilder},
};

fn setup_test_config() -> (ProviderConfig, ProviderConfig, ProviderConfig) {
    let openai_config = ProviderConfig {
        api_key: env::var("OPENAI_API_KEY").ok(),
        api_endpoint: Some("https://api.openai.com".to_string()),
        model: "gpt-3.5-turbo".to_string(),
        timeout_secs: 30,
        max_retries: 3,
        org_id: None,
        extra_config: Default::default(),
    };

    let anthropic_config = ProviderConfig {
        api_key: env::var("ANTHROPIC_API_KEY").ok(),
        api_endpoint: Some("https://api.anthropic.com".to_string()),
        model: "claude-2".to_string(),
        timeout_secs: 30,
        max_retries: 3,
        org_id: None,
        extra_config: Default::default(),
    };

    let ollama_config = ProviderConfig {
        api_key: None,
        api_endpoint: Some("http://localhost:11434".to_string()),
        model: "llama2".to_string(),
        timeout_secs: 30,
        max_retries: 3,
        org_id: None,
        extra_config: Default::default(),
    };

    (openai_config, anthropic_config, ollama_config)
}

#[tokio::test]
async fn test_multi_model_builder() {
    let (openai_config, anthropic_config, ollama_config) = setup_test_config();
    
    let result = MultiModelBuilder::new()
        .add_model("openai", openai_config.clone())
        .and_then(|b| b.add_model("anthropic", anthropic_config.clone()))
        .and_then(|b| b.add_model("ollama", ollama_config.clone()));
        
    assert!(result.is_ok(), "MultiModel builder should succeed");
    
    let multi_model = result.unwrap().build();
    assert!(multi_model.complete("Test", &Default::default()).await.is_ok());
}

#[tokio::test]
async fn test_multi_model_completion() {
    let (openai_config, anthropic_config, ollama_config) = setup_test_config();
    
    let multi_model = MultiModelBuilder::new()
        .add_model("openai", openai_config)
        .unwrap()
        .add_model("anthropic", anthropic_config)
        .unwrap()
        .add_model("ollama", ollama_config)
        .unwrap()
        .build();

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

    // Test multiple completions to ensure round-robin
    for _ in 0..3 {
        let result = multi_model.complete("What is 2+2?", &params).await;
        assert!(result.is_ok(), "Multi-model completion should succeed");
        
        let response = result.unwrap();
        assert!(!response.text.is_empty(), "Response text should not be empty");
        assert!(response.tokens_used > 0, "Response should have tokens used");
    }
}

#[tokio::test]
async fn test_multi_model_streaming() {
    use futures::StreamExt;
    
    let (openai_config, anthropic_config, ollama_config) = setup_test_config();
    
    let multi_model = MultiModelBuilder::new()
        .add_model("openai", openai_config)
        .unwrap()
        .add_model("anthropic", anthropic_config)
        .unwrap()
        .add_model("ollama", ollama_config)
        .unwrap()
        .build();

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

    let result = multi_model.complete_stream("What is 2+2?", &params).await;
    assert!(result.is_ok(), "Multi-model streaming should succeed");
    
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
async fn test_multi_model_batch() {
    let (openai_config, anthropic_config, ollama_config) = setup_test_config();
    
    let multi_model = MultiModelBuilder::new()
        .add_model("openai", openai_config)
        .unwrap()
        .add_model("anthropic", anthropic_config)
        .unwrap()
        .add_model("ollama", ollama_config)
        .unwrap()
        .build();

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
        "What is 4+4?".to_string(),
    ];

    let result = multi_model.complete_batch(&prompts, &params).await;
    assert!(result.is_ok(), "Multi-model batch completion should succeed");
    
    let responses = result.unwrap();
    assert_eq!(responses.len(), prompts.len(), "Should get response for each prompt");
    
    for response in responses {
        assert!(!response.text.is_empty(), "Response text should not be empty");
        assert!(response.tokens_used > 0, "Response should have tokens used");
    }
}

#[tokio::test]
async fn test_multi_model_error_handling() {
    let mut openai_config = ProviderConfig {
        api_key: Some("invalid_key".to_string()),
        api_endpoint: Some("https://api.openai.com".to_string()),
        model: "gpt-3.5-turbo".to_string(),
        timeout_secs: 30,
        max_retries: 1,
        org_id: None,
        extra_config: Default::default(),
    };
    
    let mut anthropic_config = ProviderConfig {
        api_key: Some("invalid_key".to_string()),
        api_endpoint: Some("https://api.anthropic.com".to_string()),
        model: "claude-2".to_string(),
        timeout_secs: 30,
        max_retries: 1,
        org_id: None,
        extra_config: Default::default(),
    };
    
    let ollama_config = ProviderConfig {
        api_key: None,
        api_endpoint: Some("http://invalid-endpoint:11434".to_string()),
        model: "llama2".to_string(),
        timeout_secs: 5,
        max_retries: 1,
        org_id: None,
        extra_config: Default::default(),
    };

    let multi_model = MultiModelBuilder::new()
        .add_model("openai", openai_config.clone())
        .unwrap()
        .add_model("anthropic", anthropic_config.clone())
        .unwrap()
        .add_model("ollama", ollama_config.clone())
        .unwrap()
        .build();

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

    let result = multi_model.complete("What is 2+2?", &params).await;
    assert!(result.is_err(), "Request with all invalid configs should fail");

    // Test with one valid config
    openai_config.api_key = env::var("OPENAI_API_KEY").ok();
    let multi_model = MultiModelBuilder::new()
        .add_model("openai", openai_config)
        .unwrap()
        .add_model("anthropic", anthropic_config)
        .unwrap()
        .add_model("ollama", ollama_config)
        .unwrap()
        .build();

    let result = multi_model.complete("What is 2+2?", &params).await;
    assert!(result.is_ok(), "Request should succeed with one valid config");
} 