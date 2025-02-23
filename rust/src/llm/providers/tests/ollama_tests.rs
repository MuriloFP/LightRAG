use super::*;
use crate::llm::providers::ollama::OllamaClient;
use crate::processing::keywords::ConversationTurn;
use crate::types::llm::{LLMConfig, LLMParams, LLMError};
use tokio::test;
use wiremock::{Mock, MockServer, ResponseTemplate};
use wiremock::matchers::{method, path};
use serde_json::json;

#[test]
async fn test_build_conversation() {
    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some("http://localhost:11434".to_string()),
        ..Default::default()
    };
    let client = OllamaClient::new(config).unwrap();

    let history = vec![
        ConversationTurn {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
        },
        ConversationTurn {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        },
        ConversationTurn {
            role: "assistant".to_string(),
            content: "Hi there!".to_string(),
        },
    ];

    let params = LLMParams {
        system_prompt: Some("Be concise.".to_string()),
        ..Default::default()
    };

    let conversation = client.build_conversation("How are you?", Some(&history), &params);

    assert!(conversation.contains("System: Be concise."));
    assert!(conversation.contains("System: You are a helpful assistant."));
    assert!(conversation.contains("User: Hello!"));
    assert!(conversation.contains("Assistant: Hi there!"));
    assert!(conversation.contains("User: How are you?"));
    assert!(conversation.ends_with("Assistant:"));
}

#[test]
async fn test_build_request() {
    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some("http://localhost:11434".to_string()),
        ..Default::default()
    };
    let client = OllamaClient::new(config).unwrap();

    let history = vec![
        ConversationTurn {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        },
    ];

    let mut params = LLMParams::default();
    params.temperature = 0.7;
    params.top_p = 0.9;
    params.max_tokens = 100;
    params.stream = false;
    params.extra_params.insert("repeat_penalty".to_string(), "1.1".to_string());

    let request = client.build_request("How are you?", Some(&history), &params);

    assert_eq!(request["model"], "llama2");
    assert!(request["prompt"].as_str().unwrap().contains("User: Hello!"));
    assert!(request["prompt"].as_str().unwrap().contains("User: How are you?"));
    assert_eq!(request["stream"], false);
    assert_eq!(request["options"]["temperature"], 0.7);
    assert_eq!(request["options"]["top_p"], 0.9);
    assert_eq!(request["options"]["num_predict"], 100);
    assert_eq!(request["options"]["repeat_penalty"], "1.1");
}

#[test]
async fn test_generate_with_history() {
    let mock_server = MockServer::start().await;

    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some(mock_server.uri()),
        ..Default::default()
    };
    let client = OllamaClient::new(config).unwrap();

    let history = vec![
        ConversationTurn {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        },
        ConversationTurn {
            role: "assistant".to_string(),
            content: "Hi there!".to_string(),
        },
    ];

    let params = LLMParams::default();

    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "response": "I'm doing well, thanks for asking!",
            "done": true,
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": 10,
            "eval_count": 20,
            "eval_duration": 900000
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let response = client.generate_with_history("How are you?", &history, &params).await.unwrap();

    assert_eq!(response.text, "I'm doing well, thanks for asking!");
    assert_eq!(response.tokens_used, 30);
    assert_eq!(response.model, "llama2");
    assert!(!response.cached);
    assert!(response.metadata.contains_key("total_duration"));
    assert!(response.metadata.contains_key("load_duration"));
    assert!(response.metadata.contains_key("eval_duration"));
}

#[test]
async fn test_error_handling() {
    let mock_server = MockServer::start().await;

    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some(mock_server.uri()),
        ..Default::default()
    };
    let client = OllamaClient::new(config).unwrap();

    // Test rate limit error
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(429).set_body_json(json!({
            "error": "rate limit exceeded, please try again later"
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let result = client.generate("test", &LLMParams::default()).await;
    assert!(matches!(result, Err(LLMError::RateLimitExceeded(_))));

    // Test token limit error
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(400).set_body_json(json!({
            "error": "context length exceeded"
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let result = client.generate("test", &LLMParams::default()).await;
    assert!(matches!(result, Err(LLMError::TokenLimitExceeded(_))));

    // Test model error
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(404).set_body_json(json!({
            "error": "model not found: nonexistent-model"
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let result = client.generate("test", &LLMParams::default()).await;
    assert!(matches!(result, Err(LLMError::ConfigError(_))));
}

#[test]
async fn test_token_validation() {
    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some("http://localhost:11434".to_string()),
        ..Default::default()
    };
    let client = OllamaClient::new(config).unwrap();

    // Create a long prompt that would exceed token limit
    let long_prompt = "test ".repeat(4000);
    let params = LLMParams {
        max_tokens: 1000,
        ..Default::default()
    };

    let result = client.validate_token_limits(&long_prompt, &params);
    assert!(matches!(result, Err(LLMError::TokenLimitExceeded(_))));
}

#[test]
async fn test_cache_with_history() {
    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some("http://localhost:11434".to_string()),
        use_cache: true,
        ..Default::default()
    };
    let client = OllamaClient::new(config).unwrap();

    let history = vec![
        ConversationTurn {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        },
    ];

    let cache_key = format!("{:?}:{}", history, "test prompt");
    
    // First call should miss cache
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "response": "cached response",
            "done": true,
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": 10,
            "eval_count": 20,
            "eval_duration": 900000
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    let first_response = client.generate_with_history("test prompt", &history, &LLMParams::default()).await.unwrap();
    assert_eq!(first_response.text, "cached response");
    assert!(!first_response.cached);

    // Second call should hit cache
    let second_response = client.generate_with_history("test prompt", &history, &LLMParams::default()).await.unwrap();
    assert_eq!(second_response.text, "cached response");
    assert!(second_response.cached);
}

#[tokio::test]
async fn test_similarity_search_caching() {
    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some("http://localhost:11434".to_string()),
        use_cache: true,
        similarity_enabled: true,
        similarity_threshold: 0.85,
        ..Default::default()
    };
    let client = OllamaClient::new(config).unwrap();

    let mock_server = MockServer::start().await;
    
    // Mock embeddings endpoint
    Mock::given(method("POST"))
        .and(path("/api/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": vec![0.1; 384]
        })))
        .expect(2)
        .mount(&mock_server)
        .await;

    // Mock generate endpoint
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "response": "test response",
            "done": true,
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": 10,
            "eval_count": 20,
            "eval_duration": 900000
        })))
        .expect(1)
        .mount(&mock_server)
        .await;

    // First request should miss cache
    let first_response = client.generate(
        "What is the meaning of life?",
        &LLMParams {
            query_params: Some(QueryParams::default()),
            ..Default::default()
        }
    ).await.unwrap();
    assert!(!first_response.cached);

    // Similar query should hit cache
    let second_response = client.generate(
        "Tell me about the meaning of life",
        &LLMParams {
            query_params: Some(QueryParams::default()),
            ..Default::default()
        }
    ).await.unwrap();
    assert!(second_response.cached);
    assert_eq!(second_response.text, first_response.text);
}

#[tokio::test]
async fn test_similarity_search_threshold() {
    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some("http://localhost:11434".to_string()),
        use_cache: true,
        similarity_enabled: true,
        similarity_threshold: 0.95, // Very high threshold
        ..Default::default()
    };
    let client = OllamaClient::new(config).unwrap();

    let mock_server = MockServer::start().await;
    
    // Mock embeddings endpoint with different embeddings
    Mock::given(method("POST"))
        .and(path("/api/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": vec![0.1; 384]
        })))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    Mock::given(method("POST"))
        .and(path("/api/embeddings"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "embedding": vec![0.9; 384]
        })))
        .up_to_n_times(1)
        .mount(&mock_server)
        .await;

    // Mock generate endpoint
    Mock::given(method("POST"))
        .and(path("/api/generate"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "response": "test response",
            "done": true,
            "total_duration": 1000000,
            "load_duration": 100000,
            "prompt_eval_count": 10,
            "eval_count": 20,
            "eval_duration": 900000
        })))
        .expect(2)
        .mount(&mock_server)
        .await;

    // First request
    let first_response = client.generate(
        "What is the meaning of life?",
        &LLMParams {
            query_params: Some(QueryParams::default()),
            ..Default::default()
        }
    ).await.unwrap();
    assert!(!first_response.cached);

    // Very different query should miss cache due to high threshold
    let second_response = client.generate(
        "Tell me about programming",
        &LLMParams {
            query_params: Some(QueryParams::default()),
            ..Default::default()
        }
    ).await.unwrap();
    assert!(!second_response.cached);
} 