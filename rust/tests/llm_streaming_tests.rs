use std::collections::HashMap;
use std::time::Duration;
use futures::StreamExt;
use tokio;

use super_lightrag::llm::{
    LLMClient, LLMConfig, LLMParams, LLMResponse,
    providers::{OpenAIClient, OllamaClient},
};
use super_lightrag::types::llm::{StreamingResponse, StreamingTiming};

#[tokio::test]
async fn test_openai_streaming() {
    let config = LLMConfig {
        model: "gpt-3.5-turbo".to_string(),
        api_endpoint: Some("https://api.openai.com/v1/chat/completions".to_string()),
        api_key: Some("test-key".to_string()),
        org_id: None,
        timeout_secs: 30,
        max_retries: 1,
        use_cache: false,
        rate_limit_config: None,
        extra_config: HashMap::new(),
    };

    let client = OpenAIClient::new(config).unwrap();
    let params = LLMParams {
        max_tokens: 50,
        temperature: 0.7,
        top_p: 1.0,
        stream: true,
        system_prompt: None,
        query_params: None,
        extra_params: HashMap::new(),
    };

    let stream = client.generate_stream("Test prompt", &params).await.unwrap();
    let mut stream = Box::pin(stream);

    let mut received_chunks = 0;
    let mut total_tokens = 0;
    let mut last_chunk_timing: Option<StreamingTiming> = None;

    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                received_chunks += 1;
                total_tokens += chunk.chunk_tokens;
                assert!(!chunk.text.is_empty() || chunk.done);
                assert!(chunk.timing.is_some());
                last_chunk_timing = chunk.timing;
            }
            Err(e) => {
                if !e.to_string().contains("API key not valid") {
                    panic!("Unexpected error: {}", e);
                }
            }
        }
    }

    // Even with invalid API key, we should have proper stream structure
    assert!(received_chunks > 0);
    assert!(total_tokens > 0);
    assert!(last_chunk_timing.is_some());
}

#[tokio::test]
async fn test_ollama_streaming() {
    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some("http://localhost:11434".to_string()),
        api_key: None,
        org_id: None,
        timeout_secs: 30,
        max_retries: 1,
        use_cache: false,
        rate_limit_config: None,
        extra_config: HashMap::new(),
    };

    let client = OllamaClient::new(config).unwrap();
    let params = LLMParams {
        max_tokens: 50,
        temperature: 0.7,
        top_p: 1.0,
        stream: true,
        system_prompt: None,
        query_params: None,
        extra_params: HashMap::new(),
    };

    let stream = client.generate_stream("Test prompt", &params).await.unwrap();
    let mut stream = Box::pin(stream);

    let mut received_chunks = 0;
    let mut total_tokens = 0;
    let mut last_chunk_timing: Option<StreamingTiming> = None;

    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                received_chunks += 1;
                total_tokens += chunk.chunk_tokens;
                assert!(!chunk.text.is_empty() || chunk.done);
                assert!(chunk.timing.is_some());
                if let Some(timing) = &chunk.timing {
                    assert!(timing.chunk_duration > 0);
                    assert!(timing.total_duration > 0);
                    assert!(timing.prompt_eval_duration.is_some());
                    assert!(timing.token_gen_duration.is_some());
                }
                last_chunk_timing = chunk.timing;

                // Check Ollama-specific metadata
                assert!(chunk.metadata.contains_key("model"));
                if !chunk.done {
                    assert!(chunk.metadata.contains_key("eval_count"));
                }
            }
            Err(e) => {
                if !e.to_string().contains("connection refused") {
                    panic!("Unexpected error: {}", e);
                }
            }
        }
    }

    // Even with connection error, we should have proper stream structure
    assert!(received_chunks > 0);
    assert!(total_tokens > 0);
    assert!(last_chunk_timing.is_some());
}

#[tokio::test]
async fn test_streaming_cancellation() {
    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some("http://localhost:11434".to_string()),
        api_key: None,
        org_id: None,
        timeout_secs: 30,
        max_retries: 1,
        use_cache: false,
        rate_limit_config: None,
        extra_config: HashMap::new(),
    };

    let client = OllamaClient::new(config).unwrap();
    let params = LLMParams {
        max_tokens: 500, // Longer response to ensure we can cancel
        temperature: 0.7,
        top_p: 1.0,
        stream: true,
        system_prompt: None,
        query_params: None,
        extra_params: HashMap::new(),
    };

    let stream = client.generate_stream("Write a long story about an adventure", &params).await.unwrap();
    let mut stream = Box::pin(stream);

    let mut received_chunks = 0;
    
    // Only process first few chunks then drop the stream
    while let Some(result) = stream.next().await {
        match result {
            Ok(chunk) => {
                received_chunks += 1;
                if received_chunks >= 3 {
                    break;
                }
            }
            Err(_) => break,
        }
    }

    // Stream should be dropped gracefully
    drop(stream);
    
    // Small delay to ensure cleanup
    tokio::time::sleep(Duration::from_millis(100)).await;
}

#[tokio::test]
async fn test_streaming_timeout() {
    let config = LLMConfig {
        model: "llama2".to_string(),
        api_endpoint: Some("http://localhost:11434".to_string()),
        api_key: None,
        org_id: None,
        timeout_secs: 1, // Very short timeout
        max_retries: 1,
        use_cache: false,
        rate_limit_config: None,
        extra_config: HashMap::new(),
    };

    let client = OllamaClient::new(config).unwrap();
    let params = LLMParams {
        max_tokens: 50,
        temperature: 0.7,
        top_p: 1.0,
        stream: true,
        system_prompt: None,
        query_params: None,
        extra_params: HashMap::new(),
    };

    let result = client.generate_stream("Test prompt", &params).await;
    
    match result {
        Ok(_) => {
            // If we got a stream, it should error on first chunk due to timeout
            let mut stream = Box::pin(result.unwrap());
            if let Some(chunk) = stream.next().await {
                assert!(chunk.is_err());
                assert!(chunk.unwrap_err().to_string().contains("timeout"));
            }
        }
        Err(e) => {
            assert!(e.to_string().contains("timeout"));
        }
    }
} 