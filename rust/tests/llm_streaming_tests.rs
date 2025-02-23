use std::collections::HashMap;
use std::time::Duration;
use futures::StreamExt;
use tokio;
use std::time::Instant;
use std::pin::Pin;
use futures::future::BoxFuture;
use futures::stream;
use bytes::Bytes;

use super_lightrag::llm::{
    LLMClient, LLMConfig, LLMParams,
    providers::{OpenAIClient, OllamaClient},
};
use super_lightrag::types::llm::StreamingTiming;
use wiremock::{Mock, MockServer, ResponseTemplate};
use wiremock::matchers::{method, path};
use serde_json::json;
use super_lightrag::llm::streaming::{StreamConfig, StreamProcessor};
use super_lightrag::llm::LLMError;

fn streaming_parser(text: &str) -> futures::future::BoxFuture<'static, Result<Option<(String, bool, std::collections::HashMap<String, String>)>, super_lightrag::llm::LLMError>> {
    let text_owned = text.to_string();
    Box::pin(async move {
        tokio::time::sleep(std::time::Duration::from_micros(100)).await;
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("test".to_string(), "value".to_string());
        let contains = text_owned.contains("chunk999");
        let formatted_text = format!("{{\"text\": \"{}\"}}", text_owned);
        Ok(Some((formatted_text, contains, metadata)))
    })
}

#[tokio::test]
async fn test_openai_streaming() {
    let config = LLMConfig {
        model: "gpt-3.5-turbo".to_string(),
        api_endpoint: Some("https://api.openai.com".to_string()),
        api_key: Some("test-key".to_string()),
        org_id: None,
        timeout_secs: 30,
        max_retries: 1,
        use_cache: false,
        rate_limit_config: None,
        similarity_threshold: 0.8,
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
        model: "openai/gpt-4".to_string(),
    };

    let result = client.generate_stream("Test prompt", &params).await;
    match result {
        Ok(stream) => {
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
                        if !e.to_string().contains("API key not valid") && !e.to_string().contains("invalid_api_key") {
                            panic!("Unexpected error: {}", e);
                        }
                        return;
                    }
                }
            }

            // Only check these if we got any chunks
            if received_chunks > 0 {
                assert!(total_tokens > 0);
                assert!(last_chunk_timing.is_some());
            }
        }
        Err(e) => {
            // Test should pass if we get an invalid API key error
            assert!(e.to_string().contains("API key not valid") || e.to_string().contains("invalid_api_key"),
                "Expected API key error, got: {}", e);
        }
    }
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
        similarity_threshold: 0.8,
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
        model: "openai/gpt-4".to_string(),
    };

    let stream = client.generate_stream("Test prompt", &params).await;
    match stream {
        Ok(stream) => {
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
                        let error_msg = e.to_string().to_lowercase();
                        let is_connection_error = error_msg.contains("connection refused") ||
                                                error_msg.contains("connection failed") ||
                                                error_msg.contains("failed to connect") ||
                                                error_msg.contains("error trying to connect") ||
                                                error_msg.contains("tcp connect error");
                        
                        assert!(is_connection_error, "Unexpected error: {}", e);
                        return; // Exit early if Ollama is not running
                    }
                }
            }

            // Only check these if we got any chunks
            if received_chunks > 0 {
                assert!(total_tokens > 0);
                assert!(last_chunk_timing.is_some());
            }
        }
        Err(e) => {
            // Skip test if Ollama is not running
            let error_msg = e.to_string().to_lowercase();
            let is_connection_error = error_msg.contains("connection refused") ||
                                    error_msg.contains("connection failed") ||
                                    error_msg.contains("failed to connect") ||
                                    error_msg.contains("error trying to connect") ||
                                    error_msg.contains("tcp connect error");
            
            assert!(is_connection_error, "Unexpected error: {}", e);
        }
    }
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
        similarity_threshold: 0.8,
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
        model: "openai/gpt-4".to_string(),
    };

    let stream = client.generate_stream("Write a long story about an adventure", &params).await;
    match stream {
        Ok(stream) => {
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
                    Err(e) => {
                        let error_msg = e.to_string().to_lowercase();
                        let is_connection_error = error_msg.contains("connection refused") ||
                                                error_msg.contains("connection failed") ||
                                                error_msg.contains("failed to connect") ||
                                                error_msg.contains("error trying to connect") ||
                                                error_msg.contains("tcp connect error");
                        
                        assert!(is_connection_error, "Unexpected error: {}", e);
                        break;
                    }
                }
            }

            // Stream should be dropped gracefully
            drop(stream);
            
            // Small delay to ensure cleanup
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        Err(e) => {
            // Skip test if Ollama is not running
            let error_msg = e.to_string().to_lowercase();
            let is_connection_error = error_msg.contains("connection refused") ||
                                    error_msg.contains("connection failed") ||
                                    error_msg.contains("failed to connect") ||
                                    error_msg.contains("error trying to connect") ||
                                    error_msg.contains("tcp connect error");
            
            assert!(is_connection_error, "Unexpected error: {}", e);
        }
    }
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
        similarity_threshold: 0.8,
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
        model: "openai/gpt-4".to_string(),
    };

    let result = client.generate_stream("Test prompt", &params).await;
    
    match result {
        Ok(stream) => {
            // If we got a stream, it should error on first chunk due to timeout
            let mut stream = Box::pin(stream);
            if let Some(chunk) = stream.next().await {
                assert!(chunk.is_err());
                let err = chunk.unwrap_err();
                let error_msg = err.to_string().to_lowercase();
                
                // Check for various timeout and connection error patterns
                let is_timeout = error_msg.contains("timeout") || 
                               error_msg.contains("timed out") ||
                               error_msg.contains("operation timed out");
                               
                let is_connection_error = error_msg.contains("connection refused") ||
                                        error_msg.contains("connection failed") ||
                                        error_msg.contains("failed to connect");
                
                assert!(is_timeout || is_connection_error,
                    "Expected timeout or connection error, got: {}", err);
            }
        }
        Err(e) => {
            let error_msg = e.to_string().to_lowercase();
            
            // Check for various timeout and connection error patterns
            let is_timeout = error_msg.contains("timeout") || 
                           error_msg.contains("timed out") ||
                           error_msg.contains("operation timed out");
                           
            let is_connection_error = error_msg.contains("connection refused") ||
                                    error_msg.contains("connection failed") ||
                                    error_msg.contains("failed to connect");
            
            assert!(is_timeout || is_connection_error,
                "Expected timeout or connection error, got: {}", e);
        }
    }
}

#[tokio::test]
async fn test_batch_processing_performance() {
    let mock_server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_json(json!({
                    "id": "chatcmpl-batch-123",
                    "object": "chat.completion",
                    "created": 1677652288,
                    "model": "gpt-3.5-turbo",
                    "choices": [
                        {"message": {"role": "assistant", "content": "response for prompt1"}, "text": "response for prompt1", "finish_reason": "stop", "index": 0},
                        {"message": {"role": "assistant", "content": "response for prompt2"}, "text": "response for prompt2", "finish_reason": "stop", "index": 1},
                        {"message": {"role": "assistant", "content": "response for prompt3"}, "text": "response for prompt3", "finish_reason": "stop", "index": 2}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30
                    }
                }))
        )
        .mount(&mock_server)
        .await;

    let config = LLMConfig {
        model: "gpt-3.5-turbo".to_string(),
        api_endpoint: Some(mock_server.uri()),
        api_key: Some("test-key".to_string()),
        org_id: None,
        timeout_secs: 30,
        max_retries: 1,
        use_cache: false,
        rate_limit_config: None,
        similarity_threshold: 0.8,
        extra_config: HashMap::new(),
    };
    let client = OpenAIClient::new(config).unwrap();

    let prompts = vec![
        "prompt1".to_string(),
        "prompt2".to_string(),
        "prompt3".to_string(),
    ];

    let params = LLMParams {
        max_tokens: 50,
        temperature: 0.7,
        top_p: 1.0,
        stream: false,
        system_prompt: None,
        query_params: None,
        extra_params: HashMap::new(),
        model: "openai/gpt-3.5-turbo".to_string(),
    };

    // Measure batch processing time
    let start_time = Instant::now();
    let responses = client.batch_generate_chat(&prompts, &params).await.unwrap();
    let batch_duration = start_time.elapsed();

    // Measure sequential processing time
    let start_time = Instant::now();
    let mut sequential_responses = Vec::new();
    for prompt in &prompts {
        let response = client.generate(prompt, &params).await.unwrap();
        sequential_responses.push(response);
    }
    let sequential_duration = start_time.elapsed();

    // Verify responses
    assert_eq!(responses.len(), prompts.len());
    assert_eq!(sequential_responses.len(), prompts.len());

    // Print performance results for manual inspection
    println!("Batch processing duration: {:?}", batch_duration);
    println!("Sequential processing duration: {:?}", sequential_duration);
}

#[tokio::test]
async fn test_streaming_performance() {
    // Test configuration with different batching settings
    let configs = vec![
        StreamConfig {
            enable_batching: false,
            ..Default::default()
        },
        StreamConfig {
            enable_batching: true,
            max_batch_size: 1024,
            max_batch_wait_ms: 10,
            ..Default::default()
        },
        StreamConfig {
            enable_batching: true,
            max_batch_size: 16384,
            max_batch_wait_ms: 50,
            ..Default::default()
        },
    ];

    // Create a large test stream generator function
    let create_test_stream = || {
        let num_chunks = 1000;
        stream::iter((0..num_chunks).map(|i| {
            Ok(Bytes::from(format!("chunk{}\n", i))) as Result<Bytes, reqwest::Error>
        }))
    };

    let mut results = Vec::new();

    // Test each configuration
    for config in configs {
        let mut processor = StreamProcessor::new(config.clone());
        let test_stream = create_test_stream();
        let start_time = Instant::now();

        let mut result_stream = processor.process_stream(test_stream, streaming_parser).await;
        let mut response_count = 0;
        let mut total_tokens = 0;

        while let Some(result) = result_stream.next().await {
            let response = result.unwrap();
            response_count += 1;
            total_tokens += response.chunk_tokens;
        }

        let duration = start_time.elapsed();
        results.push((
            config.enable_batching,
            config.max_batch_size,
            duration,
            response_count,
            total_tokens,
        ));
    }

    // Print performance results
    for (batching, batch_size, duration, responses, tokens) in results {
        println!(
            "Config: batching={}, batch_size={}, duration={:?}, responses={}, tokens={}, tokens/sec={}",
            batching,
            batch_size,
            duration,
            responses,
            tokens,
            tokens as f64 / duration.as_secs_f64(),
        );
    }
} 