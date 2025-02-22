use std::collections::HashMap;
use std::time::Duration;
use futures::StreamExt;
use tokio;

use super_lightrag::llm::{
    LLMClient, LLMConfig, LLMParams,
    providers::{OpenAIClient, OllamaClient},
};
use super_lightrag::types::llm::StreamingTiming;

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