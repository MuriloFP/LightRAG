use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use async_trait::async_trait;

use super_lightrag::{
    types::llm::{
        LLMParams, LLMResponse, LLMError, StreamingResponse,
        StreamingTiming,
    },
    llm::{Provider, ProviderConfig},
};

/// Mock provider that returns streaming responses
struct MockStreamingProvider {
    responses: Vec<String>,
    current: AtomicUsize,
    chunk_size: usize,
}

impl MockStreamingProvider {
    fn new(responses: Vec<String>, chunk_size: usize) -> Self {
        Self {
            responses,
            current: AtomicUsize::new(0),
            chunk_size,
        }
    }
}

#[async_trait]
impl Provider for MockStreamingProvider {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        Ok(())
    }

    async fn complete(&self, _prompt: &str, _params: &LLMParams) -> Result<LLMResponse, LLMError> {
        let idx = self.current.fetch_add(1, Ordering::SeqCst);
        if idx < self.responses.len() {
            Ok(LLMResponse {
                text: self.responses[idx].clone(),
                tokens_used: 100,
                model: "mock".to_string(),
                cached: false,
                context: None,
                metadata: HashMap::new(),
            })
        } else {
            Err(LLMError::RequestFailed("No more mock responses".to_string()))
        }
    }

    async fn complete_stream(
        &self,
        _prompt: &str,
        _params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        let idx = self.current.fetch_add(1, Ordering::SeqCst);
        if idx >= self.responses.len() {
            return Err(LLMError::RequestFailed("No more mock responses".to_string()));
        }

        let response = self.responses[idx].clone();
        let chunk_size = self.chunk_size;
        let (tx, rx) = mpsc::channel(32);

        tokio::spawn(async move {
            let mut chunks = Vec::new();
            let mut current_chunk = String::new();
            let mut total_tokens = 0;

            // Split response into chunks
            for c in response.chars() {
                current_chunk.push(c);
                if current_chunk.len() >= chunk_size {
                    chunks.push(current_chunk.clone());
                    current_chunk.clear();
                }
            }
            if !current_chunk.is_empty() {
                chunks.push(current_chunk);
            }

            // Send chunks as streaming responses
            for (i, chunk) in chunks.iter().enumerate() {
                let is_last = i == chunks.len() - 1;
                total_tokens += chunk.split_whitespace().count();

                let timing = StreamingTiming {
                    chunk_duration: 100u64,  // Mock duration
                    total_duration: ((i + 1) * 100) as u64,
                    prompt_eval_duration: Some(50u64),
                    token_gen_duration: Some(50u64),
                };

                let stream_response = StreamingResponse {
                    text: chunk.clone(),
                    done: is_last,
                    timing: Some(timing),
                    chunk_tokens: chunk.split_whitespace().count(),
                    total_tokens,
                    metadata: HashMap::new(),
                };

                if tx.send(Ok(stream_response)).await.is_err() {
                    break;
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn complete_batch(
        &self,
        _prompts: &[String],
        _params: &LLMParams,
    ) -> Result<Vec<LLMResponse>, LLMError> {
        Err(LLMError::RequestFailed("Batch completion not needed for tests".to_string()))
    }

    fn get_config(&self) -> &ProviderConfig {
        panic!("Config not needed for tests")
    }

    fn update_config(&mut self, _config: ProviderConfig) -> Result<(), LLMError> {
        Err(LLMError::RequestFailed("Config update not needed for tests".to_string()))
    }
}

#[tokio::test]
async fn test_basic_streaming() {
    let provider = MockStreamingProvider::new(
        vec!["Hello, this is a test response.".to_string()],
        5,  // 5 characters per chunk
    );

    let params = LLMParams::default();
    let mut stream = provider.complete_stream("test prompt", &params).await.unwrap();

    let mut combined_text = String::new();
    let mut total_chunks = 0;
    let mut last_total_tokens = 0;

    while let Some(result) = stream.next().await {
        let response = result.unwrap();
        combined_text.push_str(&response.text);
        total_chunks += 1;
        last_total_tokens = response.total_tokens;

        // Verify timing information
        let timing = response.timing.unwrap();
        assert!(timing.chunk_duration > 0);
        assert!(timing.total_duration >= timing.chunk_duration);
        assert_eq!(timing.prompt_eval_duration, Some(50));
        assert_eq!(timing.token_gen_duration, Some(50));
    }

    assert_eq!(combined_text, "Hello, this is a test response.");
    assert!(total_chunks > 1);
    assert!(last_total_tokens > 0);
}

#[tokio::test]
async fn test_empty_stream() {
    let provider = MockStreamingProvider::new(vec![], 5);
    let params = LLMParams::default();
    
    let result = provider.complete_stream("test prompt", &params).await;
    assert!(result.is_err());
    if let Err(LLMError::RequestFailed(msg)) = result {
        assert_eq!(msg, "No more mock responses");
    } else {
        panic!("Expected RequestFailed error");
    }
}

#[tokio::test]
async fn test_streaming_with_different_chunk_sizes() {
    let test_response = "This is a longer response that will be split into different sized chunks.";
    
    // Test with different chunk sizes
    for chunk_size in [3, 5, 10] {
        let provider = MockStreamingProvider::new(vec![test_response.to_string()], chunk_size);
        let params = LLMParams::default();
        let mut stream = provider.complete_stream("test prompt", &params).await.unwrap();

        let mut combined_text = String::new();
        let mut last_chunk_size = 0;

        while let Some(result) = stream.next().await {
            let response = result.unwrap();
            combined_text.push_str(&response.text);
            last_chunk_size = response.text.len();
        }

        assert_eq!(combined_text, test_response);
        assert!(last_chunk_size <= chunk_size);
    }
}

#[tokio::test]
async fn test_streaming_metadata() {
    let provider = MockStreamingProvider::new(
        vec!["Test response with metadata.".to_string()],
        5,
    );

    let params = LLMParams::default();
    let mut stream = provider.complete_stream("test prompt", &params).await.unwrap();

    let mut saw_final_chunk = false;
    let mut total_tokens = 0;

    while let Some(result) = stream.next().await {
        let response = result.unwrap();
        
        // Verify response structure
        assert!(!response.text.is_empty());
        assert!(response.chunk_tokens > 0);
        assert!(response.total_tokens >= response.chunk_tokens);
        
        // Track if we've seen the final chunk
        if response.done {
            saw_final_chunk = true;
            total_tokens = response.total_tokens;
        }
    }

    assert!(saw_final_chunk, "Should have received a final chunk");
    assert!(total_tokens > 0, "Should have counted total tokens");
} 