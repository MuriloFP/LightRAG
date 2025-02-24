use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::{Duration, Instant};
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use md5;
use futures::future::BoxFuture;
use tokio::sync::RwLock;

use crate::llm::{
    LLMError, LLMParams, LLMResponse,
    StreamingResponse,
    StreamingTiming,
    cache::{ResponseCache, MemoryCache},
    Provider,
    ProviderConfig,
    rate_limiter::{RateLimiter},
};
use crate::types::llm::{QueryParams};
use crate::processing::keywords::ConversationTurn;
use crate::processing::context::ContextBuilder;
use crate::llm::streaming::{StreamProcessor, StreamConfig};

/// Ollama API response format
#[derive(Debug, Deserialize)]
pub struct OllamaResponse {
    /// The generated text response
    pub response: String,
    
    /// Whether the generation is complete
    #[allow(dead_code)]  // Used by Ollama API but not by our implementation
    pub done: bool,
    
    /// Total duration of the request in microseconds
    pub total_duration: Option<u64>,
    
    /// Time taken to load the model in microseconds
    pub load_duration: Option<u64>,
    
    /// Number of tokens in the prompt
    pub prompt_eval_count: Option<usize>,
    
    /// Number of tokens in the response
    pub eval_count: Option<usize>,
    
    /// Time taken for token generation in microseconds
    pub eval_duration: Option<u64>,

    /// Error message if any
    pub error: Option<String>,
}

/// Ollama streaming response format
#[derive(Debug, Deserialize)]
pub struct OllamaStreamResponse {
    /// The generated text chunk
    pub response: String,
    
    /// Whether the generation is complete
    pub done: bool,

    /// Total duration of the request in microseconds
    pub total_duration: Option<u64>,
    
    /// Time taken to load the model in microseconds
    pub load_duration: Option<u64>,
    
    /// Number of tokens in the prompt
    pub prompt_eval_count: Option<usize>,
    
    /// Number of tokens in the response
    pub eval_count: Option<usize>,
    
    /// Time taken for token generation in microseconds
    pub eval_duration: Option<u64>,

    /// Error message if any
    pub error: Option<String>,
}

/// Configuration for Ollama API calls
#[derive(Debug, Clone)]
pub struct OllamaConfig {
    pub api_base: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub timeout_secs: u64,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self {
            api_base: "http://localhost:11434".to_string(),
            model: "llama2".to_string(),
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 1.0,
            timeout_secs: 30,
        }
    }
}

/// Map Ollama errors to LLMError
fn map_ollama_error(error: &str) -> LLMError {
    if error.contains("rate limit") || error.contains("too many requests") {
        LLMError::RateLimitExceeded(error.to_string())
    } else if error.contains("context length") || error.contains("token limit") {
        LLMError::TokenLimitExceeded(error.to_string())
    } else if error.contains("model") && (error.contains("not found") || error.contains("failed to load")) {
        LLMError::ConfigError(format!("Model error: {}", error))
    } else if error.contains("invalid") {
        LLMError::InvalidResponse(error.to_string())
    } else {
        LLMError::RequestFailed(error.to_string())
    }
}

/// Complete text using Ollama API
pub async fn ollama_complete(
    prompt: &str,
    params: &LLMParams,
    api_endpoint: Option<&str>,
) -> Result<LLMResponse, LLMError> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

    let url = format!("{}/api/generate", 
        api_endpoint.unwrap_or("http://localhost:11434"));

    let mut request_body = json!({
        "model": params.model.strip_prefix("ollama/").unwrap_or(&params.model),
        "prompt": prompt,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "stream": false,
    });

    // Add system prompt if provided
    if let Some(system) = &params.system_prompt {
        request_body["system"] = json!(system);
    }

    // Add extra parameters
    for (key, value) in &params.extra_params {
        request_body[key] = json!(value);
    }

    let response = client.post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

    if !response.status().is_success() {
        let error = response.text().await.unwrap_or_default();
        return Err(LLMError::RequestFailed(error));
    }

    let ollama_response: OllamaResponse = response.json()
        .await
        .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

    if let Some(error) = ollama_response.error {
        return Err(map_ollama_error(&error));
    }

    Ok(LLMResponse {
        text: ollama_response.response,
        tokens_used: ollama_response.eval_count.unwrap_or(0),
        model: params.model.clone(),
        cached: false,
        context: None,
        metadata: HashMap::new(),
    })
}

/// Complete text with streaming using Ollama API
pub async fn ollama_complete_stream(
    prompt: &str,
    params: &LLMParams,
    api_endpoint: Option<&str>,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

    let url = format!("{}/api/generate", 
        api_endpoint.unwrap_or("http://localhost:11434"));

    let mut request_body = json!({
        "model": params.model.strip_prefix("ollama/").unwrap_or(&params.model),
        "prompt": prompt,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "stream": true,
    });

    // Add system prompt if provided
    if let Some(system) = &params.system_prompt {
        request_body["system"] = json!(system);
    }

    // Add extra parameters
    for (key, value) in &params.extra_params {
        request_body[key] = json!(value);
    }

    let response = client.post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

    if !response.status().is_success() {
        let error = response.text().await.unwrap_or_default();
        return Err(LLMError::RequestFailed(error));
    }

    let (tx, rx) = mpsc::channel(32);
    let start_time = Instant::now();

    tokio::spawn(async move {
        let mut stream = response.bytes_stream();
        let mut total_tokens = 0;

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    match serde_json::from_slice::<OllamaStreamResponse>(&chunk) {
                        Ok(ollama_response) => {
                            if let Some(error) = ollama_response.error {
                                let _ = tx.send(Err(map_ollama_error(&error))).await;
                                break;
                            }

                            let timing = StreamingTiming {
                                chunk_duration: ollama_response.eval_duration.unwrap_or(0),
                                total_duration: ollama_response.total_duration.unwrap_or(0),
                                prompt_eval_duration: Some(ollama_response.prompt_eval_count.unwrap_or(0) as u64),
                                token_gen_duration: Some(ollama_response.eval_duration.unwrap_or(0)),
                            };

                            total_tokens += ollama_response.eval_count.unwrap_or(0);

                            let stream_response = StreamingResponse {
                                text: ollama_response.response,
                                done: ollama_response.done,
                                timing: Some(timing),
                                chunk_tokens: ollama_response.eval_count.unwrap_or(0),
                                total_tokens,
                                metadata: HashMap::new(),
                            };

                            if tx.send(Ok(stream_response)).await.is_err() {
                                break;
                            }

                            if ollama_response.done {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(LLMError::InvalidResponse(e.to_string()))).await;
                            break;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.send(Err(LLMError::RequestFailed(e.to_string()))).await;
                    break;
                }
            }
        }
    });

    Ok(Box::pin(ReceiverStream::new(rx)))
}

/// Complete multiple prompts in batch using Ollama API
pub async fn ollama_complete_batch(
    prompts: &[String],
    params: &LLMParams,
    api_endpoint: Option<&str>,
) -> Result<Vec<LLMResponse>, LLMError> {
    let mut responses = Vec::with_capacity(prompts.len());
    
    for prompt in prompts {
        let response = ollama_complete(prompt, params, api_endpoint).await?;
        responses.push(response);
    }
    
    Ok(responses)
}

/// Ollama provider implementation
pub struct OllamaProvider {
    config: ProviderConfig,
    client: Arc<Client>,
    rate_limiter: Arc<RwLock<Option<RateLimiter>>>,
}

impl OllamaProvider {
    pub fn new(config: ProviderConfig) -> Result<Self, LLMError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| LLMError::ConfigError(e.to_string()))?;

        Ok(Self {
            config,
            client: Arc::new(client),
            rate_limiter: Arc::new(RwLock::new(None)),
        })
    }

    // Helper function to estimate token count
    fn estimate_tokens(&self, text: &str) -> u32 {
        // Simple estimation: ~4 characters per token
        (text.len() as f32 / 4.0).ceil() as u32
    }
}

#[async_trait::async_trait]
impl Provider for OllamaProvider {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        Ok(())
    }

    async fn complete(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        // Check rate limit if configured
        if let Some(rate_limiter) = &*self.rate_limiter.read().await {
            let estimated_tokens = self.estimate_tokens(prompt) + params.max_tokens as u32;
            rate_limiter.acquire_permit(estimated_tokens).await?;
        }

        let url = format!("{}/api/generate", 
            self.config.api_endpoint.as_deref().unwrap_or("http://localhost:11434"));

        let mut request_body = json!({
            "model": params.model.strip_prefix("ollama/").unwrap_or(&params.model),
            "prompt": prompt,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": false,
        });

        // Add system prompt if provided
        if let Some(system) = &params.system_prompt {
            request_body["system"] = json!(system);
        }

        // Add extra parameters
        for (key, value) in &params.extra_params {
            request_body[key] = json!(value);
        }

        let response = self.client.post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed(error));
        }

        let ollama_response: OllamaResponse = response.json()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        if let Some(error) = ollama_response.error {
            return Err(map_ollama_error(&error));
        }

        Ok(LLMResponse {
            text: ollama_response.response,
            tokens_used: ollama_response.eval_count.unwrap_or(0),
            model: params.model.clone(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        })
    }

    async fn complete_stream(
        &self,
        prompt: &str,
        params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        // Check rate limit if configured
        if let Some(rate_limiter) = &*self.rate_limiter.read().await {
            let estimated_tokens = self.estimate_tokens(prompt) + params.max_tokens as u32;
            rate_limiter.acquire_permit(estimated_tokens).await?;
        }

        let url = format!("{}/api/generate", 
            self.config.api_endpoint.as_deref().unwrap_or("http://localhost:11434"));

        let mut request_body = json!({
            "model": params.model.strip_prefix("ollama/").unwrap_or(&params.model),
            "prompt": prompt,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": true,
        });

        // Add system prompt if provided
        if let Some(system) = &params.system_prompt {
            request_body["system"] = json!(system);
        }

        // Add extra parameters
        for (key, value) in &params.extra_params {
            request_body[key] = json!(value);
        }

        let response = self.client.post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed(error));
        }

        let (tx, rx) = mpsc::channel(32);
        let model = params.model.clone();
        let start_time = Instant::now();

        tokio::spawn(async move {
            let mut stream = response.bytes_stream();
            let mut total_tokens = 0;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        match serde_json::from_slice::<OllamaStreamResponse>(&chunk) {
                            Ok(ollama_response) => {
                                if let Some(error) = ollama_response.error {
                                    let _ = tx.send(Err(map_ollama_error(&error))).await;
                                    break;
                                }

                                let timing = StreamingTiming {
                                    chunk_duration: ollama_response.eval_duration.unwrap_or(0),
                                    total_duration: ollama_response.total_duration.unwrap_or(0),
                                    prompt_eval_duration: Some(ollama_response.prompt_eval_count.unwrap_or(0) as u64),
                                    token_gen_duration: Some(ollama_response.eval_duration.unwrap_or(0)),
                                };

                                total_tokens += ollama_response.eval_count.unwrap_or(0);

                                let stream_response = StreamingResponse {
                                    text: ollama_response.response,
                                    done: ollama_response.done,
                                    timing: Some(timing),
                                    chunk_tokens: ollama_response.eval_count.unwrap_or(0),
                                    total_tokens,
                                    metadata: HashMap::new(),
                                };

                                if tx.send(Ok(stream_response)).await.is_err() {
                                    break;
                                }

                                if ollama_response.done {
                                    break;
                                }
                            }
                            Err(e) => {
                                let _ = tx.send(Err(LLMError::InvalidResponse(e.to_string()))).await;
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(LLMError::RequestFailed(e.to_string()))).await;
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
    }

    async fn complete_batch(
        &self,
        prompts: &[String],
        params: &LLMParams,
    ) -> Result<Vec<LLMResponse>, LLMError> {
        let mut responses = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            responses.push(self.complete(prompt, params).await?);
        }
        Ok(responses)
    }

    fn get_config(&self) -> &ProviderConfig {
        &self.config
    }

    fn update_config(&mut self, config: ProviderConfig) -> Result<(), LLMError> {
        self.config = config;
        Ok(())
    }
} 