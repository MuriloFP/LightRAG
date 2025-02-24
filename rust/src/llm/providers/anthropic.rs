use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::{Duration, Instant};
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use futures::future::BoxFuture;
use tokio::sync::RwLock;

use crate::llm::{
    LLMError, LLMParams, LLMResponse,
    Provider,
    ProviderConfig,
    rate_limiter::{RateLimiter, RateLimitConfig},
};
use crate::types::llm::{StreamingResponse, StreamingTiming};
use crate::processing::keywords::ConversationTurn;
use crate::llm::streaming::{StreamProcessor, StreamConfig};

/// Anthropic API response format
#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    pub completion: String,
    pub stop_reason: Option<String>,
    pub model: String,
    #[serde(default)]
    pub usage: AnthropicUsage,
}

#[derive(Debug, Deserialize, Default)]
pub struct AnthropicUsage {
    pub input_tokens: Option<usize>,
    pub output_tokens: Option<usize>,
}

/// Anthropic streaming response format
#[derive(Debug, Deserialize)]
pub struct AnthropicStreamResponse {
    pub completion: String,
    pub stop_reason: Option<String>,
    pub model: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

/// Configuration for Anthropic API calls
#[derive(Debug, Clone)]
pub struct AnthropicConfig {
    pub api_key: String,
    pub api_base: String,
    pub model: String,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub timeout_secs: u64,
}

impl Default for AnthropicConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_base: "https://api.anthropic.com/v1".to_string(),
            model: "claude-2".to_string(),
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 1.0,
            timeout_secs: 30,
        }
    }
}

/// Build conversation context from history
fn build_conversation(prompt: &str, history: Option<&[ConversationTurn]>, system_prompt: Option<&str>) -> String {
    let mut conversation = String::new();
    
    // Add system prompt if provided
    if let Some(system) = system_prompt {
        conversation.push_str(&format!("\n\nSystem: {}", system));
    }
    
    // Add conversation history
    if let Some(history) = history {
        for turn in history {
            match turn.role.as_str() {
                "user" => conversation.push_str(&format!("\n\nHuman: {}", turn.content)),
                "assistant" => conversation.push_str(&format!("\n\nAssistant: {}", turn.content)),
                "system" => conversation.push_str(&format!("\n\nSystem: {}", turn.content)),
                _ => conversation.push_str(&format!("\n\n{}: {}", turn.role, turn.content)),
            }
        }
    }
    
    // Add current prompt
    conversation.push_str(&format!("\n\nHuman: {}\n\nAssistant:", prompt));
    
    conversation
}

/// Complete text using Anthropic API
pub async fn anthropic_complete(
    prompt: &str,
    params: &LLMParams,
    api_key: &str,
    api_endpoint: Option<&str>,
) -> Result<LLMResponse, LLMError> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

    let url = format!("{}/v1/complete", 
        api_endpoint.unwrap_or("https://api.anthropic.com"));

    let mut request_body = json!({
        "model": params.model,
        "prompt": format!("\n\nHuman: {}\n\nAssistant:", prompt),
        "max_tokens_to_sample": params.max_tokens,
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
        request_body[key] = serde_json::Value::String(value.clone());
    }

    let response = client.post(&url)
        .header("X-API-Key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

    if !response.status().is_success() {
        let error = response.text().await.unwrap_or_default();
        return Err(LLMError::RequestFailed(error));
    }

    let anthropic_response: AnthropicResponse = response.json()
        .await
        .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

    let llm_response = LLMResponse {
        text: anthropic_response.completion,
        tokens_used: anthropic_response.usage.input_tokens.unwrap_or(0) + anthropic_response.usage.output_tokens.unwrap_or(0),
        model: params.model.clone(),
        cached: false,
        context: None,
        metadata: HashMap::new(),
    };

    Ok(llm_response)
}

/// Complete text with streaming using Anthropic API
pub async fn anthropic_complete_stream(
    prompt: &str,
    params: &LLMParams,
    api_key: &str,
    api_endpoint: Option<&str>,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
    let client = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .map_err(|e| LLMError::ConfigError(e.to_string()))?;

    let url = format!("{}/v1/complete", 
        api_endpoint.unwrap_or("https://api.anthropic.com"));

    let mut request_body = json!({
        "model": params.model,
        "prompt": format!("\n\nHuman: {}\n\nAssistant:", prompt),
        "max_tokens_to_sample": params.max_tokens,
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
        request_body[key] = serde_json::Value::String(value.clone());
    }

    let response = client.post(&url)
        .header("X-API-Key", api_key)
        .header("anthropic-version", "2023-06-01")
        .json(&request_body)
        .send()
        .await
        .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

    if !response.status().is_success() {
        let error = response.text().await.unwrap_or_default();
        return Err(LLMError::RequestFailed(error));
    }

    let (tx, rx) = mpsc::channel(100);
    let mut stream = response.bytes_stream();

    tokio::spawn(async move {
        let mut total_tokens = 0;
        let start_time = Instant::now();

        while let Some(chunk_result) = stream.next().await {
            match chunk_result {
                Ok(chunk) => {
                    let chunk_str = String::from_utf8_lossy(&chunk);
                    for line in chunk_str.lines() {
                        if line.starts_with("data: ") {
                            let data = &line["data: ".len()..];
                            if data == "[DONE]" {
                                break;
                            }

                            match serde_json::from_str::<AnthropicStreamResponse>(data) {
                                Ok(stream_response) => {
                                    total_tokens += 1;
                                    let timing = StreamingTiming {
                                        chunk_duration: start_time.elapsed().as_micros() as u64,
                                        total_duration: start_time.elapsed().as_micros() as u64,
                                        prompt_eval_duration: None,
                                        token_gen_duration: None,
                                    };

                                    let _ = tx.send(Ok(StreamingResponse {
                                        text: stream_response.completion,
                                        done: stream_response.stop_reason.is_some(),
                                        timing: Some(timing),
                                        chunk_tokens: 1,
                                        total_tokens,
                                        metadata: HashMap::new(),
                                    })).await;
                                }
                                Err(e) => {
                                    let _ = tx.send(Err(LLMError::InvalidResponse(e.to_string()))).await;
                                }
                            }
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

/// Complete text completions in batch using Anthropic API
pub async fn anthropic_complete_batch(
    prompts: &[String],
    params: &LLMParams,
    api_key: &str,
    api_endpoint: Option<&str>,
) -> Result<Vec<LLMResponse>, LLMError> {
    let mut responses = Vec::with_capacity(prompts.len());
    
    for prompt in prompts {
        let response = anthropic_complete(prompt, params, api_key, api_endpoint).await?;
        responses.push(response);
    }
    
    Ok(responses)
}

/// Anthropic provider implementation
pub struct AnthropicProvider {
    config: ProviderConfig,
    client: Arc<Client>,
    rate_limiter: Arc<RwLock<Option<RateLimiter>>>,
}

impl AnthropicProvider {
    pub fn new(config: ProviderConfig) -> Result<Self, LLMError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        Ok(Self {
            config,
            client: Arc::new(client),
            rate_limiter: Arc::new(RwLock::new(None)),
        })
    }
}

#[async_trait::async_trait]
impl Provider for AnthropicProvider {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        Ok(())
    }

    async fn complete(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        // Check rate limit if configured
        if let Some(rate_limiter) = &*self.rate_limiter.read().await {
            // Estimate token count for rate limiting
            let estimated_tokens = (prompt.len() as f32 / 4.0).ceil() as u32 + params.max_tokens as u32;
            rate_limiter.acquire_permit(estimated_tokens).await?;
        }

        let url = format!("{}/v1/complete", 
            self.config.api_endpoint.as_deref().unwrap_or("https://api.anthropic.com"));

        let mut request_body = json!({
            "model": params.model,
            "prompt": format!("\n\nHuman: {}\n\nAssistant:", prompt),
            "max_tokens_to_sample": params.max_tokens,
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

        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| LLMError::ConfigError("API key not configured".to_string()))?;

        let response = self.client.post(&url)
            .header("X-API-Key", api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(LLMError::RequestFailed(error));
        }

        let anthropic_response: AnthropicResponse = response.json()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        Ok(LLMResponse {
            text: anthropic_response.completion,
            tokens_used: anthropic_response.usage.input_tokens.unwrap_or(0) + anthropic_response.usage.output_tokens.unwrap_or(0),
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
            // Estimate token count for rate limiting
            let estimated_tokens = (prompt.len() as f32 / 4.0).ceil() as u32 + params.max_tokens as u32;
            rate_limiter.acquire_permit(estimated_tokens).await?;
        }

        let url = format!("{}/v1/complete", 
            self.config.api_endpoint.as_deref().unwrap_or("https://api.anthropic.com"));

        let mut request_body = json!({
            "model": params.model,
            "prompt": format!("\n\nHuman: {}\n\nAssistant:", prompt),
            "max_tokens_to_sample": params.max_tokens,
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

        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| LLMError::ConfigError("API key not configured".to_string()))?;

        let response = self.client.post(&url)
            .header("X-API-Key", api_key)
            .header("anthropic-version", "2023-06-01")
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
                        let chunk_str = String::from_utf8_lossy(&chunk);
                        for line in chunk_str.lines() {
                            if line.starts_with("data: ") {
                                let data = &line["data: ".len()..];
                                if data == "[DONE]" {
                                    break;
                                }

                                match serde_json::from_str::<AnthropicStreamResponse>(data) {
                                    Ok(stream_response) => {
                                        total_tokens += 1;
                                        let timing = StreamingTiming {
                                            chunk_duration: start_time.elapsed().as_micros() as u64,
                                            total_duration: start_time.elapsed().as_micros() as u64,
                                            prompt_eval_duration: None,
                                            token_gen_duration: None,
                                        };

                                        let stream_response = StreamingResponse {
                                            text: stream_response.completion,
                                            done: stream_response.stop_reason.is_some(),
                                            timing: Some(timing),
                                            chunk_tokens: 1,
                                            total_tokens,
                                            metadata: HashMap::new(),
                                        };

                                        if tx.send(Ok(stream_response)).await.is_err() {
                                            break;
                                        }
                                    }
                                    Err(e) => {
                                        let _ = tx.send(Err(LLMError::InvalidResponse(e.to_string()))).await;
                                        break;
                                    }
                                }
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