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

use crate::llm::{
    LLMClient, LLMConfig, LLMError, LLMParams, LLMResponse,
    cache::{ResponseCache, InMemoryCache},
};
use crate::types::llm::{QueryParams, StreamingResponse, StreamingTiming};
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

/// Anthropic client implementation
pub struct AnthropicClient {
    /// HTTP client
    client: Client,
    
    /// Client configuration
    config: LLMConfig,
    
    /// Response cache
    cache: Option<InMemoryCache>,
}

impl AnthropicClient {
    /// Create a new Anthropic client
    pub fn new(config: LLMConfig) -> Result<Self, LLMError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| LLMError::ConfigError(e.to_string()))?;
            
        let cache = if config.use_cache {
            Some(InMemoryCache::default())
        } else {
            None
        };
        
        Ok(Self {
            client,
            config,
            cache,
        })
    }
    
    /// Build the API request URL
    fn build_url(&self) -> Result<String, LLMError> {
        let endpoint = self.config.api_endpoint.as_ref()
            .map(|s| s.as_str())
            .unwrap_or("https://api.anthropic.com");
        Ok(format!("{}/v1/complete", endpoint))
    }
    
    /// Build request headers
    fn build_headers(&self) -> Result<reqwest::header::HeaderMap, LLMError> {
        use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
        
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| LLMError::ConfigError("API key not configured".to_string()))?;
            
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(api_key)
                .map_err(|e| LLMError::ConfigError(e.to_string()))?
        );
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json")
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static("2023-06-01")
        );
        
        Ok(headers)
    }

    /// Build conversation context from history
    fn build_conversation(&self, prompt: &str, history: Option<&[ConversationTurn]>, params: &LLMParams) -> String {
        let mut conversation = String::new();
        
        // Add system prompt if provided
        if let Some(system) = &params.system_prompt {
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

    /// Build request payload with conversation history
    fn build_request(&self, prompt: &str, params: &LLMParams) -> serde_json::Value {
        let conversation = self.build_conversation(prompt, None, params);
        
        let mut request = json!({
            "prompt": conversation,
            "model": params.model.split('/').nth(1).unwrap_or("claude-2"),
            "max_tokens_to_sample": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": params.stream,
        });

        // Add any extra parameters
        for (key, value) in &params.extra_params {
            request[key] = json!(value);
        }

        request
    }

    /// Build request payload with explicit history
    fn build_request_with_history(&self, prompt: &str, history: &[ConversationTurn], params: &LLMParams) -> serde_json::Value {
        let conversation = self.build_conversation(prompt, Some(history), params);
        
        let mut request = json!({
            "prompt": conversation,
            "model": params.model.split('/').nth(1).unwrap_or("claude-2"),
            "max_tokens_to_sample": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": params.stream,
        });

        // Add any extra parameters
        for (key, value) in &params.extra_params {
            request[key] = json!(value);
        }

        request
    }

    async fn update_cache(&self, prompt: &str, response: &str) {
        if let Some(cache) = &self.cache {
            let _ = cache.put(prompt, LLMResponse {
                text: response.to_string(),
                tokens_used: 0,
                model: self.config.model.clone(),
                cached: true,
                context: None,
                metadata: Default::default(),
            }).await;
        }
    }

    /// Create embeddings for a list of texts
    pub async fn create_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ConfigError("Anthropic does not support embeddings yet".to_string()))
    }
}

#[async_trait]
impl LLMClient for AnthropicClient {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        Ok(())
    }

    async fn generate(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        // Check cache first if enabled
        if let Some(cache) = &self.cache {
            if let Some(cached) = cache.get(prompt).await {
                return Ok(cached);
            }
        }

        // Estimate tokens and check rate limits
        let estimated_tokens = prompt.split_whitespace().count() as u32 + params.max_tokens as u32;
        self.check_rate_limit(estimated_tokens).await?;

        let url = self.build_url()?;
        let headers = self.build_headers()?;
        let request = self.build_request(prompt, params);

        let response = self.client.post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        let status = response.status();
        let response_text = response.text().await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !status.is_success() {
            return Err(LLMError::RequestFailed(format!(
                "Anthropic API error ({}): {}", status, response_text
            )));
        }

        let anthropic_response: AnthropicResponse = serde_json::from_str(&response_text)
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        let total_tokens = anthropic_response.usage.input_tokens.unwrap_or(0) +
                          anthropic_response.usage.output_tokens.unwrap_or(0);

        let response = LLMResponse {
            text: anthropic_response.completion,
            tokens_used: total_tokens,
            model: anthropic_response.model,
            cached: false,
            context: None,
            metadata: HashMap::new(),
        };

        // Update cache if enabled
        if let Some(cache) = &self.cache {
            let _ = cache.put(prompt, response.clone()).await;
        }

        Ok(response)
    }

    async fn generate_with_history(
        &self,
        prompt: &str,
        history: &[ConversationTurn],
        params: &LLMParams
    ) -> Result<LLMResponse, LLMError> {
        // Check cache first if enabled
        if let Some(cache) = &self.cache {
            // Create a cache key that includes history
            let cache_key = format!("{:?}:{}", history, prompt);
            if let Some(cached) = cache.get(&cache_key).await {
                return Ok(cached);
            }
        }

        // Estimate tokens and check rate limits
        let estimated_tokens = prompt.split_whitespace().count() as u32 + 
            history.iter().map(|turn| turn.content.split_whitespace().count() as u32).sum::<u32>() +
            params.max_tokens as u32;
        
        self.check_rate_limit(estimated_tokens).await?;

        let url = self.build_url()?;
        let headers = self.build_headers()?;
        let request = self.build_request_with_history(prompt, history, params);

        let response = self.client.post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        let status = response.status();
        let response_text = response.text().await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !status.is_success() {
            return Err(LLMError::RequestFailed(format!(
                "Anthropic API error ({}): {}", status, response_text
            )));
        }

        let anthropic_response: AnthropicResponse = serde_json::from_str(&response_text)
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        let total_tokens = anthropic_response.usage.input_tokens.unwrap_or(0) +
                          anthropic_response.usage.output_tokens.unwrap_or(0);

        let response = LLMResponse {
            text: anthropic_response.completion,
            tokens_used: total_tokens,
            model: anthropic_response.model,
            cached: false,
            context: None,
            metadata: {
                let mut map = HashMap::new();
                if let Some(input_tokens) = anthropic_response.usage.input_tokens {
                    map.insert("input_tokens".to_string(), input_tokens.to_string());
                }
                if let Some(output_tokens) = anthropic_response.usage.output_tokens {
                    map.insert("output_tokens".to_string(), output_tokens.to_string());
                }
                map
            },
        };

        // Update cache if enabled
        if let Some(cache) = &self.cache {
            let cache_key = format!("{:?}:{}", history, prompt);
            let _ = cache.put(&cache_key, response.clone()).await;
        }

        Ok(response)
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        // Estimate tokens and check rate limits
        let estimated_tokens = prompt.split_whitespace().count() as u32 + params.max_tokens as u32;
        self.check_rate_limit(estimated_tokens).await?;

        let url = self.build_url()?;
        let headers = self.build_headers()?;
        let mut request = self.build_request(prompt, params);
        request["stream"] = json!(true);

        let response = self.client.post(&url)
            .headers(headers)
            .json(&request)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let error = response.text().await
                .map_err(|e| LLMError::RequestFailed(e.to_string()))?;
            return Err(LLMError::RequestFailed(error));
        }

        let stream = response.bytes_stream();
        let mut processor = StreamProcessor::new(StreamConfig {
            enable_batching: true,
            max_batch_size: 8192,
            max_batch_wait_ms: 50,
            ..Default::default()
        });

        // Create async parser for Anthropic stream format
        let parser = |text: &str| -> futures::future::BoxFuture<'static, Result<Option<(String, bool, std::collections::HashMap<String, String>)>, LLMError>> {
            let owned_text = text.to_owned();
            Box::pin(async move {
                match serde_json::from_str::<AnthropicStreamResponse>(&owned_text) {
                    Ok(response) => {
                        let is_done = response.stop_reason.is_some();
                        let mut metadata = std::collections::HashMap::new();
                        if let Some(model) = response.model {
                            metadata.insert("model".to_string(), model.to_owned());
                        }
                        Ok(Some((response.completion.to_owned(), is_done, metadata)))
                    }
                    Err(e) => Err(LLMError::InvalidResponse(e.to_string()))
                }
            })
        };

        Ok(processor.process_stream(stream, parser).await)
    }

    async fn batch_generate(
        &self,
        prompts: &[String],
        params: &LLMParams,
    ) -> Result<Vec<LLMResponse>, LLMError> {
        let mut responses = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            responses.push(self.generate(prompt, params).await?);
        }
        Ok(responses)
    }

    fn get_config(&self) -> &LLMConfig {
        &self.config
    }

    fn update_config(&mut self, config: LLMConfig) -> Result<(), LLMError> {
        self.config = config;
        Ok(())
    }
} 