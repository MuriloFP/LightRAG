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

use crate::llm::{
    LLMClient, LLMConfig, LLMError, LLMParams, LLMResponse,
    cache::{ResponseCache, InMemoryCache},
};
use crate::types::llm::{QueryParams, StreamingResponse, StreamingTiming};

/// Anthropic API response format
#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    completion: String,
    stop_reason: Option<String>,
    model: String,
    #[serde(default)]
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize, Default)]
struct AnthropicUsage {
    input_tokens: Option<usize>,
    output_tokens: Option<usize>,
}

/// Anthropic streaming response format
#[derive(Debug, Deserialize)]
struct AnthropicStreamResponse {
    completion: String,
    stop_reason: Option<String>,
    model: Option<String>,
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

    /// Build request payload
    fn build_request(&self, prompt: &str, params: &LLMParams) -> serde_json::Value {
        let mut request = json!({
            "prompt": format!("\n\nHuman: {}\n\nAssistant:", prompt),
            "model": params.model.split('/').nth(1).unwrap_or("claude-2"),
            "max_tokens_to_sample": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": params.stream,
        });

        if let Some(system) = &params.system_prompt {
            request["system"] = json!(system);
        }

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

    async fn generate_stream(
        &self,
        prompt: &str,
        params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
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

        let (tx, rx) = mpsc::channel(32);
        let mut stream = response.bytes_stream();
        let start_time = Instant::now();

        tokio::spawn(async move {
            let mut buffer = String::new();
            let mut total_tokens = 0;

            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        let chunk_str = String::from_utf8_lossy(&chunk);
                        buffer.push_str(&chunk_str);

                        // Process complete messages
                        while let Some(pos) = buffer.find('\n') {
                            let line = buffer[..pos].trim().to_string();
                            buffer = buffer[pos + 1..].to_string();

                            if line.is_empty() || line == "data: [DONE]" {
                                continue;
                            }

                            if let Some(data) = line.strip_prefix("data: ") {
                                match serde_json::from_str::<AnthropicStreamResponse>(data) {
                                    Ok(response) => {
                                        let is_done = response.stop_reason.is_some();
                                        let chunk_tokens = response.completion.split_whitespace().count();
                                        total_tokens += chunk_tokens;

                                        let timing = StreamingTiming {
                                            chunk_duration: start_time.elapsed().as_micros() as u64,
                                            total_duration: start_time.elapsed().as_micros() as u64,
                                            prompt_eval_duration: Some(Duration::from_secs(0).as_micros() as u64),
                                            token_gen_duration: Some(Duration::from_secs(0).as_micros() as u64),
                                        };

                                        let stream_response = StreamingResponse {
                                            text: response.completion,
                                            chunk_tokens,
                                            total_tokens,
                                            done: is_done,
                                            timing: Some(timing),
                                            metadata: HashMap::new(),
                                        };

                                        if tx.send(Ok(stream_response)).await.is_err() {
                                            break;
                                        }

                                        if is_done {
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