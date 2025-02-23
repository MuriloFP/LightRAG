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
use crate::processing::context::ContextBuilder;
use crate::llm::streaming::{StreamProcessor, StreamConfig};

/// OpenAI API response format
#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    text: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    total_tokens: usize,
}

/// OpenAI streaming response format
#[derive(Debug, Deserialize)]
struct OpenAIStreamResponse {
    choices: Vec<OpenAIStreamChoice>,
    usage: Option<OpenAIUsage>,
    model: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    text: String,
    finish_reason: Option<String>,
    index: Option<usize>,
}

/// OpenAI client implementation
pub struct OpenAIClient {
    /// HTTP client with connection pooling
    client: Client,
    
    /// Client configuration
    config: LLMConfig,
    
    /// Response cache
    cache: Option<InMemoryCache>,

    /// Active request counter for connection pooling
    active_requests: Arc<AtomicUsize>,
}

impl OpenAIClient {
    /// Create a new OpenAI client with connection pooling
    pub fn new(config: LLMConfig) -> Result<Self, LLMError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .pool_idle_timeout(Duration::from_secs(30))
            .pool_max_idle_per_host(10)
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
            active_requests: Arc::new(AtomicUsize::new(0)),
        })
    }
    
    /// Build the API request URL
    fn build_url(&self) -> Result<String, LLMError> {
        let endpoint = self.config.api_endpoint.as_ref()
            .ok_or_else(|| LLMError::ConfigError("API endpoint not configured".to_string()))?;
            
        Ok(format!("{}/v1/chat/completions", endpoint))
    }
    
    /// Build request headers
    fn build_headers(&self) -> Result<reqwest::header::HeaderMap, LLMError> {
        use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
        
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| LLMError::ConfigError("API key not configured".to_string()))?;
            
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| LLMError::ConfigError(e.to_string()))?
        );
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json")
        );
        
        if let Some(org_id) = &self.config.org_id {
            headers.insert(
                "OpenAI-Organization",
                HeaderValue::from_str(org_id)
                    .map_err(|e| LLMError::ConfigError(e.to_string()))?
            );
        }
        
        Ok(headers)
    }

    /// Build messages for chat completion
    fn build_messages(&self, prompt: &str, history: Option<&[ConversationTurn]>, params: &LLMParams) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();
        
        // Add system message if provided
        if let Some(system_prompt) = &params.system_prompt {
            messages.push(json!({
                "role": "system",
                "content": system_prompt
            }));
        }
        
        // Add conversation history
        if let Some(history) = history {
            for turn in history {
                messages.push(json!({
                    "role": turn.role,
                    "content": turn.content
                }));
            }
        }
        
        // Add current prompt
        messages.push(json!({
            "role": "user",
            "content": prompt
        }));
        
        messages
    }

    /// Generate with context building
    pub async fn generate_with_context_building(
        &self,
        query: &str,
        context_builder: &ContextBuilder,
        params: &LLMParams,
    ) -> Result<LLMResponse, LLMError> {
        // Check cache first with similarity search if enabled
        if let Some(cache) = &self.cache {
            if let Some(query_params) = &params.query_params {
                if let Some(embedding) = self.get_embedding(query).await? {
                    if let Some(cached_response) = cache.find_similar(
                        embedding,
                        self.config.similarity_threshold,
                    ).await {
                        return Ok(cached_response);
                    }
                }
            }
        }

        // Build context if query params are provided
        let prompt = if let Some(query_params) = &params.query_params {
            context_builder.build_context(query, query_params).await?
        } else {
            query.to_string()
        };

        // Generate response
        let response = self.generate(&prompt, params).await?;

        // Cache response with embedding if enabled
        if let Some(cache) = &self.cache {
            if let Some(embedding) = self.get_embedding(query).await? {
                let _ = cache.put_with_embedding(query, response.clone(), embedding).await;
            }
        }

        Ok(response)
    }

    /// Get embedding for a text
    async fn get_embedding(&self, text: &str) -> Result<Option<Vec<f32>>, LLMError> {
        let url = format!(
            "{}/v1/embeddings",
            self.config.api_endpoint.as_ref()
                .ok_or_else(|| LLMError::ConfigError("API endpoint not configured".to_string()))?
        );

        let response = self.client.post(&url)
            .headers(self.build_headers()?)
            .json(&json!({
                "model": "text-embedding-ada-002",
                "input": text
            }))
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if response.status().is_success() {
            let data: serde_json::Value = response.json().await
                .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

            if let Some(embedding) = data["data"][0]["embedding"].as_array() {
                let embedding: Vec<f32> = embedding.iter()
                    .filter_map(|v| v.as_f64().map(|x| x as f32))
                    .collect();
                Ok(Some(embedding))
            } else {
                Ok(None)
            }
        } else {
            Err(LLMError::RequestFailed(format!(
                "Embedding request failed: {}",
                response.text().await.unwrap_or_default()
            )))
        }
    }

    async fn check_cache(&self, prompt: &str, params: &QueryParams) -> Option<String> {
        if let Some(cache) = &self.cache {
            cache.get(prompt).await.map(|response| response.text)
        } else {
            None
        }
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

    async fn check_rate_limit(&self, estimated_tokens: u32) -> Result<(), LLMError> {
        if let Some(rate_limit_config) = &self.config.rate_limit_config {
            <Self as LLMClient>::check_rate_limit(self, estimated_tokens).await?;
        }
        Ok(())
    }

    /// Create embeddings for a list of texts
    pub async fn create_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LLMError> {
        let url = format!(
            "{}/v1/embeddings",
            self.config.api_endpoint.as_ref()
                .ok_or_else(|| LLMError::ConfigError("API endpoint not configured".to_string()))?
        );

        // Process in batches to avoid rate limits
        let batch_size = 20; // OpenAI recommends max 20 inputs per batch
        let mut all_embeddings = Vec::with_capacity(texts.len());
        
        for chunk in texts.chunks(batch_size) {
            let mut retries = 0;
            let mut last_error = None;

            while retries < self.config.max_retries {
                match self.client.post(&url)
                    .headers(self.build_headers()?)
                    .json(&json!({
                        "model": "text-embedding-3-large",
                        "input": chunk,
                        "encoding_format": "float",
                    }))
                    .send()
                    .await {
                        Ok(response) => {
                            if response.status().is_success() {
                                let data: serde_json::Value = response.json().await
                                    .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

                                let embeddings = data["data"].as_array()
                                    .ok_or_else(|| LLMError::InvalidResponse("No embeddings in response".to_string()))?
                                    .iter()
                                    .map(|item| {
                                        item["embedding"].as_array()
                                            .ok_or_else(|| LLMError::InvalidResponse("Invalid embedding format".to_string()))
                                            .map(|arr| {
                                                arr.iter()
                                                    .filter_map(|v| v.as_f64().map(|x| x as f32))
                                                    .collect::<Vec<f32>>()
                                            })
                                    })
                                    .collect::<Result<Vec<Vec<f32>>, LLMError>>()?;

                                all_embeddings.extend(embeddings);
                                break;
                            } else {
                                let error = response.text().await.unwrap_or_default();
                                if error.contains("rate_limit") {
                                    return Err(LLMError::RateLimitExceeded(error));
                                }
                                last_error = Some(LLMError::RequestFailed(error));
                            }
                        }
                        Err(e) => {
                            last_error = Some(LLMError::RequestFailed(e.to_string()));
                        }
                    }

                retries += 1;
                if retries < self.config.max_retries {
                    tokio::time::sleep(Duration::from_secs(2u64.pow(retries as u32))).await;
                }
            }

            if let Some(error) = last_error {
                return Err(error);
            }

            // Add a small delay between batches to respect rate limits
            if chunk.len() == batch_size {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }

        Ok(all_embeddings)
    }

    /// Process a batch of prompts with optimized connection pooling
    pub async fn batch_generate_chat(
        &self,
        prompts: &[String],
        params: &LLMParams,
    ) -> Result<Vec<LLMResponse>, LLMError> {
        let batch_size = 5; // Process 5 requests concurrently
        let mut responses = Vec::with_capacity(prompts.len());
        
        for chunk in prompts.chunks(batch_size) {
            let mut tasks = Vec::new();
            
            for prompt in chunk {
                let client = self.client.clone();
                let headers = self.build_headers()?;
                let url = self.build_url()?;
                let messages = self.build_messages(prompt, None, params);
                let active_requests = Arc::clone(&self.active_requests);
                let owned_params = params.clone();
                
                let task = tokio::spawn(async move {
                    // Increment active requests counter
                    active_requests.fetch_add(1, Ordering::SeqCst);
                    
                    let result = client.post(&url)
                        .headers(headers)
                        .json(&json!({
                            "model": owned_params.model.strip_prefix("openai/").unwrap_or(&owned_params.model),
                            "messages": messages,
                            "max_tokens": owned_params.max_tokens,
                            "temperature": owned_params.temperature,
                            "top_p": owned_params.top_p,
                            "stream": false,
                        }))
                        .send()
                        .await;
                        
                    // Decrement active requests counter
                    active_requests.fetch_sub(1, Ordering::SeqCst);
                    
                    result
                });
                
                tasks.push(task);
            }
            
            let chunk_responses = futures::future::join_all(tasks).await;
            
            for response in chunk_responses {
                match response {
                    Ok(Ok(res)) => {
                        if res.status().is_success() {
                            let chat_response: OpenAIChatResponse = res.json().await
                                .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;
                                
                            responses.push(LLMResponse {
                                text: chat_response.choices[0].message.content.clone(),
                                tokens_used: chat_response.usage.total_tokens,
                                model: chat_response.model,
                                cached: false,
                                context: None,
                                metadata: Default::default(),
                            });
                        } else {
                            let error = res.text().await.unwrap_or_default();
                            if error.contains("rate_limit") {
                                return Err(LLMError::RateLimitExceeded(error));
                            }
                            return Err(LLMError::RequestFailed(error));
                        }
                    }
                    Ok(Err(e)) => return Err(LLMError::RequestFailed(e.to_string())),
                    Err(e) => return Err(LLMError::RequestFailed(e.to_string())),
                }
            }
            
            // Add a small delay between batches to respect rate limits
            if chunk.len() == batch_size {
                tokio::time::sleep(Duration::from_millis(200)).await;
            }
        }
        
        Ok(responses)
    }
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChatResponse {
    pub choices: Vec<OpenAIChatChoice>,
    pub usage: OpenAIUsage,
    pub model: String,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChatChoice {
    pub message: OpenAIChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChatStreamResponse {
    pub choices: Vec<OpenAIChatStreamChoice>,
    pub usage: Option<OpenAIUsage>,
    pub model: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChatStreamChoice {
    pub delta: OpenAIChatStreamDelta,
    pub finish_reason: Option<String>,
    pub index: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct OpenAIChatStreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
}

#[async_trait]
impl LLMClient for OpenAIClient {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        Ok(())
    }

    async fn generate(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        // Check cache first
        if let Some(cache) = &self.cache {
            if let Some(cached_response) = cache.get(prompt).await {
                return Ok(cached_response);
            }
        }

        // Estimate tokens (rough estimate based on words)
        let estimated_tokens = prompt.split_whitespace().count() as u32 + params.max_tokens as u32;
        
        // Check rate limits
        self.check_rate_limit(estimated_tokens).await?;

        // Build request URL and headers
        let url = self.build_url()?;
        let headers = self.build_headers()?;

        // Build messages for chat completion
        let messages = self.build_messages(prompt, None, params);

        // Build request body
        let mut request_body = json!({
            "model": self.config.model,
            "messages": messages,
            "max_tokens": params.max_tokens,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "stream": params.stream,
        });

        // Add extra parameters
        for (key, value) in &params.extra_params {
            request_body[key] = serde_json::Value::String(value.clone());
        }

        // Make request with retries
        let mut retries = 0;
        let mut last_error = None;

        while retries < self.config.max_retries {
            match self.client.post(&url)
                .headers(headers.clone())
                .json(&request_body)
                .send()
                .await {
                    Ok(response) => {
                        if response.status().is_success() {
                            let response_data: OpenAIResponse = response.json().await
                                .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

                            let text = response_data.choices.first()
                                .ok_or_else(|| LLMError::InvalidResponse("No choices in response".to_string()))?
                                .text.clone();

                            let response = LLMResponse {
                                text,
                                tokens_used: response_data.usage.total_tokens,
                                model: response_data.model,
                                cached: false,
                                context: None,
                                metadata: HashMap::new(),
                            };

                            // Update cache
                            if let Some(cache) = &self.cache {
                                let _ = cache.put(prompt, response.clone()).await;
                            }

                            return Ok(response);
                        } else {
                            let error = response.text().await.unwrap_or_default();
                            if error.contains("rate_limit") {
                                return Err(LLMError::RateLimitExceeded(error));
                            }
                            last_error = Some(LLMError::RequestFailed(error));
                        }
                    }
                    Err(e) => {
                        last_error = Some(LLMError::RequestFailed(e.to_string()));
                    }
                }

            retries += 1;
            if retries < self.config.max_retries {
                tokio::time::sleep(Duration::from_secs(2u64.pow(retries as u32))).await;
            }
        }

        Err(last_error.unwrap_or_else(|| LLMError::RequestFailed("Unknown error".to_string())))
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

    async fn generate_stream(
        &self,
        prompt: &str,
        params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        // Check rate limits first
        let estimated_tokens = prompt.split_whitespace().count() as u32 + params.max_tokens as u32;
        let _rate_limit = self.check_rate_limit(estimated_tokens).await?;

        // Build request
        let url = self.build_url()?;
        let messages = self.build_messages(prompt, None, params);
        
        let mut request_body = json!({
            "model": params.model,
            "messages": messages,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
            "stream": true,
        });

        // Add extra parameters
        for (key, value) in &params.extra_params {
            request_body[key] = serde_json::Value::String(value.clone());
        }

        let response = self.client.post(&url)
            .headers(self.build_headers()?)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            return Err(LLMError::RequestFailed(
                response.text().await.unwrap_or_default()
            ));
        }

        let stream = response.bytes_stream();
        let mut processor = StreamProcessor::new(StreamConfig {
            enable_batching: true,
            max_batch_size: 8192,
            max_batch_wait_ms: 50,
            ..Default::default()
        });

        // Create async parser for OpenAI stream format
        let parser = |text: &str| -> futures::future::BoxFuture<'static, Result<Option<(String, bool, std::collections::HashMap<String, String>)>, LLMError>> {
            let owned_text = text.to_owned();
            Box::pin(async move {
                match serde_json::from_str::<OpenAIChatStreamResponse>(&owned_text) {
                    Ok(response) => {
                        let mut metadata = std::collections::HashMap::new();
                        if let Some(model) = response.model {
                            metadata.insert("model".to_string(), model);
                        }
                        if let Some(usage) = response.usage {
                            metadata.insert("total_tokens".to_string(), usage.total_tokens.to_string());
                        }

                        let mut text = String::new();
                        let mut is_done = false;

                        for choice in response.choices {
                            if let Some(content) = choice.delta.content {
                                text.push_str(&content);
                            }
                            if let Some(reason) = choice.finish_reason {
                                is_done = reason == "stop" || reason == "length";
                            }
                        }

                        Ok(Some((text, is_done, metadata)))
                    }
                    Err(e) => Err(LLMError::InvalidResponse(e.to_string())),
                }
            })
        };

        Ok(processor.process_stream(stream, parser).await)
    }
} 