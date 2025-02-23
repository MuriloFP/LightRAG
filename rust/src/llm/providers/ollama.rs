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

use crate::llm::{
    LLMClient, LLMConfig, LLMError, LLMParams, LLMResponse,
    cache::{ResponseCache, InMemoryCache},
};
use crate::types::llm::{QueryParams, StreamingResponse, StreamingTiming};
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
    pub total_duration: u64,
    
    /// Time taken to load the model in microseconds
    pub load_duration: u64,
    
    /// Number of tokens in the prompt
    pub prompt_eval_count: usize,
    
    /// Number of tokens in the response
    pub eval_count: usize,
    
    /// Time taken for token generation in microseconds
    pub eval_duration: u64,

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

/// Ollama client implementation
pub struct OllamaClient {
    /// HTTP client
    client: Client,
    
    /// Client configuration
    config: LLMConfig,
    
    /// Response cache
    cache: Option<InMemoryCache>,
}

impl OllamaClient {
    /// Create a new Ollama client
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
            .ok_or_else(|| LLMError::ConfigError("API endpoint not configured".to_string()))?;
            
        Ok(format!("{}/api/generate", endpoint))
    }

    /// Validate token limits
    fn validate_token_limits(&self, prompt: &str, params: &LLMParams) -> Result<(), LLMError> {
        let estimated_prompt_tokens = prompt.split_whitespace().count();
        if estimated_prompt_tokens + params.max_tokens > 4096 { // Ollama's typical token limit
            return Err(LLMError::TokenLimitExceeded(format!(
                "Total tokens ({} prompt + {} max_tokens) exceeds model limit",
                estimated_prompt_tokens,
                params.max_tokens
            )));
        }
        Ok(())
    }

    /// Map Ollama errors to LLMError
    fn map_ollama_error(&self, error: &str) -> LLMError {
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

    /// Handle response errors
    fn handle_response_error(&self, response: &OllamaResponse) -> Result<(), LLMError> {
        if let Some(error) = &response.error {
            Err(self.map_ollama_error(error))
        } else {
            Ok(())
        }
    }

    /// Handle streaming response errors
    fn handle_stream_error(&self, response: &OllamaStreamResponse) -> Result<(), LLMError> {
        if let Some(error) = &response.error {
            Err(self.map_ollama_error(error))
        } else {
            Ok(())
        }
    }

    /// Generate embeddings for text using Ollama's embedding endpoint
    async fn get_embedding(&self, text: &str) -> Result<Option<Vec<f32>>, LLMError> {
        let url = format!("{}/api/embeddings", self.config.api_endpoint
            .as_ref()
            .ok_or_else(|| LLMError::ConfigError("API endpoint not configured".to_string()))?);

        let response = self.client.post(&url)
            .json(&json!({
                "model": self.config.model,
                "prompt": text,
            }))
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(self.map_ollama_error(&error));
        }

        #[derive(Deserialize)]
        struct EmbeddingResponse {
            embedding: Vec<f32>,
        }

        let embedding_response: EmbeddingResponse = response.json()
            .await
            .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

        Ok(Some(embedding_response.embedding))
    }

    /// Check cache with similarity search
    async fn check_cache(&self, prompt: &str, _params: &QueryParams) -> Option<LLMResponse> {
        if let Some(cache) = &self.cache {
            // Try exact match first
            if let Some(cached) = cache.get(prompt).await {
                return Some(cached);
            }

            // Try similarity search if enabled
            if let Some(embedding) = self.get_embedding(prompt).await.ok().flatten() {
                if let Some(similar) = cache.find_similar(embedding, self.config.similarity_threshold).await {
                    return Some(similar);
                }
            }
        }
        None
    }

    /// Update cache with embedding
    async fn update_cache(&self, prompt: &str, response: LLMResponse) {
        if let Some(cache) = &self.cache {
            if let Some(embedding) = self.get_embedding(prompt).await.ok().flatten() {
                let _ = cache.put_with_embedding(prompt, response, embedding).await;
            } else {
                let _ = cache.put(prompt, response).await;
            }
        }
    }

    /// Build conversation context from history
    fn build_conversation(&self, prompt: &str, history: Option<&[ConversationTurn]>, params: &LLMParams) -> String {
        let mut conversation = String::new();
        
        // Add system prompt if provided
        if let Some(system) = &params.system_prompt {
            conversation.push_str(&format!("\nSystem: {}\n", system));
        }
        
        // Add conversation history
        if let Some(history) = history {
            for turn in history {
                match turn.role.as_str() {
                    "user" => conversation.push_str(&format!("\nUser: {}\n", turn.content)),
                    "assistant" => conversation.push_str(&format!("\nAssistant: {}\n", turn.content)),
                    "system" => conversation.push_str(&format!("\nSystem: {}\n", turn.content)),
                    _ => conversation.push_str(&format!("\n{}: {}\n", turn.role, turn.content)),
                }
            }
        }
        
        // Add current prompt
        conversation.push_str(&format!("\nUser: {}\nAssistant:", prompt));
        
        conversation
    }

    /// Build request payload with conversation history
    fn build_request(&self, prompt: &str, history: Option<&[ConversationTurn]>, params: &LLMParams) -> serde_json::Value {
        let conversation = self.build_conversation(prompt, history, params);
        
        let mut request = json!({
            "model": self.config.model,
            "prompt": conversation,
            "stream": params.stream,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "num_predict": params.max_tokens,
            }
        });

        // Add any extra parameters
        for (key, value) in &params.extra_params {
            request["options"][key] = json!(value);
        }

        request
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

        // Validate token limits for the full prompt
        self.validate_token_limits(&prompt, params)?;

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
}

#[async_trait]
impl LLMClient for OllamaClient {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        // Validate configuration
        if self.config.api_endpoint.is_none() {
            return Err(LLMError::ConfigError("API endpoint not configured".to_string()));
        }

        // Test model loading
        let url = self.build_url()?;
        let response = self.client.post(&url)
            .json(&json!({
                "model": self.config.model,
                "prompt": "test",
                "stream": false
            }))
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(self.map_ollama_error(&error));
        }

        Ok(())
    }
    
    async fn generate(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        // Check cache first with similarity search
        if let Some(query_params) = &params.query_params {
            if let Some(cached) = self.check_cache(prompt, query_params).await {
                return Ok(cached);
            }
        }

        // Validate token limits
        self.validate_token_limits(prompt, params)?;

        // Build request URL
        let url = self.build_url()?;

        // Build request body
        let request_body = self.build_request(prompt, None, params);

        // Make request with retries
        let mut retries = 0;
        let mut last_error = None;
        
        while retries < self.config.max_retries {
            match self.client.post(&url)
                .json(&request_body)
                .send()
                .await {
                    Ok(response) => {
                        if response.status().is_success() {
                            let ollama_response: OllamaResponse = response.json().await
                                .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

                            // Check for API-level errors
                            self.handle_response_error(&ollama_response)?;
                                
                            let response = LLMResponse {
                                text: ollama_response.response,
                                tokens_used: ollama_response.prompt_eval_count + ollama_response.eval_count,
                                model: self.config.model.clone(),
                                cached: false,
                                context: None,
                                metadata: HashMap::new(),
                            };

                            // Update cache with embedding
                            self.update_cache(prompt, response.clone()).await;

                            return Ok(response);
                        } else {
                            let error = response.text().await.unwrap_or_default();
                            if error.contains("rate limit") {
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
                tokio::time::sleep(Duration::from_secs(1 << retries)).await;
            }
        }

        Err(last_error.unwrap_or_else(|| LLMError::RequestFailed("Max retries exceeded".to_string())))
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
            if let Some(cached_response) = cache.get(&cache_key).await {
                return Ok(cached_response);
            }
        }

        // Estimate tokens and check rate limits
        let estimated_tokens = prompt.split_whitespace().count() as u32 + 
            history.iter().map(|turn| turn.content.split_whitespace().count() as u32).sum::<u32>() +
            params.max_tokens as u32;
        
        self.check_rate_limit(estimated_tokens).await?;

        // Build request
        let url = self.build_url()?;
        let request_body = self.build_request(prompt, Some(history), params);

        let response = self.client.post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if response.status().is_success() {
            let ollama_response: OllamaResponse = response.json().await
                .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

            // Check for API-level errors
            self.handle_response_error(&ollama_response)?;

            let response = LLMResponse {
                text: ollama_response.response,
                tokens_used: ollama_response.prompt_eval_count + ollama_response.eval_count,
                model: self.config.model.clone(),
                cached: false,
                context: None,
                metadata: {
                    let mut map = HashMap::new();
                    map.insert("total_duration".to_string(), ollama_response.total_duration.to_string());
                    map.insert("load_duration".to_string(), ollama_response.load_duration.to_string());
                    map.insert("eval_duration".to_string(), ollama_response.eval_duration.to_string());
                    map.insert("prompt_eval_count".to_string(), ollama_response.prompt_eval_count.to_string());
                    map.insert("eval_count".to_string(), ollama_response.eval_count.to_string());
                    map
                },
            };

            // Update cache if enabled
            if let Some(cache) = &self.cache {
                let cache_key = format!("{:?}:{}", history, prompt);
                let _ = cache.put(&cache_key, response.clone()).await;
            }

            Ok(response)
        } else {
            let error = response.text().await.unwrap_or_default();
            Err(self.map_ollama_error(&error))
        }
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        // Validate token limits
        self.validate_token_limits(prompt, params)?;

        let url = self.build_url()?;
        let model = self.config.model.clone();

        let mut request_body = json!({
            "model": model,
            "prompt": prompt,
            "stream": true,
            "options": {
                "temperature": params.temperature,
                "top_p": params.top_p,
                "num_predict": params.max_tokens,
            }
        });

        // Add system prompt if provided
        if let Some(system_prompt) = &params.system_prompt {
            request_body["system"] = json!(system_prompt);
        }

        // Add extra parameters
        for (key, value) in &params.extra_params {
            request_body["options"][key] = serde_json::Value::String(value.clone());
        }

        let response = self.client.post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let error = response.text().await.unwrap_or_default();
            return Err(self.map_ollama_error(&error));
        }

        let stream = response.bytes_stream();
        let mut processor = StreamProcessor::new(StreamConfig {
            enable_batching: true,
            max_batch_size: 8192,
            max_batch_wait_ms: 50,
            ..Default::default()
        });

        // Create async parser for Ollama stream format
        let parser = |text: &str| -> futures::future::BoxFuture<'static, Result<Option<(String, bool, std::collections::HashMap<String, String>)>, LLMError>> {
            let owned_text = text.to_owned();
            Box::pin(async move {
                match serde_json::from_str::<OllamaStreamResponse>(&owned_text) {
                    Ok(response) => {
                        let mut metadata = std::collections::HashMap::new();
                        if let Some(total_duration) = response.total_duration {
                            metadata.insert("total_duration".to_string(), total_duration.to_string());
                        }
                        if let Some(load_duration) = response.load_duration {
                            metadata.insert("load_duration".to_string(), load_duration.to_string());
                        }
                        if let Some(eval_count) = response.eval_count {
                            metadata.insert("eval_count".to_string(), eval_count.to_string());
                        }
                        if let Some(eval_duration) = response.eval_duration {
                            metadata.insert("eval_duration".to_string(), eval_duration.to_string());
                        }

                        Ok(Some((response.response, response.done, metadata)))
                    }
                    Err(e) => Err(LLMError::InvalidResponse(e.to_string())),
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