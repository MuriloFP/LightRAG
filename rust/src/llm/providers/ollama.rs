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

use crate::llm::{
    LLMClient, LLMConfig, LLMError, LLMParams, LLMResponse,
    cache::{ResponseCache, InMemoryCache},
};
use crate::types::llm::{QueryParams, StreamingResponse, StreamingTiming};
use crate::processing::keywords::ConversationTurn;
use crate::processing::context::ContextBuilder;

/// Ollama API response format
#[derive(Debug, Deserialize)]
struct OllamaResponse {
    /// The generated text response
    response: String,
    
    /// Whether the generation is complete
    #[allow(dead_code)]  // Used by Ollama API but not by our implementation
    done: bool,
    
    /// Total duration of the request in microseconds
    total_duration: u64,
    
    /// Time taken to load the model in microseconds
    load_duration: u64,
    
    /// Number of tokens in the prompt
    prompt_eval_count: usize,
    
    /// Number of tokens in the response
    eval_count: usize,
    
    /// Time taken for token generation in microseconds
    eval_duration: u64,
}

/// Ollama streaming response format
#[derive(Debug, Deserialize)]
struct OllamaStreamResponse {
    /// The generated text chunk
    response: String,
    
    /// Whether the generation is complete
    done: bool,

    /// Total duration of the request in microseconds
    total_duration: Option<u64>,
    
    /// Time taken to load the model in microseconds
    load_duration: Option<u64>,
    
    /// Number of tokens in the prompt
    prompt_eval_count: Option<usize>,
    
    /// Number of tokens in the response
    eval_count: Option<usize>,
    
    /// Time taken for token generation in microseconds
    eval_duration: Option<u64>,
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
            "{}/api/embeddings",
            self.config.api_endpoint.as_ref()
                .ok_or_else(|| LLMError::ConfigError("API endpoint not configured".to_string()))?
        );

        let response = self.client.post(&url)
            .json(&json!({
                "model": self.config.model,
                "prompt": text
            }))
            .send()
            .await
            .map_err(|e| LLMError::RequestFailed(e.to_string()))?;

        if response.status().is_success() {
            let data: serde_json::Value = response.json().await
                .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

            if let Some(embedding) = data["embedding"].as_array() {
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
                metadata: HashMap::new(),
            }).await;
        }
    }
}

#[async_trait]
impl LLMClient for OllamaClient {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        // Validate configuration
        if self.config.api_endpoint.is_none() {
            return Err(LLMError::ConfigError("API endpoint not configured".to_string()));
        }
        Ok(())
    }
    
    async fn generate(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        // Check rate limits first
        let estimated_tokens = prompt.split_whitespace().count() as u32 + params.max_tokens as u32;
        let _rate_limit = <Self as LLMClient>::check_rate_limit(self, estimated_tokens).await?;
        
        // Check cache first
        if let Some(cache) = &self.cache {
            if let Some(cached_response) = cache.get(prompt).await {
                return Ok(cached_response);
            }
        }
        
        // Build request
        let url = self.build_url()?;
        
        let mut request_body = json!({
            "model": self.config.model,
            "prompt": prompt,
            "stream": params.stream,
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
                                
                            let llm_response = LLMResponse {
                                text: ollama_response.response,
                                tokens_used: ollama_response.prompt_eval_count + ollama_response.eval_count,
                                model: self.config.model.clone(),
                                cached: false,
                                context: None,
                                metadata: {
                                    let mut map = std::collections::HashMap::new();
                                    map.insert("total_duration".to_string(), ollama_response.total_duration.to_string());
                                    map.insert("load_duration".to_string(), ollama_response.load_duration.to_string());
                                    map.insert("eval_duration".to_string(), ollama_response.eval_duration.to_string());
                                    map
                                },
                            };
                            
                            // Cache the response
                            if let Some(cache) = &self.cache {
                                let _ = cache.put(prompt, llm_response.clone()).await;
                            }
                            
                            return Ok(llm_response);
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

    async fn generate_with_history(
        &self,
        prompt: &str,
        history: &[ConversationTurn],
        params: &LLMParams
    ) -> Result<LLMResponse, LLMError> {
        let url = self.build_url()?;
        
        let mut context = String::new();
        for turn in history {
            context.push_str(&format!("{}: {}\n", turn.role, turn.content));
        }
        context.push_str(&format!("user: {}", prompt));

        let mut request_body = json!({
            "model": self.config.model,
            "prompt": context,
            "stream": params.stream,
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

        if response.status().is_success() {
            let ollama_response: OllamaResponse = response.json().await
                .map_err(|e| LLMError::InvalidResponse(e.to_string()))?;

            Ok(LLMResponse {
                text: ollama_response.response,
                tokens_used: ollama_response.prompt_eval_count + ollama_response.eval_count,
                model: self.config.model.clone(),
                cached: false,
                context: None,
                metadata: {
                    let mut map = std::collections::HashMap::new();
                    map.insert("total_duration".to_string(), ollama_response.total_duration.to_string());
                    map.insert("load_duration".to_string(), ollama_response.load_duration.to_string());
                    map.insert("eval_duration".to_string(), ollama_response.eval_duration.to_string());
                    map
                },
            })
        } else {
            Err(LLMError::RequestFailed(
                response.text().await.unwrap_or_default()
            ))
        }
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        let url = self.build_url()?;
        let start_time = Instant::now();
        let total_tokens = Arc::new(AtomicUsize::new(0));
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
            return Err(LLMError::RequestFailed(
                response.text().await.unwrap_or_default()
            ));
        }

        let (tx, rx) = mpsc::channel(100);
        let mut stream = response.bytes_stream();
        let total_tokens_clone = total_tokens.clone();

        tokio::spawn(async move {
            let mut chunk_start = Instant::now();
            
            while let Some(chunk_result) = stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        if let Ok(text) = String::from_utf8(chunk.to_vec()) {
                            if let Ok(response) = serde_json::from_str::<OllamaStreamResponse>(&text) {
                                let chunk_tokens = response.eval_count.unwrap_or_else(|| 
                                    response.response.split_whitespace().count()
                                );
                                total_tokens_clone.fetch_add(chunk_tokens, Ordering::Relaxed);
                                
                                let _ = tx.send(Ok(StreamingResponse {
                                    text: response.response,
                                    done: response.done,
                                    timing: Some(StreamingTiming {
                                        chunk_duration: chunk_start.elapsed().as_micros() as u64,
                                        total_duration: response.total_duration.unwrap_or_else(|| 
                                            start_time.elapsed().as_micros() as u64
                                        ),
                                        prompt_eval_duration: Some(response.eval_duration.unwrap_or(0)),
                                        token_gen_duration: Some(response.eval_duration.unwrap_or(0)),
                                    }),
                                    chunk_tokens,
                                    total_tokens: total_tokens_clone.load(Ordering::Relaxed),
                                    metadata: {
                                        let mut map = HashMap::new();
                                        if let Some(load_dur) = response.load_duration {
                                            map.insert("load_duration".to_string(), load_dur.to_string());
                                        }
                                        if let Some(prompt_eval_count) = response.prompt_eval_count {
                                            map.insert("prompt_eval_count".to_string(), prompt_eval_count.to_string());
                                        }
                                        if let Some(eval_count) = response.eval_count {
                                            map.insert("eval_count".to_string(), eval_count.to_string());
                                        }
                                        map.insert("model".to_string(), model.clone());
                                        map
                                    },
                                })).await;
                                
                                if response.done {
                                    break;
                                }
                                
                                chunk_start = Instant::now();
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