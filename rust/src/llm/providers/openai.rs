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

use crate::llm::{
    LLMClient, LLMConfig, LLMError, LLMParams, LLMResponse,
    cache::{ResponseCache, InMemoryCache},
};
use crate::types::llm::{QueryParams, StreamingResponse, StreamingTiming};
use crate::processing::keywords::ConversationTurn;
use crate::processing::context::ContextBuilder;

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
    /// HTTP client
    client: Client,
    
    /// Client configuration
    config: LLMConfig,
    
    /// Response cache
    cache: Option<InMemoryCache>,
}

impl OpenAIClient {
    /// Create a new OpenAI client
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
        // Implement rate limit checking logic here
        Ok(())
    }

    /// Create embeddings for a list of texts
    pub async fn create_embeddings(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LLMError> {
        // Implementation here
        unimplemented!("OpenAI create_embeddings not implemented yet")
    }
}

#[async_trait]
impl LLMClient for OpenAIClient {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        Ok(())
    }

    async fn generate(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        // Implementation here
        unimplemented!("OpenAI generate not implemented yet")
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