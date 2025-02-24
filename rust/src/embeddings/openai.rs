use async_trait::async_trait;
use reqwest::Client;
use serde::Deserialize;
use serde_json::json;
use std::time::Duration;
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::sync::Arc;

use crate::types::embeddings::{
    EmbeddingProvider, EmbeddingConfig, EmbeddingError, EmbeddingResponse,
};
use super::cache::EmbeddingCache;
use crate::llm::rate_limiter::{RateLimiter, RateLimit};

/// OpenAI API response format for embeddings
#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
    usage: OpenAIUsage,
    model: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    total_tokens: usize,
}

/// OpenAI embeddings provider implementation
pub struct OpenAIEmbeddingProvider {
    /// HTTP client
    client: Client,
    
    /// Provider configuration
    config: EmbeddingConfig,
    
    /// Embedding cache
    cache: Option<Arc<RwLock<EmbeddingCache>>>,

    /// Rate limiter
    rate_limiter: Option<Arc<RwLock<RateLimiter>>>,
}

impl OpenAIEmbeddingProvider {
    /// Create a new OpenAI embeddings provider
    pub fn new(config: EmbeddingConfig) -> Result<Self, EmbeddingError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| EmbeddingError::ConfigError(e.to_string()))?;
            
        let cache = if config.use_cache {
            Some(Arc::new(RwLock::new(EmbeddingCache::new(
                config.cache_config.clone().unwrap_or_default()
            ))))
        } else {
            None
        };

        let rate_limiter = config.rate_limit_config.as_ref().map(|rate_config| {
            let llm_config = crate::llm::rate_limiter::RateLimitConfig {
                requests_per_minute: rate_config.requests_per_minute,
                tokens_per_minute: rate_config.tokens_per_minute,
                max_concurrent: rate_config.max_concurrent,
                use_token_bucket: rate_config.use_token_bucket,
                burst_size: rate_config.burst_size,
            };
            Arc::new(RwLock::new(RateLimiter::new(llm_config)))
        });
        
        Ok(Self {
            client,
            config,
            cache,
            rate_limiter,
        })
    }
    
    /// Build request headers
    fn build_headers(&self) -> Result<reqwest::header::HeaderMap, EmbeddingError> {
        use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
        
        let api_key = self.config.api_key.as_ref()
            .ok_or_else(|| EmbeddingError::ConfigError("API key not configured".to_string()))?;
            
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(&format!("Bearer {}", api_key))
                .map_err(|e| EmbeddingError::ConfigError(e.to_string()))?
        );
        headers.insert(
            CONTENT_TYPE,
            HeaderValue::from_static("application/json")
        );
        
        if let Some(org_id) = &self.config.org_id {
            headers.insert(
                "OpenAI-Organization",
                HeaderValue::from_str(org_id)
                    .map_err(|e| EmbeddingError::ConfigError(e.to_string()))?
            );
        }
        
        Ok(headers)
    }

    /// Check rate limits
    async fn check_rate_limit(&self, tokens: u32) -> Result<Option<RateLimit>, EmbeddingError> {
        if let Some(rate_limiter) = &self.rate_limiter {
            let rate_limit = rate_limiter.write().await
                .acquire_permit(tokens)
                .await
                .map_err(|e| EmbeddingError::RateLimitExceeded(e.to_string()))?;
            Ok(Some(rate_limit))
        } else {
            Ok(None)
        }
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIEmbeddingProvider {
    async fn initialize(&mut self) -> Result<(), EmbeddingError> {
        // Validate configuration
        if self.config.api_key.is_none() {
            return Err(EmbeddingError::ConfigError("API key not configured".to_string()));
        }
        if self.config.api_endpoint.is_none() {
            return Err(EmbeddingError::ConfigError("API endpoint not configured".to_string()));
        }
        Ok(())
    }

    async fn embed(&self, text: &str) -> Result<EmbeddingResponse, EmbeddingError> {
        // Check rate limits first
        let estimated_tokens = text.split_whitespace().count() as u32;
        let _rate_limit = self.check_rate_limit(estimated_tokens).await?;

        // Check cache first
        if let Some(cache) = &self.cache {
            if let Some(cached_response) = cache.read().await.get(text) {
                return Ok(cached_response);
            }

            // Try similarity-based lookup if enabled
            if let Some(config) = &self.config.cache_config {
                if config.similarity_threshold > 0.0 {
                    // Get embedding for similarity search
                    let url = format!(
                        "{}/v1/embeddings",
                        self.config.api_endpoint.as_ref()
                            .ok_or_else(|| EmbeddingError::ConfigError("API endpoint not configured".to_string()))?
                    );

                    let response = self.client.post(&url)
                        .headers(self.build_headers()?)
                        .json(&json!({
                            "model": self.config.model,
                            "input": text
                        }))
                        .send()
                        .await
                        .map_err(|e| EmbeddingError::RequestFailed(e.to_string()))?;

                    if response.status().is_success() {
                        let openai_response: OpenAIEmbeddingResponse = response.json().await
                            .map_err(|e| EmbeddingError::InvalidResponse(e.to_string()))?;

                        // Check for similar cached responses
                        if let Some(cached_response) = cache.read().await.get_similar(
                            &openai_response.data[0].embedding,
                            config.similarity_threshold
                        ) {
                            return Ok(cached_response);
                        }

                        // No similar response found, return the new embedding
                        let response = EmbeddingResponse {
                            embedding: openai_response.data[0].embedding.clone(),
                            tokens_used: openai_response.usage.total_tokens,
                            model: openai_response.model,
                            cached: false,
                            metadata: HashMap::new(),
                        };

                        // Cache the response
                        let _ = cache.write().await.put(text.to_string(), response.clone());
                        return Ok(response);
                    } else {
                        let error = response.text().await.unwrap_or_default();
                        if error.contains("rate_limit") {
                            return Err(EmbeddingError::RateLimitExceeded(error));
                        }
                        return Err(EmbeddingError::RequestFailed(error));
                    }
                }
            }
        }

        // No cache hit, make the request
        let url = format!(
            "{}/v1/embeddings",
            self.config.api_endpoint.as_ref()
                .ok_or_else(|| EmbeddingError::ConfigError("API endpoint not configured".to_string()))?
        );

        let mut retries = 0;
        let mut last_error = None;

        while retries < self.config.max_retries {
            match self.client.post(&url)
                .headers(self.build_headers()?)
                .json(&json!({
                    "model": self.config.model,
                    "input": text
                }))
                .send()
                .await {
                    Ok(response) => {
                        if response.status().is_success() {
                            let openai_response: OpenAIEmbeddingResponse = response.json().await
                                .map_err(|e| EmbeddingError::InvalidResponse(e.to_string()))?;

                            let response = EmbeddingResponse {
                                embedding: openai_response.data[0].embedding.clone(),
                                tokens_used: openai_response.usage.total_tokens,
                                model: openai_response.model,
                                cached: false,
                                metadata: HashMap::new(),
                            };

                            // Cache the response
                            if let Some(cache) = &self.cache {
                                let _ = cache.write().await.put(text.to_string(), response.clone());
                            }

                            return Ok(response);
                        } else {
                            let error = response.text().await.unwrap_or_default();
                            if error.contains("rate_limit") {
                                return Err(EmbeddingError::RateLimitExceeded(error));
                            }
                            last_error = Some(EmbeddingError::RequestFailed(error));
                        }
                    }
                    Err(e) => {
                        last_error = Some(EmbeddingError::RequestFailed(e.to_string()));
                    }
                }

            retries += 1;
            if retries < self.config.max_retries {
                tokio::time::sleep(Duration::from_secs(2u64.pow(retries as u32))).await;
            }
        }

        Err(last_error.unwrap_or_else(|| EmbeddingError::RequestFailed("Unknown error".to_string())))
    }

    async fn batch_embed(&self, texts: &[String]) -> Result<Vec<EmbeddingResponse>, EmbeddingError> {
        let mut responses = Vec::with_capacity(texts.len());
        let batch_size = self.config.batch_size;

        for chunk in texts.chunks(batch_size) {
            let mut batch_responses = Vec::with_capacity(chunk.len());
            for text in chunk {
                batch_responses.push(self.embed(text).await?);
            }
            responses.extend(batch_responses);
        }

        Ok(responses)
    }

    fn get_config(&self) -> &EmbeddingConfig {
        &self.config
    }

    fn update_config(&mut self, config: EmbeddingConfig) -> Result<(), EmbeddingError> {
        self.config = config;
        Ok(())
    }
} 