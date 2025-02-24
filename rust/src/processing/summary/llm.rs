use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use md5;
use std::time::Duration;
use std::panic::AssertUnwindSafe;
use futures::FutureExt;

use crate::llm::{Provider, LLMParams, LLMResponse, LLMError};
use crate::llm::cache::backend::CacheBackend;
use crate::llm::cache::types::{CacheEntry, CacheValue, CacheMetadata};
use crate::processing::keywords::{KeywordExtractor, LLMKeywordExtractor, KeywordConfig};
use crate::types::llm::QueryParams;
use super::{
    ContentSummarizer, Summary, SummaryConfig, SummaryError, SummaryMetadata,
    BasicSummarizer,
};
use crate::types::error::Error;

/// Configuration for LLM-based summarization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMSummaryConfig {
    /// Base configuration for summarization
    pub base_config: SummaryConfig,
    
    /// Parameters for LLM requests
    pub llm_params: LLMParams,
    
    /// Whether to use fallback summarizer on error
    pub use_fallback: bool,
    
    /// Language for prompts (defaults to English)
    pub language: String,
    
    /// Whether to use caching
    pub use_cache: bool,
    
    /// Whether to extract keywords
    pub extract_keywords: bool,
    
    /// Maximum retries for LLM calls
    pub max_retries: usize,
    
    /// Additional metadata fields to include
    pub extra_metadata: HashMap<String, String>,
}

impl Default for LLMSummaryConfig {
    fn default() -> Self {
        Self {
            base_config: SummaryConfig::default(),
            llm_params: LLMParams {
                model: "openai/gpt-4".to_string(),
                max_tokens: 500,
                temperature: 0.3,
                top_p: 0.9,
                stream: false,
                system_prompt: Some("You are a helpful assistant that creates concise and informative summaries.".to_string()),
                query_params: None::<crate::types::llm::QueryParams>,
                extra_params: Default::default(),
            },
            use_fallback: true,
            language: "English".to_string(),
            use_cache: true,
            extract_keywords: true,
            max_retries: 3,
            extra_metadata: Default::default(),
        }
    }
}

/// LLM-based summarizer implementation
pub struct LLMSummarizer {
    /// LLM provider
    provider: Arc<Box<dyn Provider>>,
    
    /// Summarizer configuration
    config: LLMSummaryConfig,
    
    /// Keyword extractor
    keyword_extractor: Option<Arc<LLMKeywordExtractor>>,
    
    /// Fallback summarizer
    fallback: Arc<BasicSummarizer>,
    
    /// Response cache
    cache: Arc<dyn CacheBackend>,
}

impl LLMSummarizer {
    /// Create a new LLM summarizer
    pub fn new(
        provider: Box<dyn Provider>,
        config: LLMSummaryConfig,
        cache: Box<dyn CacheBackend>,
    ) -> Self {
        let keyword_extractor = if config.extract_keywords {
            let keyword_config = KeywordConfig {
                max_high_level: 5,
                max_low_level: 10,
                language: Some(config.language.clone()),
                use_llm: true,
                extra_params: Default::default(),
            };
            // Create a new provider instance for the keyword extractor
            let provider_config = provider.get_config().clone();
            let provider_type = match provider_config.model.split('/').next() {
                Some("openai") => "openai",
                Some("anthropic") => "anthropic",
                Some("ollama") => "ollama",
                _ => "openai", // default to OpenAI
            };
            if let Ok(new_provider) = crate::llm::create_provider(provider_type, provider_config) {
                Some(Arc::new(LLMKeywordExtractor::new(
                    keyword_config,
                    Arc::new(new_provider),
                    config.llm_params.clone(),
                )))
            } else {
                None
            }
        } else {
            None
        };

        Self {
            provider: Arc::new(provider),
            config: config.clone(),
            keyword_extractor,
            fallback: Arc::new(BasicSummarizer::new(config.base_config.clone())),
            cache: Arc::from(cache),
        }
    }
    
    /// Generate a prompt for the LLM
    fn generate_prompt(&self, content: &str) -> String {
        format!(
            "Please create a concise and informative summary of the following text in {language}. \
            The summary should be no longer than {max_tokens} tokens and should capture the main points \
            while maintaining readability and coherence. Focus on key concepts and important details.\n\n\
            Text to summarize:\n{content}\n\n\
            Summary:",
            language = self.config.language,
            max_tokens = self.config.llm_params.max_tokens,
            content = content
        )
    }
    
    /// Try to get summary from cache
    async fn get_cached_summary(&self, content: &str) -> Option<Summary> {
        let cache_key = format!("summary:{:x}", md5::compute(content.as_bytes()));
        if let Ok(entry) = self.cache.get(&cache_key).await {
            if let CacheValue::Response(response) = entry.value {
                return Some(Summary {
                    text: response.text.clone(),
                    metadata: SummaryMetadata {
                        original_length: content.len(),
                        summary_length: response.text.len(),
                        original_tokens: Some(response.tokens_used),
                        summary_tokens: Some(self.config.llm_params.max_tokens),
                        keywords: None,
                    },
                });
            }
        }
        None
    }
    
    /// Cache a generated summary
    async fn cache_summary(&self, content: &str, summary: &Summary) {
        let cache_key = format!("summary:{:x}", md5::compute(content.as_bytes()));
        let cache_entry = CacheEntry::new(
            cache_key,
            CacheValue::Response(LLMResponse {
                text: summary.text.clone(),
                tokens_used: summary.metadata.original_tokens.unwrap_or(0),
                model: self.config.llm_params.model.clone(),
                cached: false,
                context: None,
                metadata: HashMap::new(),
            }),
            Some(Duration::from_secs(3600)), // 1 hour TTL
            None,
        );
        let _ = self.cache.set(cache_entry).await;
    }

    async fn try_generate_summary(&self, content: &str) -> Result<Summary, SummaryError> {
        let response = self.provider
            .complete(&self.generate_prompt(content), &self.config.llm_params)
            .await
            .map_err(|e| SummaryError::GenerationFailed(e.to_string()))?;

        // Extract keywords if configured
        let keywords = if let Some(extractor) = &self.keyword_extractor {
            match extractor.extract_keywords(content).await {
                Ok(extracted) => Some(extracted.high_level.into_iter()
                    .chain(extracted.low_level.into_iter())
                    .collect()),
                Err(_) => None,
            }
        } else {
            None
        };

        let text = response.text;
        let text_len = text.len();
        let summary = Summary {
            text,
            metadata: SummaryMetadata {
                original_length: content.len(),
                summary_length: text_len,
                original_tokens: Some(response.tokens_used),
                summary_tokens: Some(self.config.llm_params.max_tokens),
                keywords,
            },
        };

        // Try fallback if enabled and summary is empty
        if summary.text.is_empty() && self.config.use_fallback {
            return self.fallback.generate_summary(content).await;
        }

        Ok(summary)
    }

    pub async fn generate_summary(&self, content: &str) -> Result<Summary, SummaryError> {
        if content.trim().is_empty() {
            return Err(SummaryError::EmptyContent);
        }

        // Prepare the prompt for summarization
        let prompt = format!("Summarize the content:\n{}", content);
        
        // Prepare LLM parameters.
        let params = LLMParams {
            model: self.config.llm_params.model.clone(),
            max_tokens: self.config.llm_params.max_tokens,
            temperature: self.config.llm_params.temperature,
            top_p: self.config.llm_params.top_p,
            stream: false,
            system_prompt: None,
            query_params: None,
            extra_params: std::collections::HashMap::new(),
        };
 
        // Attempt to call the provider.complete method inside a catch_unwind block to catch panics
        let provider_call = AssertUnwindSafe(self.provider.complete(&prompt, &params)).catch_unwind();
        let result = provider_call.await;
        
        match result {
            Ok(Ok(response)) => {
                let text = response.text;
                let summary = Summary {
                    text: text.clone(),
                    metadata: SummaryMetadata {
                        original_length: content.len(),
                        summary_length: text.len(),
                        original_tokens: Some(response.tokens_used),
                        summary_tokens: Some(self.config.llm_params.max_tokens),
                        keywords: None,
                    },
                };
                Ok(summary)
            },
            _ => {
                if self.config.use_fallback {
                    self.fallback.generate_summary(content).await
                } else {
                    Err(SummaryError::GenerationFailed("LLM provider failed".to_string()))
                }
            }
        }
    }
}

#[async_trait]
impl ContentSummarizer for LLMSummarizer {
    async fn generate_summary(&self, content: &str) -> Result<Summary, SummaryError> {
        if content.is_empty() {
            return Err(SummaryError::EmptyContent);
        }

        let cache_key = format!("summary:{:x}", md5::compute(content.as_bytes()));
        
        // Check cache first
        if self.config.use_cache {
            if let Ok(cached_entry) = self.cache.get(&cache_key).await {
                if let CacheValue::Response(response) = cached_entry.value {
                    let text = response.text;
                    let text_len = text.len();
                    return Ok(Summary {
                        text,
                        metadata: SummaryMetadata {
                            original_length: content.len(),
                            summary_length: text_len,
                            original_tokens: Some(response.tokens_used),
                            summary_tokens: Some(self.config.llm_params.max_tokens),
                            keywords: None,
                        },
                    });
                }
            }
        }

        let mut retries = 0;
        let mut last_error = None;

        while retries <= self.config.max_retries {
            match self.try_generate_summary(content).await {
                Ok(summary) => {
                    // Cache the successful result
                    if self.config.use_cache {
                        let cache_entry = CacheEntry::new(
                            cache_key.clone(),
                            CacheValue::Response(LLMResponse {
                                text: summary.text.clone(),
                                tokens_used: summary.metadata.original_tokens.unwrap_or(0),
                                model: self.config.llm_params.model.clone(),
                                cached: false,
                                context: None,
                                metadata: HashMap::new(),
                            }),
                            Some(Duration::from_secs(3600)), // 1 hour TTL
                            None,
                        );
                        let _ = self.cache.set(cache_entry).await;
                    }
                    return Ok(summary);
                }
                Err(e) => {
                    last_error = Some(e);
                    retries += 1;
                    
                    if retries <= self.config.max_retries {
                        tokio::time::sleep(std::time::Duration::from_secs(2u64.pow(retries as u32))).await;
                    }
                }
            }
        }

        // If we get here, all retries failed
        if let Some(e) = last_error {
            Err(e)
        } else {
            Err(SummaryError::GenerationFailed("All retries failed".to_string()))
        }
    }
    
    fn get_config(&self) -> &SummaryConfig {
        &self.config.base_config
    }
} 