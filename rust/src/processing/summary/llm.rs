use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;

use crate::types::llm::{LLMClient, LLMParams};
use crate::processing::keywords::{KeywordExtractor, LLMKeywordExtractor, KeywordConfig};
use super::{
    ContentSummarizer, Summary, SummaryConfig, SummaryError, SummaryMetadata,
    BasicSummarizer,
};

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

/// LLM-based implementation of content summarization
pub struct LLMSummarizer {
    /// Configuration
    config: LLMSummaryConfig,
    
    /// LLM client
    llm_client: Arc<dyn LLMClient>,
    
    /// Fallback summarizer
    fallback: BasicSummarizer,
    
    /// Keyword extractor
    keyword_extractor: Option<LLMKeywordExtractor>,
    
    /// Response cache
    cache: Option<Arc<RwLock<HashMap<String, Summary>>>>,
}

impl LLMSummarizer {
    /// Create a new LLMSummarizer with the given configuration and client
    pub fn new(config: LLMSummaryConfig, llm_client: Arc<dyn LLMClient>) -> Self {
        let keyword_extractor = if config.extract_keywords {
            Some(LLMKeywordExtractor::new(
                KeywordConfig {
                    max_high_level: 5,
                    max_low_level: 10,
                    language: Some(config.language.clone()),
                    use_llm: true,
                    extra_params: Default::default(),
                },
                Arc::clone(&llm_client),
                config.llm_params.clone(),
            ))
        } else {
            None
        };

        Self {
            config: config.clone(),
            llm_client,
            fallback: BasicSummarizer::new(config.base_config),
            keyword_extractor,
            cache: if config.use_cache { 
                Some(Arc::new(RwLock::new(HashMap::new()))) 
            } else { 
                None 
            },
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
        if let Some(cache) = &self.cache {
            let cache_read = cache.read().await;
            cache_read.get(content).cloned()
        } else {
            None
        }
    }
    
    /// Cache a generated summary
    async fn cache_summary(&self, content: &str, summary: Summary) {
        if let Some(cache) = &self.cache {
            let mut cache_write = cache.write().await;
            cache_write.insert(content.to_string(), summary);
        }
    }
}

#[async_trait]
impl ContentSummarizer for LLMSummarizer {
    async fn generate_summary(&self, content: &str) -> Result<Summary, SummaryError> {
        if content.is_empty() {
            return Err(SummaryError::EmptyContent);
        }
        
        // Check cache first
        if let Some(cached) = self.get_cached_summary(content).await {
            return Ok(cached);
        }
        
        let original_length = content.len();
        let mut retries = 0;
        
        // Try LLM-based summary with retries
        while retries < self.config.max_retries {
            match self.llm_client.generate(
                &self.generate_prompt(content),
                &self.config.llm_params
            ).await {
                Ok(llm_response) => {
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
                    
                    let summary_length = llm_response.text.len();
                    let summary = Summary {
                        text: llm_response.text,
                        metadata: SummaryMetadata {
                            original_length,
                            summary_length,
                            original_tokens: Some(llm_response.tokens_used),
                            summary_tokens: Some(self.config.llm_params.max_tokens),
                            keywords,
                        },
                    };
                    
                    // Cache the summary if enabled
                    if self.config.use_cache {
                        self.cache_summary(content, summary.clone()).await;
                    }
                    
                    return Ok(summary);
                }
                Err(_e) if retries < self.config.max_retries - 1 => {
                    retries += 1;
                    continue;
                }
                Err(_e) if self.config.use_fallback => {
                    // Use fallback summarizer on final retry
                    return self.fallback.generate_summary(content).await;
                }
                Err(e) => return Err(SummaryError::GenerationError(e.to_string())),
            }
        }
        
        // This should never be reached due to the error handling above
        Err(SummaryError::GenerationError("Maximum retries exceeded".to_string()))
    }
    
    fn get_config(&self) -> &SummaryConfig {
        &self.config.base_config
    }
} 