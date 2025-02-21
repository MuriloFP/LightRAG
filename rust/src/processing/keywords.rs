use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::types::Error;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use thiserror::Error;
use tiktoken_rs::cl100k_base;

/// Errors that can occur during keyword extraction
#[derive(Error, Debug)]
pub enum KeywordError {
    /// Error when content is empty
    #[error("Empty content")]
    EmptyContent,
    /// Error during tokenization
    #[error("Tokenization error: {0}")]
    TokenizationError(String),
    /// Error during LLM processing
    #[error("LLM error: {0}")]
    LLMError(String),
    /// Error during extraction
    #[error("Extraction error: {0}")]
    ExtractionError(String),
}

impl From<KeywordError> for Error {
    fn from(err: KeywordError) -> Self {
        Error::Storage(err.to_string())
    }
}

/// Extracted keywords with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedKeywords {
    /// High-level conceptual keywords
    pub high_level: Vec<String>,
    /// Low-level specific keywords
    pub low_level: Vec<String>,
    /// Additional metadata about the extraction
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Configuration for keyword extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeywordConfig {
    /// Maximum number of high-level keywords to extract
    pub max_high_level: usize,
    /// Maximum number of low-level keywords to extract
    pub max_low_level: usize,
    /// Language of the content
    pub language: Option<String>,
    /// Whether to use LLM for extraction
    pub use_llm: bool,
    /// Additional configuration parameters
    pub extra_params: HashMap<String, serde_json::Value>,
}

impl Default for KeywordConfig {
    fn default() -> Self {
        Self {
            max_high_level: 5,
            max_low_level: 10,
            language: None,
            use_llm: false,
            extra_params: HashMap::new(),
        }
    }
}

/// A turn in a conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// The role of the speaker (e.g., "user", "assistant")
    pub role: String,
    /// The content of the message
    pub content: String,
    /// Timestamp of the message
    pub timestamp: Option<DateTime<Utc>>,
}

/// Trait for keyword extraction
#[async_trait]
pub trait KeywordExtractor: Send + Sync {
    /// Extract keywords from the given content
    async fn extract_keywords(&self, content: &str) -> Result<ExtractedKeywords, Error>;
    
    /// Extract keywords considering conversation history
    async fn extract_keywords_with_history(
        &self, 
        content: &str,
        history: &[ConversationTurn]
    ) -> Result<ExtractedKeywords, Error>;
    
    /// Get the current configuration
    fn get_config(&self) -> &KeywordConfig;
}

/// Basic implementation of keyword extraction using TF-IDF
pub struct BasicKeywordExtractor {
    config: KeywordConfig,
}

impl BasicKeywordExtractor {
    /// Create a new BasicKeywordExtractor with the given configuration
    pub fn new(config: KeywordConfig) -> Self {
        Self { config }
    }

    /// Create a new BasicKeywordExtractor with default configuration
    pub fn default() -> Self {
        Self {
            config: KeywordConfig::default(),
        }
    }

    /// Extract keywords using TF-IDF
    fn extract_keywords_tfidf(&self, content: &str) -> Result<ExtractedKeywords, Error> {
        let bpe = cl100k_base()
            .map_err(|e| KeywordError::TokenizationError(e.to_string()))?;
        let tokens = bpe.encode_with_special_tokens(content);
        
        // Create word frequency map
        let mut word_freq: HashMap<String, usize> = HashMap::new();
        let mut total_words = 0;
        
        for token in tokens {
            let word = bpe.decode(vec![token])
                .map_err(|e| KeywordError::TokenizationError(e.to_string()))?;
            if !word.trim().is_empty() {
                *word_freq.entry(word).or_insert(0) += 1;
                total_words += 1;
            }
        }

        // Calculate TF-IDF scores (simplified since we only have one document)
        let mut word_scores: Vec<(String, f32)> = word_freq
            .into_iter()
            .map(|(word, freq)| {
                let tf = freq as f32 / total_words as f32;
                (word, tf)
            })
            .collect();

        // Sort by score
        word_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Split into high-level and low-level keywords
        let mut high_level = Vec::new();
        let mut low_level = Vec::new();

        for (word, _) in word_scores.iter().take(self.config.max_high_level) {
            high_level.push(word.clone());
        }

        for (word, _) in word_scores.iter()
            .skip(self.config.max_high_level)
            .take(self.config.max_low_level) {
            low_level.push(word.clone());
        }

        Ok(ExtractedKeywords {
            high_level,
            low_level,
            metadata: {
                let mut meta = HashMap::new();
                meta.insert("total_words".to_string(), serde_json::json!(total_words));
                meta.insert("extraction_method".to_string(), serde_json::json!("tf-idf"));
                meta
            },
        })
    }
}

#[async_trait]
impl KeywordExtractor for BasicKeywordExtractor {
    async fn extract_keywords(&self, content: &str) -> Result<ExtractedKeywords, Error> {
        if content.is_empty() {
            return Err(KeywordError::EmptyContent.into());
        }
        self.extract_keywords_tfidf(content)
    }

    async fn extract_keywords_with_history(
        &self,
        content: &str,
        history: &[ConversationTurn]
    ) -> Result<ExtractedKeywords, Error> {
        // For basic implementation, combine history with current content
        let mut combined_content = String::new();
        
        // Add history with weights based on recency
        for (i, turn) in history.iter().enumerate() {
            let weight = (i + 1) as f32 / history.len() as f32;
            combined_content.push_str(&format!("{} ", turn.content));
            if weight < 0.5 {
                break; // Only use recent history
            }
        }
        
        // Add current content
        combined_content.push_str(content);
        
        self.extract_keywords(&combined_content).await
    }

    fn get_config(&self) -> &KeywordConfig {
        &self.config
    }
}

/// LLM-based implementation of keyword extraction
pub struct LLMKeywordExtractor {
    config: KeywordConfig,
}

impl LLMKeywordExtractor {
    /// Create a new LLMKeywordExtractor with the given configuration
    pub fn new(config: KeywordConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl KeywordExtractor for LLMKeywordExtractor {
    async fn extract_keywords(&self, content: &str) -> Result<ExtractedKeywords, Error> {
        if content.is_empty() {
            return Err(KeywordError::EmptyContent.into());
        }
        // TODO: Implement LLM-based extraction once we have the LLM module
        Err(KeywordError::ExtractionError("LLM extraction not yet implemented".to_string()).into())
    }

    async fn extract_keywords_with_history(
        &self,
        content: &str,
        _history: &[ConversationTurn]
    ) -> Result<ExtractedKeywords, Error> {
        // TODO: Implement with history support
        self.extract_keywords(content).await
    }

    fn get_config(&self) -> &KeywordConfig {
        &self.config
    }
} 