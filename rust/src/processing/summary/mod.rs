mod basic;
mod llm;

pub use basic::BasicSummarizer;
pub use llm::LLMSummarizer;
pub use llm::LLMSummaryConfig;

use thiserror::Error;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;

/// Errors that can occur during summary generation
#[derive(Error, Debug)]
pub enum SummaryError {
    /// Content is empty
    #[error("Empty content")]
    EmptyContent,
    /// Error during tokenization
    #[error("Tokenization error: {0}")]
    TokenizationError(String),
    /// Error generating summary
    #[error("Generation error: {0}")]
    GenerationError(String),
}

/// Types of summary generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SummaryType {
    /// Simple truncation-based summary
    Truncation,
    /// Token-based summary using start and end tokens
    TokenBased,
    /// LLM-based summary generation
    LLMBased,
}

/// Configuration for summary generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryConfig {
    /// Maximum length of the summary in characters
    pub max_length: usize,
    /// Maximum number of tokens in the summary
    pub max_tokens: usize,
    /// Type of summary to generate
    pub summary_type: SummaryType,
    /// Language of the content (optional)
    pub language: Option<String>,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            max_length: 1000,
            max_tokens: 100,
            summary_type: SummaryType::TokenBased,
            language: None,
        }
    }
}

/// Metadata about the generated summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryMetadata {
    /// Length of the original content
    pub original_length: usize,
    /// Length of the summary
    pub summary_length: usize,
    /// Number of tokens in original content (if token-based)
    pub original_tokens: Option<usize>,
    /// Number of tokens in summary (if token-based)
    pub summary_tokens: Option<usize>,
    /// Extracted keywords (if available)
    pub keywords: Option<Vec<String>>,
}

/// A generated summary with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Summary {
    /// The summary text
    pub text: String,
    /// Metadata about the summary
    pub metadata: SummaryMetadata,
}

/// Trait for content summarization
#[async_trait]
pub trait ContentSummarizer: Send + Sync {
    /// Generate a summary of the given content
    async fn generate_summary(&self, content: &str) -> Result<Summary, SummaryError>;
    
    /// Get the current configuration
    fn get_config(&self) -> &SummaryConfig;
} 