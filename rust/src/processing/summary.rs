use thiserror::Error;
use tiktoken_rs::cl100k_base;
use serde::{Deserialize, Serialize};

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
pub trait ContentSummarizer {
    /// Generate a summary of the given content
    fn generate_summary(&self, content: &str) -> Result<Summary, SummaryError>;
    
    /// Get the current configuration
    fn get_config(&self) -> &SummaryConfig;
}

/// Basic implementation of content summarization
pub struct BasicSummarizer {
    config: SummaryConfig,
}

impl BasicSummarizer {
    /// Create a new BasicSummarizer with the given configuration
    pub fn new(config: SummaryConfig) -> Self {
        Self { config }
    }

    /// Create a new BasicSummarizer with default configuration
    pub fn default() -> Self {
        Self {
            config: SummaryConfig::default(),
        }
    }

    /// Generate a summary using truncation
    fn generate_truncation_summary(&self, content: &str) -> Result<Summary, SummaryError> {
        if content.is_empty() {
            return Err(SummaryError::EmptyContent);
        }

        let original_length = content.len();
        let summary = if content.len() <= self.config.max_length {
            content.to_string()
        } else {
            // Find last word boundary before max_length
            let truncated = content.chars()
                .take(self.config.max_length)
                .collect::<String>();
            
            // Find last word boundary
            match truncated.rfind(char::is_whitespace) {
                Some(pos) => format!("{}...", &truncated[..pos]),
                None => format!("{}...", truncated),
            }
        };

        let summary_length = summary.len();
        
        Ok(Summary {
            metadata: SummaryMetadata {
                original_length,
                summary_length,
                original_tokens: None,
                summary_tokens: None,
                keywords: None,
            },
            text: summary,
        })
    }

    /// Generate a summary using token-based approach
    fn generate_token_based_summary(&self, content: &str) -> Result<Summary, SummaryError> {
        let bpe = cl100k_base().map_err(|e| SummaryError::TokenizationError(e.to_string()))?;
        
        // Tokenize content
        let tokens = bpe.encode_with_special_tokens(content);
        let original_tokens = tokens.len();
        
        // If content is shorter than max tokens, return as is
        if original_tokens <= self.config.max_tokens {
            return Ok(Summary {
                text: content.to_string(),
                metadata: SummaryMetadata {
                    original_length: content.len(),
                    summary_length: content.len(),
                    original_tokens: Some(original_tokens),
                    summary_tokens: Some(original_tokens),
                    keywords: None,
                },
            });
        }

        // Take tokens from start and end
        let start_tokens = self.config.max_tokens / 2;
        let end_tokens = self.config.max_tokens - start_tokens;
        
        let mut summary_tokens = Vec::new();
        
        // Add start tokens
        summary_tokens.extend_from_slice(&tokens[..start_tokens]);
        
        // Add ellipsis token
        summary_tokens.push(bpe.encode_with_special_tokens("...")[0]);
        
        // Add end tokens
        let end_start = tokens.len().saturating_sub(end_tokens);
        summary_tokens.extend_from_slice(&tokens[end_start..]);
        
        // Decode summary
        let summary_text = bpe.decode(summary_tokens.clone())
            .map_err(|e| SummaryError::TokenizationError(e.to_string()))?;

        let summary_length = summary_text.len();
        let summary_tokens_len = summary_tokens.len();

        Ok(Summary {
            metadata: SummaryMetadata {
                original_length: content.len(),
                summary_length,
                original_tokens: Some(original_tokens),
                summary_tokens: Some(summary_tokens_len),
                keywords: None,
            },
            text: summary_text,
        })
    }
}

impl ContentSummarizer for BasicSummarizer {
    fn generate_summary(&self, content: &str) -> Result<Summary, SummaryError> {
        if content.is_empty() {
            return Err(SummaryError::EmptyContent);
        }

        match self.config.summary_type {
            SummaryType::Truncation => self.generate_truncation_summary(content),
            SummaryType::TokenBased => self.generate_token_based_summary(content),
        }
    }

    fn get_config(&self) -> &SummaryConfig {
        &self.config
    }
} 