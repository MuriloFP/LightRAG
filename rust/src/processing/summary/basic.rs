use async_trait::async_trait;
use tiktoken_rs::cl100k_base;

use super::{ContentSummarizer, Summary, SummaryConfig, SummaryError, SummaryMetadata, SummaryType};

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

#[async_trait]
impl ContentSummarizer for BasicSummarizer {
    async fn generate_summary(&self, content: &str) -> Result<Summary, SummaryError> {
        if content.is_empty() {
            return Err(SummaryError::EmptyContent);
        }

        match self.config.summary_type {
            SummaryType::Truncation => self.generate_truncation_summary(content),
            SummaryType::TokenBased => self.generate_token_based_summary(content),
            SummaryType::LLMBased => Err(SummaryError::GenerationError(
                "LLM-based summary not supported by BasicSummarizer".to_string()
            )),
        }
    }

    fn get_config(&self) -> &SummaryConfig {
        &self.config
    }
} 