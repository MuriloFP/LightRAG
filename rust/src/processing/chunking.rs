use crate::processing::types::{TextChunk, ChunkingError};
use tiktoken_rs::cl100k_base;
use tracing::{debug, warn};
use std::collections::HashSet;
use regex::Regex;

/// Configuration for text chunking
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Number of overlapping tokens between consecutive chunks
    pub overlap_token_size: usize,
    /// Maximum number of tokens per chunk
    pub max_token_size: usize,
    /// Optional character to split text by before tokenization
    pub split_by_character: Option<String>,
    /// Whether to only use character splitting without token-based chunking
    pub split_by_character_only: bool,
    /// Minimum chunk size in tokens
    pub min_chunk_size: usize,
    /// Whether to enable smart boundary detection
    pub smart_boundary: bool,
    /// Whether to enable chunk deduplication
    pub enable_deduplication: bool,
    /// Minimum similarity threshold for deduplication (0.0-1.0)
    pub dedup_similarity_threshold: f32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            overlap_token_size: 100,
            max_token_size: 1200,
            split_by_character: None,
            split_by_character_only: false,
            min_chunk_size: 50,
            smart_boundary: true,
            enable_deduplication: true,
            dedup_similarity_threshold: 0.85,
        }
    }
}

/// Chunks text into smaller pieces based on token size and configuration
/// 
/// # Arguments
/// * `content` - The text to chunk
/// * `config` - Configuration for chunking behavior
/// * `doc_id` - ID of the full document being chunked
/// 
/// # Returns
/// A vector of TextChunks or an error
pub async fn chunk_text(
    content: &str,
    config: &ChunkingConfig,
    doc_id: &str,
) -> Result<Vec<TextChunk>, ChunkingError> {
    // Validate input
    if content.trim().is_empty() {
        return Err(ChunkingError::EmptyInput);
    }

    let bpe = cl100k_base().map_err(|e| ChunkingError::TokenizationError(e.to_string()))?;
    let mut chunks = Vec::new();

    // Handle character-based splitting if configured
    if let Some(split_char) = &config.split_by_character {
        let raw_chunks: Vec<&str> = content.split(split_char).collect();
        
        if config.split_by_character_only {
            // Only split by character, create chunks directly
            for (idx, chunk) in raw_chunks.iter().enumerate() {
                let chunk_content = chunk.trim();
                if !chunk_content.is_empty() {
                    let tokens = bpe.encode_with_special_tokens(chunk_content);
                    if tokens.len() >= config.min_chunk_size {
                        chunks.push(TextChunk {
                            tokens: tokens.len(),
                            content: chunk_content.to_string(),
                            full_doc_id: doc_id.to_string(),
                            chunk_order_index: idx,
                        });
                    }
                }
            }
        } else {
            // Split by character first, then by tokens if needed
            for (idx, chunk) in raw_chunks.iter().enumerate() {
                let chunk_content = chunk.trim();
                if chunk_content.is_empty() {
                    continue;
                }

                let tokens = bpe.encode_with_special_tokens(chunk_content);
                if tokens.len() > config.max_token_size {
                    // Further split by tokens if too large
                    let sub_chunks = split_by_tokens(
                        chunk_content,
                        &bpe,
                        config,
                    )?;
                    
                    for (sub_idx, sub_chunk) in sub_chunks.into_iter().enumerate() {
                        chunks.push(TextChunk {
                            tokens: sub_chunk.tokens,
                            content: sub_chunk.content,
                            full_doc_id: doc_id.to_string(),
                            chunk_order_index: idx * 1000 + sub_idx, // Preserve order with sub-indexing
                        });
                    }
                } else if tokens.len() >= config.min_chunk_size {
                    chunks.push(TextChunk {
                        tokens: tokens.len(),
                        content: chunk_content.to_string(),
                        full_doc_id: doc_id.to_string(),
                        chunk_order_index: idx,
                    });
                }
            }
        }
    } else {
        // Token-based splitting only
        chunks = split_by_tokens(
            content,
            &bpe,
            config,
        )?;
        
        // Set chunk order and document ID
        for (idx, chunk) in chunks.iter_mut().enumerate() {
            chunk.chunk_order_index = idx;
            chunk.full_doc_id = doc_id.to_string();
        }
    }

    // Validate and deduplicate chunks
    chunks = validate_and_deduplicate_chunks(chunks, config)?;

    debug!(
        "Created {} chunks from document {}",
        chunks.len(),
        doc_id
    );

    Ok(chunks)
}

/// Helper function to split text by tokens with smart boundary detection
fn split_by_tokens(
    content: &str,
    bpe: &tiktoken_rs::CoreBPE,
    config: &ChunkingConfig,
) -> Result<Vec<TextChunk>, ChunkingError> {
    let tokens = bpe.encode_with_special_tokens(content);
    let mut chunks = Vec::new();
    
    // Compile regex patterns for smart boundary detection
    let sentence_end = Regex::new(r"[.!?]\s+").unwrap();
    let paragraph_end = Regex::new(r"\n\s*\n").unwrap();
    
    // Calculate chunk positions
    let mut start = 0;
    while start < tokens.len() {
        let mut end = (start + config.max_token_size).min(tokens.len());
        
        // Smart boundary detection if enabled
        if config.smart_boundary && end < tokens.len() {
            let chunk_text = bpe.decode(tokens[start..end].to_vec())
                .map_err(|e| ChunkingError::TokenizationError(e.to_string()))?;
                
            // Try to find a paragraph boundary
            if let Some(para_match) = paragraph_end.find_iter(&chunk_text).last() {
                end = start + bpe.encode_with_special_tokens(&chunk_text[..para_match.end()]).len();
            } else if let Some(sent_match) = sentence_end.find_iter(&chunk_text).last() {
                // Fall back to sentence boundary
                end = start + bpe.encode_with_special_tokens(&chunk_text[..sent_match.end()]).len();
            }
        }
        // Safeguard to ensure progress and avoid infinite loop
        if end <= start {
            end = (start + config.max_token_size).min(tokens.len());
            if end <= start {
                end = start + 1;
            }
        }
        
        let chunk_tokens = &tokens[start..end];
        let chunk_content = bpe.decode(chunk_tokens.to_vec())
            .map_err(|e| ChunkingError::TokenizationError(e.to_string()))?;
            
        // Only add chunk if it meets minimum size requirement
        if chunk_tokens.len() >= config.min_chunk_size {
            chunks.push(TextChunk {
                tokens: chunk_tokens.len(),
                content: chunk_content,
                full_doc_id: String::new(), // Will be set by caller
                chunk_order_index: 0, // Will be set by caller
            });
        }
    
        // Calculate next start position with overlap, ensuring progress
        start = if end == tokens.len() {
            end // No more tokens to process
        } else {
            std::cmp::max(end - config.overlap_token_size, start + 1)
        };
    }

    Ok(chunks)
}

/// Validate chunks and remove duplicates
fn validate_and_deduplicate_chunks(
    mut chunks: Vec<TextChunk>,
    config: &ChunkingConfig,
) -> Result<Vec<TextChunk>, ChunkingError> {
    // Remove chunks that are too small
    chunks.retain(|chunk| chunk.tokens >= config.min_chunk_size);
    
    if chunks.is_empty() {
        return Err(ChunkingError::ProcessingError("No valid chunks created".to_string()));
    }
    
    // Deduplicate chunks if enabled
    if config.enable_deduplication {
        let mut unique_chunks = Vec::new();
        let mut seen_content = HashSet::new();
        
        for chunk in chunks {
            // Calculate normalized content for comparison
            let normalized = chunk.content.to_lowercase();
            
            // Check if this is a duplicate
            if !seen_content.contains(&normalized) {
                seen_content.insert(normalized);
                unique_chunks.push(chunk);
            } else {
                warn!("Removed duplicate chunk at index {}", chunk.chunk_order_index);
            }
        }
        
        chunks = unique_chunks;
    }
    
    Ok(chunks)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_chunk_text_with_smart_boundary() {
        let config = ChunkingConfig {
            max_token_size: 100,
            overlap_token_size: 20,
            smart_boundary: true,
            split_by_character: Some("\n\n".to_string()),
            min_chunk_size: 1,
            ..Default::default()
        };
        
        let text = "This is sentence one. This is sentence two.\n\nThis is a new paragraph. And another sentence.\n\nFinal paragraph here.";
        let chunks = chunk_text(text, &config, "test-doc").await.unwrap();
        
        assert!(chunks.len() > 0);
        // Verify chunks end at sentence or paragraph boundaries
        for chunk in chunks {
            assert!(chunk.content.ends_with('.') || chunk.content.ends_with('\n'));
        }
    }

    #[tokio::test]
    async fn test_chunk_deduplication() {
        let config = ChunkingConfig {
            enable_deduplication: true,
            dedup_similarity_threshold: 0.85,
            split_by_character: Some("\n".to_string()),
            min_chunk_size: 1,
            ..Default::default()
        };
        
        let text = "Duplicate text here.\nDuplicate text here.\nUnique text.\nDuplicate text here.";
        let chunks = chunk_text(text, &config, "test-doc").await.unwrap();
        
        // Should only have two chunks after deduplication
        assert_eq!(chunks.len(), 2);
    }
} 