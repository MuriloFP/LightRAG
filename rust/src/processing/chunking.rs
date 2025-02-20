use crate::processing::types::{TextChunk, ChunkingError};
use tiktoken_rs::cl100k_base;
use tracing::debug;

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
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            overlap_token_size: 100,
            max_token_size: 1200,
            split_by_character: None,
            split_by_character_only: false,
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
                    chunks.push(TextChunk {
                        tokens: tokens.len(),
                        content: chunk_content.to_string(),
                        full_doc_id: doc_id.to_string(),
                        chunk_order_index: idx,
                    });
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
                        config.max_token_size,
                        config.overlap_token_size,
                    )?;
                    
                    for (sub_idx, sub_chunk) in sub_chunks.into_iter().enumerate() {
                        chunks.push(TextChunk {
                            tokens: sub_chunk.tokens,
                            content: sub_chunk.content,
                            full_doc_id: doc_id.to_string(),
                            chunk_order_index: idx * 1000 + sub_idx, // Preserve order with sub-indexing
                        });
                    }
                } else {
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
            config.max_token_size,
            config.overlap_token_size,
        )?;
        
        // Set chunk order and document ID
        for (idx, chunk) in chunks.iter_mut().enumerate() {
            chunk.chunk_order_index = idx;
            chunk.full_doc_id = doc_id.to_string();
        }
    }

    debug!(
        "Created {} chunks from document {}",
        chunks.len(),
        doc_id
    );

    Ok(chunks)
}

/// Helper function to split text by tokens
fn split_by_tokens(
    content: &str,
    bpe: &tiktoken_rs::CoreBPE,
    max_token_size: usize,
    overlap_token_size: usize,
) -> Result<Vec<TextChunk>, ChunkingError> {
    let tokens = bpe.encode_with_special_tokens(content);
    let mut chunks = Vec::new();
    
    // Calculate chunk positions
    for start in (0..tokens.len()).step_by(max_token_size - overlap_token_size) {
        let end = (start + max_token_size).min(tokens.len());
        let chunk_tokens = &tokens[start..end];
        
        let chunk_content = bpe.decode(chunk_tokens.to_vec())
            .map_err(|e| ChunkingError::TokenizationError(e.to_string()))?;
            
        chunks.push(TextChunk {
            tokens: chunk_tokens.len(),
            content: chunk_content,
            full_doc_id: String::new(), // Will be set by caller
            chunk_order_index: 0, // Will be set by caller
        });

        if end == tokens.len() {
            break;
        }
    }

    Ok(chunks)
} 