use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Represents a chunk of text with associated metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextChunk {
    /// Number of tokens in the chunk
    pub tokens: usize,
    /// The actual text content
    pub content: String,
    /// ID of the full document this chunk belongs to
    pub full_doc_id: String,
    /// Order index of this chunk in the original document
    pub chunk_order_index: usize,
}

/// Errors that can occur during text chunking
#[derive(Error, Debug)]
pub enum ChunkingError {
    /// Error during tokenization
    #[error("Tokenization error: {0}")]
    TokenizationError(String),
    
    /// Error when input text is empty
    #[error("Empty input text")]
    EmptyInput,
    
    /// Error when chunk size is invalid
    #[error("Invalid chunk size: {0}")]
    InvalidChunkSize(String),
    
    /// Error during text processing
    #[error("Text processing error: {0}")]
    ProcessingError(String),
} 