//! Document processing functionality
//! 
//! This module provides functionality for processing documents, including:
//! - Document status management
//! - Text chunking
//! - Content processing
//! - Document format handling

mod status;
mod chunking;
mod types;

pub use status::{
    DocumentStatus,
    DocumentMetadata,
    DocumentStatusError,
    DocumentStatusManager,
    InMemoryStatusManager,
};

pub use chunking::{
    ChunkingConfig,
    chunk_text,
};

pub use types::{
    TextChunk,
    ChunkingError,
};

// Future processing functions and types will be added here. 