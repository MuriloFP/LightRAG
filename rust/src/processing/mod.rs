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

/// Document format handling and text extraction.
/// 
/// This module provides functionality for:
/// - Detecting document formats based on file extensions
/// - Extracting text content from various document types
/// - Supporting multiple formats including:
///   - Plain text (.txt)
///   - Markdown (.md)
///   - PDF (.pdf)
///   - Word documents (.docx)
pub mod formats;

/// Content cleaning and normalization.
/// 
/// This module provides functionality for:
/// - HTML entity unescaping
/// - Control character removal
/// - Whitespace normalization
/// - UTF-8 validation
pub mod cleaning;

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

pub use formats::{
    DocumentFormat,
    FormatHandler,
    FormatError,
    detect_format,
    get_format_handler,
};

// Future processing functions and types will be added here. 