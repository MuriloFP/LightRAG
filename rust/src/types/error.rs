use thiserror::Error;
use crate::llm::cache::backend::CacheError;
use crate::llm::LLMError;
use crate::processing::SummaryError;
use crate::processing::ChunkingError;
use crate::types::embeddings::EmbeddingError;
use crate::processing::DocumentStatusError;
use tokio::sync::AcquireError;

#[derive(Debug, Error)]
pub enum Error {
    /// IO errors
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization errors
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Storage errors
    #[error("Storage error: {0}")]
    Storage(String),

    /// Vector storage specific errors
    #[error("Vector storage error: {0}")]
    VectorStorage(String),

    /// API errors
    #[error("API error: {0}")]
    Api(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    Config(String),

    /// Invalid input errors
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// XML errors
    #[error("XML error: {0}")]
    Xml(#[from] quick_xml::Error),

    /// Cache errors
    #[error("Cache error: {0}")]
    Cache(CacheError),

    /// LLM errors
    #[error("LLM error: {0}")]
    LLM(LLMError),

    /// Other errors
    #[error("Other error: {0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Error>;

// Manual implementations for From traits
impl From<CacheError> for Error {
    fn from(err: CacheError) -> Self {
        Error::Cache(err)
    }
}

impl From<LLMError> for Error {
    fn from(err: LLMError) -> Self {
        Error::LLM(err)
    }
}

// Implement conversion from Error to CacheError
impl From<Error> for CacheError {
    fn from(err: Error) -> Self {
        match err {
            Error::Cache(cache_err) => cache_err,
            Error::Storage(msg) => CacheError::StorageError(msg),
            Error::Io(io_err) => CacheError::StorageError(io_err.to_string()),
            Error::Json(json_err) => CacheError::InvalidData(json_err.to_string()),
            _ => CacheError::StorageError(err.to_string()),
        }
    }
}

// Implement conversion from Error to LLMError
impl From<Error> for LLMError {
    fn from(err: Error) -> Self {
        match err {
            Error::LLM(llm_err) => llm_err,
            Error::Api(msg) => LLMError::RequestFailed(msg),
            Error::Cache(cache_err) => LLMError::CacheError(cache_err.to_string()),
            _ => LLMError::Other(err.to_string()),
        }
    }
}

// Add From implementation for SummaryError
impl From<SummaryError> for Error {
    fn from(err: SummaryError) -> Self {
        match err {
            SummaryError::EmptyContent => Error::InvalidInput("Empty content for summary".to_string()),
            SummaryError::TokenizationError(msg) => Error::InvalidInput(format!("Tokenization error: {}", msg)),
            SummaryError::GenerationError(msg) => Error::LLM(LLMError::Other(format!("Summary generation error: {}", msg))),
            SummaryError::GenerationFailed(msg) => Error::LLM(LLMError::Other(format!("Summary generation failed: {}", msg))),
            SummaryError::KeywordExtractionFailed(msg) => Error::LLM(LLMError::Other(format!("Keyword extraction failed: {}", msg))),
        }
    }
}

// Add these implementations
impl From<DocumentStatusError> for Error {
    fn from(err: DocumentStatusError) -> Self {
        Error::Other(format!("Document status error: {}", err))
    }
}

impl From<ChunkingError> for Error {
    fn from(err: ChunkingError) -> Self {
        Error::Other(format!("Chunking error: {}", err))
    }
}

impl From<EmbeddingError> for Error {
    fn from(err: EmbeddingError) -> Self {
        Error::Other(format!("Embedding error: {}", err))
    }
}

impl From<AcquireError> for Error {
    fn from(err: AcquireError) -> Self {
        Error::Other(format!("Semaphore acquire error: {}", err))
    }
} 