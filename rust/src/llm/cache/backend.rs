use async_trait::async_trait;
use std::error::Error;
use std::fmt;
use serde::{Serialize, Deserialize};
use crate::llm::LLMError;
use super::types::{CacheEntry, CacheStats, CacheType, StorageConfig};
use std::any::Any;

/// Cache backend errors
#[derive(Debug)]
pub enum CacheError {
    /// Storage initialization failed
    InitError(String),
    
    /// Storage operation failed
    StorageError(String),
    
    /// Entry not found
    NotFound,
    
    /// Entry expired
    Expired,
    
    /// Storage quota exceeded
    QuotaExceeded,
    
    /// Invalid data
    InvalidData(String),
    
    /// Backend not available
    Unavailable(String),

    /// Compression failed
    CompressionError(String),

    /// Decompression failed
    DecompressionError(String),

    /// Encryption failed
    EncryptionError(String),

    /// Decryption failed
    DecryptionError(String),

    /// Invalid encryption key
    InvalidKey(String),

    /// Encryption not initialized
    EncryptionNotInitialized,

    /// Unsupported algorithm
    UnsupportedAlgorithm(String),

    /// Key derivation error
    KeyDerivationError(String),

    /// Operation not supported by this backend
    UnsupportedOperation(String),
}

impl fmt::Display for CacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitError(msg) => write!(f, "Cache initialization error: {}", msg),
            Self::StorageError(msg) => write!(f, "Storage error: {}", msg),
            Self::NotFound => write!(f, "Cache entry not found"),
            Self::Expired => write!(f, "Cache entry expired"),
            Self::QuotaExceeded => write!(f, "Storage quota exceeded"),
            Self::InvalidData(msg) => write!(f, "Invalid cache data: {}", msg),
            Self::Unavailable(msg) => write!(f, "Cache backend unavailable: {}", msg),
            Self::CompressionError(msg) => write!(f, "Compression error: {}", msg),
            Self::DecompressionError(msg) => write!(f, "Decompression error: {}", msg),
            Self::EncryptionError(msg) => write!(f, "Encryption error: {}", msg),
            Self::DecryptionError(msg) => write!(f, "Decryption error: {}", msg),
            Self::InvalidKey(msg) => write!(f, "Invalid encryption key: {}", msg),
            Self::EncryptionNotInitialized => write!(f, "Encryption not initialized"),
            Self::UnsupportedAlgorithm(msg) => write!(f, "Unsupported algorithm: {}", msg),
            Self::KeyDerivationError(msg) => write!(f, "Key derivation error: {}", msg),
            Self::UnsupportedOperation(msg) => write!(f, "Operation not supported: {}", msg),
        }
    }
}

impl Error for CacheError {}

impl From<CacheError> for LLMError {
    fn from(err: CacheError) -> Self {
        LLMError::CacheError(err.to_string())
    }
}

impl From<tokio_rusqlite::Error> for CacheError {
    fn from(err: tokio_rusqlite::Error) -> Self {
        CacheError::StorageError(err.to_string())
    }
}

impl From<rusqlite::Error> for CacheError {
    fn from(err: rusqlite::Error) -> Self {
        CacheError::StorageError(err.to_string())
    }
}

impl From<CacheError> for tokio_rusqlite::Error {
    fn from(err: CacheError) -> Self {
        tokio_rusqlite::Error::Other(Box::new(err))
    }
}

/// Cache backend capabilities
#[derive(Debug, Clone, Copy)]
pub struct CacheCapabilities {
    /// Supports persistence
    pub persistent: bool,
    
    /// Supports streaming
    pub streaming: bool,
    
    /// Supports compression
    pub compression: bool,
    
    /// Supports encryption
    pub encryption: bool,
    
    /// Supports transactions
    pub transactions: bool,
    
    /// Supports pub/sub
    pub pubsub: bool,
}

/// Supported encryption algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    /// AES-256-GCM (recommended)
    Aes256Gcm,
    /// ChaCha20-Poly1305
    ChaCha20Poly1305,
}

impl fmt::Display for EncryptionAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncryptionAlgorithm::Aes256Gcm => write!(f, "AES-256-GCM"),
            EncryptionAlgorithm::ChaCha20Poly1305 => write!(f, "ChaCha20-Poly1305"),
        }
    }
}

/// Unified cache backend interface
#[async_trait]
pub trait CacheBackend: Send + Sync + Any {
    /// Get the backend type
    fn backend_type(&self) -> CacheType;
    
    /// Get backend capabilities
    fn capabilities(&self) -> CacheCapabilities;
    
    /// Initialize the backend
    async fn initialize(&mut self, config: StorageConfig) -> Result<(), CacheError>;
    
    /// Get cache entry
    async fn get(&self, key: &str) -> Result<CacheEntry, CacheError>;
    
    /// Set cache entry
    async fn set(&self, entry: CacheEntry) -> Result<(), CacheError>;
    
    /// Delete cache entry
    async fn delete(&self, key: &str) -> Result<(), CacheError>;
    
    /// Check if key exists
    async fn exists(&self, key: &str) -> Result<bool, CacheError>;
    
    /// Get multiple entries
    async fn get_many(&self, keys: &[String]) -> Result<Vec<Option<CacheEntry>>, CacheError>;
    
    /// Set multiple entries
    async fn set_many(&self, entries: Vec<CacheEntry>) -> Result<(), CacheError>;
    
    /// Delete multiple entries
    async fn delete_many(&self, keys: &[String]) -> Result<(), CacheError>;
    
    /// Clear all entries
    async fn clear(&self) -> Result<(), CacheError>;
    
    /// Get cache statistics
    async fn stats(&self) -> Result<CacheStats, CacheError>;
    
    /// Cleanup expired entries
    async fn cleanup(&self) -> Result<(), CacheError>;
    
    /// Optimize storage (defragmentation, etc)
    async fn optimize(&self) -> Result<(), CacheError>;
    
    /// Backup cache data
    async fn backup(&self, path: &str) -> Result<(), CacheError>;
    
    /// Restore cache data
    async fn restore(&self, path: &str) -> Result<(), CacheError>;
    
    /// Check backend health
    async fn health_check(&self) -> Result<(), CacheError>;

    /// Get entries with priority above the threshold
    async fn get_high_priority_entries(&self, min_priority: u32, max_items: usize) -> Result<Vec<CacheEntry>, CacheError> {
        Err(CacheError::UnsupportedOperation("get_high_priority_entries not implemented".to_string()))
    }

    /// Get a batch of entries starting from an offset
    async fn get_entries_batch(&self, offset: usize, limit: usize) -> Result<Vec<CacheEntry>, CacheError>;

    /// Get this object as Any
    fn as_any(&self) -> &dyn Any where Self: Sized {
        self
    }

    /// Get this object as mutable Any
    fn as_any_mut(&mut self) -> &mut dyn Any where Self: Sized {
        self
    }

    /// Try to get this backend as a compressible cache
    fn as_compressible(&self) -> Option<&dyn CompressibleCache> {
        None
    }

    /// Try to get this backend as a mutable compressible cache
    fn as_mut_compressible(&mut self) -> Option<&mut dyn CompressibleCache> {
        None
    }

    /// Try to get this backend as an encryptable cache
    fn as_encryptable(&self) -> Option<&dyn EncryptableCache> {
        None
    }

    /// Try to get this backend as a mutable encryptable cache
    fn as_mut_encryptable(&mut self) -> Option<&mut dyn EncryptableCache> {
        None
    }
}

/// Helper trait for streaming support
#[async_trait]
pub trait StreamingCache: CacheBackend {
    /// Get streaming entry
    async fn get_stream(&self, key: &str) -> Result<Vec<u8>, CacheError>;
    
    /// Set streaming entry
    async fn set_stream(&self, key: &str, data: Vec<u8>) -> Result<(), CacheError>;
}

/// Helper trait for compression support
#[async_trait]
pub trait CompressibleCache: CacheBackend {
    /// Get compression level (0-9)
    fn compression_level(&self) -> u32;
    
    /// Set compression level
    fn set_compression_level(&mut self, level: u32);
    
    /// Compress data
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, CacheError>;
    
    /// Decompress data
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, CacheError>;
}

/// Helper trait for encryption support
#[async_trait]
pub trait EncryptableCache: Send + Sync {
    async fn encrypt(&self, data: Vec<u8>, algorithm: EncryptionAlgorithm) -> Result<Vec<u8>, CacheError>;
    async fn decrypt(&self, data: Vec<u8>, algorithm: EncryptionAlgorithm) -> Result<Vec<u8>, CacheError>;
    fn is_encryption_ready(&self) -> bool;
    fn supports_encryption(&self) -> bool;
    fn supported_algorithms(&self) -> Vec<EncryptionAlgorithm>;
}

/// Helper trait for transaction support
#[async_trait]
pub trait TransactionalCache: CacheBackend {
    /// Start transaction
    async fn begin_transaction(&self) -> Result<(), CacheError>;
    
    /// Commit transaction
    async fn commit_transaction(&self) -> Result<(), CacheError>;
    
    /// Rollback transaction
    async fn rollback_transaction(&self) -> Result<(), CacheError>;
}

/// Helper trait for pub/sub support
#[async_trait]
pub trait PubSubCache: CacheBackend {
    /// Subscribe to cache events
    async fn subscribe(&self, pattern: &str) -> Result<(), CacheError>;
    
    /// Unsubscribe from cache events
    async fn unsubscribe(&self, pattern: &str) -> Result<(), CacheError>;
    
    /// Publish cache event
    async fn publish(&self, channel: &str, message: &[u8]) -> Result<(), CacheError>;
}

/// Helper trait for downcasting cache backends
pub trait CacheBackendExt: CacheBackend {
    fn as_compressible(&self) -> Option<&dyn CompressibleCache> {
        None
    }

    fn as_mut_compressible(&mut self) -> Option<&mut dyn CompressibleCache> {
        None
    }

    fn as_encryptable(&self) -> Option<&dyn EncryptableCache> {
        None
    }

    fn as_mut_encryptable(&mut self) -> Option<&mut dyn EncryptableCache> {
        None
    }
}

impl<T: CacheBackend> CacheBackendExt for T {} 