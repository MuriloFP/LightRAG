use std::fmt;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use super::backend::{EncryptionAlgorithm, EncryptableCache, CacheError};
use async_trait::async_trait;

/// Cache backend types supported by the system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CacheType {
    /// In-memory cache (available in all environments)
    Memory,
    
    /// SQLite-based cache (native environments)
    SQLite,
    
    /// Redis-based cache (server environment)
    #[cfg(feature = "redis")]
    Redis,
    
    /// IndexedDB-based cache (browser environment)
    #[cfg(target_arch = "wasm32")]
    IndexedDB,
    
    /// LocalStorage-based cache (browser environment)
    #[cfg(target_arch = "wasm32")]
    LocalStorage,
    
    /// Tiered cache
    Tiered,
}

impl fmt::Display for CacheType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CacheType::Memory => write!(f, "memory"),
            CacheType::SQLite => write!(f, "sqlite"),
            #[cfg(feature = "redis")]
            CacheType::Redis => write!(f, "redis"),
            #[cfg(target_arch = "wasm32")]
            CacheType::IndexedDB => write!(f, "indexeddb"),
            #[cfg(target_arch = "wasm32")]
            CacheType::LocalStorage => write!(f, "localstorage"),
            CacheType::Tiered => write!(f, "tiered"),
        }
    }
}

impl CacheType {
    /// Get string representation of cache type
    pub fn as_str(&self) -> &'static str {
        match self {
            CacheType::Memory => "memory",
            CacheType::SQLite => "sqlite",
            #[cfg(feature = "redis")]
            CacheType::Redis => "redis",
            #[cfg(target_arch = "wasm32")]
            CacheType::IndexedDB => "indexeddb",
            #[cfg(target_arch = "wasm32")]
            CacheType::LocalStorage => "localstorage",
            CacheType::Tiered => "tiered",
        }
    }
}

impl Default for CacheType {
    fn default() -> Self {
        Self::Memory
    }
}

/// Cache storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Maximum memory usage in megabytes
    pub max_memory_mb: u32,
    
    /// Maximum storage size in megabytes
    pub max_storage_mb: u32,
    
    /// Path for persistent storage (if applicable)
    pub storage_path: Option<String>,
    
    /// Connection string (for Redis)
    #[cfg(feature = "redis")]
    pub connection_string: Option<String>,

    /// Compression level (0-9, 0 = disabled)
    pub compression_level: Option<u32>,
    
    /// Whether to use compression
    pub use_compression: bool,

    /// Whether to use encryption
    pub use_encryption: bool,

    /// Encryption key (if encryption is enabled)
    #[serde(skip_serializing)]
    pub encryption_key: Option<Vec<u8>>,

    /// Encryption algorithm to use
    pub encryption_algorithm: Option<EncryptionAlgorithm>,
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: 32,  // Conservative default
            max_storage_mb: 1024, // 1GB default
            storage_path: None,
            #[cfg(feature = "redis")]
            connection_string: None,
            compression_level: Some(4), // Default compression level
            use_compression: true,
            use_encryption: false,
            encryption_key: None,
            encryption_algorithm: None,
        }
    }
}

/// Cache entry metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    /// When the entry was created
    pub created_at: std::time::SystemTime,
    
    /// When the entry expires
    pub expires_at: Option<std::time::SystemTime>,
    
    /// Size of the entry in bytes
    pub size_bytes: usize,
    
    /// Number of times the entry was accessed
    pub access_count: u64,
    
    /// Last access time
    pub last_accessed: std::time::SystemTime,
    
    /// Custom metadata
    pub extra: std::collections::HashMap<String, String>,
}

impl CacheMetadata {
    pub fn new(ttl: Option<Duration>, size_bytes: usize) -> Self {
        let now = std::time::SystemTime::now();
        Self {
            created_at: now,
            expires_at: ttl.map(|d| now + d),
            size_bytes,
            access_count: 0,
            last_accessed: now,
            extra: std::collections::HashMap::new(),
        }
    }
    
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            std::time::SystemTime::now() > expires_at
        } else {
            false
        }
    }
}

/// Cache statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CacheStats {
    /// Number of items in cache
    pub item_count: usize,
    
    /// Total size in bytes
    pub total_size_bytes: usize,
    
    /// Number of cache hits
    pub hits: u64,
    
    /// Number of cache misses
    pub misses: u64,
    
    /// Number of evicted items
    pub evictions: u64,
    
    /// Number of expired items
    pub expirations: u64,
}

/// Cache entry value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheValue {
    /// Regular response
    Response(crate::llm::LLMResponse),
    
    /// Streaming response chunks
    Stream(Vec<crate::types::llm::StreamingResponse>),
    
    /// Raw bytes
    Bytes(Vec<u8>),
}

/// Priority level for cache entries
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CachePriority {
    Low,
    Normal,
    High,
    Critical,
}

impl Default for CachePriority {
    fn default() -> Self {
        Self::Normal
    }
}

/// Cache entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// Entry key
    pub key: String,
    
    /// Cached value
    pub value: CacheValue,
    
    /// Entry metadata
    pub metadata: CacheMetadata,
    
    /// Entry priority
    pub priority: CachePriority,

    /// Whether the value is encrypted
    pub is_encrypted: bool,
}

impl CacheEntry {
    pub fn new(
        key: String,
        value: CacheValue,
        ttl: Option<Duration>,
        priority: Option<CachePriority>,
    ) -> Self {
        let size_bytes = match &value {
            CacheValue::Response(r) => bincode::serialize(r).map(|b| b.len()).unwrap_or(0),
            CacheValue::Stream(s) => bincode::serialize(s).map(|b| b.len()).unwrap_or(0),
            CacheValue::Bytes(b) => b.len(),
        };
        
        Self {
            key,
            value,
            metadata: CacheMetadata::new(ttl, size_bytes),
            priority: priority.unwrap_or_default(),
            is_encrypted: false,
        }
    }
}

#[async_trait]
pub trait EncryptedEntry: Send {
    async fn new_encrypted(
        key: String,
        value: CacheValue,
        ttl: Option<Duration>,
        priority: Option<CachePriority>,
        cache: &(impl EncryptableCache + Send + Sync),
        algorithm: EncryptionAlgorithm,
    ) -> Result<CacheEntry, CacheError>;

    async fn decrypt(&mut self, cache: &(impl EncryptableCache + Send + Sync), algorithm: EncryptionAlgorithm) -> Result<(), CacheError>;
}

#[async_trait]
impl EncryptedEntry for CacheEntry {
    async fn new_encrypted(
        key: String,
        value: CacheValue,
        ttl: Option<Duration>,
        priority: Option<CachePriority>,
        cache: &(impl EncryptableCache + Send + Sync),
        algorithm: EncryptionAlgorithm,
    ) -> Result<CacheEntry, CacheError> {
        let mut entry = Self::new(key, value, ttl, priority);
        
        // Serialize and encrypt the value
        let data = bincode::serialize(&entry.value)
            .map_err(|e| CacheError::InvalidData(e.to_string()))?;
        let encrypted = cache.encrypt(data, algorithm).await?;
        entry.value = CacheValue::Bytes(encrypted);
        entry.is_encrypted = true;
        
        Ok(entry)
    }

    async fn decrypt(&mut self, cache: &(impl EncryptableCache + Send + Sync), algorithm: EncryptionAlgorithm) -> Result<(), CacheError> {
        if !self.is_encrypted {
            return Ok(());
        }

        if let CacheValue::Bytes(ref encrypted) = self.value {
            let decrypted = cache.decrypt(encrypted.to_vec(), algorithm).await?;
            let value = bincode::deserialize(&decrypted)
                .map_err(|e| CacheError::InvalidData(e.to_string()))?;
            self.value = value;
            self.is_encrypted = false;
            Ok(())
        } else {
            Err(CacheError::InvalidData("Expected encrypted bytes".into()))
        }
    }
} 