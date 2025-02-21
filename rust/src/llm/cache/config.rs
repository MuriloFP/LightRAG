use std::time::Duration;
use serde::{Deserialize, Serialize};

/// Cache backend type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CacheBackend {
    /// In-memory storage
    InMemory,
    /// Distributed storage
    Distributed(DistributedConfig),
    /// Redis storage
    Redis(RedisConfig),
}

/// Configuration for distributed cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Node addresses
    pub nodes: Vec<String>,
    /// Replication factor
    pub replication_factor: usize,
    /// Consistency level
    pub consistency_level: ConsistencyLevel,
}

/// Consistency level for distributed operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsistencyLevel {
    /// One node
    One,
    /// Quorum of nodes
    Quorum,
    /// All nodes
    All,
}

/// Configuration for Redis cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RedisConfig {
    /// Redis URL
    pub url: String,
    /// Connection pool size
    pub pool_size: usize,
}

/// Configuration for response caching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether caching is enabled
    pub enabled: bool,

    /// Time-to-live for cache entries
    pub ttl: Option<Duration>,

    /// Maximum number of entries to store
    pub max_entries: Option<usize>,

    /// Whether to enable similarity search
    pub similarity_enabled: bool,

    /// Similarity threshold for fuzzy matching (0.0 to 1.0)
    pub similarity_threshold: f32,

    /// Whether to enable streaming response caching
    pub stream_cache_enabled: bool,

    /// Maximum number of chunks to store per streaming response
    pub max_stream_chunks: Option<usize>,

    /// Whether to compress streaming chunks
    pub compress_streams: bool,

    /// Maximum age for streaming cache entries (separate from regular TTL)
    pub stream_ttl: Option<Duration>,

    /// Prefix for cache keys
    pub prefix: String,

    /// Whether to use fuzzy matching for prompts
    pub use_fuzzy_match: bool,

    /// Whether to use persistent storage
    pub use_persistent: bool,

    /// Whether to use LLM verification for cache hits
    pub use_llm_verification: bool,

    /// Prompt template for LLM verification
    pub llm_verification_prompt: Option<String>,

    /// Cache backend type
    pub backend: CacheBackend,

    /// Eviction strategy
    pub eviction_strategy: EvictionStrategy,

    /// Whether to enable metrics collection
    pub enable_metrics: bool,

    /// Whether to compress cached data
    pub use_compression: bool,

    /// Maximum size of compressed data in bytes
    pub max_compressed_size: Option<usize>,

    /// Whether to encrypt cached data
    pub use_encryption: bool,

    /// Encryption key if encryption is enabled
    pub encryption_key: Option<String>,

    /// Whether to validate cache integrity
    pub validate_integrity: bool,

    /// Whether to enable cache synchronization
    pub enable_sync: bool,

    /// Sync interval in seconds
    pub sync_interval: Option<Duration>,
}

/// Cache eviction strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvictionStrategy {
    /// Least Recently Used
    LRU,
    /// Least Frequently Used
    LFU,
    /// First In First Out
    FIFO,
    /// Random eviction
    Random,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl: Some(Duration::from_secs(3600)), // 1 hour
            max_entries: Some(1000),
            similarity_enabled: false,
            similarity_threshold: 0.8,
            stream_cache_enabled: true,
            max_stream_chunks: Some(1000),
            compress_streams: false,
            stream_ttl: Some(Duration::from_secs(1800)), // 30 minutes
            prefix: "cache".to_string(),
            use_fuzzy_match: true,
            use_persistent: false,
            use_llm_verification: false,
            llm_verification_prompt: None,
            backend: CacheBackend::InMemory,
            eviction_strategy: EvictionStrategy::LRU,
            enable_metrics: true,
            use_compression: false,
            max_compressed_size: None,
            use_encryption: false,
            encryption_key: None,
            validate_integrity: true,
            enable_sync: false,
            sync_interval: None,
        }
    }
} 