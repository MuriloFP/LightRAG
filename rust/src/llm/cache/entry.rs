use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use serde::{Serialize, Deserialize};
use crate::llm::LLMError;
use crate::llm::LLMResponse;
use lz4_flex::{compress_prepend_size, decompress_size_prepended};

/// Entry in the cache
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// The cached response
    pub response: LLMResponse,
    
    /// When the entry was created
    pub created_at: SystemTime,
    
    /// When the entry expires
    pub expires_at: Option<SystemTime>,

    /// Embedding of the prompt for similarity search
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,

    /// Number of times this entry has been accessed
    pub access_count: u64,

    /// Last access time
    pub last_accessed: SystemTime,

    /// Whether this entry has been LLM verified
    pub llm_verified: bool,

    /// Checksum for integrity validation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checksum: Option<String>,

    /// Compressed data if compression is enabled
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compressed_data: Option<Vec<u8>>,

    /// Original size before compression
    #[serde(skip_serializing_if = "Option::is_none")]
    pub original_size: Option<usize>,

    /// Metadata for custom tracking
    pub metadata: HashMap<String, String>,
}

impl CacheEntry {
    /// Create a new cache entry
    pub fn new(response: LLMResponse, ttl: Option<Duration>) -> Self {
        let now = SystemTime::now();
        Self {
            response,
            created_at: now,
            expires_at: ttl.map(|ttl| now + ttl),
            embedding: None,
            access_count: 0,
            last_accessed: now,
            llm_verified: false,
            checksum: None,
            compressed_data: None,
            original_size: None,
            metadata: HashMap::new(),
        }
    }

    /// Update access metrics
    pub fn record_access(&mut self) {
        self.access_count += 1;
        self.last_accessed = SystemTime::now();
    }

    /// Check if the entry is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            SystemTime::now() > expires_at
        } else {
            false
        }
    }

    /// Set the embedding for similarity search
    pub fn set_embedding(&mut self, embedding: Vec<f32>) {
        self.embedding = Some(embedding);
    }

    /// Compress the response data
    pub fn compress(&mut self) -> Result<(), LLMError> {
        if self.compressed_data.is_none() {
            let serialized = serde_json::to_vec(&self.response)
                .map_err(|e| LLMError::CacheError(format!("Failed to serialize response: {}", e)))?;
            
            self.original_size = Some(serialized.len());
            
            let compressed = compress_prepend_size(&serialized);
            self.compressed_data = Some(compressed);
        }
        Ok(())
    }

    /// Decompress the response data
    pub fn decompress(&mut self) -> Result<&LLMResponse, LLMError> {
        if let Some(compressed) = &self.compressed_data {
            if self.original_size.is_none() {
                return Err(LLMError::CacheError("Missing original size information".to_string()));
            }

            // If already decompressed, return the response directly
            if !self.response.text.is_empty() {
                return Ok(&self.response);
            }

            let decompressed = decompress_size_prepended(compressed)
                .map_err(|e| LLMError::CacheError(format!("Failed to decompress data: {}", e)))?;

            let response: LLMResponse = serde_json::from_slice(&decompressed)
                .map_err(|e| LLMError::CacheError(format!("Failed to deserialize response: {}", e)))?;

            // Update the response field with decompressed data
            self.response = response;
            Ok(&self.response)
        } else {
            Ok(&self.response)
        }
    }

    /// Calculate and set checksum for integrity validation
    pub fn set_checksum(&mut self) -> Result<(), LLMError> {
        let serialized = serde_json::to_vec(&self.response)
            .map_err(|e| LLMError::CacheError(format!("Failed to serialize response: {}", e)))?;
        
        use sha2::{Sha256, Digest};
        let mut hasher = Sha256::new();
        hasher.update(&serialized);
        let result = hasher.finalize();
        
        self.checksum = Some(format!("{:x}", result));
        Ok(())
    }

    /// Validate entry integrity
    pub fn validate_integrity(&self) -> Result<bool, LLMError> {
        if let Some(stored_checksum) = &self.checksum {
            let serialized = serde_json::to_vec(&self.response)
                .map_err(|e| LLMError::CacheError(format!("Failed to serialize response: {}", e)))?;
            
            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            hasher.update(&serialized);
            let result = hasher.finalize();
            let current_checksum = format!("{:x}", result);
            
            Ok(current_checksum == *stored_checksum)
        } else {
            Ok(true)
        }
    }
} 