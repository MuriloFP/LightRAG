use std::collections::HashMap;
use std::time::{SystemTime, Duration};
use crate::types::embeddings::{EmbeddingResponse, CacheConfig, EmbeddingError};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};

/// Statistics for compression and quantization
#[derive(Debug, Default)]
pub struct CacheStats {
    /// Total number of entries
    pub total_entries: usize,
    /// Total original size in bytes
    pub total_original_size: usize,
    /// Total compressed size in bytes
    pub total_compressed_size: usize,
    /// Number of entries using compression
    pub compressed_entries: usize,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Number of entries using quantization
    pub quantized_entries: usize,
    /// Average quantization error
    pub avg_quantization_error: f32,
}

/// Entry in the embedding cache
#[derive(Clone)]
struct CacheEntry {
    /// The cached response
    response: EmbeddingResponse,
    
    /// When the entry was created
    timestamp: SystemTime,

    /// Quantized embedding data
    quantized_embedding: Option<Vec<u8>>,
    
    /// Original min value for dequantization
    min_val: Option<f32>,
    
    /// Original max value for dequantization
    max_val: Option<f32>,

    /// Compressed data
    compressed_data: Option<Vec<u8>>,

    /// Original size before compression
    original_size: Option<usize>,

    /// Quantization error for this entry
    quantization_error: Option<f32>,
}

/// Cache for embedding responses with similarity-based lookup
pub struct EmbeddingCache {
    /// Cache entries
    entries: HashMap<String, CacheEntry>,
    
    /// Cache configuration
    config: CacheConfig,

    /// Cache statistics
    stats: CacheStats,
}

impl EmbeddingCache {
    /// Create a new embedding cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
            stats: CacheStats::default(),
        }
    }
    
    /// Get cache statistics
    pub fn get_stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get a cached response
    pub fn get(&self, text: &str) -> Option<EmbeddingResponse> {
        if !self.config.enabled {
            return None;
        }
        
        // Check for exact match first
        if let Some(entry) = self.entries.get(text) {
            // Check TTL
            if let Some(ttl) = self.config.ttl_seconds {
                if entry.timestamp.elapsed().unwrap().as_secs() > ttl {
                    return None;
                }
            }
            return Some(entry.response.clone());
        }
        
        None
    }

    /// Calculate optimal number of quantization bits based on embedding distribution
    fn calculate_optimal_bits(&self, embedding: &[f32]) -> u8 {
        if !self.config.use_quantization {
            return 32; // No quantization
        }

        // Calculate variance of the embedding
        let mean: f32 = embedding.iter().sum::<f32>() / embedding.len() as f32;
        let variance: f32 = embedding.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f32>() / embedding.len() as f32;

        // Adjust bits based on variance - more variance needs more bits
        let base_bits = self.config.quantization_bits;
        let variance_factor = (variance * 10.0) as u8;
        
        // Clamp between config bits and 16 bits
        (base_bits + variance_factor).clamp(base_bits, 16)
    }

    /// Quantize a floating-point embedding to n-bit integers
    fn quantize_embedding(&self, embedding: &[f32]) -> Result<(Vec<u8>, f32, f32, f32), EmbeddingError> {
        if !self.config.use_quantization {
            // If quantization is disabled, just convert to bytes
            let bytes: Vec<u8> = embedding.iter()
                .flat_map(|x| x.to_le_bytes().to_vec())
                .collect();
            return Ok((bytes, 0.0, 0.0, 0.0));
        }

        let bits = self.calculate_optimal_bits(embedding);
        let min_val = embedding.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = embedding.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        if min_val == max_val {
            return Err(EmbeddingError::QuantizationError("Zero variance in embedding".to_string()));
        }

        let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
        
        let quantized: Vec<u8> = embedding.iter()
            .map(|&x| ((x - min_val) / scale) as u8)
            .collect();

        // Calculate quantization error
        let dequantized = self.dequantize_embedding(&quantized, min_val, max_val)?;
        let error = embedding.iter()
            .zip(dequantized.iter())
            .map(|(&orig, &deq)| (orig - deq).powi(2))
            .sum::<f32>()
            .sqrt() / embedding.len() as f32;
            
        Ok((quantized, min_val, max_val, error))
    }
    
    /// Dequantize an n-bit integer embedding back to floating-point
    fn dequantize_embedding(&self, quantized: &[u8], min_val: f32, max_val: f32) -> Result<Vec<f32>, EmbeddingError> {
        if !self.config.use_quantization {
            // If quantization is disabled, just convert from bytes
            return Ok(quantized.chunks(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect());
        }

        if min_val == max_val {
            return Err(EmbeddingError::QuantizationError("Invalid min/max values".to_string()));
        }

        let bits = self.calculate_optimal_bits(&vec![min_val, max_val]);
        let scale = (max_val - min_val) / ((1 << bits) - 1) as f32;
        
        Ok(quantized.iter()
            .map(|&x| min_val + (x as f32) * scale)
            .collect())
    }

    /// Compress data using LZ4 with configurable level
    fn compress_data(&self, data: &[u8]) -> Option<Vec<u8>> {
        if !self.config.use_compression {
            return None;
        }

        let compressed = compress_prepend_size(data);
        
        // Check if compression is beneficial
        if compressed.len() >= data.len() {
            return None;
        }

        // Check if compressed size is within limits
        if let Some(max_size) = self.config.max_compressed_size {
            if compressed.len() > max_size {
                return None;
            }
        }

        Some(compressed)
    }

    /// Decompress data using LZ4
    fn decompress_data(&self, compressed: &[u8]) -> Result<Vec<u8>, EmbeddingError> {
        if !self.config.use_compression {
            return Err(EmbeddingError::CompressionError("Compression disabled".to_string()));
        }

        decompress_size_prepended(compressed)
            .map_err(|e| EmbeddingError::CompressionError(format!("Decompression failed: {}", e)))
    }

    /// Put a response in the cache with quantization and compression
    pub fn put(&mut self, text: String, response: EmbeddingResponse) -> Result<(), EmbeddingError> {
        if !self.config.enabled {
            return Ok(());
        }

        // Quantize the embedding
        let (quantized, min_val, max_val, error) = self.quantize_embedding(&response.embedding)?;
        
        // Update quantization stats
        self.stats.quantized_entries += 1;
        self.stats.avg_quantization_error = (self.stats.avg_quantization_error * (self.stats.quantized_entries - 1) as f32 + error) 
            / self.stats.quantized_entries as f32;

        // Compress the quantized data if enabled
        let (compressed_data, original_size) = if self.config.use_compression {
            if let Some(compressed) = self.compress_data(&quantized) {
                self.stats.compressed_entries += 1;
                self.stats.total_original_size += quantized.len();
                self.stats.total_compressed_size += compressed.len();
                self.stats.avg_compression_ratio = (self.stats.total_original_size as f32) / (self.stats.total_compressed_size as f32);
                (Some(compressed), Some(quantized.len()))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        
        // Check cache size limit and evict oldest entries until there is room for the new entry
        if let Some(max_size) = self.config.max_size {
            while self.entries.len() >= max_size {
                self.evict_oldest();
            }
        }

        self.entries.insert(text, CacheEntry {
            response,
            timestamp: SystemTime::now(),
            quantized_embedding: Some(quantized),
            min_val: Some(min_val),
            max_val: Some(max_val),
            compressed_data,
            original_size,
            quantization_error: Some(error),
        });

        self.stats.total_entries = self.entries.len();
        Ok(())
    }

    /// Get a cached response using similarity matching with quantized embeddings
    pub fn get_similar(&self, embedding: &[f32], threshold: f32) -> Option<EmbeddingResponse> {
        if !self.config.enabled || threshold <= 0.0 {
            return None;
        }

        let mut best_match = None;
        let mut best_similarity = threshold;

        // Quantize the query embedding
        let (query_quantized, query_min, query_max, _) = self.quantize_embedding(embedding).ok()?;

        for entry in self.entries.values() {
            // Check TTL
            if let Some(ttl) = self.config.ttl_seconds {
                if entry.timestamp.elapsed().unwrap().as_secs() > ttl {
                    continue;
                }
            }

            let stored_embedding = if let Some(compressed) = &entry.compressed_data {
                // Use compressed data if available
                match self.decompress_data(compressed) {
                    Ok(decompressed) => decompressed,
                    Err(_) => entry.quantized_embedding.as_ref()?.clone(),
                }
            } else {
                entry.quantized_embedding.as_ref()?.clone()
            };

            // Dequantize the stored embedding
            let dequantized = match self.dequantize_embedding(
                &stored_embedding,
                entry.min_val?,
                entry.max_val?
            ) {
                Ok(deq) => deq,
                Err(_) => continue,
            };

            let similarity = cosine_similarity(embedding, &dequantized);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_match = Some(entry.response.clone());
            }
        }

        best_match
    }
    
    /// Evict the oldest entries from the cache
    fn evict_oldest(&mut self) {
        if let Some(max_size) = self.config.max_size {
            if self.entries.len() >= max_size {
                if let Some(oldest_key) = self.entries.iter()
                    .min_by_key(|(_, entry)| entry.timestamp)
                    .map(|(k, _)| k.clone()) {
                    self.entries.remove(&oldest_key);
                    self.stats.total_entries = self.entries.len();
                }
            }
        }
    }

    /// Calculate the size of the cache
    pub fn size(&self) -> usize {
        self.entries.len()
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.entries.clear();
        self.stats = CacheStats::default();
    }

    /// Get the cache configuration
    pub fn get_config(&self) -> &CacheConfig {
        &self.config
    }

    /// Update the cache configuration
    pub fn update_config(&mut self, config: CacheConfig) {
        self.config = config;
        self.evict_oldest(); // Apply new limits
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}