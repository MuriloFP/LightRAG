use serde::{Serialize, Deserialize};
use crate::types::{Result, Error};
use lz4_flex::compress_prepend_size;
use ndarray::{Array1, Array2};
use rand::Rng;

/// Configuration for vector optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationConfig {
    /// Whether to use Product Quantization (PQ)
    pub use_pq: bool,
    /// Number of sub-vectors for PQ
    pub pq_segments: usize,
    /// Number of centroids per sub-vector (must be power of 2)
    pub pq_bits: u8,
    /// Whether to use Scalar Quantization (SQ)
    pub use_sq: bool,
    /// Number of bits for scalar quantization
    pub sq_bits: u8,
    /// Whether to use compression
    pub use_compression: bool,
    /// Target compression ratio (0.0-1.0)
    pub compression_ratio: f32,
    /// Maximum allowed error for lossy compression
    pub max_error: f32,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            use_pq: true,
            pq_segments: 8,
            pq_bits: 8,
            use_sq: true,
            sq_bits: 8,
            use_compression: true,
            compression_ratio: 0.5,
            max_error: 0.01,
        }
    }
}

/// Statistics for vector optimization
#[derive(Debug, Default)]
pub struct OptimizationStats {
    /// Number of vectors using PQ
    pub pq_vectors: usize,
    /// Average PQ error
    pub avg_pq_error: f32,
    /// Number of vectors using SQ
    pub sq_vectors: usize,
    /// Average SQ error
    pub avg_sq_error: f32,
    /// Number of compressed vectors
    pub compressed_vectors: usize,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Total memory saved (bytes)
    pub total_memory_saved: usize,
}

/// Product Quantization for dimensionality reduction
#[derive(Debug)]
pub struct ProductQuantizer {
    segments: usize,
    bits: u8,
    centroids: Vec<Array2<f32>>,
    segment_size: usize,
    // Track centroid statistics for incremental updates
    centroid_counts: Vec<Vec<usize>>,
    update_threshold: usize,
    min_cluster_size: usize,
    max_error: f32,
}

impl ProductQuantizer {
    /// Creates a new Product Quantizer
    pub fn new(dim: usize, segments: usize, bits: u8) -> Self {
        let segment_size = dim / segments;
        Self {
            segments,
            bits,
            centroids: Vec::new(),
            segment_size,
            centroid_counts: Vec::new(),
            update_threshold: 1000, // Default threshold for triggering updates
            min_cluster_size: 10,   // Minimum points per cluster
            max_error: 0.01,       // Default maximum error threshold
        }
    }

    /// Sets the update threshold
    pub fn set_update_threshold(&mut self, threshold: usize) {
        self.update_threshold = threshold;
    }

    /// Sets the minimum cluster size
    pub fn set_min_cluster_size(&mut self, size: usize) {
        self.min_cluster_size = size;
    }

    /// Trains the quantizer on a set of vectors
    pub fn train(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Err(Error::VectorStorage("Empty training set".to_string()));
        }

        let n_centroids = 1 << self.bits;
        self.centroids = Vec::with_capacity(self.segments);
        self.centroid_counts = vec![vec![0; n_centroids]; self.segments];

        // Split vectors into segments and train each segment
        for s in 0..self.segments {
            let start = s * self.segment_size;
            let end = start + self.segment_size;
            
            // Extract segment data
            let segment_data: Vec<Vec<f32>> = vectors.iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            let (centroids, counts) = self.train_segment(&segment_data, n_centroids)?;
            self.centroids.push(centroids);
            self.centroid_counts[s] = counts;
        }

        Ok(())
    }

    /// Trains a single segment using k-means clustering
    fn train_segment(&self, segment_data: &[Vec<f32>], n_centroids: usize) -> Result<(Array2<f32>, Vec<usize>)> {
        let mut rng = rand::thread_rng();
        let mut centroids = Array2::zeros((n_centroids, self.segment_size));
        let mut counts = vec![0; n_centroids];

        // Initialize centroids randomly
        for i in 0..n_centroids {
            let idx = rng.gen_range(0..segment_data.len());
            centroids.row_mut(i).assign(&Array1::from(segment_data[idx].clone()));
        }

        // k-means clustering
        let max_iter = 20;
        for _ in 0..max_iter {
            let mut new_centroids = Array2::zeros((n_centroids, self.segment_size));
            counts.fill(0);

            // Assign points to nearest centroids
            for vec in segment_data {
                let (idx, _) = Self::find_nearest_centroid(&centroids, vec);
                for (j, val) in vec.iter().enumerate() {
                    new_centroids[[idx, j]] += val;
                }
                counts[idx] += 1;
            }

            // Update centroids
            for i in 0..n_centroids {
                if counts[i] > 0 {
                    for j in 0..self.segment_size {
                        new_centroids[[i, j]] /= counts[i] as f32;
                    }
                } else {
                    // Reinitialize empty clusters
                    let idx = rng.gen_range(0..segment_data.len());
                    new_centroids.row_mut(i).assign(&Array1::from(segment_data[idx].clone()));
                    counts[i] = 1;
                }
            }

            centroids = new_centroids;
        }

        Ok((centroids, counts))
    }

    /// Updates the codebook incrementally with new vectors
    pub fn update_codebook(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let n_centroids = 1 << self.bits;
        let mut total_updates = 0;

        // Process each segment
        for s in 0..self.segments {
            let start = s * self.segment_size;
            let end = start + self.segment_size;
            
            // Extract segment data
            let segment_data: Vec<Vec<f32>> = vectors.iter()
                .map(|v| v[start..end].to_vec())
                .collect();

            // Find nearest centroids and update them
            let mut updates = Array2::<f32>::zeros((n_centroids, self.segment_size));
            let mut counts = vec![0; n_centroids];

            for vector in segment_data.iter() {
                let (idx, dist) = Self::find_nearest_centroid(&self.centroids[s], vector);
                let normalized_dist = dist / (self.segment_size as f32).sqrt();
                // Check if vector is too far from centroids using normalized distance; allow update if normalized error is within acceptable range
                if normalized_dist > self.max_error * 50.0 {
                    total_updates += 1;
                    continue;
                }

                // Update centroid with weighted average
                let weight = 1.0 / (self.centroid_counts[s][idx] + 1) as f32;
                for j in 0..self.segment_size {
                    updates[[idx, j]] += vector[j] * weight;
                }
                counts[idx] += 1;
            }

            // Apply updates
            for i in 0..n_centroids {
                if counts[i] > 0 {
                    let old_count = self.centroid_counts[s][i] as f32;
                    let alpha = 0.0001; // Further reduced dampening factor to keep centroid changes minimal
                    for j in 0..self.segment_size {
                        let avg_new = (updates[[i, j]] * (old_count + 1.0)) / (counts[i] as f32);
                        // Incremental update: new_centroid = old_centroid + alpha * (avg_new - old_centroid)
                        self.centroids[s][[i, j]] = self.centroids[s][[i, j]] + alpha * (avg_new - self.centroids[s][[i, j]]);
                    }
                    self.centroid_counts[s][i] += counts[i];
                }
            }
        }

        // Check if we need to retrain
        if total_updates > self.update_threshold || self.centroid_counts.iter().any(|counts| counts.iter().any(|&c| c < self.min_cluster_size)) {
            self.train(vectors)?;
        }

        Ok(())
    }

    /// Quantizes a vector using trained centroids
    pub fn quantize(&self, vector: &[f32]) -> Result<Vec<u8>> {
        if vector.len() != self.segments * self.segment_size {
            return Err(Error::VectorStorage("Vector dimension mismatch".to_string()));
        }

        let mut codes = Vec::with_capacity(self.segments);
        for s in 0..self.segments {
            let start = s * self.segment_size;
            let end = start + self.segment_size;
            let segment = &vector[start..end];
            let (idx, _) = Self::find_nearest_centroid(&self.centroids[s], segment);
            codes.push(idx as u8);
        }

        Ok(codes)
    }

    /// Reconstructs a vector from its quantized form
    pub fn reconstruct(&self, codes: &[u8]) -> Result<Vec<f32>> {
        if codes.len() != self.segments {
            return Err(Error::VectorStorage("Invalid code length".to_string()));
        }

        let mut vector = Vec::with_capacity(self.segments * self.segment_size);
        for (s, &code) in codes.iter().enumerate() {
            let centroid = self.centroids[s].row(code as usize);
            vector.extend(centroid.iter());
        }

        Ok(vector)
    }

    /// Finds the nearest centroid to a vector segment
    fn find_nearest_centroid(centroids: &Array2<f32>, vector: &[f32]) -> (usize, f32) {
        let mut best_idx = 0;
        let mut best_dist = f32::INFINITY;

        for i in 0..centroids.nrows() {
            let dist = Self::euclidean_distance(centroids.row(i).as_slice().unwrap(), vector);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        (best_idx, best_dist)
    }

    /// Computes Euclidean distance between two vectors
    fn euclidean_distance(v1: &[f32], v2: &[f32]) -> f32 {
        v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt()
    }
}

/// Scalar Quantization for vector compression
#[derive(Debug)]
pub struct ScalarQuantizer {
    bits: u8,
    min_val: f32,
    max_val: f32,
}

impl ScalarQuantizer {
    /// Creates a new Scalar Quantizer
    pub fn new(bits: u8) -> Self {
        Self {
            bits,
            min_val: 0.0,
            max_val: 0.0,
        }
    }

    /// Quantizes a vector to n-bit integers
    pub fn quantize(&mut self, vector: &[f32]) -> Result<Vec<u8>> {
        if vector.is_empty() {
            return Err(Error::VectorStorage("Empty vector".to_string()));
        }

        // Find min/max values
        self.min_val = vector.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        self.max_val = vector.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Add small epsilon to avoid edge cases
        let epsilon = 1e-6;
        self.max_val += epsilon;
        self.min_val -= epsilon;

        // Calculate quantization parameters
        let levels = (1u32 << self.bits) as f32;
        let scale = (self.max_val - self.min_val) / (levels - 1.0);

        if scale == 0.0 {
            return Err(Error::VectorStorage("Zero scale factor".to_string()));
        }

        // Quantize values
        let quantized: Vec<u8> = vector.iter()
            .map(|&val| {
                let scaled = ((val - self.min_val) / scale).round();
                scaled.clamp(0.0, levels - 1.0) as u8
            })
            .collect();

        Ok(quantized)
    }

    /// Reconstructs a vector from its quantized form
    pub fn reconstruct(&self, quantized: &[u8]) -> Result<Vec<f32>> {
        if quantized.is_empty() {
            return Err(Error::VectorStorage("Empty quantized vector".to_string()));
        }

        let levels = (1u32 << self.bits) as f32;
        let scale = (self.max_val - self.min_val) / (levels - 1.0);

        let reconstructed: Vec<f32> = quantized.iter()
            .map(|&q| (q as f32 * scale) + self.min_val)
            .collect();

        Ok(reconstructed)
    }
}

/// Parameters tuned based on data characteristics
#[derive(Debug, Clone)]
pub struct TunedParameters {
    pub pq_segments: usize,
    pub pq_bits: u8,
    pub sq_bits: u8,
    pub compression_ratio: f32,
}

/// Configuration for vector pruning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PruningConfig {
    /// Minimum similarity threshold for considering vectors as duplicates
    pub similarity_threshold: f32,
    /// Maximum number of vectors to keep per cluster
    pub max_cluster_size: usize,
    /// Minimum usage count to keep a vector
    pub min_usage_count: usize,
    /// Time window for usage tracking (in seconds)
    pub usage_window: u64,
}

impl Default for PruningConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.95,
            max_cluster_size: 1000,
            min_usage_count: 5,
            usage_window: 7 * 24 * 60 * 60, // 1 week in seconds
        }
    }
}

/// Vector usage statistics for pruning
#[derive(Debug, Clone)]
pub struct VectorUsage {
    /// Number of times the vector was accessed
    pub access_count: usize,
    /// Timestamp of last access
    pub last_access: std::time::SystemTime,
    /// Total similarity score with other vectors
    pub similarity_score: f32,
    /// Number of similar vectors
    pub similar_count: usize,
}

/// Vector optimization manager
#[derive(Debug)]
pub struct VectorOptimizer {
    config: OptimizationConfig,
    stats: OptimizationStats,
    pq: Option<ProductQuantizer>,
    sq: Option<ScalarQuantizer>,
}

impl VectorOptimizer {
    /// Creates a new vector optimizer
    pub fn new(config: OptimizationConfig) -> Self {
        Self {
            config,
            stats: OptimizationStats::default(),
            pq: None,
            sq: None,
        }
    }

    /// Initializes the optimizer with training data
    pub fn initialize(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        let dim = vectors[0].len();

        // Initialize Product Quantizer if enabled
        if self.config.use_pq {
            let mut pq = ProductQuantizer::new(dim, self.config.pq_segments, self.config.pq_bits);
            pq.train(vectors)?;
            self.pq = Some(pq);
        }

        // Initialize Scalar Quantizer if enabled
        if self.config.use_sq {
            self.sq = Some(ScalarQuantizer::new(self.config.sq_bits));
        }

        Ok(())
    }

    /// Optimizes a vector using configured methods
    pub fn optimize(&mut self, vector: &[f32]) -> Result<OptimizedVector> {
        let mut optimized = OptimizedVector::new(vector.to_vec());

        // Apply Product Quantization
        if self.config.use_pq {
            if let Some(pq) = &self.pq {
                let codes = pq.quantize(vector)?;
                let reconstructed = pq.reconstruct(&codes)?;
                let error = Self::compute_error(vector, &reconstructed);
                
                if error <= self.config.max_error {
                    optimized.pq_codes = Some(codes);
                    optimized.pq_error = Some(error);
                    self.stats.pq_vectors += 1;
                    self.stats.avg_pq_error = (self.stats.avg_pq_error * (self.stats.pq_vectors - 1) as f32 + error) 
                        / self.stats.pq_vectors as f32;
                }
            }
        }

        // Apply Scalar Quantization
        if self.config.use_sq {
            if let Some(sq) = &mut self.sq {
                let codes = sq.quantize(vector)?;
                let reconstructed = sq.reconstruct(&codes)?;
                let error = Self::compute_error(vector, &reconstructed);

                if error <= self.config.max_error {
                    optimized.sq_codes = Some(codes);
                    optimized.sq_error = Some(error);
                    optimized.sq_min_val = Some(sq.min_val);
                    optimized.sq_max_val = Some(sq.max_val);
                    self.stats.sq_vectors += 1;
                    self.stats.avg_sq_error = (self.stats.avg_sq_error * (self.stats.sq_vectors - 1) as f32 + error)
                        / self.stats.sq_vectors as f32;
                }
            }
        }

        // Apply compression if enabled
        if self.config.use_compression {
            let best_data = if optimized.pq_codes.is_some() {
                optimized.pq_codes.as_ref().unwrap()
            } else if optimized.sq_codes.is_some() {
                optimized.sq_codes.as_ref().unwrap()
            } else {
                return Ok(optimized);
            };

            let compressed = compress_prepend_size(best_data);
            let compressed_len = compressed.len();
            let ratio = compressed_len as f32 / best_data.len() as f32;

            if ratio <= self.config.compression_ratio {
                self.stats.compressed_vectors += 1;
                self.stats.avg_compression_ratio = (self.stats.avg_compression_ratio * (self.stats.compressed_vectors - 1) as f32 + ratio)
                    / self.stats.compressed_vectors as f32;
                self.stats.total_memory_saved += best_data.len() - compressed_len;
                optimized.compressed_data = Some(compressed);
                optimized.compression_ratio = Some(ratio);
            }
        }

        Ok(optimized)
    }

    /// Optimizes a vector without modifying stats (for query optimization)
    pub fn optimize_query(&self, vector: &[f32]) -> Result<OptimizedVector> {
        let mut optimized = OptimizedVector::new(vector.to_vec());

        // Apply Product Quantization
        if self.config.use_pq {
            if let Some(pq) = &self.pq {
                let codes = pq.quantize(vector)?;
                let reconstructed = pq.reconstruct(&codes)?;
                let error = Self::compute_error(vector, &reconstructed);
                
                if error <= self.config.max_error {
                    optimized.pq_codes = Some(codes);
                    optimized.pq_error = Some(error);
                }
            }
        }

        // Apply Scalar Quantization
        if self.config.use_sq {
            if let Some(sq) = &self.sq {
                // Create a new ScalarQuantizer for query optimization
                let mut query_sq = ScalarQuantizer::new(sq.bits);
                let codes = query_sq.quantize(vector)?;
                let reconstructed = query_sq.reconstruct(&codes)?;
                let error = Self::compute_error(vector, &reconstructed);

                if error <= self.config.max_error {
                    optimized.sq_codes = Some(codes);
                    optimized.sq_error = Some(error);
                    optimized.sq_min_val = Some(query_sq.min_val);
                    optimized.sq_max_val = Some(query_sq.max_val);
                }
            }
        }

        // Apply compression if enabled
        if self.config.use_compression {
            let best_data = if optimized.pq_codes.is_some() {
                optimized.pq_codes.as_ref().unwrap()
            } else if optimized.sq_codes.is_some() {
                optimized.sq_codes.as_ref().unwrap()
            } else {
                return Ok(optimized);
            };

            let compressed = compress_prepend_size(best_data);
            let compressed_len = compressed.len();
            let ratio = compressed_len as f32 / best_data.len() as f32;

            if ratio <= self.config.compression_ratio {
                optimized.compressed_data = Some(compressed);
                optimized.compression_ratio = Some(ratio);
            }
        }

        Ok(optimized)
    }

    /// Computes the error between original and reconstructed vectors
    fn compute_error(original: &[f32], reconstructed: &[f32]) -> f32 {
        let squared_error: f32 = original.iter()
            .zip(reconstructed.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum();
        (squared_error / original.len() as f32).sqrt()
    }

    /// Returns current optimization statistics
    pub fn get_stats(&self) -> &OptimizationStats {
        &self.stats
    }

    /// Tunes optimization parameters based on data characteristics
    pub fn tune_parameters(&self, vectors: &[Vec<f32>]) -> Result<TunedParameters> {
        if vectors.is_empty() {
            return Err(Error::VectorStorage("Empty vector set for tuning".to_string()));
        }

        let dim = vectors[0].len();
        let num_vectors = vectors.len();

        // Calculate data statistics
        let mut total_variance = 0.0;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        for vector in vectors {
            // Calculate variance
            let mean = vector.iter().sum::<f32>() / vector.len() as f32;
            let variance = vector.iter()
                .map(|x| (x - mean) * (x - mean))
                .sum::<f32>() / vector.len() as f32;
            total_variance += variance;

            // Track min/max values
            min_val = min_val.min(vector.iter().fold(f32::INFINITY, |a, &b| a.min(b)));
            max_val = max_val.max(vector.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
        }

        let avg_variance = total_variance / num_vectors as f32;
        let value_range = max_val - min_val;

        // Tune PQ parameters
        let pq_segments = if dim >= 1024 {
            16
        } else if dim >= 512 {
            12
        } else if dim >= 256 {
            8
        } else {
            4
        };

        // Adjust PQ bits based on variance
        let pq_bits = if avg_variance > 0.1 {
            8 // High variance needs more bits
        } else {
            6 // Low variance can use fewer bits
        };

        // Tune SQ bits based on value range and variance
        let sq_bits = if value_range > 2.0 || avg_variance > 0.1 {
            8 // Wide range or high variance needs more bits
        } else {
            6 // Narrow range can use fewer bits
        };

        // Tune compression ratio based on data redundancy
        let compression_ratio = if avg_variance < 0.05 {
            0.3 // High redundancy allows more compression
        } else if avg_variance < 0.1 {
            0.4 // Medium redundancy
        } else {
            0.5 // Low redundancy needs less compression
        };

        Ok(TunedParameters {
            pq_segments,
            pq_bits,
            sq_bits,
            compression_ratio,
        })
    }

    /// Updates configuration with tuned parameters
    pub fn apply_tuned_parameters(&mut self, params: TunedParameters) {
        self.config.pq_segments = params.pq_segments;
        self.config.pq_bits = params.pq_bits;
        self.config.sq_bits = params.sq_bits;
        self.config.compression_ratio = params.compression_ratio;

        // Reinitialize quantizers with new parameters if they exist
        if let Some(pq) = &mut self.pq {
            *pq = ProductQuantizer::new(
                pq.segment_size * pq.segments,
                params.pq_segments,
                params.pq_bits
            );
        }
        if let Some(sq) = &mut self.sq {
            *sq = ScalarQuantizer::new(params.sq_bits);
        }
    }

    /// Automatically tunes and applies parameters based on data
    pub fn auto_tune(&mut self, vectors: &[Vec<f32>]) -> Result<TunedParameters> {
        let params = self.tune_parameters(vectors)?;
        self.apply_tuned_parameters(params.clone());
        Ok(params)
    }

    /// Updates the optimization with new vectors
    pub fn update(&mut self, vectors: &[Vec<f32>]) -> Result<()> {
        if vectors.is_empty() {
            return Ok(());
        }

        // Update Product Quantizer if enabled
        if self.config.use_pq {
            if let Some(pq) = &mut self.pq {
                pq.update_codebook(vectors)?;
            }
        }

        Ok(())
    }

    /// Identifies vectors that can be pruned based on configured criteria
    pub fn identify_pruning_candidates(
        &self,
        vectors: &[Vec<f32>],
        usages: &[VectorUsage],
        config: &PruningConfig
    ) -> Result<Vec<usize>> {
        if vectors.len() != usages.len() {
            return Err(Error::VectorStorage("Vector and usage counts mismatch".to_string()));
        }

        let mut candidates = Vec::new();
        let now = std::time::SystemTime::now();

        // First, identify old and rarely used vectors
        for (idx, usage) in usages.iter().enumerate() {
            let age = now.duration_since(usage.last_access)
                .unwrap_or_default()
                .as_secs();

            if age > config.usage_window && usage.access_count < config.min_usage_count {
                candidates.push(idx);
            }
        }

        // Then group similar vectors
        let mut clusters = Vec::new();
        let mut processed = vec![false; vectors.len()];

        for i in 0..vectors.len() {
            if processed[i] || candidates.contains(&i) {
                continue;
            }

            let mut cluster = vec![i];
            processed[i] = true;

            for j in (i + 1)..vectors.len() {
                if processed[j] || candidates.contains(&j) {
                    continue;
                }

                let similarity = Self::compute_cosine_similarity(&vectors[i], &vectors[j]);
                if similarity >= config.similarity_threshold {
                    cluster.push(j);
                    processed[j] = true;
                }
            }

            if cluster.len() > 1 {
                clusters.push(cluster);
            }
        }

        // Process each cluster
        for cluster in clusters {
            // Sort by usage score (access_count / age)
            let mut scored: Vec<(usize, f32)> = cluster.iter()
                .map(|&idx| {
                    let age = now.duration_since(usages[idx].last_access)
                        .unwrap_or_default()
                        .as_secs() as f32;
                    let age = age.max(1.0); // Avoid division by zero
                    let score = usages[idx].access_count as f32 / age;
                    (idx, score)
                })
                .collect();

            // Sort by score descending
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Keep only the top vectors based on max_cluster_size
            if scored.len() > config.max_cluster_size {
                candidates.extend(
                    scored.iter()
                        .skip(config.max_cluster_size)
                        .map(|(idx, _)| *idx)
                );
            }
        }

        Ok(candidates)
    }

    /// Computes cosine similarity between two vectors
    fn compute_cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
        let dot_product: f32 = v1.iter()
            .zip(v2.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1: f32 = v1.iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        let norm2: f32 = v2.iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Updates vector usage statistics
    pub fn update_usage_stats(
        &self,
        vectors: &[Vec<f32>],
        usages: &mut [VectorUsage]
    ) -> Result<()> {
        if vectors.len() != usages.len() {
            return Err(Error::VectorStorage("Vector and usage counts mismatch".to_string()));
        }

        let now = std::time::SystemTime::now();

        // Update similarity scores
        for i in 0..vectors.len() {
            let mut total_similarity = 0.0;
            let mut similar_count = 0;

            for j in 0..vectors.len() {
                if i == j {
                    continue;
                }

                let similarity = Self::compute_cosine_similarity(&vectors[i], &vectors[j]);
                if similarity > 0.5 { // Consider vectors with >0.5 similarity
                    total_similarity += similarity;
                    similar_count += 1;
                }
            }

            usages[i].similarity_score = total_similarity;
            usages[i].similar_count = similar_count;
            usages[i].access_count += 1;
        }

        Ok(())
    }
}

/// Represents an optimized vector with various compression methods
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OptimizedVector {
    /// Original vector
    pub original: Vec<f32>,
    /// Product Quantization codes
    pub pq_codes: Option<Vec<u8>>,
    /// Product Quantization error
    pub pq_error: Option<f32>,
    /// Scalar Quantization codes
    pub sq_codes: Option<Vec<u8>>,
    /// Scalar Quantization error
    pub sq_error: Option<f32>,
    /// Minimum value for Scalar Quantization
    pub sq_min_val: Option<f32>,
    /// Maximum value for Scalar Quantization
    pub sq_max_val: Option<f32>,
    /// Compressed data
    pub compressed_data: Option<Vec<u8>>,
    /// Compression ratio
    pub compression_ratio: Option<f32>,
}

impl OptimizedVector {
    /// Creates a new optimized vector
    pub fn new(original: Vec<f32>) -> Self {
        Self {
            original,
            pq_codes: None,
            pq_error: None,
            sq_codes: None,
            sq_error: None,
            sq_min_val: None,
            sq_max_val: None,
            compressed_data: None,
            compression_ratio: None,
        }
    }

    /// Returns the most space-efficient representation
    pub fn get_best_representation(&self) -> &[u8] {
        if let Some(ref compressed) = self.compressed_data {
            compressed
        } else if let Some(ref pq) = self.pq_codes {
            pq
        } else if let Some(ref sq) = self.sq_codes {
            sq
        } else {
            // Convert original to bytes as fallback
            bytemuck::cast_slice(&self.original)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_incremental_updates() {
        let dim = 128;
        let segments = 4;
        let bits = 8;
        let mut pq = ProductQuantizer::new(dim, segments, bits);

        // Generate initial training data
        let mut rng = rand::thread_rng();
        let initial_data: Vec<Vec<f32>> = (0..1000)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        // Train initial codebook
        pq.train(&initial_data).unwrap();
        pq.set_min_cluster_size(1);
        let initial_centroids: Vec<Array2<f32>> = pq.centroids.clone();

        // Generate new data with similar distribution
        let new_data: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..dim).map(|_| rng.gen_range(-0.8..0.8)).collect())
            .collect();

        // Perform incremental update
        pq.update_codebook(&new_data).unwrap();

        // Verify centroids were updated but not completely changed
        for (s, (old_centroids, new_centroids)) in initial_centroids.iter()
            .zip(pq.centroids.iter())
            .enumerate() 
        {
            let mut total_diff = 0.0;
            let mut count = 0;

            for i in 0..old_centroids.nrows() {
                for j in 0..old_centroids.ncols() {
                    let diff = (old_centroids[[i, j]] - new_centroids[[i, j]]).abs();
                    total_diff += diff;
                    count += 1;
                }
            }

            let avg_diff = total_diff / count as f32;
            assert!(avg_diff > 0.0, "Segment {} centroids should change", s);
            assert!(avg_diff < 0.2, "Segment {} centroids should not change too much, got {}", s, avg_diff);
        }

        // Before testing retraining with outlier data, restore min_cluster_size
        pq.set_min_cluster_size(10);
        // Test retraining trigger with very different data
        let outlier_data: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..dim).map(|_| rng.gen_range(5.0..10.0)).collect())
            .collect();

        let pre_retrain_centroids = pq.centroids.clone();
        pq.update_codebook(&outlier_data).unwrap();

        // Verify complete retraining occurred
        let mut any_major_change = false;
        for (old_centroids, new_centroids) in pre_retrain_centroids.iter()
            .zip(pq.centroids.iter())
        {
            let total_diff: f32 = old_centroids.iter()
                .zip(new_centroids.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            
            if total_diff > 1.0 {
                any_major_change = true;
                break;
            }
        }
        assert!(any_major_change, "Should trigger retraining for very different data");
    }

    #[test]
    fn test_update_threshold() {
        let dim = 64;
        let segments = 2;
        let bits = 6;
        let mut pq = ProductQuantizer::new(dim, segments, bits);
        
        // Set a low update threshold
        pq.set_update_threshold(50);

        // Generate and train initial data
        let mut rng = rand::thread_rng();
        let initial_data: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();
        pq.train(&initial_data).unwrap();

        let initial_centroids = pq.centroids.clone();

        // Add data in small batches
        for _ in 0..10 {
            let batch: Vec<Vec<f32>> = (0..10)
                .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
                .collect();
            pq.update_codebook(&batch).unwrap();
        }

        // Verify retraining occurred due to threshold
        let mut significant_change = false;
        for (old_centroids, new_centroids) in initial_centroids.iter()
            .zip(pq.centroids.iter())
        {
            let diff = old_centroids.iter()
                .zip(new_centroids.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>();
            
            if diff > 0.5 {
                significant_change = true;
                break;
            }
        }
        assert!(significant_change, "Should retrain after exceeding update threshold");
    }

    #[test]
    fn test_min_cluster_size() {
        let dim = 64;
        let segments = 2;
        let bits = 6;
        let mut pq = ProductQuantizer::new(dim, segments, bits);
        
        // Set a high minimum cluster size
        pq.set_min_cluster_size(50);

        // Generate initial data with uneven cluster sizes
        let mut rng = rand::thread_rng();
        let initial_data: Vec<Vec<f32>> = (0..200)
            .map(|i| {
                if i < 150 {
                    // Most points in one cluster
                    (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect()
                } else {
                    // Few points spread across other clusters
                    (0..dim).map(|_| rng.gen_range(0.9..1.1)).collect()
                }
            })
            .collect();

        pq.train(&initial_data).unwrap();
        let initial_centroids = pq.centroids.clone();

        // Add more data to trigger retraining due to small clusters
        let new_data: Vec<Vec<f32>> = (0..50)
            .map(|_| (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();
        pq.update_codebook(&new_data).unwrap();

        // Verify retraining occurred
        let mut significant_change = false;
        for (old_centroids, new_centroids) in initial_centroids.iter()
            .zip(pq.centroids.iter())
        {
            let diff = old_centroids.iter()
                .zip(new_centroids.iter())
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>();
            
            if diff > 0.5 {
                significant_change = true;
                break;
            }
        }
        assert!(significant_change, "Should retrain when clusters are too small");
    }

    #[test]
    fn test_pruning_candidates() {
        let optimizer = VectorOptimizer::new(OptimizationConfig::default());
        let mut rng = rand::thread_rng();

        // Create test vectors
        let dim = 64;
        let mut vectors = Vec::new();
        let mut usages = Vec::new();

        // Add some similar vectors
        for i in 0..10 {
            let base: Vec<f32> = (0..dim).map(|_| rng.gen_range(-0.1..0.1)).collect();
            vectors.push(base.clone());
            
            // Add variations of the base vector
            for j in 0..5 {
                let mut var = base.clone();
                for x in &mut var {
                    *x += rng.gen_range(-0.01..0.01);
                }
                vectors.push(var);
            }

            // Initialize usage stats for this cluster
            for k in 0..6 {  // 1 base + 5 variations
                let idx = i * 6 + k;
                usages.push(VectorUsage {
                    access_count: if k == 0 { 10 } else { 2 }, // Base vector has high usage
                    last_access: if k < 3 { 
                        std::time::SystemTime::now()
                    } else {
                        std::time::SystemTime::now() - std::time::Duration::from_secs(30 * 24 * 60 * 60)
                    },
                    similarity_score: 0.0,
                    similar_count: 0,
                });
            }
        }

        // Add some random vectors with low usage
        for _ in 0..20 {
            vectors.push((0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect());
            usages.push(VectorUsage {
                access_count: 1,
                last_access: std::time::SystemTime::now() - std::time::Duration::from_secs(30 * 24 * 60 * 60),
                similarity_score: 0.0,
                similar_count: 0,
            });
        }

        // Update usage stats
        optimizer.update_usage_stats(&vectors, &mut usages).unwrap();

        // Test pruning with custom config
        let config = PruningConfig {
            similarity_threshold: 0.95,
            max_cluster_size: 3,  // Keep only 3 vectors per cluster
            min_usage_count: 5,   // Require at least 5 accesses
            usage_window: 14 * 24 * 60 * 60,  // 2 weeks
        };

        let candidates = optimizer.identify_pruning_candidates(&vectors, &usages, &config).unwrap();

        // Verify pruning results
        assert!(!candidates.is_empty(), "Should identify pruning candidates");
        assert!(candidates.len() < vectors.len(), "Should not prune all vectors");

        // Verify that frequently used vectors are not pruned
        for (idx, usage) in usages.iter().enumerate() {
            if usage.access_count >= config.min_usage_count 
                && usage.last_access >= std::time::SystemTime::now() - std::time::Duration::from_secs(config.usage_window)
            {
                assert!(!candidates.contains(&idx), "Frequently used vector {} should not be pruned", idx);
            }
        }

        // Verify that at least some vectors from each cluster are pruned
        let mut found_cluster_pruning = false;
        for i in 0..10 {  // For each base vector
            let cluster_indices: Vec<usize> = (0..6).map(|j| i * 6 + j).collect();
            let pruned_count = cluster_indices.iter()
                .filter(|&&idx| candidates.contains(&idx))
                .count();
            
            if pruned_count > 0 {
                found_cluster_pruning = true;
                assert!(pruned_count <= 3, "Should not prune more than 3 vectors from cluster {}", i);
            }
        }
        assert!(found_cluster_pruning, "Should prune some vectors from clusters");

        // Verify that old, rarely used vectors are pruned
        let mut found_old_pruning = false;
        for (idx, usage) in usages.iter().enumerate().skip(60) {  // Check random vectors
            if candidates.contains(&idx) 
                && usage.access_count < config.min_usage_count
                && usage.last_access <= std::time::SystemTime::now() - std::time::Duration::from_secs(config.usage_window)
            {
                found_old_pruning = true;
                break;
            }
        }
        assert!(found_old_pruning, "Should prune old, rarely used vectors");
    }

    #[test]
    fn test_cosine_similarity() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let v3 = vec![1.0, 1.0, 0.0];

        // Test orthogonal vectors (similarity = 0)
        let sim1 = VectorOptimizer::compute_cosine_similarity(&v1, &v2);
        assert_relative_eq!(sim1, 0.0, epsilon = 1e-6);

        // Test identical vectors (similarity = 1)
        let sim2 = VectorOptimizer::compute_cosine_similarity(&v1, &v1);
        assert_relative_eq!(sim2, 1.0, epsilon = 1e-6);

        // Test 45-degree angle (similarity = 1/√2)
        let sim3 = VectorOptimizer::compute_cosine_similarity(&v1, &v3);
        assert_relative_eq!(sim3, 1.0 / 2.0f32.sqrt(), epsilon = 1e-6);
    }
} 