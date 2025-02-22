use super::{VectorOptimizer, OptimizationConfig, VectorData, NanoVectorStorage};
use crate::types::Result;
use rand::Rng;
use std::time::{Duration, Instant};
use serde::{Serialize, Deserialize};

/// Benchmark configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of vectors to generate
    pub num_vectors: usize,
    /// Vector dimension
    pub vector_dim: usize,
    /// Number of queries to run
    pub num_queries: usize,
    /// Number of nearest neighbors to retrieve
    pub top_k: usize,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Number of benchmark iterations
    pub benchmark_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_vectors: 10000,
            vector_dim: 768,
            num_queries: 100,
            top_k: 10,
            warmup_iterations: 3,
            benchmark_iterations: 5,
        }
    }
}

/// Benchmark results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Storage metrics
    pub storage: StorageMetrics,
    /// Query performance metrics
    pub query: QueryMetrics,
    /// Optimization metrics
    pub optimization: OptimizationMetrics,
}

/// Storage-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageMetrics {
    /// Total size of original vectors (bytes)
    pub original_size: usize,
    /// Total size of optimized vectors (bytes)
    pub optimized_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Memory savings (bytes)
    pub memory_saved: usize,
}

/// Query performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryMetrics {
    /// Average query latency without optimization
    pub baseline_latency: Duration,
    /// Average query latency with optimization
    pub optimized_latency: Duration,
    /// Latency improvement ratio
    pub latency_improvement: f32,
    /// Average recall@k
    pub recall_at_k: f32,
}

/// Optimization-related metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationMetrics {
    /// Average PQ error
    pub avg_pq_error: f32,
    /// Average SQ error
    pub avg_sq_error: f32,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Total memory saved (bytes)
    pub total_memory_saved: usize,
}

/// Generates random test vectors
fn generate_test_vectors(num_vectors: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    let mut vectors = Vec::with_capacity(num_vectors);
    
    for _ in 0..num_vectors {
        let vector: Vec<f32> = (0..dim)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        vectors.push(vector);
    }
    
    vectors
}

/// Runs benchmarks for vector optimization
pub async fn run_benchmarks(config: BenchmarkConfig) -> Result<BenchmarkResults> {
    // Generate test vectors
    let vectors = generate_test_vectors(config.num_vectors, config.vector_dim);
    let query_vectors = generate_test_vectors(config.num_queries, config.vector_dim);
    
    // Create storage instances
    let mut baseline_storage = NanoVectorStorage::new_for_testing(None)?;
    let mut optimized_storage = NanoVectorStorage::new_for_testing(Some(OptimizationConfig::default()))?;
    
    // Insert vectors
    let vector_data: Vec<VectorData> = vectors.iter().enumerate().map(|(i, v)| {
        VectorData {
            id: format!("vec_{}", i),
            vector: v.clone(),
            metadata: Default::default(),
            created_at: std::time::SystemTime::now(),
            optimized: None,
        }
    }).collect();
    
    baseline_storage.upsert(vector_data.clone()).await?;
    optimized_storage.upsert(vector_data).await?;
    
    // Warmup
    for _ in 0..config.warmup_iterations {
        for query in &query_vectors {
            baseline_storage.query(query.clone(), config.top_k).await?;
            optimized_storage.query(query.clone(), config.top_k).await?;
        }
    }
    
    // Benchmark queries
    let mut baseline_total = Duration::default();
    let mut optimized_total = Duration::default();
    let mut recall_sum = 0.0;
    
    for _ in 0..config.benchmark_iterations {
        for query in &query_vectors {
            // Baseline query
            let start = Instant::now();
            let baseline_results = baseline_storage.query(query.clone(), config.top_k).await?;
            baseline_total += start.elapsed();
            
            // Optimized query
            let start = Instant::now();
            let optimized_results = optimized_storage.query(query.clone(), config.top_k).await?;
            optimized_total += start.elapsed();
            
            // Calculate recall
            let baseline_ids: Vec<&str> = baseline_results.iter().map(|r| r.id.as_str()).collect();
            let optimized_ids: Vec<&str> = optimized_results.iter().map(|r| r.id.as_str()).collect();
            let common = baseline_ids.iter().filter(|id| optimized_ids.contains(id)).count();
            recall_sum += common as f32 / config.top_k as f32;
        }
    }
    
    // Calculate metrics
    let total_queries = (config.benchmark_iterations * config.num_queries) as u32;
    let baseline_latency = baseline_total / total_queries;
    let optimized_latency = optimized_total / total_queries;
    let latency_improvement = 1.0 - (optimized_latency.as_secs_f32() / baseline_latency.as_secs_f32());
    let recall_at_k = recall_sum / total_queries as f32;
    
    // Get optimization stats
    let opt_stats = optimized_storage.get_optimization_stats().unwrap();
    
    // Calculate storage metrics
    let original_size = vectors.iter().map(|v| v.len() * std::mem::size_of::<f32>()).sum();
    let optimized_size = original_size - opt_stats.total_memory_saved;
    let compression_ratio = optimized_size as f32 / original_size as f32;
    
    Ok(BenchmarkResults {
        storage: StorageMetrics {
            original_size,
            optimized_size,
            compression_ratio,
            memory_saved: opt_stats.total_memory_saved,
        },
        query: QueryMetrics {
            baseline_latency,
            optimized_latency,
            latency_improvement,
            recall_at_k,
        },
        optimization: OptimizationMetrics {
            avg_pq_error: opt_stats.avg_pq_error,
            avg_sq_error: opt_stats.avg_sq_error,
            avg_compression_ratio: opt_stats.avg_compression_ratio,
            total_memory_saved: opt_stats.total_memory_saved,
        },
    })
}

/// Helper function to format benchmark results as a string
pub fn format_benchmark_results(results: &BenchmarkResults) -> String {
    let mut output = String::new();
    
    output.push_str("\n=== Vector Optimization Benchmark Results ===\n\n");
    
    // Storage metrics
    output.push_str("Storage Metrics:\n");
    output.push_str(&format!("- Original Size: {:.2} MB\n", results.storage.original_size as f32 / 1_000_000.0));
    output.push_str(&format!("- Optimized Size: {:.2} MB\n", results.storage.optimized_size as f32 / 1_000_000.0));
    output.push_str(&format!("- Compression Ratio: {:.2}%\n", results.storage.compression_ratio * 100.0));
    output.push_str(&format!("- Memory Saved: {:.2} MB\n\n", results.storage.memory_saved as f32 / 1_000_000.0));
    
    // Query metrics
    output.push_str("Query Performance:\n");
    output.push_str(&format!("- Baseline Latency: {:.2} ms\n", results.query.baseline_latency.as_secs_f32() * 1000.0));
    output.push_str(&format!("- Optimized Latency: {:.2} ms\n", results.query.optimized_latency.as_secs_f32() * 1000.0));
    output.push_str(&format!("- Latency Improvement: {:.2}%\n", results.query.latency_improvement * 100.0));
    output.push_str(&format!("- Recall@k: {:.2}%\n\n", results.query.recall_at_k * 100.0));
    
    // Optimization metrics
    output.push_str("Optimization Metrics:\n");
    output.push_str(&format!("- Avg PQ Error: {:.4}\n", results.optimization.avg_pq_error));
    output.push_str(&format!("- Avg SQ Error: {:.4}\n", results.optimization.avg_sq_error));
    output.push_str(&format!("- Avg Compression Ratio: {:.2}%\n", results.optimization.avg_compression_ratio * 100.0));
    output.push_str(&format!("- Total Memory Saved: {:.2} MB\n", results.optimization.total_memory_saved as f32 / 1_000_000.0));
    
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::vector::optimization::OptimizationConfig;

    #[tokio::test]
    async fn test_vector_optimization_benchmarks() {
        // Create a small benchmark config for testing
        let config = BenchmarkConfig {
            num_vectors: 1000,
            vector_dim: 128,
            num_queries: 10,
            top_k: 5,
            warmup_iterations: 1,
            benchmark_iterations: 2,
        };

        // Run benchmarks
        let results = run_benchmarks(config).await.unwrap();

        // Basic validation
        assert!(results.storage.compression_ratio > 0.0);
        assert!(results.storage.compression_ratio < 1.0);
        assert!(results.storage.memory_saved > 0);
        
        assert!(results.query.baseline_latency > Duration::from_nanos(0));
        assert!(results.query.optimized_latency > Duration::from_nanos(0));
        assert!(results.query.recall_at_k > 0.0);
        assert!(results.query.recall_at_k <= 1.0);
        
        assert!(results.optimization.avg_pq_error >= 0.0);
        assert!(results.optimization.avg_sq_error >= 0.0);
        assert!(results.optimization.avg_compression_ratio > 0.0);
        assert!(results.optimization.total_memory_saved > 0);

        // Format results
        let formatted = format_benchmark_results(&results);
        assert!(formatted.contains("Vector Optimization Benchmark Results"));
        assert!(formatted.contains("Storage Metrics"));
        assert!(formatted.contains("Query Performance"));
        assert!(formatted.contains("Optimization Metrics"));
    }

    #[test]
    fn test_generate_test_vectors() {
        let num_vectors = 100;
        let dim = 64;
        let vectors = generate_test_vectors(num_vectors, dim);

        assert_eq!(vectors.len(), num_vectors);
        assert!(vectors.iter().all(|v| v.len() == dim));
        assert!(vectors.iter().all(|v| v.iter().all(|x| *x >= -1.0 && *x <= 1.0)));
    }

    #[test]
    fn test_parameter_tuning() {
        // Create test vectors with different characteristics
        let mut rng = rand::thread_rng();
        
        // High variance vectors
        let high_variance: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..512).map(|_| rng.gen_range(-2.0..2.0)).collect())
            .collect();
            
        // Low variance vectors
        let low_variance: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..512).map(|_| rng.gen_range(-0.1..0.1)).collect())
            .collect();

        // Test tuning with high variance data
        let optimizer = VectorOptimizer::new(OptimizationConfig::default());
        let high_var_params = optimizer.tune_parameters(&high_variance).unwrap();
        
        // High variance should result in more bits and segments
        assert_eq!(high_var_params.pq_segments, 12); // For 512 dim
        assert_eq!(high_var_params.pq_bits, 8);      // High variance
        assert_eq!(high_var_params.sq_bits, 8);      // Wide range
        assert!((high_var_params.compression_ratio - 0.5).abs() < f32::EPSILON); // Low compression

        // Test tuning with low variance data
        let low_var_params = optimizer.tune_parameters(&low_variance).unwrap();
        
        // Low variance should allow for more compression
        assert_eq!(low_var_params.pq_segments, 12);  // For 512 dim
        assert_eq!(low_var_params.pq_bits, 6);       // Low variance
        assert_eq!(low_var_params.sq_bits, 6);       // Narrow range
        assert!((low_var_params.compression_ratio - 0.3).abs() < f32::EPSILON); // High compression

        // Test auto-tuning
        let mut optimizer = VectorOptimizer::new(OptimizationConfig::default());
        let tuned_params = optimizer.auto_tune(&high_variance).unwrap();
        
        // Verify parameters were applied
        assert_eq!(optimizer.config.pq_segments, tuned_params.pq_segments);
        assert_eq!(optimizer.config.pq_bits, tuned_params.pq_bits);
        assert_eq!(optimizer.config.sq_bits, tuned_params.sq_bits);
        assert!((optimizer.config.compression_ratio - tuned_params.compression_ratio).abs() < f32::EPSILON);
    }
} 