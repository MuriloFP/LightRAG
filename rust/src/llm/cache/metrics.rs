use std::sync::atomic::{AtomicUsize, Ordering};

/// Cache metrics
#[derive(Debug)]
pub struct CacheMetrics {
    /// Number of cache hits
    pub hits: AtomicUsize,
    
    /// Number of cache misses
    pub misses: AtomicUsize,
    
    /// Number of evictions
    pub evictions: AtomicUsize,
    
    /// Current cache size
    pub size: AtomicUsize,
    
    /// Number of similarity matches
    pub similarity_matches: AtomicUsize,
    
    /// Number of LLM verifications
    pub llm_verifications: AtomicUsize,
}

impl Default for CacheMetrics {
    fn default() -> Self {
        Self {
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
            evictions: AtomicUsize::new(0),
            size: AtomicUsize::new(0),
            similarity_matches: AtomicUsize::new(0),
            llm_verifications: AtomicUsize::new(0),
        }
    }
}

impl CacheMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all metrics to zero
    pub fn reset(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
        self.evictions.store(0, Ordering::Relaxed);
        self.size.store(0, Ordering::Relaxed);
        self.similarity_matches.store(0, Ordering::Relaxed);
        self.llm_verifications.store(0, Ordering::Relaxed);
    }

    /// Get hit rate as percentage
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits.load(Ordering::Relaxed) as f64;
        let total = hits + self.misses.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            (hits / total) * 100.0
        } else {
            0.0
        }
    }

    /// Get eviction rate as percentage
    pub fn eviction_rate(&self) -> f64 {
        let evictions = self.evictions.load(Ordering::Relaxed) as f64;
        let total = self.size.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            (evictions / total) * 100.0
        } else {
            0.0
        }
    }

    /// Get similarity match rate as percentage
    pub fn similarity_match_rate(&self) -> f64 {
        let matches = self.similarity_matches.load(Ordering::Relaxed) as f64;
        let total = self.hits.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            (matches / total) * 100.0
        } else {
            0.0
        }
    }

    /// Get verification rate as percentage
    pub fn verification_rate(&self) -> f64 {
        let verifications = self.llm_verifications.load(Ordering::Relaxed) as f64;
        let total = self.hits.load(Ordering::Relaxed) as f64;
        if total > 0.0 {
            (verifications / total) * 100.0
        } else {
            0.0
        }
    }
} 