use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use async_trait::async_trait;

use super::{ResponseCache, CacheConfig, CacheEntry, CacheMetrics};
use crate::llm::{LLMError, LLMResponse};

/// In-memory implementation of ResponseCache
pub struct InMemoryCache {
    /// Cache entries
    entries: Arc<RwLock<HashMap<String, CacheEntry>>>,
    
    /// Cache configuration
    config: CacheConfig,
    
    /// Cache metrics
    metrics: Arc<CacheMetrics>,
}

impl Default for InMemoryCache {
    fn default() -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config: CacheConfig::default(),
            metrics: Arc::new(CacheMetrics::default()),
        }
    }
}

impl InMemoryCache {
    /// Create a new in-memory cache
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: Arc::new(RwLock::new(HashMap::new())),
            config,
            metrics: Arc::new(CacheMetrics::default()),
        }
    }

    /// Find similar entry using vector similarity
    pub async fn find_similar_entry(&self, embedding: &[f32], threshold: f32) -> Option<LLMResponse> {
        if !self.config.use_fuzzy_match {
            return None;
        }

        let entries = self.entries.read().unwrap();
        let mut best_match = None;
        let mut best_similarity = threshold;

        for entry in entries.values() {
            if let Some(entry_embedding) = &entry.embedding {
                let similarity = cosine_similarity(embedding, entry_embedding);
                if similarity > best_similarity {
                    best_similarity = similarity;
                    best_match = Some(entry.response.clone());
                }
            }
        }

        if best_match.is_some() {
            self.metrics.similarity_matches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        best_match
    }

    /// Put an entry with embedding for similarity search
    pub async fn put_with_embedding(&self, prompt: &str, response: LLMResponse, embedding: Vec<f32>) -> Result<(), LLMError> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut entries = self.entries.write().unwrap();
        let mut entry = CacheEntry::new(response, self.config.ttl);
        entry.embedding = Some(embedding);
        entries.insert(prompt.to_string(), entry);
        self.metrics.size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[async_trait]
impl ResponseCache for InMemoryCache {
    async fn get(&self, prompt: &str) -> Option<LLMResponse> {
        if !self.config.enabled {
            return None;
        }

        let entries = self.entries.read().unwrap();
        if let Some(entry) = entries.get(prompt) {
            if entry.is_expired() {
                self.metrics.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                None
            } else {
                self.metrics.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                Some(entry.response.clone())
            }
        } else {
            self.metrics.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }

    async fn put(&self, prompt: &str, response: LLMResponse) -> Result<(), LLMError> {
        if !self.config.enabled {
            return Ok(());
        }

        let mut entries = self.entries.write().unwrap();
        let entry = CacheEntry::new(response, self.config.ttl);
        entries.insert(prompt.to_string(), entry);
        self.metrics.size.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    async fn cleanup(&self) -> Result<(), LLMError> {
        let mut entries = self.entries.write().unwrap();
        let before_len = entries.len();
        entries.retain(|_, entry| !entry.is_expired());
        let removed = before_len - entries.len();
        
        if removed > 0 {
            self.metrics.evictions.fetch_add(removed, std::sync::atomic::Ordering::Relaxed);
            self.metrics.size.fetch_sub(removed, std::sync::atomic::Ordering::Relaxed);
        }
        
        Ok(())
    }

    async fn clear(&self) -> Result<(), LLMError> {
        let mut entries = self.entries.write().unwrap();
        entries.clear();
        self.metrics.size.store(0, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    fn get_config(&self) -> &CacheConfig {
        &self.config
    }

    fn update_config(&mut self, config: CacheConfig) -> Result<(), LLMError> {
        self.config = config;
        Ok(())
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mut dot_product = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..a.len() {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a.sqrt() * norm_b.sqrt())
} 