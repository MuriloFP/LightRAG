pub mod hnsw;

use crate::types::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use crate::storage::vector::hnsw::HNSWIndex;

/// Extended VectorStorage trait with vector operations.
#[async_trait]
pub trait VectorStorage: Send + Sync {
    async fn initialize(&mut self) -> Result<()>;
    async fn finalize(&mut self) -> Result<()>;
    async fn query(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>>;
    async fn upsert(&mut self, data: Vec<VectorData>) -> Result<UpsertResponse>;
    async fn delete(&mut self, ids: Vec<String>) -> Result<()>;
}

/// Data structure to represent a vector and its metadata.
#[derive(Clone, Debug)]
pub struct VectorData {
    pub id: String,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, Value>,
}

/// Data structure for query results.
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: String,
    pub distance: f32,
    pub metadata: HashMap<String, Value>,
}

/// Data structure for the upsert response.
#[derive(Clone, Debug)]
pub struct UpsertResponse {
    pub inserted: Vec<String>,
    pub updated: Vec<String>,
}

/// Helper function to compute cosine similarity between two vectors.
fn cosine_similarity(v1: &[f32], v2: &[f32]) -> f32 {
    let dot: f32 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1 = v1.iter().map(|a| a * a).sum::<f32>().sqrt();
    let norm2 = v2.iter().map(|a| a * a).sum::<f32>().sqrt();
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot / (norm1 * norm2)
    }
}

/// A basic implementation of vector storage mimicking NanoVectorDB-RS.
pub struct NanoVectorStorage {
    /// In-memory storage of vectors.
    storage: Vec<VectorData>,
    /// Cosine similarity threshold for filtering query results.
    threshold: f32,
    /// HNSW index for approximate nearest neighbor search
    hnsw: HNSWIndex,
}

impl NanoVectorStorage {
    pub fn new(_config: &crate::types::Config) -> Result<Self> {
        // Default threshold set to 0.2, and HNSW with one layer, ef_construction=128, m=16
        Ok(NanoVectorStorage {
            storage: Vec::new(),
            threshold: 0.2,
            hnsw: HNSWIndex::new(1, 128, 16),
        })
    }
}

#[async_trait]
impl VectorStorage for NanoVectorStorage {
    async fn initialize(&mut self) -> Result<()> {
        // For this in-memory storage, nothing extra is needed.
        Ok(())
    }

    async fn finalize(&mut self) -> Result<()> {
        // For this in-memory storage, nothing extra is needed.
        Ok(())
    }

    async fn query(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>> {
        // Use the HNSW index for querying. Currently, HNSWIndex::query performs a linear scan (placeholder).
        let hnsw_results = self.hnsw.query(query.as_slice(), top_k, 128);
        let results: Vec<SearchResult> = hnsw_results.into_iter().map(|(id, sim)| {
            SearchResult { id, distance: sim, metadata: HashMap::new() }
        }).collect();
        Ok(results)
    }

    async fn upsert(&mut self, data: Vec<VectorData>) -> Result<UpsertResponse> {
        let mut inserted = Vec::new();
        let mut updated = Vec::new();
        for new_data in data {
            if let Some(existing) = self.storage.iter_mut().find(|v| v.id == new_data.id) {
                *existing = new_data.clone();
                updated.push(existing.id.clone());
            } else {
                #[cfg(feature = "mobile")]
                {
                    // For mobile optimization, shrink the vector's capacity to reduce memory overhead
                    let mut optimized_vector = new_data.vector.clone();
                    optimized_vector.shrink_to_fit();
                    let optimized_new_data = VectorData {
                        id: new_data.id.clone(),
                        vector: optimized_vector.clone(),
                        metadata: new_data.metadata.clone(),
                    };
                    self.storage.push(optimized_new_data.clone());
                    // Add a new node to the HNSW index using the optimized vector
                    let node = crate::storage::vector::hnsw::HNSWNode::new(&new_data.id, optimized_vector);
                    let _ = self.hnsw.add_node(node);
                    inserted.push(new_data.id.clone());
                }
                #[cfg(not(feature = "mobile"))]
                {
                    self.storage.push(new_data.clone());
                    // Add a new node to the HNSW index with the original vector
                    let node = crate::storage::vector::hnsw::HNSWNode::new(&new_data.id, new_data.vector.clone());
                    let _ = self.hnsw.add_node(node);
                    inserted.push(new_data.id.clone());
                }
            }
        }
        Ok(UpsertResponse { inserted, updated })
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        // Remove vectors from the in-memory storage
        self.storage.retain(|v| !ids.contains(&v.id));

        // Update the HNSW index by deleting corresponding nodes
        self.hnsw.delete_nodes(&ids);

        Ok(())
    }
} 