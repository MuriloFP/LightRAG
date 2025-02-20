/// Vector storage module providing efficient similarity search capabilities.
/// 
/// This module implements vector storage and similarity search functionality using:
/// - HNSW (Hierarchical Navigable Small World) index for fast approximate nearest neighbor search
/// - In-memory vector storage with persistence
/// - Cosine similarity-based search
/// - Metadata management for vectors
/// 
/// The module supports:
/// - Vector insertion and updates
/// - Efficient similarity queries
/// - Batch operations
/// - Persistence to disk
/// - Automatic index maintenance

/// HNSW (Hierarchical Navigable Small World) implementation for approximate nearest neighbor search.
/// 
/// This module provides:
/// - Multi-layer graph structure for efficient search
/// - Logarithmic time complexity for queries
/// - Configurable index parameters
/// - Persistence capabilities
pub mod hnsw;

use crate::types::Result;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use crate::storage::vector::hnsw::HNSWIndex;
use serde::{Serialize, Deserialize};
use std::fs;
use std::path::PathBuf;

/// Extended VectorStorage trait with vector operations.
#[async_trait]
pub trait VectorStorage: Send + Sync {
    /// Initializes the vector storage system.
    /// 
    /// This method should be called before any other operations to:
    /// - Load persisted data from disk
    /// - Set up any required data structures
    /// - Initialize the HNSW index
    /// 
    /// # Returns
    /// A Result indicating success or failure of the initialization
    async fn initialize(&mut self) -> Result<()>;

    /// Finalizes the vector storage system.
    /// 
    /// This method should be called when shutting down to:
    /// - Persist data to disk
    /// - Clean up resources
    /// - Save the HNSW index
    /// 
    /// # Returns
    /// A Result indicating success or failure of the finalization
    async fn finalize(&mut self) -> Result<()>;

    /// Performs a similarity search query.
    /// 
    /// # Arguments
    /// * `query` - The query vector to search for
    /// * `top_k` - Number of most similar vectors to return
    /// 
    /// # Returns
    /// A Result containing a vector of SearchResult ordered by similarity
    async fn query(&self, query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>>;

    /// Inserts or updates vectors in the storage.
    /// 
    /// # Arguments
    /// * `data` - Vector of VectorData containing vectors and their metadata
    /// 
    /// # Returns
    /// A Result containing an UpsertResponse with lists of inserted and updated IDs
    async fn upsert(&mut self, data: Vec<VectorData>) -> Result<UpsertResponse>;

    /// Deletes vectors from the storage.
    /// 
    /// # Arguments
    /// * `ids` - Vector of IDs to delete
    /// 
    /// # Returns
    /// A Result indicating success or failure of the deletion
    async fn delete(&mut self, ids: Vec<String>) -> Result<()>;
}

/// Data structure to represent a vector and its metadata.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorData {
    /// Unique identifier for the vector.
    pub id: String,
    /// The vector representation as a list of 32-bit floats.
    pub vector: Vec<f32>,
    /// Additional metadata associated with the vector.
    pub metadata: HashMap<String, Value>,
}

/// Data structure for query results.
#[derive(Clone, Debug)]
pub struct SearchResult {
    /// ID of the matched vector.
    pub id: String,
    /// Similarity score between query and matched vector.
    pub distance: f32,
    /// Metadata associated with the matched vector.
    pub metadata: HashMap<String, Value>,
}

/// Data structure for the upsert response.
#[derive(Clone, Debug)]
pub struct UpsertResponse {
    /// List of IDs for newly inserted vectors.
    pub inserted: Vec<String>,
    /// List of IDs for updated vectors.
    pub updated: Vec<String>,
}

/// A basic implementation of vector storage mimicking NanoVectorDB-RS.
#[derive(Debug)]
pub struct NanoVectorStorage {
    /// In-memory storage of vectors.
    storage: Vec<VectorData>,
    /// Cosine similarity threshold for filtering query results.
    /// Results with similarity below this threshold will be filtered out.
    threshold: f32,
    /// HNSW index for approximate nearest neighbor search.
    hnsw: HNSWIndex,
    /// File path for vector storage persistence.
    storage_path: PathBuf,
}

impl NanoVectorStorage {
    /// Creates a new NanoVectorStorage instance.
    /// 
    /// # Arguments
    /// * `config` - Configuration object containing working directory information
    /// 
    /// # Returns
    /// A Result containing the new NanoVectorStorage instance or an error
    pub fn new(config: &crate::types::Config) -> Result<Self> {
        // Create HNSW index with default parameters
        let mut hnsw = HNSWIndex::new(1, 128, 16);
        
        // Set up persistence paths
        let hnsw_path = config.working_dir.join("hnsw_index.json");
        let storage_path = config.working_dir.join("vector_storage.json");
        
        // Ensure working directory exists
        if let Some(parent) = storage_path.parent() {
            fs::create_dir_all(parent)?;
        }
        
        hnsw.set_persistence_path(hnsw_path);
        
        Ok(NanoVectorStorage {
            storage: Vec::new(),
            threshold: 0.0, // Initialize with no threshold filtering
            hnsw,
            storage_path,
        })
    }

    /// Loads vector storage from file.
    /// 
    /// # Returns
    /// A Result indicating success or failure of the load operation
    fn load_storage(&mut self) -> Result<()> {
        if self.storage_path.exists() {
            let content = fs::read_to_string(&self.storage_path)?;
            if !content.trim().is_empty() {
                self.storage = serde_json::from_str(&content)?;
            }
        }
        Ok(())
    }

    /// Saves vector storage to file.
    /// 
    /// # Returns
    /// A Result indicating success or failure of the save operation
    fn save_storage(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.storage)?;
        fs::write(&self.storage_path, content)?;
        Ok(())
    }

    /// Normalizes a vector in-place by dividing by its L2 norm.
    /// 
    /// # Arguments
    /// * `vector` - Vector to normalize
    fn normalize_vector(vector: &mut Vec<f32>) {
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in vector.iter_mut() {
                *x /= norm;
            }
        }
    }
}

#[async_trait]
impl VectorStorage for NanoVectorStorage {
    async fn initialize(&mut self) -> Result<()> {
        // Load both HNSW index and vector storage from files
        self.hnsw.load()?;
        self.load_storage()?;
        Ok(())
    }

    async fn finalize(&mut self) -> Result<()> {
        // Save both HNSW index and vector storage to files
        self.hnsw.save()?;
        self.save_storage()?;
        Ok(())
    }

    async fn query(&self, mut query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>> {
        // Normalize the query vector
        Self::normalize_vector(&mut query);

        // Use the HNSW index for querying with proper ef_search parameter
        let hnsw_results = self.hnsw.query(query.as_slice(), top_k * 2, 128);
        
        // Convert candidates to SearchResult format
        let mut results: Vec<SearchResult> = hnsw_results.into_iter()
            // Only apply threshold filtering if it's greater than 0
            .filter(|(_id, sim)| self.threshold <= 0.0 || *sim >= self.threshold)
            .map(|(id, sim)| {
                let metadata = self.storage
                    .iter()
                    .find(|v| v.id == id)
                    .map(|v| v.metadata.clone())
                    .unwrap_or_default();
                SearchResult { id, distance: sim, metadata }
            })
            .collect();

        // Truncate to top_k after filtering
        results.truncate(top_k);
        Ok(results)
    }

    async fn upsert(&mut self, data: Vec<VectorData>) -> Result<UpsertResponse> {
        let mut inserted = Vec::new();
        let mut updated = Vec::new();

        for mut vector_data in data {
            // Normalize the vector
            Self::normalize_vector(&mut vector_data.vector);

            let mut exists = false;
            for existing in &mut self.storage {
                if existing.id == vector_data.id {
                    *existing = vector_data.clone();
                    updated.push(existing.id.clone());
                    exists = true;

                    // Update corresponding HNSW node
                    for node in self.hnsw.nodes.iter_mut() {
                        if node.id == vector_data.id {
                            node.vector = vector_data.vector.clone();
                            break;
                        }
                    }
                    break;
                }
            }

            if !exists {
                // Create HNSW node and add to index
                let node = hnsw::HNSWNode::new(&vector_data.id, vector_data.vector.clone());
                self.hnsw.add_node(node);
                
                self.storage.push(vector_data.clone());
                inserted.push(vector_data.id);
            }
        }

        // Save changes to disk
        self.save_storage()?;

        Ok(UpsertResponse { inserted, updated })
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        // Remove vectors from the in-memory storage
        self.storage.retain(|v| !ids.contains(&v.id));

        // Update the HNSW index by deleting corresponding nodes
        self.hnsw.delete_nodes(&ids);

        // Save changes to disk
        self.save_storage()?;

        Ok(())
    }
} 