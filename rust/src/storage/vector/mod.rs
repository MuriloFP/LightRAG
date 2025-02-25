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

pub mod optimization;
pub use optimization::{OptimizationConfig, OptimizationStats, VectorOptimizer, OptimizedVector};

use crate::types::Result;
use crate::types::Error;
use crate::utils::compute_mdhash_id;
use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use crate::storage::vector::hnsw::HNSWIndex;
use serde::{Serialize, Deserialize};
use std::fs;
use std::path::PathBuf;
use tracing;

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
    /// Creation timestamp
    #[serde(default = "std::time::SystemTime::now")]
    pub created_at: std::time::SystemTime,
    /// Optimized vector data if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimized: Option<OptimizedVector>,
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
    /// Maximum batch size for operations
    max_batch_size: usize,
    /// Lock for file operations
    save_lock: tokio::sync::Mutex<()>,
    /// Vector optimizer for compression and quantization
    optimizer: Option<VectorOptimizer>,
}

impl NanoVectorStorage {
    /// Creates a new NanoVectorStorage instance.
    pub fn new(config: &crate::types::Config) -> Result<Self> {
        // Get configuration values with defaults
        let threshold = config.get_f32("vector_db_storage.cosine_threshold")
            .unwrap_or(0.2);
        let max_batch_size = config.get_usize("vector_db_storage.batch_size")
            .unwrap_or(32);
        
        // Get HNSW parameters from config
        let max_layers = config.get_usize("vector_db_storage.hnsw.max_layers").unwrap_or(1);
        let ef_construction = config.get_usize("vector_db_storage.hnsw.construction_ef").unwrap_or(128);
        let m = config.get_usize("vector_db_storage.hnsw.m").unwrap_or(16);
        let batch_size = config.get_usize("vector_db_storage.hnsw.batch_size").unwrap_or(100);
        let sync_threshold = config.get_usize("vector_db_storage.hnsw.sync_threshold").unwrap_or(1000);
        
        // Get optimization parameters from config
        let use_optimization = config.get_bool("vector_db_storage.optimization.enabled").unwrap_or(true);
        let optimizer = if use_optimization {
            let opt_config = OptimizationConfig {
                use_pq: config.get_bool("vector_db_storage.optimization.use_pq").unwrap_or(true),
                pq_segments: config.get_usize("vector_db_storage.optimization.pq_segments").unwrap_or(8),
                pq_bits: config.get_u8("vector_db_storage.optimization.pq_bits").unwrap_or(8),
                use_sq: config.get_bool("vector_db_storage.optimization.use_sq").unwrap_or(true),
                sq_bits: config.get_u8("vector_db_storage.optimization.sq_bits").unwrap_or(8),
                use_compression: config.get_bool("vector_db_storage.optimization.use_compression").unwrap_or(true),
                compression_ratio: config.get_f32("vector_db_storage.optimization.compression_ratio").unwrap_or(0.5),
                max_error: config.get_f32("vector_db_storage.optimization.max_error").unwrap_or(0.01),
            };
            Some(VectorOptimizer::new(opt_config))
        } else {
            None
        };
        
        tracing::info!(
            threshold = threshold,
            batch_size = max_batch_size,
            max_layers = max_layers,
            ef_construction = ef_construction,
            m = m,
            use_optimization = use_optimization,
            "Initializing NanoVectorStorage"
        );
        
        // Create HNSW index with configured parameters
        let mut hnsw = HNSWIndex::new(max_layers, ef_construction, m);
        hnsw.batch_size = batch_size;
        hnsw.sync_threshold = sync_threshold;
        
        // If a dimension is provided in config.extra_config under the key "dim", set it for the HNSW index
        if let Some(dim) = config.get_usize("dim") {
            hnsw.set_dimension(dim);
        }
        
        // Set up persistence paths
        let storage_path = config.working_dir.join("vector_storage.json");
        
        // Ensure working directory exists
        if let Some(parent) = storage_path.parent() {
            fs::create_dir_all(parent).map_err(|e| 
                Error::VectorStorage(format!("Failed to create directory: {}", e)))?;
        }
        
        hnsw.set_persistence_path(storage_path.clone());
        
        Ok(NanoVectorStorage {
            storage: Vec::new(),
            threshold,
            hnsw,
            storage_path,
            max_batch_size,
            save_lock: tokio::sync::Mutex::new(()),
            optimizer,
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
                let loaded: Vec<VectorData> = serde_json::from_str(&content)?;
                
                // Normalize all vectors before storing
                self.storage = loaded.into_iter().map(|mut v| {
                    Self::normalize_vector(&mut v.vector);
                    v
                }).collect();
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

    /// Gets vectors by their IDs with optional field filtering.
    /// 
    /// # Arguments
    /// * `ids` - List of vector IDs to retrieve
    /// * `fields` - Optional list of metadata fields to include
    /// 
    /// # Returns
    /// A Result containing a vector of VectorData for the requested IDs
    pub async fn get_with_fields(&self, ids: &[String], fields: Option<&[String]>) -> Result<Vec<VectorData>> {
        let _guard = self.save_lock.lock().await;
        let mut results = Vec::new();
        
        for id in ids {
            if let Some(data) = self.storage.iter().find(|v| v.id == *id) {
                let mut filtered_data = data.clone();
                
                // If fields are specified, filter metadata
                if let Some(field_list) = fields {
                    let filtered_metadata: HashMap<String, Value> = data.metadata
                        .iter()
                        .filter(|(k, _)| field_list.contains(&k.to_string()))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    filtered_data.metadata = filtered_metadata;
                }
                
                results.push(filtered_data);
            }
        }
        
        Ok(results)
    }

    /// Deletes an entity and its vector by entity name.
    pub async fn delete_entity(&mut self, entity_name: &str) -> Result<()> {
        let entity_id = compute_mdhash_id(entity_name, "ent-");

        // Check if entity exists before deleting
        if let Some(_) = self.storage.iter().find(|v| v.id == entity_id) {
            self.delete(vec![entity_id]).await?;
            tracing::debug!("Successfully deleted entity {}", entity_name);
        } else {
            tracing::debug!("Entity {} not found in storage", entity_name);
        }
        Ok(())
    }

    /// Deletes all relations associated with an entity.
    pub async fn delete_entity_relation(&mut self, entity_name: &str) -> Result<()> {
        // Find all relations where entity is source or target
        let ids_to_delete: Vec<String> = self.storage.iter()
            .filter(|v| {
                v.metadata.get("src_id").and_then(|v| v.as_str()) == Some(entity_name) ||
                v.metadata.get("tgt_id").and_then(|v| v.as_str()) == Some(entity_name)
            })
            .map(|v| v.id.clone())
            .collect();

        if !ids_to_delete.is_empty() {
            self.delete(ids_to_delete).await?;
            tracing::debug!("Deleted relations for entity {}", entity_name);
        } else {
            tracing::debug!("No relations found for entity {}", entity_name);
        }
        Ok(())
    }

    /// Initializes vector optimization with training data.
    pub fn initialize_optimization(&mut self) -> Result<()> {
        if let Some(optimizer) = &mut self.optimizer {
            let training_vectors: Vec<Vec<f32>> = self.storage.iter()
                .map(|v| v.vector.clone())
                .collect();
            
            if !training_vectors.is_empty() {
                optimizer.initialize(&training_vectors)?;
                tracing::info!("Vector optimization initialized with {} training vectors", training_vectors.len());
            }
        }
        Ok(())
    }

    /// Gets optimization statistics.
    pub fn get_optimization_stats(&self) -> Option<&OptimizationStats> {
        self.optimizer.as_ref().map(|opt| opt.get_stats())
    }

    /// Creates a new instance for testing with optional optimization
    #[cfg(test)]
    pub fn new_for_testing(opt_config: Option<OptimizationConfig>) -> Result<Self> {
        use std::env::temp_dir;
        
        let mut config = crate::types::Config::default();
        config.working_dir = temp_dir();
        
        let mut storage = Self::new(&config)?;
        if let Some(opt_config) = opt_config {
            storage.optimizer = Some(VectorOptimizer::new(opt_config));
        }
        
        Ok(storage)
    }
}

#[async_trait]
impl VectorStorage for NanoVectorStorage {
    async fn initialize(&mut self) -> Result<()> {
        tracing::debug!("Initializing NanoVectorStorage");
        // Load storage first to get dimension
        self.load_storage()?;
        tracing::debug!("Loaded {} vectors from storage", self.storage.len());

        // Set dimension in HNSW index if we have vectors
        if let Some(first) = self.storage.first() {
            let dim = first.vector.len();
            tracing::debug!("Setting HNSW dimension to {}", dim);
            self.hnsw.set_dimension(dim);
        }

        // Now load HNSW index
        self.hnsw.load()?;
        tracing::debug!("Loaded HNSW index with {} nodes", self.hnsw.nodes.len());

        // If the HNSW index is empty but we have stored vectors, rebuild the index
        if self.hnsw.nodes.is_empty() && !self.storage.is_empty() {
            tracing::debug!("Rebuilding HNSW index from {} stored vectors", self.storage.len());
            // Add all vectors to the index
            for vector_data in &self.storage {
                let mut normalized_vector = vector_data.vector.clone();
                Self::normalize_vector(&mut normalized_vector);
                let node = hnsw::HNSWNode::new(&vector_data.id, normalized_vector);
                self.hnsw.add_node(node)?;
            }
            tracing::debug!("Finished rebuilding HNSW index");
        }

        // Initialize vector optimization with loaded data
        self.initialize_optimization()?;
        tracing::debug!("Vector storage initialization complete");
        
        Ok(())
    }

    async fn finalize(&mut self) -> Result<()> {
        println!("NanoVectorStorage: Starting finalize");
        
        // Acquire lock for file operations
        println!("NanoVectorStorage: Acquiring save lock");
        let _guard = self.save_lock.lock().await;
        println!("NanoVectorStorage: Save lock acquired");
        
        // Save HNSW index
        println!("NanoVectorStorage: Saving HNSW index with {} nodes", self.hnsw.nodes.len());
        match self.hnsw.save() {
            Ok(_) => println!("NanoVectorStorage: HNSW index saved successfully"),
            Err(e) => println!("NanoVectorStorage: Error saving HNSW index: {:?}", e),
        }
        
        // Save vector storage
        println!("NanoVectorStorage: Saving vector storage with {} vectors", self.storage.len());
        match self.save_storage() {
            Ok(_) => println!("NanoVectorStorage: Vector storage saved successfully"),
            Err(e) => println!("NanoVectorStorage: Error saving vector storage: {:?}", e),
        }
        
        println!("NanoVectorStorage: Finalize completed");
        Ok(())
    }

    async fn query(&self, mut query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>> {
        tracing::debug!("Querying vector storage with top_k={}", top_k);
        // Normalize the query vector
        Self::normalize_vector(&mut query);
        tracing::debug!("Normalized query vector: {:?}", query);

        // Use the HNSW index for querying with proper ef_search parameter
        let hnsw_results = self.hnsw.query(query.as_slice(), top_k, 128)?;
        tracing::debug!("HNSW returned {} results", hnsw_results.len());
        
        // Map all candidates to SearchResult format
        let mut candidates: Vec<SearchResult> = hnsw_results.into_iter().map(|(id, sim)| {
            let vector_data = self.storage
                .iter()
                .find(|v| v.id == id)
                .unwrap();
            
            SearchResult {
                id: id.clone(),
                distance: sim,
                metadata: vector_data.metadata.clone(),
            }
        }).collect();

        // Sort candidates by descending similarity
        candidates.sort_by(|a, b| b.distance.partial_cmp(&a.distance).unwrap());
        tracing::debug!("Found {} candidates before filtering", candidates.len());

        // If a threshold is set, filter the candidates strictly
        if self.threshold > 0.0 {
            let filtered: Vec<SearchResult> = candidates.into_iter().filter(|r| r.distance >= self.threshold).collect();
            tracing::debug!("Filtered to {} results with threshold {}", filtered.len(), self.threshold);
            Ok(filtered.into_iter().take(top_k).collect())
        } else {
            let len = candidates.len();
            let results = candidates.into_iter().take(top_k).collect();
            tracing::debug!("Returning {} results from {} candidates", top_k, len);
            Ok(results)
        }
    }

    async fn upsert(&mut self, data: Vec<VectorData>) -> Result<UpsertResponse> {
        tracing::debug!("Upserting {} vectors", data.len());
        let mut inserted = Vec::new();
        let mut updated = Vec::new();

        // Process in batches
        for chunk in data.chunks(self.max_batch_size) {
            tracing::debug!("Processing batch of {} vectors", chunk.len());
            for mut vector_data in chunk.to_vec() {
                // Set creation time for new vectors
                let is_new = !self.storage.iter().any(|v| v.id == vector_data.id);
                if is_new {
                    vector_data.created_at = std::time::SystemTime::now();
                }
                
                // Normalize the vector
                Self::normalize_vector(&mut vector_data.vector);
                tracing::debug!("Normalized vector for {}: {:?}", vector_data.id, vector_data.vector);

                let mut exists = false;
                for existing in &mut self.storage {
                    if existing.id == vector_data.id {
                        let created_at = existing.created_at;
                        *existing = vector_data.clone();
                        existing.created_at = created_at;
                        updated.push(existing.id.clone());
                        exists = true;
                        tracing::debug!("Updating existing vector {}", vector_data.id);
                        // Instead of directly updating the HNSW node, remove and re-add it to update connections
                        self.hnsw.delete_nodes(&[vector_data.id.clone()]);
                        let new_node = hnsw::HNSWNode::new(&vector_data.id, vector_data.vector.clone());
                        self.hnsw.add_node(new_node)?;
                        break;
                    }
                }

                if !exists {
                    tracing::debug!("Inserting new vector {}", vector_data.id);
                    // Create HNSW node and add to index
                    let node = hnsw::HNSWNode::new(&vector_data.id, vector_data.vector.clone());
                    self.hnsw.add_node(node)?;
                    
                    self.storage.push(vector_data.clone());
                    inserted.push(vector_data.id);
                }
            }
        }

        tracing::debug!("Upsert complete: {} inserted, {} updated", inserted.len(), updated.len());
        // Save changes to disk with lock
        let _guard = self.save_lock.lock().await;
        self.save_storage()?;
        drop(_guard);

        Ok(UpsertResponse { inserted, updated })
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        // Remove vectors from the in-memory storage
        self.storage.retain(|v| !ids.contains(&v.id));

        // Update the HNSW index by deleting corresponding nodes
        self.hnsw.delete_nodes(&ids);

        // Save changes to disk with lock
        let _guard = self.save_lock.lock().await;
        self.save_storage()?;
        drop(_guard);

        Ok(())
    }
} 