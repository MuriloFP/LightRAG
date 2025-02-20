use anyhow::{Result, Context};
use rand::Rng;
use serde::{Serialize, Deserialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;
use md5;

/// Storage container for vector data records.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Storage {
    /// Collection of vector data records.
    pub data: Vec<Data>,
    /// Additional metadata storage
    #[serde(default)]
    pub additional_data: HashMap<String, serde_json::Value>,
}

/// Represents a single vector data record with metadata.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Data {
    /// Unique identifier for the vector record.
    pub __id__: String,
    /// The vector representation of the data point.
    pub __vector__: Vec<f64>,
    /// Timestamp when the record was created.
    pub __created_at__: f64,
    /// Additional metadata associated with the vector.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Main vector database implementation supporting vector storage and similarity search.
#[derive(Debug)]
pub struct NanoVectorDB {
    /// Dimension of the vectors stored in the database.
    pub embedding_dim: usize,
    /// Distance metric used for similarity calculations (e.g., "cosine").
    pub metric: String,
    /// Path to the file where vector data is persisted.
    pub storage_file: String,
    /// In-memory storage of vector data.
    pub storage: Storage,
    /// Minimum similarity threshold for vector matches.
    pub cosine_threshold: f64,
    /// Maximum number of vectors to process in a single batch operation.
    pub max_batch_size: usize,
}

impl NanoVectorDB {
    /// Creates a new NanoVectorDB instance.
    /// 
    /// # Arguments
    /// * `embedding_dim` - Dimension of vectors to be stored
    /// * `metric` - Distance metric to use for similarity calculations
    /// * `storage_file` - Path to file where vector data will be persisted
    /// 
    /// # Returns
    /// A Result containing the new NanoVectorDB instance or an error
    pub fn new(embedding_dim: usize, metric: &str, storage_file: &str) -> Result<Self> {
        let storage = if Path::new(storage_file).exists() {
            let content = fs::read_to_string(storage_file)
                .with_context(|| format!("Failed to read file {}", storage_file))?;
            if content.trim().is_empty() {
                Storage::default()
            } else {
                serde_json::from_str(&content)
                    .with_context(|| "Failed to parse JSON")?
            }
        } else {
            Storage::default()
        };
        Ok(NanoVectorDB {
            embedding_dim,
            metric: metric.to_string(),
            storage_file: storage_file.to_string(),
            storage,
            cosine_threshold: 0.2, // Default threshold matching LightRAG
            max_batch_size: 32,    // Default batch size
        })
    }

    /// Gets additional metadata stored in the database.
    pub fn get_additional_data(&self) -> &HashMap<String, serde_json::Value> {
        &self.storage.additional_data
    }

    /// Stores additional metadata in the database.
    pub fn store_additional_data(&mut self, data: HashMap<String, serde_json::Value>) {
        self.storage.additional_data = data;
    }

    /// Gets vectors by their IDs.
    pub fn get(&self, ids: &[String]) -> Vec<Data> {
        let id_set: std::collections::HashSet<_> = ids.iter().collect();
        self.storage.data.iter()
            .filter(|data| id_set.contains(&data.__id__))
            .cloned()
            .collect()
    }

    /// Returns the number of vectors in the database.
    pub fn len(&self) -> usize {
        self.storage.data.len()
    }

    /// Returns whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.storage.data.is_empty()
    }

    /// Queries the database for similar vectors.
    pub fn query(
        &self,
        query: &[f64],
        top_k: usize,
        better_than_threshold: Option<f64>,
        filter_lambda: Option<Box<dyn Fn(&Data) -> bool>>,
    ) -> Vec<Data> {
        // Normalize query vector for cosine similarity
        let query_norm = (query.iter().map(|x| x * x).sum::<f64>()).sqrt();
        let query = if query_norm > 0.0 {
            query.iter().map(|x| x / query_norm).collect::<Vec<_>>()
        } else {
            query.to_vec()
        };

        // Filter data if lambda provided
        let filtered_data: Vec<_> = if let Some(filter) = filter_lambda {
            self.storage.data.iter()
                .filter(|data| filter(data))
                .collect()
        } else {
            self.storage.data.iter().collect()
        };

        // Calculate similarities and sort
        let mut results: Vec<_> = filtered_data.iter()
            .map(|data| {
                let similarity = cosine_similarity(&query, &data.__vector__);
                (*data, similarity)
            })
            .filter(|(_, sim)| {
                better_than_threshold.map_or(true, |threshold| *sim >= threshold)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Take top_k results and format them with all required fields
        results.into_iter()
            .take(top_k)
            .map(|(data, similarity)| {
                let mut result = data.clone();
                result.metadata.insert("__metrics__".to_string(), json!(similarity));
                result.metadata.insert("id".to_string(), json!(data.__id__.clone()));
                result.metadata.insert("distance".to_string(), json!(similarity));
                result.metadata.insert("created_at".to_string(), json!(data.__created_at__));
                result
            })
            .collect()
    }

    /// Upserts (inserts or updates) vector records into the database.
    pub fn upsert(&mut self, records: Vec<Data>) -> serde_json::Value {
        let mut inserted = Vec::new();
        let mut updated = Vec::new();

        for mut record in records {
            // Normalize vector for cosine similarity
            if self.metric == "cosine" {
                let norm = record.__vector__.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 0.0 {
                    for x in record.__vector__.iter_mut() {
                        *x /= norm;
                    }
                }
            }

            let id = record.__id__.clone();
            let mut exists = false;

            // Update existing record
            for existing in &mut self.storage.data {
                if existing.__id__ == id {
                    *existing = record.clone();
                    updated.push(id.clone());
                    exists = true;
                    break;
                }
            }

            // Insert new record
            if !exists {
                self.storage.data.push(record);
                inserted.push(id);
            }
        }

        json!({
            "insert": inserted,
            "update": updated
        })
    }

    /// Deletes vector records with the specified IDs from the database.
    /// 
    /// # Arguments
    /// * `ids` - Array of record IDs to delete
    /// 
    /// # Returns
    /// Result indicating success or failure of the operation
    pub fn delete(&mut self, ids: &[String]) -> Result<()> {
        self.storage.data.retain(|data| !ids.contains(&data.__id__));
        self.save()?;
        Ok(())
    }

    /// Deletes a vector record associated with a specific entity.
    /// 
    /// # Arguments
    /// * `entity_name` - Name of the entity to delete
    /// 
    /// # Returns
    /// Result indicating success or failure of the operation
    pub fn delete_entity(&mut self, entity_name: &str) -> Result<()> {
        let entity_id = format!("ent-{:x}", md5::compute(entity_name.as_bytes()));
        self.delete(&[entity_id])
    }

    /// Deletes all vector records that have a relation to the specified entity.
    /// 
    /// # Arguments
    /// * `entity_name` - Name of the entity whose relations should be deleted
    /// 
    /// # Returns
    /// Result indicating success or failure of the operation
    pub fn delete_entity_relation(&mut self, entity_name: &str) -> Result<()> {
        self.storage.data.retain(|data| {
            !(data.metadata.get("src_id").and_then(|v| v.as_str()) == Some(entity_name) ||
              data.metadata.get("tgt_id").and_then(|v| v.as_str()) == Some(entity_name))
        });
        self.save()
    }

    /// Saves the current state of the database to disk.
    /// 
    /// # Returns
    /// Result indicating success or failure of the save operation
    pub fn save(&self) -> Result<()> {
        let serialized = serde_json::to_string_pretty(&self.storage)?;
        let mut file = fs::File::create(&self.storage_file)
            .with_context(|| format!("Failed to create file {}", self.storage_file))?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }
}

/// Computes cosine similarity between two vectors.
fn cosine_similarity(v1: &[f64], v2: &[f64]) -> f64 {
    let dot_product: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f64 = v1.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm2: f64 = v2.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm1 == 0.0 || norm2 == 0.0 {
        0.0
    } else {
        dot_product / (norm1 * norm2)
    }
}

/// Multi-tenant vector database that manages separate vector storage instances for different tenants.
#[derive(Debug)]
pub struct MultiTenantNanoVDB {
    /// Dimension of vectors stored across all tenant databases.
    pub embedding_dim: usize,
    /// Distance metric used for similarity calculations across all tenant databases.
    pub metric: String,
    /// Map of tenant IDs to their respective vector database instances.
    pub tenants: HashMap<String, NanoVectorDB>,
    /// Directory where tenant-specific storage files are kept.
    pub storage_dir: String,
    /// Maximum number of tenants allowed in the system.
    pub max_tenants: usize,
}

impl MultiTenantNanoVDB {
    /// Creates a new multi-tenant vector database instance.
    /// 
    /// # Arguments
    /// * `embedding_dim` - Dimension of vectors to be stored
    /// * `metric` - Distance metric to use for similarity calculations
    /// * `max_tenants` - Maximum number of tenants allowed
    /// * `storage_dir` - Directory where tenant data will be stored
    /// 
    /// # Returns
    /// A new MultiTenantNanoVDB instance
    pub fn new(embedding_dim: usize, metric: &str, max_tenants: usize, storage_dir: &str) -> Self {
        if !Path::new(storage_dir).exists() {
            let _ = fs::create_dir_all(storage_dir);
        }
        MultiTenantNanoVDB {
            embedding_dim,
            metric: metric.to_string(),
            tenants: HashMap::new(),
            storage_dir: storage_dir.to_string(),
            max_tenants,
        }
    }

    /// Creates a new tenant and returns its ID.
    /// 
    /// # Returns
    /// A string containing the newly created tenant's ID
    pub fn create_tenant(&mut self) -> String {
        let tenant_id = format!("{}", rand::thread_rng().gen::<u32>());
        let tenant_file = Path::new(&self.storage_dir).join(Self::jsonfile_from_id(&tenant_id));
        let tenant = NanoVectorDB::new(self.embedding_dim, &self.metric, tenant_file.to_str().unwrap()).unwrap();
        self.tenants.insert(tenant_id.clone(), tenant);
        tenant_id
    }

    /// Gets a mutable reference to a tenant's vector database.
    /// 
    /// # Arguments
    /// * `tenant_id` - ID of the tenant to retrieve
    /// 
    /// # Returns
    /// Option containing a mutable reference to the tenant's database if it exists
    pub fn get_tenant(&mut self, tenant_id: &str) -> Option<&mut NanoVectorDB> {
        self.tenants.get_mut(tenant_id)
    }

    /// Deletes a tenant and its associated data.
    /// 
    /// # Arguments
    /// * `tenant_id` - ID of the tenant to delete
    pub fn delete_tenant(&mut self, tenant_id: &str) {
        if self.tenants.remove(tenant_id).is_some() {
            let _tenant_file = Path::new(&self.storage_dir).join(Self::jsonfile_from_id(tenant_id));
            let _ = fs::remove_file(_tenant_file);
        }
    }

    /// Saves the state of all tenant databases to disk.
    pub fn save(&self) {
        for (tenant_id, tenant) in &self.tenants {
            let _tenant_file = Path::new(&self.storage_dir).join(Self::jsonfile_from_id(tenant_id));
            let _ = tenant.save();
        }
    }

    /// Generates a JSON file name for a given tenant ID.
    /// 
    /// # Arguments
    /// * `tenant_id` - ID of the tenant
    /// 
    /// # Returns
    /// A string containing the JSON file name for the tenant
    pub fn jsonfile_from_id(tenant_id: &str) -> String {
        format!("tenant_{}.json", tenant_id)
    }
} 