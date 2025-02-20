//! NanoVectorDB: A lightweight vector database implementation in Rust
//!
//! This module is an adaptation of the Python nano_vectordb.

use ndarray::{Array2, ArrayView1};
use serde::{Serialize, Deserialize};
use serde_json;
use std::fs;
use std::path::Path;
use base64;
use md5;
use std::error::Error;
use anyhow::{Result, Context};
use std::io::Write;
use rand::Rng;
use std::collections::HashMap;

// For safe casting of slices between f32 and u8, use bytemuck
use bytemuck::{cast_slice, Pod, Zeroable};

// Constants for field names
pub const F_ID: &str = "__id__";
pub const F_VECTOR: &str = "__vector__";
pub const F_METRICS: &str = "__metrics__";

/// A record of vector data
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Data {
    pub __id__: String,
    /// The vector as a Vec<f32>
    pub __vector__: Vec<f32>,
    // Additional fields can be added here
}

/// The database storage structure
#[derive(Serialize, Deserialize, Debug)]
pub struct DataBase {
    pub embedding_dim: usize,
    pub data: Vec<Data>,
    /// The matrix stored as a flattened Vec<f32>
    pub matrix: Vec<f32>,
}

/// Convert an Array2<f32> to a base64 encoded string
pub fn array_to_buffer_string(array: &Array2<f32>) -> String {
    let slice = array.as_slice().expect("Array should be contiguous");
    base64::encode(cast_slice(slice))
}

/// Convert a base64 encoded string back to an Array2<f32> with the given embedding dimension
pub fn buffer_string_to_array(base64_str: &str, embedding_dim: usize) -> Array2<f32> {
    let bytes = base64::decode(base64_str).expect("Failed to decode base64 string");
    // Cast bytes to f32 slice
    let vec: Vec<f32> = bytemuck::cast_slice(&bytes).to_vec();
    let rows = vec.len() / embedding_dim;
    Array2::from_shape_vec((rows, embedding_dim), vec).expect("Failed to reshape array")
}

/// Compute the MD5 hash of a 1D array view (vector)
pub fn hash_ndarray(a: &ArrayView1<f32>) -> String {
    let slice = a.as_slice().expect("Array view should be contiguous");
    format!("{:x}", md5::compute(cast_slice(slice)))
}

/// Normalize each row of the 2D array
pub fn normalize(a: &Array2<f32>) -> Array2<f32> {
    let mut normalized = a.clone();
    for mut row in normalized.genrows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 {
            row.mapv_inplace(|x| x / norm);
        }
    }
    normalized
}

/// NanoVectorDB is a lightweight vector database
#[derive(Debug)]
pub struct NanoVectorDB {
    pub embedding_dim: usize,
    pub metric: String, // Currently only "cosine" is supported
    pub storage_file: String,
    pub storage: DataBase,
}

impl NanoVectorDB {
    /// Create a new NanoVectorDB instance
    pub fn new(embedding_dim: usize, metric: &str, storage_file: &str) -> Result<Self> {
        let storage = if Path::new(storage_file).exists() {
            let content = fs::read_to_string(storage_file)
                .with_context(|| format!("Failed to read file {}", storage_file))?;
            if content.trim().is_empty() {
                DataBase {
                    embedding_dim,
                    data: Vec::new(),
                    matrix: Vec::new(),
                }
            } else {
                serde_json::from_str(&content)
                    .with_context(|| "Failed to parse JSON")?
            }
        } else {
            DataBase {
                embedding_dim,
                data: Vec::new(),
                matrix: Vec::new(),
            }
        };
        Ok(NanoVectorDB {
            embedding_dim,
            metric: metric.to_string(),
            storage_file: storage_file.to_string(),
            storage,
        })
    }

    /// Pre-process the storage: normalize the matrix if cosine similarity is used
    pub fn pre_process(&mut self) {
        if self.metric == "cosine" {
            if !self.storage.matrix.is_empty() {
                let rows = self.storage.matrix.len() / self.embedding_dim;
                let arr = Array2::from_shape_vec((rows, self.embedding_dim), self.storage.matrix.clone())
                    .expect("Failed to create array from matrix");
                let normalized = normalize(&arr);
                self.storage.matrix = normalized.into_raw_vec();
            }
        }
    }

    /// Upsert data into the database
    /// For simplicity, this method assumes that each Data record contains its vector in __vector__ field.
    /// new_vectors is a vector of Array2<f32> corresponding to the new vector representations to be inserted.
    pub fn upsert(&mut self, records: Vec<Data>, _arrays: Vec<Array2<f32>>) -> serde_json::Value {
        let mut inserted = Vec::new();
        for record in records {
            let mut exists = false;
            for existing in &mut self.storage.data {
                if existing.__id__ == record.__id__ {
                    *existing = record.clone();
                    exists = true;
                    break;
                }
            }
            if !exists {
                self.storage.data.push(record.clone());
                inserted.push(record.__id__);
            }
        }
        serde_json::json!({ "insert": inserted })
    }

    /// Save the current state to the storage file
    pub fn save(&self) -> Result<()> {
        let serialized = serde_json::to_string_pretty(&self.storage)?;
        let mut file = fs::File::create(&self.storage_file)
            .with_context(|| format!("Failed to create file {}", self.storage_file))?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }

    // Additional methods such as query, get, delete etc. can be implemented similarly.
}

/// MultiTenantNanoVDB manages multiple NanoVectorDB instances across tenants
#[derive(Debug)]
pub struct MultiTenantNanoVDB {
    pub embedding_dim: usize,
    pub metric: String,
    pub tenants: HashMap<String, NanoVectorDB>,
    pub storage_dir: String,
    pub max_tenants: usize,
}

impl MultiTenantNanoVDB {
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

    /// Generate a JSON file name for a tenant
    pub fn jsonfile_from_id(tenant_id: &str) -> String {
        format!("tenant_{}.json", tenant_id)
    }

    /// Create a new tenant and load into cache
    pub fn create_tenant(&mut self) -> String {
        let tenant_id = format!("{}", rand::thread_rng().gen::<u32>());
        let tenant_file = Path::new(&self.storage_dir).join(Self::jsonfile_from_id(&tenant_id));
        let tenant = NanoVectorDB::new(self.embedding_dim, &self.metric, tenant_file.to_str().unwrap()).unwrap();
        self.tenants.insert(tenant_id.clone(), tenant);
        tenant_id
    }

    /// Get tenant by id, loading from file if necessary
    pub fn get_tenant(&mut self, tenant_id: &str) -> Option<&mut NanoVectorDB> {
        self.tenants.get_mut(tenant_id)
    }

    /// Delete a tenant
    pub fn delete_tenant(&mut self, tenant_id: &str) {
        self.tenants.remove(tenant_id);
        let tenant_file = Path::new(&self.storage_dir).join(Self::jsonfile_from_id(tenant_id));
        if Path::new(&tenant_file).exists() {
            let _ = fs::remove_file(tenant_file);
        }
    }

    /// Save all tenants
    pub fn save(&self) {
        for (tenant_id, tenant) in &self.tenants {
            let tenant_file = Path::new(&self.storage_dir).join(Self::jsonfile_from_id(tenant_id));
            let _ = tenant.save();
        }
    }
} 