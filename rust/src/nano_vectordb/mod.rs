use anyhow::{Result, Context};
use ndarray::Array2;
use rand::Rng;
use serde::{Serialize, Deserialize};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Storage {
    pub data: Vec<Data>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Data {
    pub __id__: String,
    pub __vector__: Vec<f64>,
}

#[derive(Debug)]
pub struct NanoVectorDB {
    pub embedding_dim: usize,
    pub metric: String,
    pub storage_file: String,
    pub storage: Storage,
}

impl NanoVectorDB {
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
        })
    }

    pub fn upsert(&mut self, records: Vec<Data>, _arrays: Vec<Array2<f64>>) -> serde_json::Value {
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
        json!({ "insert": inserted })
    }

    pub fn save(&self) -> Result<()> {
        let serialized = serde_json::to_string_pretty(&self.storage)?;
        let mut file = fs::File::create(&self.storage_file)
            .with_context(|| format!("Failed to create file {}", self.storage_file))?;
        file.write_all(serialized.as_bytes())?;
        Ok(())
    }
}

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

    pub fn create_tenant(&mut self) -> String {
        let tenant_id = format!("{}", rand::thread_rng().gen::<u32>());
        let tenant_file = Path::new(&self.storage_dir).join(Self::jsonfile_from_id(&tenant_id));
        let tenant = NanoVectorDB::new(self.embedding_dim, &self.metric, tenant_file.to_str().unwrap()).unwrap();
        self.tenants.insert(tenant_id.clone(), tenant);
        tenant_id
    }

    pub fn get_tenant(&mut self, tenant_id: &str) -> Option<&mut NanoVectorDB> {
        self.tenants.get_mut(tenant_id)
    }

    pub fn delete_tenant(&mut self, tenant_id: &str) {
        if self.tenants.remove(tenant_id).is_some() {
            let tenant_file = Path::new(&self.storage_dir).join(Self::jsonfile_from_id(tenant_id));
            let _ = fs::remove_file(tenant_file);
        }
    }

    pub fn save(&self) {
        for (tenant_id, tenant) in &self.tenants {
            let tenant_file = Path::new(&self.storage_dir).join(Self::jsonfile_from_id(tenant_id));
            let _ = tenant.save();
        }
    }

    pub fn jsonfile_from_id(tenant_id: &str) -> String {
        format!("tenant_{}.json", tenant_id)
    }
} 