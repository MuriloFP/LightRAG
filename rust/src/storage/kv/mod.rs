use crate::types::{Result, Config};
use serde_json::Value;
use std::collections::{HashMap, HashSet};

use async_trait::async_trait;

/// Trait for Key-Value Storage operations
#[async_trait]
pub trait KVStorage: Send + Sync {
    /// Initialize the storage (e.g., load from file)
    async fn initialize(&mut self) -> Result<()>;
    
    /// Finalize the storage (e.g., persist data to file)
    async fn finalize(&mut self) -> Result<()>;
    
    /// Get a value by its id
    async fn get_by_id(&self, key: &str) -> Result<Option<String>>;
    
    /// Get values for a list of ids
    async fn get_by_ids(&self, keys: &[String]) -> Result<Vec<Option<String>>>;
    
    /// Given a set of keys, return those that don't exist in the storage
    async fn filter_keys(&self, pattern: &str) -> Result<Vec<String>>;
    
    /// Upsert (insert/update) data into storage
    async fn upsert(&mut self, key: &str, value: String) -> Result<()>;
}

pub struct JsonKVStorage {
    data: HashMap<String, String>,
    file_path: String,
}

impl JsonKVStorage {
    pub fn new(config: &Config) -> Result<Self> {
        // Build file path from working directory in config
        let file_path = config.working_dir.join("json_kv_storage.json");
        // Ensure the working directory exists
        std::fs::create_dir_all(&config.working_dir)?;
        Ok(JsonKVStorage {
            data: HashMap::new(),
            file_path: file_path.to_string_lossy().to_string(),
        })
    }
}

#[async_trait]
impl KVStorage for JsonKVStorage {
    async fn initialize(&mut self) -> Result<()> {
        use tokio::fs;
        use std::io;
        // Attempt to read the file; if not found, start with empty data
        match fs::read_to_string(&self.file_path).await {
            Ok(content) => {
                if content.trim().is_empty() {
                    self.data = HashMap::new();
                } else {
                    self.data = serde_json::from_str(&content)?;
                }
            },
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    self.data = HashMap::new();
                } else {
                    return Err(e.into());
                }
            }
        }
        Ok(())
    }

    async fn finalize(&mut self) -> Result<()> {
        use tokio::fs;
        // Serialize the data map to pretty JSON and write it to file
        let content = serde_json::to_string_pretty(&self.data)?;
        fs::write(&self.file_path, content).await?;
        Ok(())
    }

    async fn get_by_id(&self, key: &str) -> Result<Option<String>> {
        Ok(self.data.get(key).cloned())
    }

    async fn get_by_ids(&self, keys: &[String]) -> Result<Vec<Option<String>>> {
        let results = keys.iter().map(|k| self.data.get(k).cloned()).collect();
        Ok(results)
    }

    async fn filter_keys(&self, pattern: &str) -> Result<Vec<String>> {
        let results = self
            .data
            .keys()
            .filter(|k| k.contains(pattern))
            .cloned()
            .collect();
        Ok(results)
    }

    async fn upsert(&mut self, key: &str, value: String) -> Result<()> {
        self.data.insert(key.to_string(), value);
        Ok(())
    }
} 