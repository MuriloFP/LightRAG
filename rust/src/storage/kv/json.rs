use crate::storage::kv::KVStorage;
use crate::types::Result;
use async_trait::async_trait;
use serde_json::{self, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tokio::fs;
use tokio::sync::RwLock;
use std::io;

/// JSON-based KV storage implementation
pub struct JsonKVStorage {
    /// The file where data is persisted
    file_name: PathBuf,
    /// In-memory data stored as a JSON object
    data: RwLock<HashMap<String, Value>>,
}

impl JsonKVStorage {
    /// Create a new JSON KV storage with a given file name
    pub async fn new<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let file_name = file_path.as_ref().to_path_buf();
        // Try to load existing data from file; if error, assume empty data
        let data = match fs::read_to_string(&file_name).await {
            Ok(content) => {
                if content.is_empty() {
                    HashMap::new()
                } else {
                    serde_json::from_str::<HashMap<String, Value>>(&content).unwrap_or_else(|_| HashMap::new())
                }
            },
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    HashMap::new()
                } else {
                    return Err(e.into());
                }
            }
        };

        Ok(JsonKVStorage {
            file_name,
            data: RwLock::new(data),
        })
    }

    /// Helper function to persist data to file asynchronously
    async fn persist(&self) -> Result<()> {
        let data = self.data.read().await;
        let content = serde_json::to_string_pretty(&*data)?;
        fs::write(&self.file_name, content).await?;
        Ok(())
    }
}

#[async_trait]
impl KVStorage for JsonKVStorage {
    async fn initialize(&mut self) -> Result<()> {
        // Re-load data from file in case of external changes
        let content = match fs::read_to_string(&self.file_name).await {
            Ok(content) => content,
            Err(e) => {
                if e.kind() == io::ErrorKind::NotFound {
                    String::new()
                } else {
                    return Err(e.into());
                }
            }
        };

        let loaded: HashMap<String, Value> = if content.is_empty() {
            HashMap::new()
        } else {
            serde_json::from_str(&content)?
        };
        let mut data_lock = self.data.write().await;
        *data_lock = loaded;
        Ok(())
    }

    async fn finalize(&mut self) -> Result<()> {
        self.persist().await
    }

    async fn get_by_id(&self, id: &str) -> Result<Option<Value>> {
        let data = self.data.read().await;
        Ok(data.get(id).cloned())
    }

    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Value>> {
        let data = self.data.read().await;
        Ok(ids.iter().filter_map(|id| data.get(id).cloned()).collect())
    }

    async fn filter_keys(&self, keys: &HashSet<String>) -> Result<HashSet<String>> {
        let data = self.data.read().await;
        let existing: HashSet<String> = data.keys().cloned().collect();
        Ok(keys.difference(&existing).cloned().collect())
    }

    async fn upsert(&mut self, new_data: HashMap<String, Value>) -> Result<()> {
        if new_data.is_empty() {
            return Ok(());
        }
        let mut data = self.data.write().await;
        // Mimic Python behavior: insert key only if it does not exist
        for (k, v) in new_data.into_iter() {
            if !data.contains_key(&k) {
                data.insert(k, v);
            }
        }
        Ok(())
    }
} 