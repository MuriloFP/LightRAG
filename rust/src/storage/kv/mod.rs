use crate::types::error::Result;
use crate::types::{Config};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use tracing::info;
use tokio::sync::Mutex;
use async_trait::async_trait;

/// Trait for Key-Value Storage operations
#[async_trait]
pub trait KVStorage: Send + Sync {
    /// Initialize the storage (e.g., load from file)
    async fn initialize(&mut self) -> Result<()>;
    
    /// Finalize the storage (e.g., persist data to file)
    async fn finalize(&mut self) -> Result<()>;
    
    /// Get a value by its id
    async fn get_by_id(&self, id: &str) -> Result<Option<HashMap<String, Value>>>;
    
    /// Get values for a list of ids
    /// Returns a list of dictionaries, where each dictionary contains the full data for the ID
    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<HashMap<String, Value>>>;
    
    /// Given a set of keys, return those that don't exist in the storage
    async fn filter_keys(&self, keys: &HashSet<String>) -> Result<HashSet<String>>;
    
    /// Upsert (insert/update) data into storage
    /// Only inserts keys that don't already exist
    async fn upsert(&mut self, data: HashMap<String, HashMap<String, Value>>) -> Result<()>;

    /// Delete documents by their IDs
    async fn delete(&mut self, doc_ids: &[String]) -> Result<()>;

    /// Drop all data from storage
    async fn drop(&mut self) -> Result<()>;

    /// Clear all data from storage and persist the empty state
    async fn clear(&mut self) -> Result<()> {
        self.drop().await?;
        self.finalize().await
    }

    /// Called after indexing operations are complete
    /// Default implementation persists data
    async fn index_done_callback(&mut self) -> Result<()> {
        self.finalize().await
    }
}

/// JSON-based KV storage implementation
pub struct JsonKVStorage {
    /// The file where data is persisted
    file_path: String,
    /// The namespace for this storage instance
    namespace: String,
    /// In-memory data stored as a JSON object
    data: HashMap<String, HashMap<String, Value>>,
    /// Lock for thread safety, matching LightRAG's asyncio.Lock()
    lock: Mutex<()>,
}

impl JsonKVStorage {
    /// Creates a new JSON-based key-value storage instance.
    /// 
    /// # Arguments
    /// * `config` - Configuration object containing working directory information
    /// * `namespace` - Namespace to isolate this storage instance's data
    /// 
    /// # Returns
    /// A Result containing the new JsonKVStorage instance or an error
    pub fn new(config: &Config, namespace: &str) -> Result<Self> {
        // Build file path from working directory in config
        let file_path = config.working_dir.join(format!("kv_store_{}.json", namespace));
        // Ensure the working directory exists
        std::fs::create_dir_all(&config.working_dir)?;
        Ok(JsonKVStorage {
            data: HashMap::new(),
            namespace: namespace.to_string(),
            file_path: file_path.to_string_lossy().to_string(),
            lock: Mutex::new(()),
        })
    }
}

#[async_trait]
impl KVStorage for JsonKVStorage {
    async fn initialize(&mut self) -> Result<()> {
        use tokio::fs;
        use std::io;

        // Acquire lock for thread safety
        let _guard = self.lock.lock().await;

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
        info!(namespace = %self.namespace, count = self.data.len(), "Loaded KV storage");
        Ok(())
    }

    async fn finalize(&mut self) -> Result<()> {
        use tokio::fs;
        let _guard = self.lock.lock().await;
        // Serialize the data map to pretty JSON and write it to file
        let content = serde_json::to_string_pretty(&self.data)?;
        fs::write(&self.file_path, content).await?;
        Ok(())
    }

    async fn get_by_id(&self, id: &str) -> Result<Option<HashMap<String, Value>>> {
        let _guard = self.lock.lock().await;
        Ok(self.data.get(id).cloned())
    }

    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<HashMap<String, Value>>> {
        let _guard = self.lock.lock().await;
        let mut result = Vec::new();
        for id in ids {
            if let Some(data) = self.data.get(id) {
                result.push(data.clone());
            }
        }
        Ok(result)
    }

    async fn filter_keys(&self, keys: &HashSet<String>) -> Result<HashSet<String>> {
        let _guard = self.lock.lock().await;
        let existing: HashSet<String> = self.data.keys().cloned().collect();
        Ok(keys.difference(&existing).cloned().collect())
    }

    async fn upsert(&mut self, data: HashMap<String, HashMap<String, Value>>) -> Result<()> {
        let _guard = self.lock.lock().await;
        info!(namespace = %self.namespace, count = data.len(), "Inserting data");
        if data.is_empty() {
            return Ok(());
        }
        // Match LightRAG's behavior: only insert keys that don't exist
        let left_data: HashMap<_, _> = data.into_iter()
            .filter(|(k, _)| !self.data.contains_key(k))
            .collect();
        self.data.extend(left_data);
        Ok(())
    }

    async fn delete(&mut self, doc_ids: &[String]) -> Result<()> {
        let _guard = self.lock.lock().await;
        info!(namespace = %self.namespace, count = doc_ids.len(), "Deleting documents");
        for doc_id in doc_ids {
            self.data.remove(doc_id);
        }
        drop(_guard); // Drop the guard before calling index_done_callback
        self.index_done_callback().await
    }

    async fn drop(&mut self) -> Result<()> {
        let _guard = self.lock.lock().await;
        info!(namespace = %self.namespace, "Dropping all data");
        self.data.clear();
        drop(_guard); // Drop the guard before potential future operations
        Ok(())
    }

    async fn index_done_callback(&mut self) -> Result<()> {
        self.finalize().await
    }
}

/// Module for tracking and managing document processing status.
/// 
/// This module provides functionality to:
/// - Track document processing states (Pending, Processing, Completed, Failed)
/// - Store and retrieve document metadata
/// - Query document status counts
/// - Filter documents by status
pub mod doc_status; 