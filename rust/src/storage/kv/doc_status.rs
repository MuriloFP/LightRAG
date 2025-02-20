/// Module for tracking and managing document processing status.
/// 
/// This module provides functionality to:
/// - Track document processing states (Pending, Processing, Completed, Failed)
/// - Store and retrieve document metadata
/// - Query document status counts
/// - Filter documents by status
use crate::types::Result;
use crate::types::Config;
use crate::storage::kv::JsonKVStorage;
use serde_json::Value;
use std::collections::HashMap;
use tracing::info;
use crate::storage::kv::KVStorage;

/// Represents the possible states of document processing.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DocStatus {
    /// Document is queued but processing has not started.
    Pending,
    /// Document is currently being processed.
    Processing,
    /// Document has been successfully processed.
    Completed,
    /// Document processing encountered an error.
    Failed,
}

/// Represents the processing status and metadata of a document.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DocProcessingStatus {
    /// Current status of the document as a string representation.
    /// This field typically contains one of the DocStatus values as a string.
    pub status: String,
    /// Additional metadata associated with the document's processing.
    /// This can include timestamps, error messages, processing metrics, etc.
    pub metadata: HashMap<String, Value>,
}

/// JSON Document Status Storage implementation.
/// This wraps the standard JSON KV Storage but overrides some methods
/// to update (instead of only inserting) data, and adds additional methods.
pub struct JsonDocStatusStorage {
    inner: JsonKVStorage,
}

impl JsonDocStatusStorage {
    /// Creates a new JsonDocStatusStorage given a config and namespace.
    pub fn new(config: &Config, namespace: &str) -> Result<Self> {
        let inner = JsonKVStorage::new(config, namespace)?;
        Ok(JsonDocStatusStorage { inner })
    }

    /// Initializes the storage by delegating to inner storage.
    pub async fn initialize(&mut self) -> Result<()> {
        self.inner.initialize().await
    }

    /// Finalizes the storage by delegating to inner storage.
    pub async fn finalize(&mut self) -> Result<()> {
        self.inner.finalize().await
    }

    /// Gets a document by its id.
    pub async fn get_by_id(&self, id: &str) -> Result<Option<HashMap<String, Value>>> {
        self.inner.get_by_id(id).await
    }

    /// Gets documents by a list of ids.
    pub async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<HashMap<String, Value>>> {
        self.inner.get_by_ids(ids).await
    }

    /// Deletes documents by their IDs.
    pub async fn delete(&mut self, doc_ids: &[String]) -> Result<()> {
        self.inner.delete(doc_ids).await
    }

    /// Drops all data from the storage.
    pub async fn drop(&mut self) -> Result<()> {
        self.inner.drop().await
    }

    /// Upserts (inserts or updates) data into storage.
    /// Unlike the standard JsonKVStorage upsert which only inserts new keys,
    /// this implementation updates existing keys with new values.
    pub async fn upsert(&mut self, data: HashMap<String, HashMap<String, Value>>) -> Result<()> {
        // Acquire the lock from inner storage
        let _guard = self.inner.lock.lock().await;
        info!(namespace = %self.inner.namespace, count = data.len(), "DocStatus upsert: Inserting/Updating data");
        // For each key-value pair in data, update (or insert) directly in the inner map
        for (k, v) in data.into_iter() {
            self.inner.data.insert(k, v);
        }
        // Drop the guard to release the immutable borrow before calling a mutable method
        drop(_guard);
        // Persist changes by calling index_done_callback
        self.inner.index_done_callback().await
    }

    /// Returns counts of documents in each status.
    /// Assumes that each document is stored as a HashMap where a key "status" holds a string value.
    pub async fn get_status_counts(&self) -> Result<HashMap<String, i32>> {
        let _guard = self.inner.lock.lock().await;
        let mut counts: HashMap<String, i32> = HashMap::new();
        for doc in self.inner.data.values() {
            if let Some(status_value) = doc.get("status") {
                if let Some(status_str) = status_value.as_str() {
                    *counts.entry(status_str.to_string()).or_insert(0) += 1;
                }
            }
        }
        Ok(counts)
    }

    /// Returns all documents with the specified status.
    /// The provided status is compared as a string against the "status" field of each document.
    /// Documents matching the status are converted into DocProcessingStatus instances.
    pub async fn get_docs_by_status(&self, status: &str) -> Result<HashMap<String, DocProcessingStatus>> {
        let _guard = self.inner.lock.lock().await;
        let mut result: HashMap<String, DocProcessingStatus> = HashMap::new();
        for (doc_id, doc) in self.inner.data.iter() {
            if let Some(status_value) = doc.get("status") {
                if let Some(status_str) = status_value.as_str() {
                    if status_str == status {
                        // Create a DocProcessingStatus from the document
                        let dps = DocProcessingStatus {
                            status: status_str.to_string(),
                            metadata: doc.clone(),
                        };
                        result.insert(doc_id.clone(), dps);
                    }
                }
            }
        }
        Ok(result)
    }
} 