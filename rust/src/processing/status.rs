use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use thiserror::Error;

/// Document status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentStatus {
    /// Document is pending processing
    Pending,
    /// Document is currently being processed
    Processing,
    /// Document has been successfully processed
    Completed,
    /// Document processing has failed
    Failed,
}

/// Document status error types
#[derive(Error, Debug)]
pub enum DocumentStatusError {
    /// Invalid status transition attempted
    #[error("Invalid status transition from {from:?} to {to:?}")]
    InvalidTransition {
        /// Current status
        from: DocumentStatus,
        /// Attempted new status
        to: DocumentStatus,
    },
    /// Document not found
    #[error("Document not found: {0}")]
    DocumentNotFound(String),
    /// Storage error
    #[error("Storage error: {0}")]
    StorageError(String),
}

/// Document metadata struct
#[derive(Debug, Clone)]
pub struct DocumentMetadata {
    /// Unique identifier for the document
    pub id: String,
    /// Current status of the document
    pub status: DocumentStatus,
    /// When the document was created
    pub created_at: DateTime<Utc>,
    /// When the document was last updated
    pub updated_at: DateTime<Utc>,
    /// Content summary (truncated version of the content)
    pub content_summary: String,
    /// Length of the original content
    pub content_length: usize,
    /// Number of chunks the document was split into (if processed)
    pub chunks_count: Option<usize>,
    /// Error message if processing failed
    pub error: Option<String>,
    /// Additional custom metadata
    pub custom_metadata: HashMap<String, String>,
}

impl DocumentMetadata {
    /// Create new document metadata
    pub fn new(id: String, content_summary: String, content_length: usize) -> Self {
        let now = Utc::now();
        Self {
            id,
            status: DocumentStatus::Pending,
            created_at: now,
            updated_at: now,
            content_summary,
            content_length,
            chunks_count: None,
            error: None,
            custom_metadata: HashMap::new(),
        }
    }

    /// Update the status with validation
    pub fn update_status(&mut self, new_status: DocumentStatus) -> Result<(), DocumentStatusError> {
        // Validate the transition
        if !Self::is_valid_transition(self.status, new_status) {
            return Err(DocumentStatusError::InvalidTransition {
                from: self.status,
                to: new_status,
            });
        }

        self.status = new_status;
        self.updated_at = Utc::now();
        Ok(())
    }

    /// Check if a status transition is valid
    fn is_valid_transition(from: DocumentStatus, to: DocumentStatus) -> bool {
        use DocumentStatus::*;
        match (from, to) {
            // From Pending
            (Pending, Processing) => true,
            (Pending, _) => false,
            
            // From Processing
            (Processing, Completed) => true,
            (Processing, Failed) => true,
            (Processing, _) => false,
            
            // From Completed (terminal state)
            (Completed, _) => false,
            
            // From Failed (can retry)
            (Failed, Processing) => true,
            (Failed, _) => false,
        }
    }
}

/// Document status manager trait
#[async_trait::async_trait]
pub trait DocumentStatusManager: Send + Sync {
    /// Get metadata for a document by ID
    async fn get_metadata(&self, id: &str) -> Result<Option<DocumentMetadata>, DocumentStatusError>;
    
    /// Update metadata for a document
    async fn update_metadata(&self, metadata: DocumentMetadata) -> Result<(), DocumentStatusError>;
    
    /// Get all documents with a specific status
    async fn get_by_status(&self, status: DocumentStatus) -> Result<Vec<DocumentMetadata>, DocumentStatusError>;
    
    /// Get counts of documents by status
    async fn get_status_counts(&self) -> Result<HashMap<DocumentStatus, usize>, DocumentStatusError>;
    
    /// Delete metadata for a document
    async fn delete_metadata(&self, id: &str) -> Result<(), DocumentStatusError>;
}

/// In-memory implementation of DocumentStatusManager
pub struct InMemoryStatusManager {
    documents: Arc<RwLock<HashMap<String, DocumentMetadata>>>,
}

impl InMemoryStatusManager {
    /// Create a new in-memory status manager
    pub fn new() -> Self {
        Self {
            documents: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryStatusManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl DocumentStatusManager for InMemoryStatusManager {
    async fn get_metadata(&self, id: &str) -> Result<Option<DocumentMetadata>, DocumentStatusError> {
        let documents = self.documents.read().await;
        Ok(documents.get(id).cloned())
    }

    async fn update_metadata(&self, metadata: DocumentMetadata) -> Result<(), DocumentStatusError> {
        let mut documents = self.documents.write().await;
        documents.insert(metadata.id.clone(), metadata);
        Ok(())
    }

    async fn get_by_status(&self, status: DocumentStatus) -> Result<Vec<DocumentMetadata>, DocumentStatusError> {
        let documents = self.documents.read().await;
        Ok(documents
            .values()
            .filter(|doc| doc.status == status)
            .cloned()
            .collect())
    }

    async fn get_status_counts(&self) -> Result<HashMap<DocumentStatus, usize>, DocumentStatusError> {
        let documents = self.documents.read().await;
        let mut counts = HashMap::new();
        
        // Initialize counts for all statuses to 0
        for status in [
            DocumentStatus::Pending,
            DocumentStatus::Processing,
            DocumentStatus::Completed,
            DocumentStatus::Failed,
        ] {
            counts.insert(status, 0);
        }
        
        // Count documents by status
        for doc in documents.values() {
            *counts.entry(doc.status).or_insert(0) += 1;
        }
        
        Ok(counts)
    }

    async fn delete_metadata(&self, id: &str) -> Result<(), DocumentStatusError> {
        let mut documents = self.documents.write().await;
        documents.remove(id);
        Ok(())
    }
} 