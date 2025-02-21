use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use thiserror::Error;
use tokio::sync::RwLock;
use std::sync::Arc;
use sha2::{Sha256, Digest};

use crate::processing::summary::{
    ContentSummarizer, Summary, SummaryConfig,
};
use crate::processing::keywords::{
    KeywordExtractor, ExtractedKeywords, KeywordConfig,
};

/// Errors that can occur during document processing
#[derive(Error, Debug)]
pub enum DocumentError {
    /// Error when document is empty
    #[error("Empty document")]
    EmptyDocument,
    /// Error during processing
    #[error("Processing error: {0}")]
    ProcessingError(String),
    /// Error with storage
    #[error("Storage error: {0}")]
    StorageError(String),
    /// Invalid document status transition
    #[error("Invalid status transition: {0} -> {1}")]
    InvalidStatusTransition(String, String),
}

/// Status of document processing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DocumentStatus {
    /// Document is pending processing
    Pending,
    /// Document is currently being processed
    Processing,
    /// Document has been processed successfully
    Completed,
    /// Document processing failed
    Failed,
}

/// Document processing status with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentProcessingStatus {
    /// Current status
    pub status: DocumentStatus,
    /// Content hash
    pub content_hash: String,
    /// Length of content
    pub content_length: usize,
    /// Summary of content
    pub summary: Option<Summary>,
    /// Keywords extracted from content
    pub keywords: Option<ExtractedKeywords>,
    /// Number of chunks (if processed)
    pub chunks_count: Option<usize>,
    /// Error message (if failed)
    pub error_message: Option<String>,
    /// When the document was created
    pub created_at: DateTime<Utc>,
    /// When the document was last updated
    pub updated_at: DateTime<Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Configuration for document processing
#[derive(Debug, Clone)]
pub struct DocumentProcessingConfig {
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    /// Whether to process in parallel
    pub parallel_processing: bool,
    /// Maximum retries per document
    pub max_retries: usize,
    /// Whether to deduplicate content
    pub enable_deduplication: bool,
    /// Summary configuration
    pub summary_config: SummaryConfig,
    /// Keyword configuration
    pub keyword_config: KeywordConfig,
}

impl Default for DocumentProcessingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 10,
            parallel_processing: true,
            max_retries: 3,
            enable_deduplication: true,
            summary_config: SummaryConfig::default(),
            keyword_config: KeywordConfig::default(),
        }
    }
}

/// Document processor that handles batching and status tracking
pub struct DocumentProcessor {
    /// Configuration
    config: DocumentProcessingConfig,
    /// Content summarizer
    summarizer: Arc<dyn ContentSummarizer>,
    /// Keyword extractor
    keyword_extractor: Arc<dyn KeywordExtractor>,
    /// Storage for document status
    status_store: Arc<RwLock<HashMap<String, DocumentProcessingStatus>>>,
}

impl DocumentProcessor {
    /// Create a new DocumentProcessor
    pub fn new(
        config: DocumentProcessingConfig,
        summarizer: Arc<dyn ContentSummarizer>,
        keyword_extractor: Arc<dyn KeywordExtractor>,
    ) -> Self {
        Self {
            config,
            summarizer,
            keyword_extractor,
            status_store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Generate a unique ID for a document
    pub fn generate_doc_id(&self, content: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(content.as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Check if content is duplicate
    pub async fn is_duplicate(&self, content: &str) -> Result<bool, DocumentError> {
        if !self.config.enable_deduplication {
            return Ok(false);
        }

        let doc_id = self.generate_doc_id(content);
        let store = self.status_store.read().await;
        Ok(store.contains_key(&doc_id))
    }

    /// Update document status
    pub async fn update_status(
        &self,
        doc_id: &str,
        status: DocumentStatus,
        error: Option<String>,
        chunks_count: Option<usize>,
    ) -> Result<(), DocumentError> {
        let mut store = self.status_store.write().await;
        
        if let Some(doc_status) = store.get_mut(doc_id) {
            // Validate status transition
            match (doc_status.status, status) {
                (DocumentStatus::Pending, DocumentStatus::Processing) |
                (DocumentStatus::Processing, DocumentStatus::Completed) |
                (DocumentStatus::Processing, DocumentStatus::Failed) => {
                    doc_status.status = status;
                    doc_status.updated_at = Utc::now();
                    if let Some(err) = error {
                        doc_status.error_message = Some(err);
                    }
                    if let Some(count) = chunks_count {
                        doc_status.chunks_count = Some(count);
                    }
                    Ok(())
                }
                _ => Err(DocumentError::InvalidStatusTransition(
                    format!("{:?}", doc_status.status),
                    format!("{:?}", status),
                )),
            }
        } else {
            Err(DocumentError::StorageError(format!("Document {} not found", doc_id)))
        }
    }

    /// Process a single document
    pub async fn process_document(&self, content: &str) -> Result<String, DocumentError> {
        if content.is_empty() {
            return Err(DocumentError::EmptyDocument);
        }

        let doc_id = self.generate_doc_id(content);

        // Check for duplicates
        if self.is_duplicate(content).await? {
            return Ok(doc_id);
        }

        // Initialize document status
        let status = DocumentProcessingStatus {
            status: DocumentStatus::Pending,
            content_hash: doc_id.clone(),
            content_length: content.len(),
            summary: None,
            keywords: None,
            chunks_count: None,
            error_message: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            metadata: HashMap::new(),
        };

        // Store initial status
        self.status_store.write().await.insert(doc_id.clone(), status);

        // Update to processing
        self.update_status(&doc_id, DocumentStatus::Processing, None, None).await?;

        // Process the document
        match self.process_content(content).await {
            Ok((summary, keywords)) => {
                let mut store = self.status_store.write().await;
                if let Some(status) = store.get_mut(&doc_id) {
                    status.summary = Some(summary);
                    status.keywords = Some(keywords);
                    status.status = DocumentStatus::Completed;
                    status.updated_at = Utc::now();
                }
                Ok(doc_id)
            }
            Err(e) => {
                self.update_status(
                    &doc_id,
                    DocumentStatus::Failed,
                    Some(e.to_string()),
                    None,
                ).await?;
                Err(e)
            }
        }
    }

    async fn process_content(&self, content: &str) -> Result<(Summary, ExtractedKeywords), DocumentError> {
        let summary = self.summarizer
            .generate_summary(content)
            .await
            .map_err(|e| DocumentError::ProcessingError(e.to_string()))?;

        let keywords = self.keyword_extractor
            .extract_keywords(content)
            .await
            .map_err(|e| DocumentError::ProcessingError(format!("Keyword extraction error: {}", e)))?;

        Ok((summary, keywords))
    }

    /// Process multiple documents in batch
    pub async fn process_batch(&self, documents: Vec<String>) -> Result<Vec<String>, DocumentError> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        let batch_size = documents.len().min(self.config.max_batch_size);
        let mut results = Vec::with_capacity(batch_size);

        if self.config.parallel_processing {
            let mut tasks = Vec::with_capacity(batch_size);
            for doc in documents {
                let summarizer = Arc::clone(&self.summarizer);
                let keyword_extractor = Arc::clone(&self.keyword_extractor);
                let status_store = Arc::clone(&self.status_store);
                let config = self.config.clone();
                
                tasks.push(tokio::spawn(async move {
                    let processor = DocumentProcessor {
                        config,
                        summarizer,
                        keyword_extractor,
                        status_store,
                    };
                    processor.process_document(&doc).await
                }));
            }

            for task in tasks {
                match task.await {
                    Ok(result) => results.push(result?),
                    Err(e) => return Err(DocumentError::ProcessingError(e.to_string())),
                }
            }
        } else {
            for doc in documents {
                results.push(self.process_document(&doc).await?);
            }
        }

        Ok(results)
    }

    /// Get status counts
    pub async fn get_status_counts(&self) -> HashMap<DocumentStatus, usize> {
        let store = self.status_store.read().await;
        let mut counts = HashMap::new();
        
        for status in store.values() {
            *counts.entry(status.status).or_insert(0) += 1;
        }
        
        counts
    }

    /// Get documents by status
    pub async fn get_docs_by_status(&self, status: DocumentStatus) -> Vec<(String, DocumentProcessingStatus)> {
        let store = self.status_store.read().await;
        store
            .iter()
            .filter(|(_, doc)| doc.status == status)
            .map(|(id, status)| (id.clone(), status.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::processing::summary::{SummaryError, SummaryMetadata};
    use crate::processing::keywords::ConversationTurn;

    struct MockSummarizer {
        config: SummaryConfig,
    }

    impl MockSummarizer {
        fn default() -> Self {
            Self {
                config: SummaryConfig::default(),
            }
        }
    }

    #[async_trait]
    impl ContentSummarizer for MockSummarizer {
        async fn generate_summary(&self, _content: &str) -> Result<Summary, SummaryError> {
            Ok(Summary {
                text: "Mock summary".to_string(),
                metadata: SummaryMetadata {
                    original_length: 100,
                    summary_length: 20,
                    original_tokens: Some(10),
                    summary_tokens: Some(5),
                    keywords: None,
                },
            })
        }

        fn get_config(&self) -> &SummaryConfig {
            &self.config
        }
    }

    struct MockKeywordExtractor {
        config: KeywordConfig,
    }

    impl MockKeywordExtractor {
        fn default() -> Self {
            Self {
                config: KeywordConfig::default(),
            }
        }
    }

    #[async_trait]
    impl KeywordExtractor for MockKeywordExtractor {
        async fn extract_keywords(&self, _content: &str) -> Result<ExtractedKeywords, crate::types::Error> {
            Ok(ExtractedKeywords {
                high_level: vec!["mock".to_string()],
                low_level: vec!["test".to_string()],
                metadata: HashMap::new(),
            })
        }

        async fn extract_keywords_with_history(
            &self,
            content: &str,
            _history: &[ConversationTurn],
        ) -> Result<ExtractedKeywords, crate::types::Error> {
            self.extract_keywords(content).await
        }

        fn get_config(&self) -> &KeywordConfig {
            &self.config
        }
    }

    #[tokio::test]
    async fn test_document_processing() {
        let config = DocumentProcessingConfig::default();
        let summarizer = Arc::new(MockSummarizer::default());
        let keyword_extractor = Arc::new(MockKeywordExtractor::default());
        
        let processor = DocumentProcessor::new(config, summarizer, keyword_extractor);
        
        let content = "Test document content";
        let doc_id = processor.process_document(content).await.unwrap();
        
        let store = processor.status_store.read().await;
        let status = store.get(&doc_id).unwrap();
        
        assert_eq!(status.status, DocumentStatus::Completed);
        assert!(status.summary.is_some());
        assert!(status.keywords.is_some());
    }

    #[tokio::test]
    async fn test_duplicate_detection() {
        let mut config = DocumentProcessingConfig::default();
        config.enable_deduplication = true;
        
        let summarizer = Arc::new(MockSummarizer::default());
        let keyword_extractor = Arc::new(MockKeywordExtractor::default());
        
        let processor = DocumentProcessor::new(config, summarizer, keyword_extractor);
        
        let content = "Test document content";
        
        // First processing should succeed
        let doc_id1 = processor.process_document(content).await.unwrap();
        
        // Second processing of the same content should return the same ID
        let doc_id2 = processor.process_document(content).await.unwrap();
        
        assert_eq!(doc_id1, doc_id2);
        
        let counts = processor.get_status_counts().await;
        assert_eq!(counts.get(&DocumentStatus::Completed), Some(&1));
    }
} 