use super_lightrag::processing::{
    DocumentMetadata,
    DocumentStatus,
    DocumentStatusError,
    DocumentStatusManager,
    InMemoryStatusManager,
};

#[tokio::test]
async fn test_document_metadata_creation() {
    let metadata = DocumentMetadata::new(
        "test-doc-1".to_string(),
        "Test document summary".to_string(),
        100,
    );

    assert_eq!(metadata.id, "test-doc-1");
    assert_eq!(metadata.status, DocumentStatus::Pending);
    assert_eq!(metadata.content_summary, "Test document summary");
    assert_eq!(metadata.content_length, 100);
    assert!(metadata.chunks_count.is_none());
    assert!(metadata.error.is_none());
    assert!(metadata.custom_metadata.is_empty());
}

#[tokio::test]
async fn test_valid_status_transitions() {
    let mut metadata = DocumentMetadata::new(
        "test-doc-2".to_string(),
        "Test document".to_string(),
        100,
    );

    // Test valid transitions
    assert!(metadata.update_status(DocumentStatus::Processing).is_ok());
    assert!(metadata.update_status(DocumentStatus::Completed).is_ok());

    // Create new metadata for testing failure path
    let mut metadata = DocumentMetadata::new(
        "test-doc-3".to_string(),
        "Test document".to_string(),
        100,
    );
    assert!(metadata.update_status(DocumentStatus::Processing).is_ok());
    assert!(metadata.update_status(DocumentStatus::Failed).is_ok());
    // Test retry capability
    assert!(metadata.update_status(DocumentStatus::Processing).is_ok());
}

#[tokio::test]
async fn test_invalid_status_transitions() {
    let mut metadata = DocumentMetadata::new(
        "test-doc-4".to_string(),
        "Test document".to_string(),
        100,
    );

    // Test invalid transitions
    assert!(matches!(
        metadata.update_status(DocumentStatus::Completed),
        Err(DocumentStatusError::InvalidTransition { .. })
    ));

    assert!(matches!(
        metadata.update_status(DocumentStatus::Failed),
        Err(DocumentStatusError::InvalidTransition { .. })
    ));
}

#[tokio::test]
async fn test_in_memory_manager() {
    let manager = InMemoryStatusManager::new();

    // Test document creation and retrieval
    let metadata = DocumentMetadata::new(
        "test-doc-5".to_string(),
        "Test document".to_string(),
        100,
    );
    assert!(manager.update_metadata(metadata.clone()).await.is_ok());
    
    let retrieved = manager.get_metadata("test-doc-5").await.unwrap().unwrap();
    assert_eq!(retrieved.id, "test-doc-5");
    assert_eq!(retrieved.status, DocumentStatus::Pending);

    // Test status counts
    let counts = manager.get_status_counts().await.unwrap();
    assert_eq!(counts.get(&DocumentStatus::Pending).unwrap(), &1);
    assert_eq!(counts.get(&DocumentStatus::Processing).unwrap(), &0);
    assert_eq!(counts.get(&DocumentStatus::Completed).unwrap(), &0);
    assert_eq!(counts.get(&DocumentStatus::Failed).unwrap(), &0);

    // Test get by status
    let pending_docs = manager.get_by_status(DocumentStatus::Pending).await.unwrap();
    assert_eq!(pending_docs.len(), 1);
    assert_eq!(pending_docs[0].id, "test-doc-5");

    // Test deletion
    assert!(manager.delete_metadata("test-doc-5").await.is_ok());
    assert!(manager.get_metadata("test-doc-5").await.unwrap().is_none());
} 