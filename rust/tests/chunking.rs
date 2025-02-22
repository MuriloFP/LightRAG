use super_lightrag::processing::{ChunkingConfig, chunk_text};
use super_lightrag::utils::compute_mdhash_id;

#[tokio::test]
async fn test_basic_chunking() {
    let content = "This is a test document. It should be split into chunks based on tokens.";
    let config = ChunkingConfig {
        max_token_size: 5,
        overlap_token_size: 2,
        min_chunk_size: 1,
        ..Default::default()
    };
    let doc_id = compute_mdhash_id(content, "doc-");

    let chunks = chunk_text(content, &config, &doc_id).await.unwrap();
    assert!(!chunks.is_empty());
    assert!(chunks.len() > 1);
    assert_eq!(chunks[0].full_doc_id, doc_id);
}

#[tokio::test]
async fn test_character_based_splitting() {
    let content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph.";
    let config = ChunkingConfig {
        split_by_character: Some("\n\n".to_string()),
        split_by_character_only: true,
        min_chunk_size: 1,
        ..Default::default()
    };
    let doc_id = compute_mdhash_id(content, "doc-");

    let chunks = chunk_text(content, &config, &doc_id).await.unwrap();
    assert_eq!(chunks.len(), 3);
    assert_eq!(chunks[0].content, "First paragraph.");
    assert_eq!(chunks[1].content, "Second paragraph.");
    assert_eq!(chunks[2].content, "Third paragraph.");
}

#[tokio::test]
async fn test_empty_input() {
    let content = "";
    let config = ChunkingConfig::default();
    let doc_id = "test";

    let result = chunk_text(content, &config, doc_id).await;
    assert!(matches!(result, Err(super_lightrag::processing::ChunkingError::EmptyInput)));
} 