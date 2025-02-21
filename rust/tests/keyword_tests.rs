use super_lightrag::processing::{
    KeywordConfig,
    KeywordExtractor,
    BasicKeywordExtractor,
    LLMKeywordExtractor,
    ConversationTurn,
};
use chrono::Utc;

#[tokio::test]
async fn test_basic_keyword_config() {
    let config = KeywordConfig::default();
    assert_eq!(config.max_high_level, 5);
    assert_eq!(config.max_low_level, 10);
    assert!(config.language.is_none());
    assert!(!config.use_llm);
    assert!(config.extra_params.is_empty());
}

#[tokio::test]
async fn test_empty_content() {
    let extractor = BasicKeywordExtractor::default();
    let result = extractor.extract_keywords("").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_basic_extraction() {
    let config = KeywordConfig {
        max_high_level: 3,
        max_low_level: 5,
        language: Some("en".to_string()),
        use_llm: false,
        extra_params: Default::default(),
    };
    
    let extractor = BasicKeywordExtractor::new(config);
    let content = "Artificial Intelligence and Machine Learning are transforming technology.";
    
    let result = extractor.extract_keywords(content).await.unwrap();
    
    // Check that we got the expected number of keywords
    assert_eq!(result.high_level.len(), 3);
    assert!(result.low_level.len() <= 5);
    
    // Check that metadata contains expected fields
    assert!(result.metadata.contains_key("total_words"));
    assert_eq!(result.metadata.get("extraction_method").unwrap(), "tf-idf");
}

#[tokio::test]
async fn test_conversation_history() {
    let extractor = BasicKeywordExtractor::default();
    let content = "What are the latest developments?";
    
    let history = vec![
        ConversationTurn {
            role: "user".to_string(),
            content: "Tell me about AI".to_string(),
            timestamp: Some(Utc::now()),
        },
        ConversationTurn {
            role: "assistant".to_string(),
            content: "AI is a broad field of computer science...".to_string(),
            timestamp: Some(Utc::now()),
        },
    ];
    
    let result = extractor.extract_keywords_with_history(content, &history).await.unwrap();
    
    // Check that we got some keywords
    assert!(!result.high_level.is_empty());
    assert!(result.high_level.len() <= extractor.get_config().max_high_level);
    assert!(result.low_level.len() <= extractor.get_config().max_low_level);
    
    // Check that metadata contains expected fields
    assert!(result.metadata.contains_key("total_words"));
    assert_eq!(result.metadata.get("extraction_method").unwrap(), "tf-idf");
}

#[tokio::test]
async fn test_llm_extractor_creation() {
    let config = KeywordConfig {
        use_llm: true,
        ..Default::default()
    };
    
    let extractor = LLMKeywordExtractor::new(config);
    let result = extractor.extract_keywords("test content").await;
    // Should fail since LLM extraction is not implemented yet
    assert!(result.is_err());
} 