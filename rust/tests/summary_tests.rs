use super_lightrag::processing::summary::{
    BasicSummarizer,
    ContentSummarizer,
    SummaryConfig,
    SummaryType,
};

#[tokio::test]
async fn test_truncation_summary() {
    let config = SummaryConfig {
        max_length: 10,
        max_tokens: 4,
        summary_type: SummaryType::Truncation,
        language: None,
    };
    
    let summarizer = BasicSummarizer::new(config);
    let content = "The quick brown fox jumps over the lazy dog";
    
    let summary = summarizer.generate_summary(content).await.unwrap();
    assert_eq!(summary.text, "The quick...");
    assert_eq!(summary.metadata.original_length, content.len());
    assert_eq!(summary.metadata.summary_length, "The quick...".len());
}

#[tokio::test]
async fn test_token_based_summary() {
    let config = SummaryConfig {
        max_length: 100,
        max_tokens: 4,
        summary_type: SummaryType::TokenBased,
        language: None,
    };
    
    let summarizer = BasicSummarizer::new(config);
    let content = "The quick brown fox jumps over the lazy dog";
    
    let summary = summarizer.generate_summary(content).await.unwrap();
    assert_eq!(summary.text, "The quick... lazy dog");
    assert!(summary.metadata.original_tokens.unwrap() > 4);
    assert_eq!(summary.metadata.summary_tokens.unwrap(), 5);
}

#[tokio::test]
async fn test_empty_content() {
    let summarizer = BasicSummarizer::default();
    let result = summarizer.generate_summary("").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_short_content_truncation() {
    let config = SummaryConfig {
        max_length: 100,
        max_tokens: 4,
        summary_type: SummaryType::Truncation,
        language: None,
    };
    let summarizer = BasicSummarizer::new(config);
    let content = "Short text";
    let summary = summarizer.generate_summary(content).await.unwrap();
    assert_eq!(summary.text, content);
    assert_eq!(summary.metadata.original_length, content.len());
    assert_eq!(summary.metadata.summary_length, content.len());
}

#[tokio::test]
async fn test_short_content_token_based() {
    let config = SummaryConfig {
        max_length: 100,
        max_tokens: 10,
        summary_type: SummaryType::TokenBased,
        language: None,
    };
    
    let summarizer = BasicSummarizer::new(config);
    let content = "Short text";
    let summary = summarizer.generate_summary(content).await.unwrap();
    assert_eq!(summary.text, content);
    assert!(summary.metadata.original_tokens.unwrap() <= 10);
    assert_eq!(summary.metadata.original_tokens, summary.metadata.summary_tokens);
}

#[tokio::test]
async fn test_basic_summarizer() {
    let config = SummaryConfig {
        max_length: 100,
        max_tokens: 50,
        summary_type: SummaryType::Truncation,
        language: None,
    };
    let summarizer = BasicSummarizer::new(config);

    let content = "This is a test content that needs to be summarized. \
                   It contains multiple sentences and should be processed correctly.";

    let summary = summarizer.generate_summary(content).await.unwrap();

    assert!(summary.text.len() <= 100);
    assert_eq!(summary.metadata.original_length, content.len());
    assert!(summary.metadata.summary_length <= 100);
} 