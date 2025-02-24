use std::sync::Arc;
use std::collections::HashMap;
use chrono::Utc;
use async_trait::async_trait;
use futures::Stream;
use std::pin::Pin;

use super_lightrag::{
    types::{
        llm::{LLMParams, LLMResponse, LLMError, StreamingResponse},
        Error,
    },
    processing::keywords::{
        KeywordExtractor, KeywordConfig, ExtractedKeywords,
        ConversationTurn, BasicKeywordExtractor, LLMKeywordExtractor
    },
    llm::{Provider, ProviderConfig},
};

/// Mock LLM provider for testing
struct MockProvider {
    responses: Vec<String>,
    current: std::sync::atomic::AtomicUsize,
}

impl MockProvider {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    async fn initialize(&mut self) -> std::result::Result<(), LLMError> {
        Ok(())
    }

    async fn complete(&self, _prompt: &str, _params: &LLMParams) -> std::result::Result<LLMResponse, LLMError> {
        let idx = self.current.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if idx < self.responses.len() {
            Ok(LLMResponse {
                text: self.responses[idx].clone(),
                tokens_used: 100,
                model: "mock".to_string(),
                cached: false,
                context: None,
                metadata: HashMap::new(),
            })
        } else {
            Err(LLMError::RequestFailed("No more mock responses".to_string()))
        }
    }

    async fn complete_stream(
        &self,
        _prompt: &str,
        _params: &LLMParams,
    ) -> std::result::Result<Pin<Box<dyn Stream<Item = std::result::Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        Err(LLMError::RequestFailed("Streaming not needed for tests".to_string()))
    }

    async fn complete_batch(
        &self,
        _prompts: &[String],
        _params: &LLMParams,
    ) -> std::result::Result<Vec<LLMResponse>, LLMError> {
        Err(LLMError::RequestFailed("Batch completion not needed for tests".to_string()))
    }

    fn get_config(&self) -> &ProviderConfig {
        panic!("Config not needed for tests")
    }

    fn update_config(&mut self, _config: ProviderConfig) -> std::result::Result<(), LLMError> {
        Err(LLMError::RequestFailed("Config update not needed for tests".to_string()))
    }
}

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
async fn test_llm_keyword_extraction() {
    let mock_response = r#"{
        "high_level_keywords": ["artificial intelligence", "technology", "innovation"],
        "low_level_keywords": ["machine learning", "neural networks", "data science", "algorithms"]
    }"#.to_string();

    let config = KeywordConfig {
        max_high_level: 5,
        max_low_level: 10,
        language: Some("English".to_string()),
        use_llm: true,
        extra_params: HashMap::new(),
    };

    let provider = Arc::new(Box::new(MockProvider::new(vec![mock_response])) as Box<dyn Provider>);
    let llm_params = LLMParams {
        model: "test-model".to_string(),
        max_tokens: 100,
        temperature: 0.3,
        top_p: 0.9,
        stream: false,
        system_prompt: Some("Extract keywords from the text.".to_string()),
        query_params: None,
        extra_params: HashMap::new(),
    };

    let extractor = LLMKeywordExtractor::new(config, provider, llm_params);
    let content = "AI and machine learning are revolutionizing technology through innovative algorithms and neural networks.";
    
    let result = extractor.extract_keywords(content).await.unwrap();
    
    // Check high-level keywords
    assert_eq!(result.high_level.len(), 3);
    assert!(result.high_level.contains(&"artificial intelligence".to_string()));
    assert!(result.high_level.contains(&"technology".to_string()));
    assert!(result.high_level.contains(&"innovation".to_string()));
    
    // Check low-level keywords
    assert_eq!(result.low_level.len(), 4);
    assert!(result.low_level.contains(&"machine learning".to_string()));
    assert!(result.low_level.contains(&"neural networks".to_string()));
    assert!(result.low_level.contains(&"data science".to_string()));
    assert!(result.low_level.contains(&"algorithms".to_string()));
    
    // Check metadata
    assert!(result.metadata.contains_key("extraction_method"));
    assert_eq!(result.metadata.get("extraction_method").unwrap(), "llm");
    assert!(result.metadata.contains_key("timestamp"));
}

#[tokio::test]
async fn test_llm_empty_content() {
    let config = KeywordConfig {
        max_high_level: 5,
        max_low_level: 10,
        language: Some("English".to_string()),
        use_llm: true,
        extra_params: HashMap::new(),
    };

    let provider = Arc::new(Box::new(MockProvider::new(vec![])) as Box<dyn Provider>);
    let llm_params = LLMParams::default();
    let extractor = LLMKeywordExtractor::new(config, provider, llm_params);

    let result = extractor.extract_keywords("").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_llm_with_conversation_history() {
    let mock_response = r#"{
        "high_level_keywords": ["artificial intelligence", "machine learning", "conversation"],
        "low_level_keywords": ["context", "history", "dialogue", "interaction"]
    }"#.to_string();

    let config = KeywordConfig {
        max_high_level: 5,
        max_low_level: 10,
        language: Some("English".to_string()),
        use_llm: true,
        extra_params: HashMap::new(),
    };

    let provider = Arc::new(Box::new(MockProvider::new(vec![mock_response])) as Box<dyn Provider>);
    let llm_params = LLMParams::default();
    let extractor = LLMKeywordExtractor::new(config, provider, llm_params);

    let content = "Tell me more about that.";
    let history = vec![
        ConversationTurn {
            role: "user".to_string(),
            content: "How does AI work?".to_string(),
            timestamp: Some(Utc::now()),
        },
        ConversationTurn {
            role: "assistant".to_string(),
            content: "AI systems use machine learning algorithms...".to_string(),
            timestamp: Some(Utc::now()),
        },
    ];

    let result = extractor.extract_keywords_with_history(content, &history).await.unwrap();

    // Check high-level keywords
    assert_eq!(result.high_level.len(), 3);
    assert!(result.high_level.contains(&"artificial intelligence".to_string()));
    assert!(result.high_level.contains(&"machine learning".to_string()));
    assert!(result.high_level.contains(&"conversation".to_string()));

    // Check low-level keywords
    assert_eq!(result.low_level.len(), 4);
    assert!(result.low_level.contains(&"context".to_string()));
    assert!(result.low_level.contains(&"history".to_string()));
    assert!(result.low_level.contains(&"dialogue".to_string()));
    assert!(result.low_level.contains(&"interaction".to_string()));

    // Check metadata
    assert!(result.metadata.contains_key("extraction_method"));
    assert_eq!(result.metadata.get("extraction_method").unwrap(), "llm");
    assert!(result.metadata.contains_key("timestamp"));
}

#[tokio::test]
async fn test_llm_invalid_response() {
    let mock_response = "Invalid JSON response".to_string();

    let config = KeywordConfig {
        max_high_level: 5,
        max_low_level: 10,
        language: Some("English".to_string()),
        use_llm: true,
        extra_params: HashMap::new(),
    };

    let provider = Arc::new(Box::new(MockProvider::new(vec![mock_response])) as Box<dyn Provider>);
    let llm_params = LLMParams::default();
    let extractor = LLMKeywordExtractor::new(config, provider, llm_params);

    let result = extractor.extract_keywords("Test content").await;
    assert!(result.is_err());
} 