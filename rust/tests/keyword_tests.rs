use super_lightrag::processing::{
    KeywordConfig,
    KeywordExtractor,
    BasicKeywordExtractor,
    LLMKeywordExtractor,
    ConversationTurn,
};
use chrono::Utc;
use std::sync::Arc;
use std::collections::HashMap;
use async_trait::async_trait;
use super_lightrag::types::llm::{LLMClient, LLMConfig, LLMParams, LLMResponse, LLMError};

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

/// Mock LLM client for testing
struct MockLLMClient {
    config: LLMConfig,
}

impl MockLLMClient {
    fn new() -> Self {
        Self {
            config: LLMConfig::default(),
        }
    }
}

#[async_trait]
impl LLMClient for MockLLMClient {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        Ok(())
    }

    async fn generate(&self, _prompt: &str, _params: &LLMParams) -> Result<LLMResponse, LLMError> {
        Ok(LLMResponse {
            text: r#"{
                "high_level_keywords": ["test", "mock"],
                "low_level_keywords": ["detail1", "detail2"]
            }"#.to_string(),
            tokens_used: 10,
            model: "openai/gpt-4".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        })
    }

    async fn batch_generate(
        &self,
        prompts: &[String],
        params: &LLMParams,
    ) -> Result<Vec<LLMResponse>, LLMError> {
        let mut responses = Vec::new();
        for _ in prompts {
            responses.push(self.generate("", params).await?);
        }
        Ok(responses)
    }

    fn get_config(&self) -> &LLMConfig {
        &self.config
    }

    fn update_config(&mut self, config: LLMConfig) -> Result<(), LLMError> {
        self.config = config;
        Ok(())
    }
}

#[tokio::test]
async fn test_llm_keyword_extraction() {
    let config = KeywordConfig {
        max_high_level: 5,
        max_low_level: 10,
        language: Some("English".to_string()),
        use_llm: true,
        extra_params: HashMap::new(),
    };

    let llm_client = Arc::new(MockLLMClient::new());
    let llm_params = LLMParams {
        max_tokens: 100,
        temperature: 0.3,
        top_p: 0.9,
        stream: false,
        system_prompt: Some("Extract keywords from the text.".to_string()),
        query_params: None,
        extra_params: HashMap::new(),
        model: "openai/gpt-4".to_string(),
    };

    let extractor = LLMKeywordExtractor::new(config, llm_client, llm_params);

    let content = "This is a test content for keyword extraction.";
    let keywords = extractor.extract_keywords(content).await.unwrap();

    assert_eq!(keywords.high_level, vec!["test", "mock"]);
    assert_eq!(keywords.low_level, vec!["detail1", "detail2"]);
}

#[tokio::test]
async fn test_keyword_extraction_with_history() {
    let config = KeywordConfig {
        max_high_level: 5,
        max_low_level: 10,
        language: Some("English".to_string()),
        use_llm: true,
        extra_params: HashMap::new(),
    };

    let llm_client = Arc::new(MockLLMClient::new());
    let llm_params = LLMParams {
        max_tokens: 100,
        temperature: 0.3,
        top_p: 0.9,
        stream: false,
        system_prompt: Some("Extract keywords considering conversation history.".to_string()),
        query_params: None,
        extra_params: HashMap::new(),
        model: "openai/gpt-4".to_string(),
    };

    let extractor = LLMKeywordExtractor::new(config, llm_client, llm_params);

    let content = "This is the current message.";
    let history = vec![
        ConversationTurn {
            role: "user".to_string(),
            content: "Previous message 1".to_string(),
            timestamp: None,
        },
        ConversationTurn {
            role: "assistant".to_string(),
            content: "Previous message 2".to_string(),
            timestamp: None,
        },
    ];

    let keywords = extractor.extract_keywords_with_history(content, &history).await.unwrap();

    assert_eq!(keywords.high_level, vec!["test", "mock"]);
    assert_eq!(keywords.low_level, vec!["detail1", "detail2"]);
} 