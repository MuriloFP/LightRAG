use std::sync::Arc;
use async_trait::async_trait;
use once_cell::sync::Lazy;
use std::collections::HashMap;
use super_lightrag::{
    llm::{LLMClient, LLMConfig, LLMError, LLMParams, LLMResponse},
    processing::summary::{
        ContentSummarizer, LLMSummarizer, LLMSummaryConfig,
    },
};
use super_lightrag::llm::rate_limiter::RateLimitConfig;

static EMPTY_METADATA: Lazy<HashMap<String, String>> = Lazy::new(HashMap::new);

/// Mock LLM client for testing
struct MockLLMClient {
    config: LLMConfig,
}

impl MockLLMClient {
    fn new() -> Self {
        Self {
            config: LLMConfig {
                model: String::new(),
                api_endpoint: None,
                api_key: None,
                org_id: None,
                timeout_secs: 30,
                max_retries: 3,
                use_cache: false,
                rate_limit_config: Some(RateLimitConfig::default()),
                extra_config: HashMap::new(),
            },
        }
    }
}

#[async_trait]
impl LLMClient for MockLLMClient {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        Ok(())
    }

    async fn generate(&self, _prompt: &str, _params: &LLMParams) -> Result<LLMResponse, LLMError> {
        // Simple mock that returns a fixed summary
        Ok(LLMResponse {
            text: "This is a mock summary of the content.".to_string(),
            tokens_used: 8,
            model: "mock-model".to_string(),
            cached: false,
            context: None,
            metadata: EMPTY_METADATA.clone(),
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
async fn test_llm_summarizer() {
    let config = LLMSummaryConfig::default();
    let llm_client = Arc::new(MockLLMClient::new());
    let summarizer = LLMSummarizer::new(config, llm_client);

    let content = "This is a test content that needs to be summarized. \
                   It contains multiple sentences and should be processed by the LLM.";

    let summary = summarizer.generate_summary(content).await.unwrap();

    assert_eq!(summary.text, "This is a mock summary of the content.");
    assert!(summary.metadata.original_length > 0);
    assert!(summary.metadata.summary_length > 0);
    assert!(summary.metadata.original_tokens.is_some());
    assert!(summary.metadata.summary_tokens.is_some());
}

#[tokio::test]
async fn test_empty_content() {
    let config = LLMSummaryConfig::default();
    let llm_client = Arc::new(MockLLMClient::new());
    let summarizer = LLMSummarizer::new(config, llm_client);

    let result = summarizer.generate_summary("").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_fallback_behavior() {
    // Create a failing mock client
    struct FailingMockClient;

    static EMPTY_CONFIG: Lazy<LLMConfig> = Lazy::new(|| LLMConfig {
        model: String::new(),
        api_endpoint: None,
        api_key: None,
        org_id: None,
        timeout_secs: 30,
        max_retries: 3,
        use_cache: false,
        rate_limit_config: Some(RateLimitConfig::default()),
        extra_config: HashMap::new(),
    });

    #[async_trait]
    impl LLMClient for FailingMockClient {
        async fn initialize(&mut self) -> Result<(), LLMError> {
            Ok(())
        }

        async fn generate(&self, _: &str, _: &LLMParams) -> Result<LLMResponse, LLMError> {
            Err(LLMError::RequestFailed("Mock failure".to_string()))
        }

        async fn batch_generate(
            &self,
            _: &[String],
            _: &LLMParams,
        ) -> Result<Vec<LLMResponse>, LLMError> {
            Err(LLMError::RequestFailed("Mock failure".to_string()))
        }

        fn get_config(&self) -> &LLMConfig {
            &EMPTY_CONFIG
        }

        fn update_config(&mut self, _: LLMConfig) -> Result<(), LLMError> {
            Ok(())
        }
    }

    let mut config = LLMSummaryConfig::default();
    config.use_fallback = true;
    let llm_client = Arc::new(FailingMockClient);
    let summarizer = LLMSummarizer::new(config, llm_client);

    let content = "Test content for fallback behavior";
    let summary = summarizer.generate_summary(content).await.unwrap();

    // Should get a basic summary instead
    assert!(summary.text.contains("Test content"));
} 