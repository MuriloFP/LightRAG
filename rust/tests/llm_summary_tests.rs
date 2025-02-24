use std::sync::Arc;
use async_trait::async_trait;
use std::collections::HashMap;
use futures::Stream;
use std::pin::Pin;
use std::sync::LazyLock;

use super_lightrag::{
    llm::{LLMError, LLMParams, LLMResponse, StreamingResponse, Provider, ProviderConfig},
    processing::summary::{
        ContentSummarizer, LLMSummarizer, LLMSummaryConfig, SummaryError,
    },
    llm::cache::memory::MemoryCache,
};

/// Mock provider for testing
struct MockProvider {
    config: ProviderConfig,
}

impl MockProvider {
    fn new() -> Self {
        Self {
            config: ProviderConfig {
                model: "test-model".to_string(),
                api_endpoint: None,
                api_key: None,
                org_id: None,
                timeout_secs: 30,
                max_retries: 3,
                extra_config: HashMap::new(),
            },
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        Ok(())
    }

    async fn complete(&self, _prompt: &str, _params: &LLMParams) -> Result<LLMResponse, LLMError> {
        Ok(LLMResponse {
            text: "This is a mock summary of the content.".to_string(),
            tokens_used: 8,
            model: "mock-model".to_string(),
            cached: false,
            context: None,
            metadata: HashMap::new(),
        })
    }

    async fn complete_stream(
        &self,
        _prompt: &str,
        _params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        unimplemented!("Streaming not implemented for mock provider")
    }

    async fn complete_batch(
        &self,
        prompts: &[String],
        params: &LLMParams,
    ) -> Result<Vec<LLMResponse>, LLMError> {
        let mut responses = Vec::new();
        for _ in prompts {
            responses.push(self.complete("", params).await?);
        }
        Ok(responses)
    }

    fn get_config(&self) -> &ProviderConfig {
        &self.config
    }

    fn update_config(&mut self, config: ProviderConfig) -> Result<(), LLMError> {
        self.config = config;
        Ok(())
    }
}

#[tokio::test]
async fn test_llm_summarizer() {
    let config = LLMSummaryConfig::default();
    let provider = Box::new(MockProvider::new());
    let cache = Box::new(MemoryCache::new());
    let summarizer = LLMSummarizer::new(provider, config, cache);

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
    let provider = Box::new(MockProvider::new());
    let cache = Box::new(MemoryCache::new());
    let summarizer = LLMSummarizer::new(provider, config, cache);

    let result = summarizer.generate_summary("").await;
    assert!(matches!(result, Err(SummaryError::EmptyContent)));
}

#[tokio::test]
async fn test_fallback_behavior() {
    struct FailingProvider;

    #[async_trait]
    impl Provider for FailingProvider {
        async fn initialize(&mut self) -> Result<(), LLMError> {
            Ok(())
        }

        async fn complete(&self, _: &str, _: &LLMParams) -> Result<LLMResponse, LLMError> {
            Err(LLMError::RequestFailed("Mock failure".to_string()))
        }

        async fn complete_stream(
            &self,
            _: &str,
            _: &LLMParams,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
            unimplemented!()
        }

        async fn complete_batch(
            &self,
            _: &[String],
            _: &LLMParams,
        ) -> Result<Vec<LLMResponse>, LLMError> {
            unimplemented!()
        }

        fn get_config(&self) -> &ProviderConfig {
            static DUMMY_CONFIG: LazyLock<ProviderConfig> = LazyLock::new(|| ProviderConfig {
                model: "dummy-model".to_string(),
                api_endpoint: None,
                api_key: None,
                org_id: None,
                timeout_secs: 30,
                max_retries: 3,
                extra_config: std::collections::HashMap::new(),
            });
            &*DUMMY_CONFIG
        }

        fn update_config(&mut self, _: ProviderConfig) -> Result<(), LLMError> {
            Ok(())
        }
    }

    let mut config = LLMSummaryConfig::default();
    config.use_fallback = true;

    let provider = Box::new(FailingProvider);
    let cache = Box::new(MemoryCache::new());
    let summarizer = LLMSummarizer::new(provider, config, cache);

    let content = "Test content";
    let summary = summarizer.generate_summary(content).await.unwrap();

    // Should get basic summary from fallback
    assert!(summary.text.contains(content));
}

#[tokio::test]
async fn test_summarize_with_mock() -> Result<(), SummaryError> {
    let config = LLMSummaryConfig::default();
    let provider = Box::new(MockProvider::new());
    let cache = Box::new(MemoryCache::new());
    let summarizer = LLMSummarizer::new(provider, config, cache);

    let content = "This is a test content that needs to be summarized.";
    let summary = summarizer.generate_summary(content).await?;

    assert_eq!(summary.text, "This is a mock summary of the content.");
    Ok(())
}

#[tokio::test]
async fn test_summarization() {
    let config = LLMSummaryConfig::default();
    let provider = Box::new(MockProvider::new());
    let cache = Box::new(MemoryCache::new());
    let summarizer = LLMSummarizer::new(provider, config, cache);

    let content = "This is a test content that needs to be summarized.";
    let summary = summarizer.generate_summary(content).await.unwrap();

    assert_eq!(summary.text, "This is a mock summary of the content.");
} 