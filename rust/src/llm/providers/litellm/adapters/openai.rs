use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::sync::Arc;

use crate::llm::{
    LLMError, LLMParams, LLMResponse, LLMConfig,
};
use crate::types::llm::{StreamingResponse, LLMClient};
use crate::llm::providers::openai::OpenAIClient;
use crate::llm::ProviderAdapter;
use crate::llm::LLMProvider;

/// OpenAI provider adapter implementation
pub struct OpenAIAdapter {
    client: OpenAIClient,
    config: Arc<LLMConfig>,
}

impl OpenAIAdapter {
    /// Create a new OpenAI adapter
    pub fn new(client: OpenAIClient, config: LLMConfig) -> Result<Self, LLMError> {
        if config.api_key.is_none() {
            return Err(LLMError::ConfigError("OpenAI API key is required".to_string()));
        }
        Ok(Self {
            client,
            config: Arc::new(config),
        })
    }

    /// Validate request parameters for OpenAI.
    pub fn validate_params(&self, params: &LLMParams) -> Result<(), LLMError> {
        let (provider, _) = LLMProvider::parse(&params.model);
        if provider != LLMProvider::OpenAI {
            return Err(LLMError::ConfigError(format!("Invalid provider for OpenAI adapter: {:?}", provider)));
        }
        if params.temperature < 0.0 || params.temperature > 1.0 {
            return Err(LLMError::ConfigError("Temperature must be between 0.0 and 1.0 for OpenAI".to_string()));
        }
        if params.top_p < 0.0 || params.top_p > 1.0 {
            return Err(LLMError::ConfigError("Top-p must be between 0.0 and 1.0 for OpenAI".to_string()));
        }
        Ok(())
    }

    /// Map errors from OpenAI API to LLMError.
    pub fn map_error(&self, error: impl std::error::Error) -> LLMError {
        let e = error.to_string();
        if e.contains("rate limit") {
            LLMError::RateLimitExceeded(e)
        } else if e.contains("invalid_request") || e.contains("Invalid model") {
            LLMError::ConfigError(e)
        } else if e.contains("token limit") {
            LLMError::TokenLimitExceeded(e)
        } else {
            LLMError::RequestFailed(e)
        }
    }
}

#[async_trait]
impl ProviderAdapter for OpenAIAdapter {
    async fn complete(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        self.client.generate(prompt, params)
            .await
            .map_err(|e| e)
    }

    async fn complete_stream(&self, prompt: &str, params: &LLMParams) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        self.client.generate_stream(prompt, params)
            .await
            .map_err(|e| e)
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LLMError> {
        self.client.create_embeddings(texts).await
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn max_tokens(&self) -> usize {
        // Return a large default value and let the API handle actual token limits
        usize::MAX
    }

    fn provider_name(&self) -> &str {
        "OpenAI"
    }

    fn get_config(&self) -> &LLMConfig {
        &self.config
    }
} 