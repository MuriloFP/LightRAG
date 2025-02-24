use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::sync::Arc;

use crate::llm::{
    LLMError, LLMParams, LLMResponse,
    Provider,
    ProviderConfig,
};
use crate::types::llm::StreamingResponse;
use crate::llm::providers::anthropic::AnthropicProvider;
use crate::llm::ProviderAdapter;
use crate::llm::LLMProvider;

/// Anthropic provider adapter implementation
pub struct AnthropicAdapter {
    provider: AnthropicProvider,
    config: Arc<ProviderConfig>,
}

impl AnthropicAdapter {
    /// Create a new Anthropic adapter
    pub fn new(client: AnthropicProvider, config: ProviderConfig) -> Result<Self, LLMError> {
        // Validate configuration
        if config.api_key.is_none() {
            return Err(LLMError::ConfigError("Anthropic API key is required".to_string()));
        }

        Ok(Self { 
            provider: client,
            config: Arc::new(config),
        })
    }

    /// Map Anthropic errors to LLMError
    pub fn map_error(&self, error: impl std::error::Error) -> LLMError {
        match error.to_string() {
            e if e.contains("rate_limit") => LLMError::RateLimitExceeded(e),
            e if e.contains("invalid_request") => LLMError::ConfigError(e),
            e if e.contains("token_limit") => LLMError::TokenLimitExceeded(e),
            e => LLMError::RequestFailed(e),
        }
    }

    /// Validate request parameters
    pub fn validate_params(&self, params: &LLMParams) -> Result<(), LLMError> {
        // Check if model is supported
        let (provider, _) = LLMProvider::parse(&params.model);
        if provider != LLMProvider::Anthropic {
            return Err(LLMError::ConfigError(
                format!("Invalid provider for Anthropic adapter: {:?}", provider)
            ));
        }

        // Validate temperature (Anthropic uses 0.0 to 1.0)
        if params.temperature < 0.0 || params.temperature > 1.0 {
            return Err(LLMError::ConfigError(
                "Temperature must be between 0.0 and 1.0 for Anthropic".to_string()
            ));
        }

        // Validate top_p
        if params.top_p < 0.0 || params.top_p > 1.0 {
            return Err(LLMError::ConfigError(
                "Top-p must be between 0.0 and 1.0".to_string()
            ));
        }

        Ok(())
    }
}

#[async_trait]
impl Provider for AnthropicAdapter {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        self.provider.initialize().await
    }

    async fn complete(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        self.provider.complete(prompt, params).await
    }

    async fn complete_stream(
        &self,
        prompt: &str,
        params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        self.provider.complete_stream(prompt, params).await
    }

    async fn complete_batch(
        &self,
        prompts: &[String],
        params: &LLMParams,
    ) -> Result<Vec<LLMResponse>, LLMError> {
        self.provider.complete_batch(prompts, params).await
    }

    fn get_config(&self) -> &ProviderConfig {
        &self.config
    }

    fn update_config(&mut self, config: ProviderConfig) -> Result<(), LLMError> {
        self.config = Arc::new(config);
        Ok(())
    }
}

#[async_trait]
impl ProviderAdapter for AnthropicAdapter {
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LLMError> {
        self.provider.create_embeddings(texts).await
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn max_tokens(&self) -> usize {
        // Return a large default value and let the API handle actual token limits
        usize::MAX
    }

    fn provider_name(&self) -> &str {
        "Anthropic"
    }
} 