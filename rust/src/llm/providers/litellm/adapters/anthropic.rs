use async_trait::async_trait;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::sync::Arc;

use crate::llm::{
    LLMError, LLMParams, LLMResponse,
    LLMConfig,
};
use crate::types::llm::{StreamingResponse, LLMClient};
use crate::llm::providers::anthropic::AnthropicClient;
use crate::llm::ProviderAdapter;
use crate::llm::LLMProvider;

/// Anthropic provider adapter implementation
pub struct AnthropicAdapter {
    client: AnthropicClient,
    config: Arc<LLMConfig>,
}

impl AnthropicAdapter {
    /// Create a new Anthropic adapter
    pub fn new(client: AnthropicClient, config: LLMConfig) -> Result<Self, LLMError> {
        // Validate configuration
        if config.api_key.is_none() {
            return Err(LLMError::ConfigError("Anthropic API key is required".to_string()));
        }

        Ok(Self { 
            client,
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
impl ProviderAdapter for AnthropicAdapter {
    async fn complete(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        // Validate parameters
        self.validate_params(params)?;

        // Call Anthropic client and map errors
        self.client.generate(prompt, params)
            .await
            .map_err(|e| self.map_error(e))
    }

    async fn complete_stream(
        &self,
        prompt: &str,
        params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        // Validate parameters
        self.validate_params(params)?;

        // Create streaming channel
        let (tx, rx) = mpsc::channel(32);
        let mut stream = self.client.generate_stream(prompt, params)
            .await
            .map_err(|e| self.map_error(e))?;

        // Clone necessary data for error mapping
        let provider_name = self.provider_name().to_string();
        let error_mapper = move |e| {
            match e {
                LLMError::RequestFailed(msg) => LLMError::RequestFailed(format!("{} API error: {}", provider_name, msg)),
                LLMError::RateLimitExceeded(msg) => LLMError::RateLimitExceeded(format!("{} rate limit: {}", provider_name, msg)),
                LLMError::TokenLimitExceeded(msg) => LLMError::TokenLimitExceeded(format!("{} token limit: {}", provider_name, msg)),
                LLMError::InvalidResponse(msg) => LLMError::InvalidResponse(format!("{} invalid response: {}", provider_name, msg)),
                LLMError::ConfigError(msg) => LLMError::ConfigError(format!("{} config error: {}", provider_name, msg)),
                LLMError::CacheError(msg) => LLMError::CacheError(format!("{} cache error: {}", provider_name, msg)),
            }
        };
        
        // Spawn task to handle streaming
        tokio::spawn(async move {
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        if tx.send(Ok(chunk)).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(error_mapper(e))).await;
                        break;
                    }
                }
            }
        });

        Ok(Box::pin(ReceiverStream::new(rx)))
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
        "Anthropic"
    }

    fn get_config(&self) -> &LLMConfig {
        self.config.as_ref()
    }
} 