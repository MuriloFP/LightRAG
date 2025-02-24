use std::sync::{Arc, Mutex};
use futures::Stream;
use std::pin::Pin;
use std::collections::HashMap;

use crate::llm::{
    LLMParams,
    LLMResponse,
    LLMError,
    StreamingResponse,
    Provider,
    ProviderConfig,
};

/// A model instance with its configuration
#[derive(Clone)]
pub struct Model {
    pub provider: Arc<Box<dyn Provider + Send + Sync>>,
    pub config: ProviderConfig,
}

impl Model {
    pub fn new(provider: Box<dyn Provider + Send + Sync>, config: ProviderConfig) -> Self {
        Self { 
            provider: Arc::new(provider),
            config,
        }
    }
}

/// Distributes requests across multiple language models
pub struct MultiModel {
    models: Vec<Model>,
    current_model: Arc<Mutex<usize>>,
}

impl MultiModel {
    pub fn new(models: Vec<Model>) -> Self {
        Self {
            models,
            current_model: Arc::new(Mutex::new(0)),
        }
    }

    fn next_model(&self) -> Result<Model, LLMError> {
        let mut current = self.current_model.lock().unwrap();
        *current = (*current + 1) % self.models.len();
        Ok(self.models[*current].clone())
    }

    pub async fn complete(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        let model = self.next_model()?;
        model.provider.complete(prompt, params).await
    }

    pub async fn complete_stream(
        &self,
        prompt: &str,
        params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        let model = self.next_model()?;
        model.provider.complete_stream(prompt, params).await
    }

    pub async fn complete_batch(
        &self,
        prompts: &[String],
        params: &LLMParams,
    ) -> Result<Vec<LLMResponse>, LLMError> {
        let model = self.next_model()?;
        model.provider.complete_batch(prompts, params).await
    }
}

/// Builder for creating a MultiModel instance
pub struct MultiModelBuilder {
    models: Vec<Model>,
}

impl MultiModelBuilder {
    pub fn new() -> Self {
        Self { models: Vec::new() }
    }

    pub fn add_model(mut self, provider_type: &str, config: ProviderConfig) -> Result<Self, LLMError> {
        let provider = crate::llm::create_provider(provider_type, config.clone())?;
        self.models.push(Model::new(provider, config));
        Ok(self)
    }

    pub fn build(self) -> MultiModel {
        MultiModel::new(self.models)
    }
}

impl Default for MultiModelBuilder {
    fn default() -> Self {
        Self::new()
    }
} 