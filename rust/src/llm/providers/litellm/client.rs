use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::time::{Duration, Instant};
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::env;

use crate::llm::{
    LLMClient, LLMConfig, LLMError, LLMParams, LLMResponse,
    cache::{ResponseCache, InMemoryCache},
};
use crate::types::llm::{QueryParams, StreamingResponse, StreamingTiming};
use crate::processing::keywords::ConversationTurn;
use crate::processing::context::ContextBuilder;

/// Supported LLM providers
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum LLMProvider {
    OpenAI,
    Anthropic,
    Ollama,
    Azure,
    Custom(String),
}

impl LLMProvider {
    /// Get the base URL for a provider
    pub fn base_url(&self) -> &str {
        match self {
            LLMProvider::OpenAI => "https://api.openai.com/v1",
            LLMProvider::Anthropic => "https://api.anthropic.com/v1",
            LLMProvider::Ollama => "http://localhost:11434",
            LLMProvider::Azure => "https://api.cognitive.microsoft.com",
            LLMProvider::Custom(_) => "", // Must be provided in config
        }
    }

    /// Parse a provider string (e.g., "openai/gpt-4")
    pub fn parse(provider_string: &str) -> (Self, String) {
        if let Some((provider, model)) = provider_string.split_once('/') {
            let provider = match provider.to_lowercase().as_str() {
                "openai" => Self::OpenAI,
                "anthropic" => Self::Anthropic,
                "ollama" => Self::Ollama,
                "azure" => Self::Azure,
                custom => Self::Custom(custom.to_string()),
            };
            (provider, model.to_string())
        } else {
            // Default to treating as direct model name with OpenAI
            (Self::OpenAI, provider_string.to_string())
        }
    }
}

/// Provider-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// API endpoint override
    pub api_base: Option<String>,
    
    /// API key
    pub api_key: Option<String>,
    
    /// Organization ID (if applicable)
    pub org_id: Option<String>,
    
    /// Default model to use
    pub default_model: Option<String>,
    
    /// Provider-specific parameters
    pub extra_params: HashMap<String, String>,
}

/// LiteLLM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteLLMConfig {
    /// Default API endpoint
    pub api_base: String,
    
    /// Default API key
    pub api_key: String,
    
    /// Provider-specific configurations
    pub provider_configs: HashMap<String, ProviderConfig>,
    
    /// Default parameters for requests
    pub default_parameters: LLMParams,
    
    /// Fallback providers in order of preference
    pub fallback_providers: Vec<String>,
    
    /// Whether to use caching
    pub use_cache: bool,
    
    /// Cache configuration
    pub cache_config: Option<HashMap<String, String>>,
    
    /// Request timeout in seconds
    pub timeout_secs: u64,
    
    /// Maximum retries per request
    pub max_retries: u32,
}

impl Default for LiteLLMConfig {
    fn default() -> Self {
        Self {
            api_base: String::new(),
            api_key: String::new(),
            provider_configs: HashMap::new(),
            default_parameters: LLMParams::default(),
            fallback_providers: Vec::new(),
            use_cache: false,
            cache_config: None,
            timeout_secs: 30,
            max_retries: 3,
        }
    }
}

impl LiteLLMConfig {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self, LLMError> {
        let mut config = Self::default();
        
        // Load base configuration
        config.api_base = env::var("LITELLM_API_BASE")
            .unwrap_or_else(|_| String::new());
            
        config.api_key = env::var("LITELLM_API_KEY")
            .unwrap_or_else(|_| String::new());
            
        config.timeout_secs = env::var("LITELLM_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);
            
        config.max_retries = env::var("LITELLM_MAX_RETRIES")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
            
        config.use_cache = env::var("LITELLM_USE_CACHE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(false);
            
        // Load provider configurations
        for provider in ["OPENAI", "ANTHROPIC", "OLLAMA", "AZURE"] {
            let prefix = format!("LITELLM_{}_", provider);
            
            if let Ok(api_key) = env::var(format!("{}API_KEY", prefix)) {
                let mut provider_config = ProviderConfig {
                    api_key: Some(api_key),
                    api_base: env::var(format!("{}API_BASE", prefix)).ok(),
                    org_id: env::var(format!("{}ORG_ID", prefix)).ok(),
                    default_model: env::var(format!("{}DEFAULT_MODEL", prefix)).ok(),
                    extra_params: HashMap::new(),
                };
                
                // Load any extra parameters (format: LITELLM_PROVIDER_PARAM_NAME=value)
                for (key, value) in env::vars() {
                    if key.starts_with(&format!("{}_PARAM_", prefix)) {
                        let param_name = key.strip_prefix(&format!("{}_PARAM_", prefix))
                            .unwrap()
                            .to_lowercase();
                        provider_config.extra_params.insert(param_name, value);
                    }
                }
                
                config.provider_configs.insert(provider.to_lowercase(), provider_config);
            }
        }
        
        // Load fallback providers
        if let Ok(fallbacks) = env::var("LITELLM_FALLBACK_PROVIDERS") {
            config.fallback_providers = fallbacks.split(',')
                .map(|s| s.trim().to_string())
                .collect();
        }
        
        // Load cache configuration if enabled
        if config.use_cache {
            let mut cache_config = HashMap::new();
            
            for (key, value) in env::vars() {
                if key.starts_with("LITELLM_CACHE_") {
                    let config_key = key.strip_prefix("LITELLM_CACHE_")
                        .unwrap()
                        .to_lowercase();
                    cache_config.insert(config_key, value);
                }
            }
            
            if !cache_config.is_empty() {
                config.cache_config = Some(cache_config);
            }
        }
        
        Ok(config)
    }
}

/// Provider adapter trait
#[async_trait]
pub trait ProviderAdapter: Send + Sync {
    /// Generate completion
    async fn complete(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError>;
    
    /// Generate completion with streaming
    async fn complete_stream(
        &self,
        prompt: &str,
        params: &LLMParams
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError>;
    
    /// Generate embeddings
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LLMError>;
    
    /// Whether streaming is supported
    fn supports_streaming(&self) -> bool;
    
    /// Maximum tokens supported
    fn max_tokens(&self) -> usize;
    
    /// Get provider name
    fn provider_name(&self) -> &str;

    /// Get provider configuration
    fn get_config(&self) -> &LLMConfig;
}

/// LiteLLM client implementation
pub struct LiteLLMClient {
    /// HTTP client
    client: Client,
    
    /// LiteLLM configuration
    config: LiteLLMConfig,
    
    /// Publicly exposed LLMConfig converted from LiteLLMConfig
    public_config: LLMConfig,
    
    /// Response cache
    cache: Option<InMemoryCache>,
    
    /// Active provider adapters
    providers: HashMap<String, Box<dyn ProviderAdapter>>,
}

impl LiteLLMClient {
    /// Create a new LiteLLM client
    pub fn new(config: LiteLLMConfig) -> Result<Self, LLMError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| LLMError::ConfigError(e.to_string()))?;
            
        let cache = if config.use_cache {
            Some(InMemoryCache::default())
        } else {
            None
        };
        
        Ok(Self {
            client,
            public_config: convert_config(&config),
            config,
            cache,
            providers: HashMap::new(),
        })
    }
    
    /// Register a provider adapter
    pub fn register_provider(&mut self, name: &str, adapter: Box<dyn ProviderAdapter>) {
        self.providers.insert(name.to_string(), adapter);
    }
    
    /// Get provider adapter by name
    pub fn get_provider_adapter(&self, key: &str) -> Option<&Box<dyn ProviderAdapter>> {
        self.providers.get(key)
    }
}

/// Add a helper function to convert LiteLLMConfig to LLMConfig
fn convert_config(config: &LiteLLMConfig) -> LLMConfig {
    LLMConfig {
        model: config.default_parameters.model.clone(),
        api_endpoint: if config.api_base.is_empty() { None } else { Some(config.api_base.clone()) },
        api_key: if config.api_key.is_empty() { None } else { Some(config.api_key.clone()) },
        org_id: None,
        timeout_secs: config.timeout_secs,
        max_retries: config.max_retries,
        use_cache: config.use_cache,
        rate_limit_config: None,
        similarity_threshold: 0.8,
        extra_config: config.cache_config.clone().unwrap_or_default(),
    }
}

#[async_trait]
impl LLMClient for LiteLLMClient {
    async fn initialize(&mut self) -> Result<(), LLMError> {
        // Validate configuration: check if api_base is empty instead of is_none()
        if self.config.api_base.is_empty() && self.config.provider_configs.is_empty() {
            return Err(LLMError::ConfigError("Either api_base or provider_configs must be set".to_string()));
        }
        Ok(())
    }
    
    async fn generate(&self, prompt: &str, params: &LLMParams) -> Result<LLMResponse, LLMError> {
        // Parse provider from model string
        let (provider, _model) = LLMProvider::parse(&params.model);
        // Convert provider enum to lowercase string key
        let provider_key = match provider {
            LLMProvider::OpenAI => "openai".to_string(),
            LLMProvider::Anthropic => "anthropic".to_string(),
            LLMProvider::Ollama => "ollama".to_string(),
            LLMProvider::Azure => "azure".to_string(),
            LLMProvider::Custom(ref name) => name.clone(),
        };

        // Get provider adapter using the helper method
        let adapter = self.get_provider_adapter(&provider_key)
            .ok_or_else(|| LLMError::ConfigError(format!("No adapter found for provider: {:?}", provider)))?;

        // Try primary provider
        match adapter.complete(prompt, params).await {
            Ok(response) => Ok(response),
            Err(e) => {
                // Try fallback providers if configured (fallback_providers is a Vec<String>)
                for fallback in &self.config.fallback_providers {
                    if let Some(fallback_adapter) = self.get_provider_adapter(fallback) {
                        match fallback_adapter.complete(prompt, params).await {
                            Ok(response) => return Ok(response),
                            Err(_) => continue,
                        }
                    }
                }
                Err(e)
            }
        }
    }
    
    async fn generate_stream(
        &self,
        prompt: &str,
        params: &LLMParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        // Parse provider from model string
        let (provider, _model) = LLMProvider::parse(&params.model);
        let provider_key = match provider {
            LLMProvider::OpenAI => "openai".to_string(),
            LLMProvider::Anthropic => "anthropic".to_string(),
            LLMProvider::Ollama => "ollama".to_string(),
            LLMProvider::Azure => "azure".to_string(),
            LLMProvider::Custom(ref name) => name.clone(),
        };

        // Get provider adapter using the helper method
        let adapter = self.get_provider_adapter(&provider_key)
            .ok_or_else(|| LLMError::ConfigError(format!("No adapter found for provider: {:?}", provider)))?;

        // Try primary provider
        match adapter.complete_stream(prompt, params).await {
            Ok(stream) => Ok(stream),
            Err(e) => {
                // Try fallback providers if configured
                for fallback in &self.config.fallback_providers {
                    if let Some(fallback_adapter) = self.get_provider_adapter(fallback) {
                        match fallback_adapter.complete_stream(prompt, params).await {
                            Ok(stream) => return Ok(stream),
                            Err(_) => continue,
                        }
                    }
                }
                Err(e)
            }
        }
    }
    
    async fn batch_generate(
        &self,
        prompts: &[String],
        params: &LLMParams,
    ) -> Result<Vec<LLMResponse>, LLMError> {
        let mut responses = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let response = self.generate(prompt, params).await?;
            responses.push(response);
        }
        Ok(responses)
    }
    
    fn get_config(&self) -> &LLMConfig {
        &self.public_config
    }
    
    fn update_config(&mut self, config: LLMConfig) -> Result<(), LLMError> {
        self.public_config = config.clone();
        // Update fields in LiteLLMConfig that correspond to LLMConfig
        self.config.api_base = self.public_config.api_endpoint.clone().unwrap_or_default();
        self.config.api_key = self.public_config.api_key.clone().unwrap_or_default();
        self.config.timeout_secs = self.public_config.timeout_secs;
        self.config.max_retries = self.public_config.max_retries;
        self.config.use_cache = self.public_config.use_cache;
        Ok(())
    }
} 