use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use serde_json::json;

/// Provider type for configuration
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum Provider {
    OpenAI,
    Anthropic,
    Ollama,
    Custom(String),
}

impl Provider {
    /// Get default API base URL for provider
    pub fn default_api_base(&self) -> String {
        match self {
            Provider::OpenAI => "https://api.openai.com/v1".to_string(),
            Provider::Anthropic => "https://api.anthropic.com/v1".to_string(),
            Provider::Ollama => "http://localhost:11434".to_string(),
            Provider::Custom(_) => String::new(),
        }
    }

    /// Parse provider from string (e.g., "openai/gpt-4" -> (Provider::OpenAI, "gpt-4"))
    pub fn parse(provider_string: &str) -> (Self, String) {
        if let Some((provider, model)) = provider_string.split_once('/') {
            let provider = match provider.to_lowercase().as_str() {
                "openai" => Self::OpenAI,
                "anthropic" => Self::Anthropic,
                "ollama" => Self::Ollama,
                custom => Self::Custom(custom.to_string()),
            };
            (provider, model.to_string())
        } else {
            // Default to treating as direct model name with OpenAI
            (Self::OpenAI, provider_string.to_string())
        }
    }
}

/// Cache configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Whether to enable caching
    pub enabled: bool,
    /// Cache type (in-memory or redis)
    pub cache_type: String,
    /// Redis URL if using Redis
    pub redis_url: Option<String>,
    /// Maximum cache size
    pub max_size: usize,
    /// Cache TTL in seconds
    pub ttl_secs: u64,
    /// Whether to use similarity search for cache lookups
    pub use_similarity: bool,
    /// Similarity threshold for cache hits
    pub similarity_threshold: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cache_type: "in-memory".to_string(),
            redis_url: None,
            max_size: 10000,
            ttl_secs: 3600,
            use_similarity: true,
            similarity_threshold: 0.95,
        }
    }
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests per minute
    pub requests_per_minute: u32,
    /// Maximum tokens per minute
    pub tokens_per_minute: u32,
    /// Maximum concurrent requests
    pub max_concurrent: u32,
    /// Whether to use token bucket algorithm
    pub use_token_bucket: bool,
    /// Burst size for token bucket
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 60,
            tokens_per_minute: 90000,
            max_concurrent: 10,
            use_token_bucket: true,
            burst_size: 10,
        }
    }
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum number of retries
    pub max_retries: u32,
    /// Initial retry delay in seconds
    pub initial_retry_delay_secs: u64,
    /// Maximum retry delay in seconds
    pub max_retry_delay_secs: u64,
    /// Whether to use exponential backoff
    pub use_exponential_backoff: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_retry_delay_secs: 1,
            max_retry_delay_secs: 30,
            use_exponential_backoff: true,
        }
    }
}

/// Unified configuration for all LLM providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedConfig {
    /// Provider type
    pub provider: Provider,
    /// API key
    pub api_key: Option<String>,
    /// API base URL override
    pub api_base: Option<String>,
    /// Organization ID (for OpenAI)
    pub org_id: Option<String>,
    /// Default model to use
    pub model: String,
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling
    pub temperature: f32,
    /// Top-p (nucleus) sampling
    pub top_p: f32,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Cache configuration
    pub cache_config: CacheConfig,
    /// Rate limit configuration
    pub rate_limit_config: Option<RateLimitConfig>,
    /// Retry configuration
    pub retry_config: RetryConfig,
    /// Provider-specific parameters
    pub extra_params: HashMap<String, String>,
}

impl UnifiedConfig {
    /// Create a new configuration for a specific provider
    pub fn new(provider: Provider, api_key: Option<String>, model: String) -> Self {
        Self {
            provider: provider.clone(),
            api_key,
            api_base: Some(provider.default_api_base()),
            org_id: None,
            model,
            max_tokens: 1024,
            temperature: 0.7,
            top_p: 1.0,
            timeout_secs: 30,
            cache_config: CacheConfig::default(),
            rate_limit_config: Some(RateLimitConfig::default()),
            retry_config: RetryConfig::default(),
            extra_params: HashMap::new(),
        }
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        let provider_str = std::env::var("LLM_PROVIDER").unwrap_or_else(|_| "openai".to_string());
        let (provider, model) = if let Some(model) = std::env::var("LLM_MODEL").ok() {
            (Provider::parse(&provider_str).0, model)
        } else {
            Provider::parse(&provider_str)
        };

        let api_key = std::env::var("LLM_API_KEY")
            .or_else(|_| match provider {
                Provider::OpenAI => std::env::var("OPENAI_API_KEY"),
                Provider::Anthropic => std::env::var("ANTHROPIC_API_KEY"),
                Provider::Ollama => Ok(String::new()),
                Provider::Custom(_) => std::env::var("LLM_API_KEY"),
            })
            .ok();

        let mut config = Self::new(provider, api_key, model);

        // Override with environment variables if present
        if let Ok(api_base) = std::env::var("LLM_API_BASE") {
            config.api_base = Some(api_base);
        }
        if let Ok(org_id) = std::env::var("LLM_ORG_ID") {
            config.org_id = Some(org_id);
        }
        if let Ok(max_tokens) = std::env::var("LLM_MAX_TOKENS") {
            if let Ok(tokens) = max_tokens.parse() {
                config.max_tokens = tokens;
            }
        }
        if let Ok(temperature) = std::env::var("LLM_TEMPERATURE") {
            if let Ok(temp) = temperature.parse() {
                config.temperature = temp;
            }
        }
        if let Ok(top_p) = std::env::var("LLM_TOP_P") {
            if let Ok(p) = top_p.parse() {
                config.top_p = p;
            }
        }
        if let Ok(timeout) = std::env::var("LLM_TIMEOUT_SECS") {
            if let Ok(secs) = timeout.parse() {
                config.timeout_secs = secs;
            }
        }

        config
    }

    /// Convert to provider-specific configuration
    pub fn to_provider_config(&self) -> Result<HashMap<String, serde_json::Value>, String> {
        let mut config = HashMap::new();
        
        // Common parameters
        config.insert("model".to_string(), json!(self.model));
        config.insert("max_tokens".to_string(), json!(self.max_tokens));
        config.insert("temperature".to_string(), json!(self.temperature));
        config.insert("top_p".to_string(), json!(self.top_p));
        
        // Provider-specific parameters
        match self.provider {
            Provider::OpenAI => {
                if let Some(org_id) = &self.org_id {
                    config.insert("organization".to_string(), json!(org_id));
                }
            }
            Provider::Anthropic => {
                // Anthropic uses different parameter names
                config.insert("max_tokens_to_sample".to_string(), json!(self.max_tokens));
            }
            Provider::Ollama => {
                config.insert("num_predict".to_string(), json!(self.max_tokens));
            }
            Provider::Custom(_) => {
                // Use parameters as-is for custom providers
            }
        }
        
        // Add any extra parameters
        for (key, value) in &self.extra_params {
            config.insert(key.clone(), json!(value));
        }
        
        Ok(config)
    }
}

impl Default for UnifiedConfig {
    fn default() -> Self {
        Self::new(
            Provider::OpenAI,
            None,
            "gpt-3.5-turbo".to_string(),
        )
    }
}

/// Configuration for LLM clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// API key
    pub api_key: Option<String>,
    /// API endpoint
    pub api_endpoint: Option<String>,
    /// Model name
    pub model: String,
    /// Organization ID (for OpenAI)
    pub org_id: Option<String>,
    /// Request timeout in seconds
    pub timeout_secs: u64,
    /// Maximum retries
    pub max_retries: u32,
    /// Whether to use caching
    pub use_cache: bool,
    /// Similarity threshold for cache hits
    pub similarity_threshold: f32,
    /// Additional configuration values
    pub extra_config: HashMap<String, String>,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            api_endpoint: None,
            model: "gpt-4".to_string(),
            org_id: None,
            timeout_secs: 30,
            max_retries: 3,
            use_cache: true,
            similarity_threshold: 0.95,
            extra_config: HashMap::new(),
        }
    }
} 