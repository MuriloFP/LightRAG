/// OpenAI API client implementation.
/// 
/// Provides integration with OpenAI's GPT models through their REST API.
/// Supports:
/// - Text completion
/// - Response caching
/// - Rate limiting and retries
pub mod openai;

/// Ollama API client implementation.
/// 
/// Provides integration with locally hosted language models through Ollama.
/// Supports:
/// - Text generation
/// - Custom model loading
/// - Local inference
pub mod ollama;

/// Anthropic API client implementation.
///
/// Provides integration with Anthropic's Claude models through their REST API.
/// Supports:
/// - Text completion
/// - Response streaming
/// - System prompts
pub mod anthropic;

/// LiteLLM proxy client implementation.
///
/// Provides unified interface for multiple LLM providers.
/// Supports:
/// - Multiple provider routing
/// - Unified configuration
/// - Provider fallbacks
/// - Consistent API interface
pub mod litellm;

pub use openai::OpenAIClient;
pub use ollama::OllamaClient;
pub use anthropic::AnthropicClient;
pub use litellm::{
    LiteLLMClient,
    LiteLLMConfig,
    LLMProvider,
    ProviderAdapter,
    ProviderConfig,
    OpenAIAdapter,
    AnthropicAdapter,
}; 