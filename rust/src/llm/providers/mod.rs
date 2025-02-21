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

pub use openai::OpenAIClient;
pub use ollama::OllamaClient; 