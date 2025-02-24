/// OpenAI API integration.
/// 
/// Provides integration with OpenAI's GPT models through their REST API.
/// Supports:
/// - Text completion via openai_complete()
/// - Streaming via openai_complete_stream()
/// - Batch completion via openai_complete_batch()
pub mod openai;

/// Ollama API integration.
/// 
/// Provides integration with locally hosted language models through Ollama.
/// Supports:
/// - Text generation via ollama_complete()
/// - Streaming via ollama_complete_stream()
/// - Batch completion via ollama_complete_batch()
pub mod ollama;

/// Anthropic API integration.
///
/// Provides integration with Anthropic's Claude models through their REST API.
/// Supports:
/// - Text completion via anthropic_complete()
/// - Streaming via anthropic_complete_stream()
/// - Batch completion via anthropic_complete_batch()
pub mod anthropic;

// Re-export function-based APIs
pub use openai::{openai_complete, openai_complete_stream, openai_complete_batch};
pub use ollama::{ollama_complete, ollama_complete_stream, ollama_complete_batch};
pub use anthropic::{anthropic_complete, anthropic_complete_stream, anthropic_complete_batch}; 