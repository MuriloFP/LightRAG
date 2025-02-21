use crate::types::embeddings::EmbeddingProvider;

/// OpenAI embeddings provider implementation
pub mod openai;

/// Ollama embeddings provider implementation
pub mod ollama;

// Re-export providers
pub use openai::OpenAIEmbeddingProvider;
pub use ollama::OllamaEmbeddingProvider;

// Re-export core types from types::embeddings
pub use crate::types::embeddings::*;

/// Cache implementation for embeddings
pub mod cache;

// Future embedding providers will be added here 