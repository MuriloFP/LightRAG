// Core cache system tests
mod cache;

// LLM provider tests
mod providers;

// Embedding tests
mod embedding_tests;

// LLM summary tests
mod llm_summary_tests;

// Re-export test modules
pub use cache::*;
pub use providers::*;
pub use embedding_tests::*;
pub use llm_summary_tests::*;

// Will add other test categories as we implement them:
// mod benchmarks;     // Performance benchmarks
// mod stress;         // Stress tests
// mod integration;    // Integration tests
// mod security;       // Security tests
// mod utils;          // Test utilities 