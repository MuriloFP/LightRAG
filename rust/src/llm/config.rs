use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LLMConfig {
    /// Model to use
    pub model: String,

    /// API endpoint
    pub api_endpoint: String,

    /// API key
    pub api_key: String,

    /// Organization ID
    pub org_id: Option<String>,

    /// Timeout in seconds
    pub timeout_secs: u64,

    /// Maximum number of tokens to generate
    pub max_tokens: Option<u32>,

    /// Temperature for sampling
    pub temperature: Option<f32>,

    /// Top-p for nucleus sampling
    pub top_p: Option<f32>,

    /// Frequency penalty
    pub frequency_penalty: Option<f32>,

    /// Presence penalty
    pub presence_penalty: Option<f32>,

    /// Similarity threshold for cache matching
    pub similarity_threshold: f32,
}

impl Default for LLMConfig {
    fn default() -> Self {
        Self {
            model: "gpt-3.5-turbo".to_string(),
            api_endpoint: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            org_id: None,
            timeout_secs: 30,
            max_tokens: Some(2048),
            temperature: Some(0.7),
            top_p: Some(1.0),
            frequency_penalty: Some(0.0),
            presence_penalty: Some(0.0),
            similarity_threshold: 0.8,
        }
    }
} 