use std::fmt;
use redis::RedisError;
use serde_json::Error as JsonError;

#[derive(Debug)]
pub enum LLMError {
    ApiError(String),
    CacheError(String),
    ConfigError(String),
    IoError(String),
    SerializationError(String),
    ValidationError(String),
    RateLimitExceeded(String),
    RequestFailed(String),
    InvalidResponse(String),
}

impl fmt::Display for LLMError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LLMError::ApiError(msg) => write!(f, "API error: {}", msg),
            LLMError::CacheError(msg) => write!(f, "Cache error: {}", msg),
            LLMError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            LLMError::IoError(msg) => write!(f, "IO error: {}", msg),
            LLMError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            LLMError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            LLMError::RateLimitExceeded(msg) => write!(f, "Rate limit exceeded: {}", msg),
            LLMError::RequestFailed(msg) => write!(f, "Request failed: {}", msg),
            LLMError::InvalidResponse(msg) => write!(f, "Invalid response: {}", msg),
        }
    }
}

impl std::error::Error for LLMError {}

impl From<RedisError> for LLMError {
    fn from(err: RedisError) -> Self {
        LLMError::CacheError(err.to_string())
    }
}

impl From<JsonError> for LLMError {
    fn from(err: JsonError) -> Self {
        LLMError::SerializationError(err.to_string())
    }
}

impl From<std::io::Error> for LLMError {
    fn from(err: std::io::Error) -> Self {
        LLMError::IoError(err.to_string())
    }
}

impl From<reqwest::Error> for LLMError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            LLMError::RequestFailed(format!("Request timed out: {}", err))
        } else if err.is_connect() {
            LLMError::RequestFailed(format!("Connection failed: {}", err))
        } else {
            LLMError::RequestFailed(err.to_string())
        }
    }
} 