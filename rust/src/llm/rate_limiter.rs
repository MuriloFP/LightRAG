use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during rate limiting
#[derive(Error, Debug)]
pub enum RateLimitError {
    /// Rate limit exceeded
    #[error("Rate limit exceeded: {0}")]
    LimitExceeded(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Configuration for rate limiting
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
            tokens_per_minute: 90_000,
            max_concurrent: 10,
            use_token_bucket: true,
            burst_size: 5,
        }
    }
}

/// Token bucket rate limiter
pub struct TokenBucketLimiter {
    /// Current token count
    tokens: f64,
    
    /// Maximum token count
    max_tokens: f64,
    
    /// Token refill rate per second
    refill_rate: f64,
    
    /// Last refill time
    last_refill: Instant,
}

impl TokenBucketLimiter {
    /// Create a new token bucket limiter
    pub fn new(max_tokens: u32, tokens_per_second: f64) -> Self {
        Self {
            tokens: max_tokens as f64,
            max_tokens: max_tokens as f64,
            refill_rate: tokens_per_second,
            last_refill: Instant::now(),
        }
    }
    
    /// Try to acquire tokens
    pub fn try_acquire(&mut self, tokens: u32) -> bool {
        self.refill();
        
        if self.tokens >= tokens as f64 {
            self.tokens -= tokens as f64;
            true
        } else {
            false
        }
    }
    
    /// Refill tokens based on elapsed time
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.last_refill = now;
        
        self.tokens = (self.tokens + self.refill_rate * elapsed).min(self.max_tokens);
    }
}

/// Sliding window rate limiter
pub struct SlidingWindowLimiter {
    /// Window of request timestamps
    window: VecDeque<Instant>,
    
    /// Window duration
    duration: Duration,
    
    /// Maximum requests in window
    max_requests: u32,
}

impl SlidingWindowLimiter {
    /// Create a new sliding window limiter
    pub fn new(duration: Duration, max_requests: u32) -> Self {
        Self {
            window: VecDeque::with_capacity(max_requests as usize),
            duration,
            max_requests,
        }
    }
    
    /// Try to acquire permission for a request
    pub fn try_acquire(&mut self) -> bool {
        let now = Instant::now();
        
        // Remove expired timestamps
        while let Some(timestamp) = self.window.front() {
            if now.duration_since(*timestamp) > self.duration {
                self.window.pop_front();
            } else {
                break;
            }
        }
        
        if self.window.len() < self.max_requests as usize {
            self.window.push_back(now);
            true
        } else {
            false
        }
    }
}

/// Combined rate limiter using both token bucket and sliding window
pub struct RateLimiter {
    /// Configuration
    config: RateLimitConfig,
    
    /// Token bucket for token-based limiting
    token_bucket: Arc<RwLock<TokenBucketLimiter>>,
    
    /// Sliding window for request-based limiting
    sliding_window: Arc<RwLock<SlidingWindowLimiter>>,
    
    /// Current concurrent requests
    concurrent_requests: Arc<RwLock<u32>>,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(config: RateLimitConfig) -> Self {
        let token_bucket = TokenBucketLimiter::new(
            config.tokens_per_minute,
            config.tokens_per_minute as f64 / 60.0,
        );
        
        let sliding_window = SlidingWindowLimiter::new(
            Duration::from_secs(60),
            config.requests_per_minute,
        );
        
        Self {
            config,
            token_bucket: Arc::new(RwLock::new(token_bucket)),
            sliding_window: Arc::new(RwLock::new(sliding_window)),
            concurrent_requests: Arc::new(RwLock::new(0)),
        }
    }
    
    /// Try to acquire permission for a request with token count
    pub async fn try_acquire(&self, tokens: u32) -> Result<RateLimit, RateLimitError> {
        // Check concurrent requests
        let mut concurrent = self.concurrent_requests.write().await;
        if *concurrent >= self.config.max_concurrent {
            return Err(RateLimitError::LimitExceeded(
                "Maximum concurrent requests exceeded".to_string()
            ));
        }
        
        // Check token bucket if enabled
        if self.config.use_token_bucket {
            let mut bucket = self.token_bucket.write().await;
            if !bucket.try_acquire(tokens) {
                return Err(RateLimitError::LimitExceeded(
                    "Token limit exceeded".to_string()
                ));
            }
        }
        
        // Check sliding window
        let mut window = self.sliding_window.write().await;
        if !window.try_acquire() {
            return Err(RateLimitError::LimitExceeded(
                "Request limit exceeded".to_string()
            ));
        }
        
        // Increment concurrent requests
        *concurrent += 1;
        
        Ok(RateLimit::new(self.concurrent_requests.clone()))
    }
}

/// RAII guard for rate limit
pub struct RateLimit {
    concurrent_requests: Arc<tokio::sync::RwLock<u32>>,
}

impl RateLimit {
    pub fn new(concurrent_requests: Arc<tokio::sync::RwLock<u32>>) -> Self {
        Self { concurrent_requests }
    }
}

impl Drop for RateLimit {
    fn drop(&mut self) {
        // Decrement concurrent requests in a blocking context
        // This is safe because Drop is called when the guard goes out of scope
        if let Ok(mut concurrent) = self.concurrent_requests.try_write() {
            *concurrent = concurrent.saturating_sub(1);
        }
    }
} 