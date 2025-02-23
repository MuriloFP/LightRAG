use std::time::{Duration, Instant};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::mpsc;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use tokio_stream::wrappers::ReceiverStream;
use std::collections::HashMap;
use bytes::BytesMut;
use futures::future::BoxFuture;

use crate::types::llm::{StreamingResponse, StreamingTiming};
use crate::llm::LLMError;

/// Configuration for stream processing
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Channel buffer size for streaming responses
    pub channel_buffer_size: usize,
    
    /// Whether to enable chunk batching
    pub enable_batching: bool,
    
    /// Maximum batch size in bytes
    pub max_batch_size: usize,
    
    /// Maximum batch wait time in milliseconds
    pub max_batch_wait_ms: u64,
    
    /// Whether to enable compression
    pub enable_compression: bool,

    /// Initial buffer capacity
    pub initial_buffer_capacity: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            channel_buffer_size: 100,
            enable_batching: true,
            max_batch_size: 16384, // 16KB
            max_batch_wait_ms: 50,  // 50ms
            enable_compression: false,
            initial_buffer_capacity: 4096, // 4KB
        }
    }
}

/// Optimized stream processor for LLM responses
pub struct StreamProcessor {
    config: StreamConfig,
    start_time: Instant,
    total_tokens: Arc<AtomicUsize>,
    current_batch: BytesMut,
    last_batch_time: Instant,
}

impl StreamProcessor {
    /// Create a new stream processor
    pub fn new(config: StreamConfig) -> Self {
        Self {
            config: config.clone(),
            start_time: Instant::now(),
            total_tokens: Arc::new(AtomicUsize::new(0)),
            current_batch: BytesMut::with_capacity(config.initial_buffer_capacity),
            last_batch_time: Instant::now(),
        }
    }

    /// Process a stream of bytes into StreamingResponse chunks with optimized batching
    pub async fn process_stream<S, F>(
        &mut self,
        stream: S,
        parser: F,
    ) -> Pin<Box<dyn Stream<Item = Result<StreamingResponse, LLMError>> + Send>>
    where
        S: Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + Unpin + 'static,
        F: Fn(&str) -> BoxFuture<'static, Result<Option<(String, bool, HashMap<String, String>)>, LLMError>> + Send + Sync + 'static,
    {
        let (tx, rx) = mpsc::channel(self.config.channel_buffer_size);
        let total_tokens = self.total_tokens.clone();
        let config = self.config.clone();
        let start_time = self.start_time;

        let mut pinned_stream = Box::pin(stream);
        
        tokio::spawn(async move {
            let mut buffer = BytesMut::with_capacity(config.initial_buffer_capacity);
            let mut batch_start = Instant::now();
            let mut batch_size = 0;
            
            while let Some(result) = pinned_stream.next().await {
                match result {
                    Ok(bytes) => {
                        buffer.extend_from_slice(&bytes);
                        batch_size += bytes.len();
                        
                        let should_process = if config.enable_batching {
                            batch_size >= config.max_batch_size || 
                            batch_start.elapsed() >= Duration::from_millis(config.max_batch_wait_ms)
                        } else {
                            true
                        };
                        
                        if should_process {
                            let mut processed = 0;
                            let buffer_str = String::from_utf8_lossy(&buffer[..]);
                            
                            for line in buffer_str.split_inclusive('\n') {
                                if line.ends_with('\n') {
                                    if let Ok(Some((text, done, metadata))) = parser(line.trim()).await {
                                        let chunk_tokens = text.split_whitespace().count();
                                        total_tokens.fetch_add(chunk_tokens, Ordering::Relaxed);
                                        
                                        let timing = StreamingTiming {
                                            chunk_duration: batch_start.elapsed().as_micros() as u64,
                                            total_duration: start_time.elapsed().as_micros() as u64,
                                            prompt_eval_duration: Some(0),
                                            token_gen_duration: Some(0),
                                        };
                                        
                                        let stream_response = StreamingResponse {
                                            text,
                                            done,
                                            timing: Some(timing),
                                            chunk_tokens,
                                            total_tokens: total_tokens.load(Ordering::Relaxed),
                                            metadata,
                                        };
                                        
                                        if tx.send(Ok(stream_response)).await.is_err() {
                                            return;
                                        }
                                        
                                        processed += line.len();
                                        
                                        if done {
                                            return;
                                        }
                                    }
                                }
                            }
                            
                            if processed > 0 {
                                buffer.split_to(processed);
                            }
                            
                            batch_size = buffer.len();
                            batch_start = Instant::now();
                            
                            if buffer.capacity() > config.initial_buffer_capacity * 2 {
                                let mut new_buffer = BytesMut::with_capacity(config.initial_buffer_capacity);
                                new_buffer.extend_from_slice(&buffer[..]);
                                buffer = new_buffer;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = tx.send(Err(LLMError::RequestFailed(e.to_string()))).await;
                        break;
                    }
                }
            }

            // Flush any remaining buffered data after stream ends
            if !buffer.is_empty() {
                let buffer_str = String::from_utf8_lossy(&buffer[..]);
                for line in buffer_str.split_inclusive('\n') {
                    if line.ends_with('\n') {
                        if let Ok(Some((text, done, metadata))) = parser(line.trim()).await {
                            let chunk_tokens = text.split_whitespace().count();
                            total_tokens.fetch_add(chunk_tokens, Ordering::Relaxed);

                            let timing = StreamingTiming {
                                chunk_duration: batch_start.elapsed().as_micros() as u64,
                                total_duration: start_time.elapsed().as_micros() as u64,
                                prompt_eval_duration: Some(0),
                                token_gen_duration: Some(0),
                            };

                            let stream_response = StreamingResponse {
                                text,
                                done,
                                timing: Some(timing),
                                chunk_tokens,
                                total_tokens: total_tokens.load(Ordering::Relaxed),
                                metadata,
                            };

                            let _ = tx.send(Ok(stream_response)).await;

                            if done {
                                break;
                            }
                        }
                    }
                }
            }
        });

        Box::pin(ReceiverStream::new(rx))
    }
}

/// Helper trait for stream compression
#[cfg(feature = "compression")]
pub trait StreamCompression {
    /// Compress a chunk of data
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>, LLMError>;
    
    /// Decompress a chunk of data
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>, LLMError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;
    use bytes::Bytes;
    use reqwest::Error as ReqwestError;
    use std::time::Duration;
    
    #[tokio::test]
    async fn test_stream_processor() {
        let config = StreamConfig {
            enable_batching: true,
            max_batch_size: 10,
            max_batch_wait_ms: 10,
            ..Default::default()
        };
        
        let mut processor = StreamProcessor::new(config);
        
        // Create a test stream with bytes
        let test_stream = stream::iter(vec![
            Ok(Bytes::from("chunk1\n")),
            Ok(Bytes::from("chunk2\n")),
            Ok(Bytes::from("chunk3\n")),
        ]);
        
        // Create a parser that properly returns a BoxFuture
        let parser = move |text: &str| -> BoxFuture<'static, Result<Option<(String, bool, HashMap<String, String>)>, LLMError>> {
            let text = text.to_string(); // Clone the text to own it
            Box::pin(async move {
                let mut metadata = HashMap::new();
                metadata.insert("test".to_string(), "value".to_string());
                Ok(Some((text.clone(), text == "chunk3", metadata)))
            })
        };
        
        let mut result_stream = processor.process_stream(test_stream, parser).await;
        
        // Verify the results
        let mut results = Vec::new();
        while let Some(result) = result_stream.next().await {
            results.push(result.unwrap());
        }
        
        assert!(!results.is_empty());
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].text, "chunk1");
        assert_eq!(results[1].text, "chunk2");
        assert_eq!(results[2].text, "chunk3");
    }

    #[tokio::test]
    async fn test_streaming_performance() {
        // Test configuration with different batching settings
        let configs = vec![
            StreamConfig {
                enable_batching: false,
                ..Default::default()
            },
            StreamConfig {
                enable_batching: true,
                max_batch_size: 1024,
                max_batch_wait_ms: 10,
                ..Default::default()
            },
            StreamConfig {
                enable_batching: true,
                max_batch_size: 16384,
                max_batch_wait_ms: 50,
                ..Default::default()
            },
        ];

        // Create a stream generator function
        let create_test_stream = || {
            let num_chunks = 1000;
            stream::iter((0..num_chunks).map(|i| {
                Ok(Bytes::from(format!("chunk{}\n", i))) as Result<Bytes, ReqwestError>
            }))
        };

        // Create a parser that properly returns a BoxFuture
        let parser = move |text: &str| -> BoxFuture<'static, Result<Option<(String, bool, HashMap<String, String>)>, LLMError>> {
            let text = text.to_string(); // Clone the text to own it
            Box::pin(async move {
                tokio::time::sleep(Duration::from_micros(100)).await;
                let mut metadata = HashMap::new();
                metadata.insert("test".to_string(), "value".to_string());
                Ok(Some((
                    text.clone(),
                    text.contains("chunk999"),
                    metadata,
                )))
            })
        };

        let mut results = Vec::new();

        // Test each configuration
        for config in configs {
            let mut processor = StreamProcessor::new(config.clone());
            let test_stream = create_test_stream();
            let start_time = Instant::now();

            let mut result_stream = processor.process_stream(test_stream, parser).await;
            let mut response_count = 0;
            let mut total_tokens = 0;

            while let Some(result) = result_stream.next().await {
                let response = result.unwrap();
                response_count += 1;
                total_tokens += response.chunk_tokens;
            }

            let duration = start_time.elapsed();
            results.push((
                config.enable_batching,
                config.max_batch_size,
                duration,
                response_count,
                total_tokens,
            ));
        }

        // Verify performance improvements
        assert!(results[0].2 > results[1].2, "Batching should be faster than no batching");
        assert!(results[1].2 >= results[2].2, "Larger batches should be at least as fast");

        // Print performance results
        for (batching, batch_size, duration, responses, tokens) in results {
            println!(
                "Config: batching={}, batch_size={}, duration={:?}, responses={}, tokens={}, tokens/sec={}",
                batching,
                batch_size,
                duration,
                responses,
                tokens,
                tokens as f64 / duration.as_secs_f64(),
            );
        }
    }
} 