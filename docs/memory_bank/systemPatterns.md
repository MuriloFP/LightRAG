# SuperLightRAG - System Patterns

## Core Architecture

### Storage Components
1. **JSON KV Storage**
   - Simple file-based storage using JSON
   - Handles document storage, chunks, and caching
   - Rust implementation using `serde_json`

2. **NanoVectorDB-RS**
   - Rust implementation of NanoVectorDB
   - In-memory vector storage with persistence
   - Optimized for mobile resources

3. **Petgraph Graph Storage**
   - Graph operations using petgraph
   - Efficient relationship mapping
   - Memory-optimized graph traversal
   - Native Rust implementation
   - Strongly typed data structures:
     - `NodeData`: ID and attribute map
     - `EdgeData`: Weight, description, keywords
   - JSON-based persistence
   - Advanced features:
     - Pattern matching
     - A* pathfinding
     - Graph algorithm support
   - Node embedding support:
     - Basic vector embedding
     - Node2Vec integration (planned)
   - Graph operations:
     - CRUD operations
     - Batch operations
     - Cascading deletes
     - Graph stabilization
     - Default edge properties
   - Performance optimizations:
     - Memory-efficient storage
     - Optimized query paths
     - Batch operation support

4. **Storage Pattern Comparison with LightRAG**
   - LightRAG supports multiple backends:
     - NetworkX (in-memory)
     - Gremlin (TinkerPop)
     - PostgreSQL (AGE)
     - MongoDB
     - TiDB
     - Oracle
   - SuperLightRAG focuses on single efficient implementation:
     - Petgraph for performance
     - Memory optimization
     - Native Rust benefits
   - Key differences:
     - Strong typing vs dynamic typing
     - Single vs multiple backends
     - File-based vs varied persistence
     - Enhanced graph algorithms

### Core Processing Pipeline
1. Document Processing
   - Text extraction and cleaning
   - Chunking with configurable sizes
   - Metadata extraction

2. Vector Processing
   - API-based embeddings generation
   - Vector indexing and storage
   - Similarity search optimization

3. Graph Building
   - Entity relationship mapping
   - Graph construction and updates
   - Efficient traversal paths

### API Integration
1. **LLM Integration**
   - OpenAI API support
   - Anthropic API support
   - Custom API provider support

2. **Embedding Services**
   - OpenAI Embeddings
   - Custom embedding endpoints
   - Caching for efficiency

## Technical Decisions

### Language & Framework
- **Rust**: Core implementation language
- **Cross-platform support**: iOS, Android, Desktop
- **WASM compatibility**: Web deployment option

### Dependencies
- `serde_json`: JSON handling
- `petgraph`: Graph operations
- `reqwest`: API calls
- `tokio`: Async runtime

### Performance Optimizations
1. Batch processing for embeddings
2. Efficient memory management
3. Persistent caching
4. Incremental updates 

## Rate Limiting

### Token Bucket Algorithm
- Used for token-based rate limiting
- Allows for burst handling while maintaining average rate
- Configurable token refill rate and bucket size
- Implemented in `RateLimiter` struct

### Sliding Window Algorithm
- Used for request-based rate limiting
- Maintains accurate request counts over time windows
- Prevents request bunching at window boundaries
- Implemented in `SlidingWindowLimiter` struct

### Concurrent Request Limiting
- RAII-based approach using `RateLimit` guard
- Automatically decrements count when guard is dropped
- Thread-safe using `RwLock`
- Prevents resource exhaustion

### Rate Limit Configuration
- Configurable limits per provider
- Separate limits for tokens and requests
- Burst allowance configuration
- Default values tuned for common use cases

## Caching System

The caching system is designed to provide efficient and flexible caching for LLM responses. It supports multiple backend implementations and advanced features like similarity search and distributed caching.

### Cache Architecture

The system uses a trait-based design with multiple implementations:

1. `ResponseCache` trait - Core interface defining cache operations
2. `InMemoryCache` - Local in-memory implementation
3. `RedisCache` - Distributed Redis-based implementation
4. `PersistentCache` - Persistent storage implementation

### Cache Features

- Thread-safe operations using `Arc` and `RwLock`
- RAII-based resource management
- Configurable TTL and size limits
- Automatic cleanup of expired entries
- Similarity matching using embeddings
- LLM verification of cache hits
- Multiple eviction strategies (LRU, LFU, FIFO, Random)
- Compression support
- Integrity validation
- Metrics collection

### Distributed Caching

The Redis implementation provides:

- Connection pooling with r2d2
- Configurable Redis URL and pool size
- Separate storage of embeddings for fast similarity search
- Atomic operations for thread safety
- Automatic key prefixing and namespacing
- TTL-based expiration
- Batch operations for efficiency

### Cache Configuration

The `CacheConfig` struct provides extensive configuration options:

```rust
pub struct CacheConfig {
    pub enabled: bool,
    pub max_entries: usize,
    pub ttl: Option<Duration>,
    pub use_fuzzy_match: bool,
    pub similarity_threshold: f32,
    pub use_persistent: bool,
    pub use_llm_verification: bool,
    pub llm_verification_prompt: Option<String>,
    pub backend: CacheBackend,
    pub eviction_strategy: EvictionStrategy,
    pub enable_metrics: bool,
    pub use_compression: bool,
    pub max_compressed_size: Option<usize>,
    pub use_encryption: bool,
    pub encryption_key: Option<String>,
    pub validate_integrity: bool,
    pub enable_sync: bool,
    pub sync_interval: Option<Duration>,
}
```

### Cache Operations

#### Basic Operations
- `get(prompt)` - Retrieve cached response
- `put(prompt, response)` - Store response in cache
- `cleanup()` - Remove expired entries
- `clear()` - Clear all entries

#### Advanced Operations
- `put_with_embedding(prompt, response, embedding)` - Store with similarity search
- `find_similar_entry(embedding)` - Find similar cached responses
- `verify_with_llm(entry)` - Verify cache hit with LLM

#### Distributed Operations
- Connection pooling
- Key-based operations
- Batch operations
- Atomic updates
- TTL management
- Embedding storage

### Error Handling

The system uses a custom `LLMError` type with specific variants for cache operations:
- Connection errors
- Serialization errors
- Redis command errors
- Configuration errors
- Validation errors

### Performance Optimizations

1. Memory Management
   - Connection pooling
   - Separate storage for embeddings
   - Compression support
   - Configurable limits

2. Concurrency
   - Thread-safe operations
   - Atomic updates
   - Connection pooling
   - Lock-free metrics

3. Caching Strategies
   - Multiple eviction policies
   - TTL-based expiration
   - Similarity matching
   - Batch operations

### Monitoring

The system collects metrics via `CacheMetrics`:
- Cache hits/misses
- Evictions
- Cache size
- Similarity matches
- LLM verifications

These metrics are atomic and can be accessed without locking.

## Provider Integration

### LLM Providers
- Abstract trait for provider implementations
- Common configuration structure
- Rate limiting integration
- Caching integration
- Error handling and retries

### Embedding Providers
- Abstract trait for provider implementations
- Shared configuration structure
- Rate limiting integration
- Similarity-based caching
- Batch operations support

## Error Handling

### Rate Limit Errors
- Specific error types for rate limiting
- Automatic retry with backoff
- Clear error messages
- Proper error propagation

### Cache Errors
- TTL expiration handling
- Size limit handling
- Thread safety errors
- Invalid configuration errors

## Future Enhancements

### Distributed Caching
- Redis integration planned
- Cache synchronization
- Distributed locking
- Cross-node invalidation

### Cache Monitoring
- Hit/miss ratio tracking
- Size monitoring
- Performance metrics
- Health checks

### Cache Persistence
- Disk-based storage
- Backup/restore
- Migration support
- Data integrity checks 