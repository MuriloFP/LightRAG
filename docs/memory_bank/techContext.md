# SuperLightRAG - Technical Context

## Development Environment

### Core Technologies
- **Language**: Rust 1.75+ (2021 edition)
- **Build System**: Cargo
- **Package Manager**: Cargo/crates.io

### Cross-Platform Support
- **Desktop**: Windows, macOS, Linux
- **Mobile**: iOS (11+), Android (API 21+)
- **Web**: WASM (optional)

### Development Tools
- **IDE**: VS Code with rust-analyzer
- **Testing**: cargo test
- **Documentation**: rustdoc
- **CI/CD**: GitHub Actions

## Technical Requirements

### Storage
1. **JSON KV Storage**
   - File-based persistence
   - Atomic operations
   - Concurrent access support
   - Mobile storage optimization

2. **NanoVectorDB-RS**
   - Maximum memory usage: 512MB
   - Vector dimension support: up to 1536
   - Index type: HNSW
   - Persistence format: Binary

3. **Graph Storage**
   - Maximum nodes: 100,000
   - Maximum edges: 500,000
   - Memory-mapped storage
   - Efficient traversal using petgraph
   - Persistence format: JSON/Binary
   - CRUD operations support
   - Query optimization

### API Integration
- **Rate Limiting**: Configurable per provider
- **Batch Processing**: Configurable sizes
- **Timeout Handling**: Configurable with retries
- **Error Handling**: Comprehensive error types

### Performance Targets
- **Memory Usage**: < 512MB total
- **Storage Size**: < 1GB for typical use
- **Query Latency**: < 100ms (p95)
- **Batch Processing**: 1000 docs/minute

### Mobile Constraints
- **Memory Usage**: < 200MB on mobile
- **Storage**: < 500MB on mobile
- **Battery Impact**: Minimal background processing
- **Network**: Efficient batch operations

## Dependencies

### Core
```toml
[dependencies]
serde = "1.0"
serde_json = "1.0"
tokio = { version = "1.0", features = ["full"] }
petgraph = "0.6"
reqwest = { version = "0.11", features = ["json"] }
```

### Optional
```toml
[features]
wasm = ["wasm-bindgen"]
mobile = ["mobile-optimizations"]
```

## Security Considerations
- API key management
- Data encryption at rest
- Secure storage access
- Input validation

## Compatibility Requirements
- Rust stable channel
- No unsafe code (when possible)
- Cross-platform file paths
- Mobile filesystem access 