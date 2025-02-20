# SuperLightRAG

A lightweight, cross-platform implementation of LightRAG in Rust, optimized for mobile and desktop use.

## Project Structure

```
rust/
├── Cargo.toml              # Main package manifest
├── src/
│   ├── lib.rs             # Library root
│   ├── storage/           # Storage implementations
│   │   ├── kv/           # JSON KV storage
│   │   ├── vector/       # NanoVectorDB-RS
│   │   └── graph/        # petgraph integration
│   ├── processing/       # Document processing
│   │   ├── chunking.rs   # Text chunking
│   │   ├── metadata.rs   # Metadata extraction
│   │   └── cleaning.rs   # Text cleaning
│   ├── api/             # API integrations
│   │   ├── llm/         # LLM providers
│   │   └── embeddings/  # Embedding providers
│   ├── types/           # Common types and traits
│   └── utils/           # Utility functions
├── examples/            # Usage examples
├── benches/            # Performance benchmarks
└── tests/              # Integration tests
```

## Development

### Requirements
- Rust 1.75+
- Cargo
- (Optional) Android NDK for mobile builds
- (Optional) Xcode for iOS builds

### Getting Started
```bash
cd rust
cargo build
```

### Running Tests
```bash
cargo test
```

### Mobile Development
See [MOBILE.md](./MOBILE.md) for mobile platform setup instructions.

## Features
- JSON-based document storage
- Efficient vector search with NanoVectorDB-RS
- Graph operations with petgraph
- Cross-platform support (Desktop + Mobile)
- API-first approach for AI operations 