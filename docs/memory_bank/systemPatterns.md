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