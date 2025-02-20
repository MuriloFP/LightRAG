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