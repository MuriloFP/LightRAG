//! Storage module providing persistent data storage capabilities.
//! 
//! This module implements three core storage components:
//! - Key-Value storage for document metadata and content
//! - Vector storage for embeddings and similarity search
//! - Graph storage for entity relationships and knowledge graphs
//! 
//! Each component is optimized for its specific use case while maintaining
//! cross-platform compatibility and efficient resource usage.

/// Key-Value storage implementation for document metadata and content.
/// 
/// Features:
/// - JSON-based persistence
/// - Efficient key-based lookups
/// - Namespace support for data isolation
/// - Caching for frequently accessed data
pub mod kv;

/// Vector storage implementation for embeddings and similarity search.
/// 
/// Features:
/// - HNSW-based approximate nearest neighbor search
/// - Efficient vector operations and normalization
/// - Metadata association with vectors
/// - Persistence with automatic index maintenance
pub mod vector;

/// Graph storage implementation for entity relationships.
/// 
/// Features:
/// - Petgraph-based graph operations
/// - Relationship mapping and traversal
/// - Pattern matching and subgraph extraction
/// - Efficient graph persistence
pub mod graph; 