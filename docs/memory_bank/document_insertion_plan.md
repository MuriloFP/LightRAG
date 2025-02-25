# SuperLightRAG Document Insertion Implementation Plan

## Overview

This document outlines the implementation plan for adding document insertion capabilities to SuperLightRAG. Document insertion is a critical component that processes input text, extracts relevant information, and stores it in various storage components for later retrieval.

## Implementation Goals

1. Implement various document insertion methods matching the Python implementation
2. Support both single and batch document processing
3. Enable custom chunk insertion for pre-processed documents
4. Support custom knowledge graph insertion
5. Integrate with existing document processors and storage components

## Core Insertion Method Signatures

```rust
impl SuperLightRAG {
    /// Insert a single document into the system
    pub async fn insert(&self, text: &str) -> Result<String> {
        // Implementation details below
    }
    
    /// Insert multiple documents in batch
    pub async fn insert_batch(&self, texts: Vec<String>) -> Result<Vec<String>> {
        // Implementation details below
    }
    
    /// Insert pre-chunked document
    pub async fn insert_custom_chunks(&self, full_text: &str, chunks: Vec<String>) -> Result<String> {
        // Implementation details below
    }
    
    /// Insert custom knowledge graph
    pub async fn insert_custom_kg(&self, custom_kg: &KnowledgeGraph) -> Result<()> {
        // Implementation details below
    }
}
```

## Implementation Steps

### Step 1: Implement Basic Insert Method

```rust
pub async fn insert(&self, text: &str) -> Result<String> {
    // 1. Generate document ID
    let doc_id = generate_document_id(text);
    
    // 2. Check for duplicates
    let existing = self.check_document_exists(&doc_id).await?;
    if existing {
        return Ok(doc_id); // Return existing ID if document already exists
    }
    
    // 3. Track document status
    self.update_document_status(&doc_id, DocumentStatus::Processing, None).await?;
    
    // 4. Process document
    let result = self.process_document(text, &doc_id).await;
    
    // 5. Update status based on processing result
    match result {
        Ok(_) => {
            self.update_document_status(&doc_id, DocumentStatus::Completed, None).await?;
            Ok(doc_id)
        }
        Err(e) => {
            self.update_document_status(
                &doc_id, 
                DocumentStatus::Failed, 
                Some(e.to_string())
            ).await?;
            Err(e)
        }
    }
}

async fn process_document(&self, text: &str, doc_id: &str) -> Result<()> {
    // 1. Chunk the document
    let chunks = processing::chunking::chunk_text(
        text,
        processing::chunking::ChunkingConfig {
            chunk_size: self.config.chunk_size,
            chunk_overlap: self.config.chunk_overlap,
            ..Default::default()
        },
    )?;
    
    // 2. Store full document in KV storage
    let mut full_doc_data = HashMap::new();
    full_doc_data.insert("content".to_string(), serde_json::Value::String(text.to_string()));
    full_doc_data.insert("created_at".to_string(), serde_json::Value::String(chrono::Utc::now().to_rfc3339()));
    
    let mut data = HashMap::new();
    data.insert(doc_id.to_string(), full_doc_data);
    
    let mut kv = self.kv_storage.write().await;
    kv.upsert(data).await?;
    
    // 3. Process chunks and store them
    let mut chunk_data = HashMap::new();
    let mut vector_data = Vec::new();
    
    for (i, chunk) in chunks.iter().enumerate() {
        let chunk_id = format!("chunk-{}-{}", doc_id, i);
        
        // Store chunk metadata
        let mut chunk_metadata = HashMap::new();
        chunk_metadata.insert("content".to_string(), serde_json::Value::String(chunk.text.clone()));
        chunk_metadata.insert("doc_id".to_string(), serde_json::Value::String(doc_id.to_string()));
        chunk_metadata.insert("index".to_string(), serde_json::Value::Number(serde_json::Number::from(i)));
        chunk_metadata.insert("start_char".to_string(), serde_json::Value::Number(serde_json::Number::from(chunk.start_char)));
        chunk_metadata.insert("end_char".to_string(), serde_json::Value::Number(serde_json::Number::from(chunk.end_char)));
        
        chunk_data.insert(chunk_id.clone(), chunk_metadata.clone());
        
        // Generate embedding for the chunk
        let embedding = self.generate_embedding(&chunk.text).await?;
        
        // Create vector data
        let vector = storage::vector::VectorData {
            id: chunk_id.clone(),
            vector: embedding,
            metadata: chunk_metadata,
        };
        
        vector_data.push(vector);
    }
    
    // 4. Store chunk data in KV storage
    kv.upsert(chunk_data).await?;
    
    // 5. Store vectors in vector storage
    let mut vector_storage = self.vector_storage.write().await;
    vector_storage.upsert(vector_data).await?;
    
    // 6. Extract entities and build graph
    self.extract_entities_and_build_graph(text, doc_id).await?;
    
    // 7. Finalize processing
    self.finalize_document_processing(doc_id).await?;
    
    Ok(())
}
```

### Step 2: Implement Entity Extraction and Graph Building

```rust
async fn extract_entities_and_build_graph(&self, text: &str, doc_id: &str) -> Result<()> {
    // 1. Create entity extractor
    let entity_extractor = match self.entity_extractor.as_ref() {
        Some(extractor) => extractor.clone(),
        None => {
            // Create default extractor
            let llm_provider = self.get_llm_provider().await?;
            Arc::new(processing::entity::LLMEntityExtractor::new(llm_provider))
        }
    };
    
    // 2. Extract entities and relationships
    let extraction_result = entity_extractor.extract_entities(text).await?;
    
    // 3. Store entities as nodes in graph
    let mut graph = self.graph_storage.write().await;
    
    for entity in &extraction_result.entities {
        let mut attributes = HashMap::new();
        attributes.insert("name".to_string(), serde_json::Value::String(entity.name.clone()));
        attributes.insert("type".to_string(), serde_json::Value::String(entity.entity_type.to_string()));
        attributes.insert("description".to_string(), serde_json::Value::String(entity.description.clone()));
        attributes.insert("source_id".to_string(), serde_json::Value::String(doc_id.to_string()));
        
        // Add entity as a node to the graph
        graph.upsert_node(&entity.id, attributes).await?;
    }
    
    // 4. Store relationships as edges
    for relation in &extraction_result.relationships {
        let edge_data = storage::graph::EdgeData {
            weight: 1.0,
            label: relation.relation_type.clone(),
            attributes: {
                let mut m = HashMap::new();
                m.insert("source_id".to_string(), serde_json::Value::String(doc_id.to_string()));
                m.insert("description".to_string(), serde_json::Value::String(relation.description.clone()));
                m
            },
        };
        
        // Add relationship as an edge to the graph
        graph.upsert_edge(&relation.source_id, &relation.target_id, edge_data).await?;
    }
    
    // 5. Store entity vectors for semantic search
    let mut entity_vectors = Vec::new();
    
    for entity in &extraction_result.entities {
        let content = format!("{} {}", entity.name, entity.description);
        let embedding = self.generate_embedding(&content).await?;
        
        let vector = storage::vector::VectorData {
            id: format!("entity-{}", entity.id),
            vector: embedding,
            metadata: {
                let mut m = HashMap::new();
                m.insert("content".to_string(), serde_json::Value::String(content));
                m.insert("entity_name".to_string(), serde_json::Value::String(entity.name.clone()));
                m.insert("entity_type".to_string(), serde_json::Value::String(entity.entity_type.to_string()));
                m.insert("source_id".to_string(), serde_json::Value::String(doc_id.to_string()));
                m
            },
        };
        
        entity_vectors.push(vector);
    }
    
    // Store entity vectors
    if !entity_vectors.is_empty() {
        let mut vector_storage = self.vector_storage.write().await;
        vector_storage.upsert(entity_vectors).await?;
    }
    
    Ok(())
}
```

### Step 3: Implement Batch Document Insertion

```rust
pub async fn insert_batch(&self, texts: Vec<String>) -> Result<Vec<String>> {
    // 1. Set up progress tracking
    let total = texts.len();
    let processed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let mut doc_ids = Vec::with_capacity(total);
    
    // 2. Process documents in parallel with controlled concurrency
    let semaphore = Arc::new(tokio::sync::Semaphore::new(4)); // Limit to 4 concurrent operations
    
    let mut tasks = Vec::new();
    
    for text in texts {
        let permit = semaphore.clone().acquire_owned().await?;
        let processed_clone = processed.clone();
        let self_clone = self.clone();
        
        let task = tokio::spawn(async move {
            let result = self_clone.insert(&text).await;
            
            // Track progress
            let current = processed_clone.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            tracing::info!("Processed document {}/{}", current, total);
            
            // Release permit
            drop(permit);
            
            result
        });
        
        tasks.push(task);
    }
    
    // 3. Collect results
    for task in tasks {
        match task.await {
            Ok(Ok(doc_id)) => {
                doc_ids.push(doc_id);
            }
            Ok(Err(e)) => {
                tracing::error!("Error processing document: {}", e);
            }
            Err(e) => {
                tracing::error!("Task error: {}", e);
            }
        }
    }
    
    Ok(doc_ids)
}
```

### Step 4: Implement Custom Chunk Insertion

```rust
pub async fn insert_custom_chunks(&self, full_text: &str, chunks: Vec<String>) -> Result<String> {
    // 1. Generate document ID
    let doc_id = generate_document_id(full_text);
    
    // 2. Check for duplicates
    let existing = self.check_document_exists(&doc_id).await?;
    if existing {
        return Ok(doc_id); // Return existing ID if document already exists
    }
    
    // 3. Track document status
    self.update_document_status(&doc_id, DocumentStatus::Processing, None).await?;
    
    // 4. Store full document in KV storage
    let mut full_doc_data = HashMap::new();
    full_doc_data.insert("content".to_string(), serde_json::Value::String(full_text.to_string()));
    full_doc_data.insert("created_at".to_string(), serde_json::Value::String(chrono::Utc::now().to_rfc3339()));
    
    let mut data = HashMap::new();
    data.insert(doc_id.to_string(), full_doc_data);
    
    let mut kv = self.kv_storage.write().await;
    kv.upsert(data).await?;
    
    // 5. Process custom chunks
    let mut chunk_data = HashMap::new();
    let mut vector_data = Vec::new();
    
    for (i, chunk) in chunks.into_iter().enumerate() {
        let chunk_id = format!("chunk-{}-{}", doc_id, i);
        
        // Store chunk metadata
        let mut chunk_metadata = HashMap::new();
        chunk_metadata.insert("content".to_string(), serde_json::Value::String(chunk.clone()));
        chunk_metadata.insert("doc_id".to_string(), serde_json::Value::String(doc_id.to_string()));
        chunk_metadata.insert("index".to_string(), serde_json::Value::Number(serde_json::Number::from(i)));
        
        chunk_data.insert(chunk_id.clone(), chunk_metadata.clone());
        
        // Generate embedding for the chunk
        let embedding = self.generate_embedding(&chunk).await?;
        
        // Create vector data
        let vector = storage::vector::VectorData {
            id: chunk_id.clone(),
            vector: embedding,
            metadata: chunk_metadata,
        };
        
        vector_data.push(vector);
    }
    
    // 6. Store chunk data in KV storage
    kv.upsert(chunk_data).await?;
    
    // 7. Store vectors in vector storage
    let mut vector_storage = self.vector_storage.write().await;
    vector_storage.upsert(vector_data).await?;
    
    // 8. Finalize processing
    self.update_document_status(&doc_id, DocumentStatus::Completed, None).await?;
    
    Ok(doc_id)
}
```

### Step 5: Implement Custom Knowledge Graph Insertion

```rust
pub async fn insert_custom_kg(&self, custom_kg: &KnowledgeGraph) -> Result<()> {
    // 1. Store nodes in graph
    let mut graph = self.graph_storage.write().await;
    
    for node in &custom_kg.nodes {
        graph.upsert_node(&node.id, node.attributes.clone()).await?;
    }
    
    // 2. Store edges
    for edge in &custom_kg.edges {
        graph.upsert_edge(&edge.source, &edge.target, edge.data.clone()).await?;
    }
    
    // 3. Store vector representations for nodes
    let mut vectors = Vec::new();
    
    for node in &custom_kg.nodes {
        if let Some(content) = node.attributes.get("content").and_then(|v| v.as_str()) {
            let embedding = self.generate_embedding(content).await?;
            
            let vector = storage::vector::VectorData {
                id: format!("node-{}", node.id),
                vector: embedding,
                metadata: node.attributes.clone(),
            };
            
            vectors.push(vector);
        }
    }
    
    // 4. Store vectors
    if !vectors.is_empty() {
        let mut vector_storage = self.vector_storage.write().await;
        vector_storage.upsert(vectors).await?;
    }
    
    // 5. Store document chunks from KG if provided
    if let Some(chunks) = &custom_kg.chunks {
        let mut chunk_data = HashMap::new();
        let mut chunk_vectors = Vec::new();
        
        for (i, chunk) in chunks.iter().enumerate() {
            if let Some(content) = chunk.attributes.get("content").and_then(|v| v.as_str()) {
                let chunk_id = format!("chunk-kg-{}", i);
                
                // Store chunk metadata
                let mut metadata = chunk.attributes.clone();
                metadata.insert("index".to_string(), serde_json::Value::Number(serde_json::Number::from(i)));
                
                chunk_data.insert(chunk_id.clone(), metadata.clone());
                
                // Generate embedding for the chunk
                let embedding = self.generate_embedding(content).await?;
                
                // Create vector data
                let vector = storage::vector::VectorData {
                    id: chunk_id.clone(),
                    vector: embedding,
                    metadata,
                };
                
                chunk_vectors.push(vector);
            }
        }
        
        // Store chunk data in KV storage
        let mut kv = self.kv_storage.write().await;
        kv.upsert(chunk_data).await?;
        
        // Store vectors in vector storage
        let mut vector_storage = self.vector_storage.write().await;
        vector_storage.upsert(chunk_vectors).await?;
    }
    
    Ok(())
}
```

### Step 6: Helper Methods

```rust
fn generate_document_id(text: &str) -> String {
    // Create a hash of the content
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("doc-{:x}", hasher.finalize())
}

async fn check_document_exists(&self, doc_id: &str) -> Result<bool> {
    let kv = self.kv_storage.read().await;
    let result = kv.get_by_id(doc_id).await?;
    Ok(result.is_some())
}

async fn update_document_status(
    &self, 
    doc_id: &str, 
    status: DocumentStatus,
    error: Option<String>,
) -> Result<()> {
    // Get the status storage
    let mut status_storage = match self.status_manager.as_ref() {
        Some(manager) => manager.clone(),
        None => {
            // Create in-memory status manager if none exists
            Arc::new(RwLock::new(processing::InMemoryStatusManager::new()))
        }
    };
    
    // Update status
    let mut status_manager = status_storage.write().await;
    status_manager.update_status(doc_id, status, error).await?;
    
    Ok(())
}

async fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
    // Use configured embedding provider or create default
    let embedding_provider = match self.embedding_provider.as_ref() {
        Some(provider) => provider.clone(),
        None => {
            // Create default provider
            Arc::new(embeddings::BasicEmbeddingProvider::new())
        }
    };
    
    let embeddings = embedding_provider.generate_embeddings(&[text.to_string()]).await?;
    
    if embeddings.is_empty() {
        return Err(Error::ProcessingError("Failed to generate embedding".to_string()));
    }
    
    Ok(embeddings[0].clone())
}

async fn finalize_document_processing(&self, doc_id: &str) -> Result<()> {
    // Additional finalization steps can be added here
    // For example:
    // - Notify any listeners that processing is complete
    // - Update document stats
    // - Run post-processing hooks
    
    tracing::info!("Document processing finalized for {}", doc_id);
    Ok(())
}
```

## Required Struct Field Additions

```rust
pub struct SuperLightRAG {
    // Existing fields...
    kv_storage: Arc<RwLock<Box<dyn storage::kv::KVStorage>>>,
    vector_storage: Arc<RwLock<Box<dyn storage::vector::VectorStorage>>>,
    graph_storage: Arc<RwLock<Box<dyn storage::graph::GraphStorage>>>,
    config: Arc<types::Config>,
    
    // New fields for document processing
    entity_extractor: Option<Arc<dyn processing::entity::EntityExtractor>>,
    embedding_provider: Option<Arc<dyn embeddings::EmbeddingProvider>>,
    llm_provider: Option<Arc<dyn llm::LLMProvider>>,
    llm_cache: Option<Arc<dyn llm::cache::LLMCache>>,
    status_manager: Option<Arc<RwLock<dyn processing::DocumentStatusManager>>>,
}
```

## Knowledge Graph Data Structures

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraph {
    /// List of nodes in the graph
    pub nodes: Vec<KnowledgeGraphNode>,
    
    /// List of edges connecting the nodes
    pub edges: Vec<KnowledgeGraphEdge>,
    
    /// Optional document chunks associated with the KG
    pub chunks: Option<Vec<KnowledgeGraphNode>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphNode {
    /// Unique identifier for the node
    pub id: String,
    
    /// Node attributes
    pub attributes: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphEdge {
    /// Source node ID
    pub source: String,
    
    /// Target node ID
    pub target: String,
    
    /// Edge data
    pub data: storage::graph::EdgeData,
}
```

## Builder Pattern Methods

To make SuperLightRAG more customizable, we'll add builder pattern methods:

```rust
impl SuperLightRAG {
    // Existing methods...
    
    /// Set a custom entity extractor
    pub fn with_entity_extractor(mut self, extractor: Arc<dyn processing::entity::EntityExtractor>) -> Self {
        self.entity_extractor = Some(extractor);
        self
    }
    
    /// Set a custom embedding provider
    pub fn with_embedding_provider(mut self, provider: Arc<dyn embeddings::EmbeddingProvider>) -> Self {
        self.embedding_provider = Some(provider);
        self
    }
    
    /// Set a custom LLM provider
    pub fn with_llm_provider(mut self, provider: Arc<dyn llm::LLMProvider>) -> Self {
        self.llm_provider = Some(provider);
        self
    }
    
    /// Set a custom LLM cache
    pub fn with_llm_cache(mut self, cache: Arc<dyn llm::cache::LLMCache>) -> Self {
        self.llm_cache = Some(cache);
        self
    }
    
    /// Set a custom document status manager
    pub fn with_status_manager(mut self, manager: Arc<RwLock<dyn processing::DocumentStatusManager>>) -> Self {
        self.status_manager = Some(manager);
        self
    }
}
```

## Testing Strategy

1. **Unit Tests**:
   - Test document ID generation and deduplication
   - Test entity extraction and relationship building
   - Test embedding generation
   - Test storage operations

2. **Integration Tests**:
   - Test end-to-end document insertion
   - Test batch insertion with various document sizes
   - Test custom chunk insertion
   - Test custom KG insertion

3. **Performance Tests**:
   - Measure insertion throughput
   - Benchmark memory usage during insertion
   - Test with large documents and complex knowledge graphs

## Implementation Timeline

1. **Week 1**: Implement basic document insertion and helper methods
2. **Week 2**: Implement entity extraction and graph building
3. **Week 3**: Implement batch insertion and custom chunk insertion
4. **Week 4**: Implement custom KG insertion, testing, and optimization

## Conclusion

This implementation plan provides a detailed roadmap for adding comprehensive document insertion functionality to SuperLightRAG. By following this plan, we'll create a robust document processing system that matches the capabilities of the Python implementation while leveraging Rust's performance and safety features. 