# SuperLightRAG API Design

## Overview

This document outlines the high-level design for the user-facing API of the SuperLightRAG system. The API is designed to provide a clean, intuitive interface to the underlying functionality while maintaining the performance and cross-platform benefits of Rust.

## Core API Philosophy

The SuperLightRAG API follows these key principles:

1. **Simplicity** - Straightforward methods that are easy to understand and use
2. **Consistency** - Similar patterns across different operations
3. **Composability** - Components that can be used independently or together
4. **Concurrency** - First-class support for asynchronous operations
5. **Cross-platform** - Works on desktop, mobile, and web environments

## Main API Components

### 1. Document Management

```rust
// Document insertion
async fn insert(&self, text: &str) -> Result<String>;
async fn insert_batch(&self, texts: Vec<String>) -> Result<Vec<String>>;
async fn insert_custom_chunks(&self, full_text: &str, chunks: Vec<String>) -> Result<String>;
async fn insert_custom_kg(&self, custom_kg: &KnowledgeGraph) -> Result<()>;

// Document deletion and management
async fn delete_document(&self, doc_id: &str) -> Result<()>;
async fn get_document_status(&self, doc_id: &str) -> Result<DocumentStatus>;
async fn get_all_document_ids(&self) -> Result<Vec<String>>;
```

### 2. Querying and RAG

```rust
// Basic querying
async fn query(&self, query: &str, params: &QueryParams) -> Result<String>;
async fn query_with_keywords(&self, query: &str, params: &QueryParams) -> Result<String>;

// Context retrieval
async fn get_context(&self, query: &str, params: &QueryParams) -> Result<Vec<String>>;
async fn get_context_with_scores(&self, query: &str, params: &QueryParams) -> Result<Vec<(String, f32)>>;

// Streaming support
async fn query_stream(&self, query: &str, params: &QueryParams) -> Result<impl Stream<Item = Result<String>>>;
```

### 3. Knowledge Graph Operations

```rust
// Graph operations
async fn get_entity(&self, entity_id: &str) -> Result<Option<Entity>>;
async fn get_entity_relationships(&self, entity_id: &str) -> Result<Vec<Relationship>>;
async fn search_entities(&self, query: &str, limit: usize) -> Result<Vec<Entity>>;
async fn get_graph_visualization(&self, central_entity: Option<&str>, depth: usize) -> Result<GraphVisualization>;
```

### 4. Utility Operations

```rust
// Configuration and system management
async fn initialize(&self) -> Result<()>;
async fn finalize(&self) -> Result<()>;
async fn clear_storage(&self) -> Result<()>;
async fn get_storage_stats(&self) -> Result<StorageStats>;
async fn update_config(&self, config: Config) -> Result<()>;
```

## Integration Points

### LLM Integration

SuperLightRAG supports multiple LLM providers through a standardized interface:

```rust
// LLM configuration
fn with_llm_provider(&mut self, provider: Box<dyn LLMProvider>) -> &mut Self;
fn with_embedding_provider(&mut self, provider: Box<dyn EmbeddingProvider>) -> &mut Self;

// Direct LLM access
async fn generate_text(&self, prompt: &str, params: &LLMParams) -> Result<String>;
async fn generate_embeddings(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
```

### Storage Extension

SuperLightRAG allows custom storage implementations:

```rust
fn with_kv_storage(&mut self, storage: Box<dyn KVStorage>) -> &mut Self;
fn with_vector_storage(&mut self, storage: Box<dyn VectorStorage>) -> &mut Self;
fn with_graph_storage(&mut self, storage: Box<dyn GraphStorage>) -> &mut Self;
```

## Usage Examples

### Basic Usage

```rust
use super_lightrag::{SuperLightRAG, QueryParams, QueryMode};

#[tokio::main]
async fn main() -> Result<()> {
    // Create a new instance with default configuration
    let rag = SuperLightRAG::new().await?;
    
    // Initialize (creates necessary directories and storage)
    rag.initialize().await?;
    
    // Insert documents
    let doc_id = rag.insert("This is a sample document about AI technology.").await?;
    println!("Inserted document with ID: {}", doc_id);
    
    // Query the system
    let response = rag.query(
        "What is this document about?", 
        &QueryParams {
            mode: QueryMode::Hybrid,
            top_k: 3,
            ..Default::default()
        }
    ).await?;
    
    println!("Response: {}", response);
    
    // Clean up
    rag.finalize().await?;
    
    Ok(())
}
```

### Advanced Usage

```rust
use super_lightrag::{
    SuperLightRAG, Config, QueryParams, QueryMode, 
    LLMParams, KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge
};

#[tokio::main]
async fn main() -> Result<()> {
    // Create with custom configuration
    let mut config = Config::default();
    config.working_dir = PathBuf::from("./data");
    config.max_memory = 1024 * 1024 * 1024; // 1GB
    
    let rag = SuperLightRAG::with_config(config).await?;
    rag.initialize().await?;
    
    // Insert a custom knowledge graph
    let custom_kg = KnowledgeGraph {
        nodes: vec![
            KnowledgeGraphNode {
                id: "company_a".to_string(),
                attributes: {
                    let mut m = HashMap::new();
                    m.insert("name".to_string(), json!("Company A"));
                    m.insert("type".to_string(), json!("Organization"));
                    m.insert("description".to_string(), json!("A technology company"));
                    m
                },
            },
            KnowledgeGraphNode {
                id: "product_x".to_string(),
                attributes: {
                    let mut m = HashMap::new();
                    m.insert("name".to_string(), json!("Product X"));
                    m.insert("type".to_string(), json!("Product"));
                    m.insert("description".to_string(), json!("A product made by Company A"));
                    m
                },
            },
        ],
        edges: vec![
            KnowledgeGraphEdge {
                source: "company_a".to_string(),
                target: "product_x".to_string(),
                data: EdgeData {
                    weight: 1.0,
                    label: "MANUFACTURES".to_string(),
                    attributes: {
                        let mut m = HashMap::new();
                        m.insert("since".to_string(), json!(2020));
                        m
                    },
                },
            },
        ],
    };
    
    rag.insert_custom_kg(&custom_kg).await?;
    
    // Query with streaming response
    let mut stream = rag.query_stream(
        "What products does Company A make?",
        &QueryParams {
            mode: QueryMode::Global,
            ..Default::default()
        }
    ).await?;
    
    while let Some(chunk) = stream.next().await {
        print!("{}", chunk?);
    }
    
    // Get entity relationships
    let relationships = rag.get_entity_relationships("company_a").await?;
    for rel in relationships {
        println!("{} --[{}]--> {}", rel.source, rel.relation_type, rel.target);
    }
    
    rag.finalize().await?;
    
    Ok(())
}
```

## Web and Mobile Integration

SuperLightRAG is designed to work well in web and mobile environments:

### Web (WebAssembly)

```rust
#[wasm_bindgen]
pub struct WebLightRAG {
    rag: SuperLightRAG,
}

#[wasm_bindgen]
impl WebLightRAG {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Result<WebLightRAG, JsValue> {
        // Initialize with IndexedDB storage for web
        let config = Config::default();
        let rag = SuperLightRAG::with_config(config)
            .with_kv_storage(Box::new(IndexedDBStorage::new("docs")))
            .with_vector_storage(Box::new(WebVectorStorage::new()))
            .with_graph_storage(Box::new(InMemoryGraphStorage::new()));
            
        Ok(WebLightRAG { rag })
    }
    
    #[wasm_bindgen]
    pub async fn insert(&self, text: &str) -> Result<String, JsValue> {
        self.rag.insert(text).await.map_err(|e| JsValue::from_str(&e.to_string()))
    }
    
    #[wasm_bindgen]
    pub async fn query(&self, query: &str) -> Result<String, JsValue> {
        let params = QueryParams::default();
        self.rag.query(query, &params).await.map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
```

### Mobile (React Native / Flutter)

```rust
// For React Native or Flutter, we provide FFI bindings
#[no_mangle]
pub extern "C" fn lightrag_create() -> *mut SuperLightRAG {
    let rag = SuperLightRAG::new()
        .with_config(MobileConfig::default())
        .with_kv_storage(Box::new(SqliteStorage::new("docs.db")))
        .with_vector_storage(Box::new(MobileVectorStorage::new()));
        
    Box::into_raw(Box::new(rag))
}

#[no_mangle]
pub extern "C" fn lightrag_insert(rag_ptr: *mut SuperLightRAG, text: *const c_char) -> *const c_char {
    let rag = unsafe { &*rag_ptr };
    let text_str = unsafe { CStr::from_ptr(text).to_str().unwrap() };
    
    // Use a runtime to execute async function
    let runtime = Runtime::new().unwrap();
    let doc_id = runtime.block_on(async {
        rag.insert(text_str).await.unwrap_or_else(|_| "error".to_string())
    });
    
    CString::new(doc_id).unwrap().into_raw()
}
```

## API Evolution and Future Directions

The API is designed to evolve in the following directions:

1. **Visualization Tools** - Adding built-in visualization for knowledge graphs and query processes
2. **Advanced Chunking** - More sophisticated document chunking strategies
3. **Multi-modal Support** - Extending beyond text to handle images and other data types
4. **Cross-document Analysis** - Tools for connecting information across multiple documents
5. **User Feedback Integration** - Learning from user interactions to improve results
6. **Offline Support** - Robust operation without constant network connectivity

## Implementation Strategy

The implementation will be phased:

1. **Phase 1**: Core document operations and basic querying
2. **Phase 2**: Knowledge graph operations and advanced querying
3. **Phase 3**: Streaming, visualization, and platform-specific optimizations
4. **Phase 4**: Multi-modal support and extended capabilities

Each phase will include:
- API implementation
- Comprehensive tests
- Documentation
- Example applications

## Conclusion

The SuperLightRAG API is designed to provide a powerful yet simple interface to advanced RAG capabilities, while leveraging Rust's performance and safety features. The design prioritizes both usability and extensibility, allowing for a wide range of applications across desktop, web, and mobile platforms. 