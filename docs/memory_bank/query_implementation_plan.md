# SuperLightRAG Query Implementation Plan

## Overview

This document outlines the implementation plan for adding querying capabilities to SuperLightRAG. Query functionality is a critical component that allows users to retrieve relevant information from their documents based on semantic similarity, keyword matching, and knowledge graph traversal.

## Implementation Goals

1. Implement various query modes (Naive, Local, Global, Hybrid, Mix) matching the Python implementation
2. Ensure efficient retrieval with appropriate scoring mechanisms
3. Support streaming responses for large result sets
4. Allow for flexible query parameters to customize retrieval
5. Integrate with existing document processors and storage components

## Core Query Method Signatures

```rust
impl SuperLightRAG {
    /// Query the system using the specified parameters
    pub async fn query(&self, query_text: &str, params: &QueryParams) -> Result<String> {
        // Implementation details below
    }
    
    /// Query with streaming response
    pub async fn query_stream(&self, query_text: &str, params: &QueryParams) 
        -> Result<impl Stream<Item = Result<String>>> {
        // Implementation details below
    }
    
    /// Get raw context without LLM processing
    pub async fn get_context(&self, query_text: &str, params: &QueryParams) 
        -> Result<Vec<String>> {
        // Implementation details below
    }
    
    /// Get context with relevance scores
    pub async fn get_context_with_scores(&self, query_text: &str, params: &QueryParams) 
        -> Result<Vec<(String, f32)>> {
        // Implementation details below
    }
    
    /// Query with separate keyword extraction step
    pub async fn query_with_keywords(&self, query_text: &str, params: &QueryParams) 
        -> Result<String> {
        // Implementation details below
    }
}
```

## Implementation Steps

### Step 1: Implement Basic Query Method

```rust
pub async fn query(&self, query_text: &str, params: &QueryParams) -> Result<String> {
    // 1. Create the appropriate query processor based on the mode
    let factory = processing::query::QueryProcessorFactory::new(
        self.vector_storage.clone(),
        self.graph_storage.clone(),
        Arc::new(processing::keywords::BasicKeywordExtractor::new(
            processing::keywords::KeywordConfig::default()
        )),
    );
    
    let processor = factory.create_processor(params.mode.clone(), params.clone());
    
    // 2. Process the query to get context
    let query_result = processor.process_query(
        query_text, 
        params,
        params.conversation_history.as_deref(),
    ).await?;
    
    // 3. If only context is needed, return early
    if params.only_need_context {
        return Ok(query_result.context_chunks.join("\n\n"));
    }
    
    // 4. Build prompt with context
    let context = query_result.context_chunks.join("\n\n");
    let llm_prompt = self.build_llm_prompt(query_text, &context, params).await?;
    
    // 5. If only prompt is needed, return early
    if params.only_need_prompt {
        return Ok(llm_prompt);
    }
    
    // 6. Get LLM provider
    let llm_provider = self.get_llm_provider().await?;
    
    // 7. Generate response from LLM
    let llm_params = llm::LLMParams {
        max_tokens: params.max_tokens.unwrap_or(1000),
        temperature: params.temperature.unwrap_or(0.7),
        stream: false,
        system_prompt: Some(llm_prompt),
        ..Default::default()
    };
    
    let response = llm_provider.generate(query_text, &llm_params).await?;
    
    // 8. Cache response if needed
    if let Some(cache) = &self.llm_cache {
        let cache_key = format!("query:{}:{}", query_text, params.mode);
        cache.set(&cache_key, &response).await?;
    }
    
    Ok(response)
}
```

### Step 2: Implement Query Processor Factory and Processors

The query processor factory is already implemented and tested in the codebase, but we need to integrate it with the SuperLightRAG struct.

### Step 3: Implement Streaming Query Support

```rust
pub async fn query_stream(&self, query_text: &str, params: &mut QueryParams) 
    -> Result<impl Stream<Item = Result<String>>> {
    // Set streaming to true
    params.stream = true;
    
    // Get context same as in regular query
    let factory = processing::query::QueryProcessorFactory::new(
        self.vector_storage.clone(),
        self.graph_storage.clone(),
        Arc::new(processing::keywords::BasicKeywordExtractor::new(
            processing::keywords::KeywordConfig::default()
        )),
    );
    
    let processor = factory.create_processor(params.mode.clone(), params.clone());
    let query_result = processor.process_query(
        query_text, 
        params,
        params.conversation_history.as_deref(),
    ).await?;
    
    // Build prompt with context
    let context = query_result.context_chunks.join("\n\n");
    let llm_prompt = self.build_llm_prompt(query_text, &context, params).await?;
    
    // Get LLM provider
    let llm_provider = self.get_llm_provider().await?;
    
    // Generate streaming response
    let llm_params = llm::LLMParams {
        max_tokens: params.max_tokens.unwrap_or(1000),
        temperature: params.temperature.unwrap_or(0.7),
        stream: true,
        system_prompt: Some(llm_prompt),
        ..Default::default()
    };
    
    let stream = llm_provider.generate_stream(query_text, &llm_params).await?;
    
    Ok(stream)
}
```

### Step 4: Implement Context Retrieval Methods

```rust
pub async fn get_context(&self, query_text: &str, params: &mut QueryParams) 
    -> Result<Vec<String>> {
    // Force only_need_context to true
    let mut modified_params = params.clone();
    modified_params.only_need_context = true;
    
    // Create processor and process query
    let factory = processing::query::QueryProcessorFactory::new(
        self.vector_storage.clone(),
        self.graph_storage.clone(),
        Arc::new(processing::keywords::BasicKeywordExtractor::new(
            processing::keywords::KeywordConfig::default()
        )),
    );
    
    let processor = factory.create_processor(
        modified_params.mode.clone(), 
        modified_params.clone()
    );
    
    let query_result = processor.process_query(
        query_text, 
        &modified_params,
        modified_params.conversation_history.as_deref(),
    ).await?;
    
    Ok(query_result.context_chunks)
}

pub async fn get_context_with_scores(&self, query_text: &str, params: &mut QueryParams) 
    -> Result<Vec<(String, f32)>> {
    // Similar to get_context but return with scores
    let mut modified_params = params.clone();
    modified_params.only_need_context = true;
    
    let factory = processing::query::QueryProcessorFactory::new(
        self.vector_storage.clone(),
        self.graph_storage.clone(),
        Arc::new(processing::keywords::BasicKeywordExtractor::new(
            processing::keywords::KeywordConfig::default()
        )),
    );
    
    let processor = factory.create_processor(
        modified_params.mode.clone(), 
        modified_params.clone()
    );
    
    let query_result = processor.process_query(
        query_text, 
        &modified_params,
        modified_params.conversation_history.as_deref(),
    ).await?;
    
    let result = query_result.context_chunks.into_iter()
        .zip(query_result.relevance_scores)
        .collect();
        
    Ok(result)
}
```

### Step 5: Implement Keyword-Based Query

```rust
pub async fn query_with_keywords(&self, query_text: &str, params: &QueryParams) 
    -> Result<String> {
    // Extract keywords first
    let keyword_extractor = processing::keywords::LLMKeywordExtractor::new(
        Arc::new(llm::provider::OpenAIProvider::new("gpt-4".to_string())),
        processing::keywords::KeywordConfig::default(),
    );
    
    let extracted_keywords = keyword_extractor.extract_keywords(query_text).await?;
    
    // Create a new params with the extracted keywords
    let mut new_params = params.clone();
    new_params.hl_keywords = Some(extracted_keywords.high_level);
    new_params.ll_keywords = Some(extracted_keywords.low_level);
    
    // Call the regular query method with updated params
    self.query(query_text, &new_params).await
}
```

### Step 6: Helper Methods

```rust
async fn build_llm_prompt(&self, query: &str, context: &str, params: &QueryParams) -> Result<String> {
    // Build prompt based on query mode and context
    let template = match params.mode {
        QueryMode::Naive => templates::NAIVE_PROMPT,
        QueryMode::Local => templates::LOCAL_PROMPT,
        QueryMode::Global => templates::GLOBAL_PROMPT,
        QueryMode::Hybrid | QueryMode::Mix => templates::HYBRID_PROMPT,
    };
    
    // Format the template with the context and query
    let mut prompt = template.replace("{{CONTEXT}}", context);
    prompt = prompt.replace("{{QUERY}}", query);
    prompt = prompt.replace("{{RESPONSE_TYPE}}", &params.response_type);
    
    // Add conversation history if available
    if let Some(history) = &params.conversation_history {
        let history_str = history.iter()
            .map(|turn| format!("{}: {}", turn.role, turn.content))
            .collect::<Vec<_>>()
            .join("\n");
            
        prompt = prompt.replace("{{HISTORY}}", &history_str);
    } else {
        prompt = prompt.replace("{{HISTORY}}", "");
    }
    
    Ok(prompt)
}

async fn get_llm_provider(&self) -> Result<Arc<dyn llm::LLMProvider>> {
    // Return configured LLM provider or create a default one
    if let Some(provider) = &self.llm_provider {
        Ok(provider.clone())
    } else {
        // Create a default provider, preferring local if available
        #[cfg(feature = "local-llm")]
        {
            Ok(Arc::new(llm::provider::LocalProvider::new(
                "models/mixtral-8x7b-instruct.gguf",
                llm::provider::LocalProviderConfig::default(),
            )?))
        }
        
        #[cfg(not(feature = "local-llm"))]
        {
            // Fallback to OpenAI if no local LLM
            Ok(Arc::new(llm::provider::OpenAIProvider::new(
                "gpt-3.5-turbo",
            )))
        }
    }
}
```

## Prompt Templates

We'll need to implement various prompt templates for different query modes:

```rust
pub mod templates {
    pub const NAIVE_PROMPT: &str = r#"
You are an AI assistant tasked with providing accurate and helpful information.
Based on the following context, answer the user's query.

CONTEXT:
{{CONTEXT}}

CONVERSATION HISTORY:
{{HISTORY}}

Please provide a response in the format: {{RESPONSE_TYPE}}
"#;

    pub const LOCAL_PROMPT: &str = r#"
You are an AI assistant tasked with providing accurate and helpful information.
Based on the following local context from relevant text chunks, answer the user's query.

LOCAL CONTEXT:
{{CONTEXT}}

CONVERSATION HISTORY:
{{HISTORY}}

Please provide a response in the format: {{RESPONSE_TYPE}}
"#;

    pub const GLOBAL_PROMPT: &str = r#"
You are an AI assistant tasked with providing accurate and helpful information.
Based on the following global context from a knowledge graph, answer the user's query.

KNOWLEDGE GRAPH CONTEXT:
{{CONTEXT}}

CONVERSATION HISTORY:
{{HISTORY}}

Please provide a response in the format: {{RESPONSE_TYPE}}
"#;

    pub const HYBRID_PROMPT: &str = r#"
You are an AI assistant tasked with providing accurate and helpful information.
Based on the following context, which combines relevant text chunks and knowledge graph information, answer the user's query.

COMBINED CONTEXT:
{{CONTEXT}}

CONVERSATION HISTORY:
{{HISTORY}}

Please provide a response in the format: {{RESPONSE_TYPE}}
"#;
}
```

## Testing Strategy

1. **Unit Tests**:
   - Test each query processor implementation
   - Test prompt building and formatting
   - Test parameter handling and validation

2. **Integration Tests**:
   - Test end-to-end query functionality with mock data
   - Test different query modes with realistic documents
   - Test streaming functionality

3. **Performance Tests**:
   - Measure query latency under different loads
   - Benchmark memory usage during query processing
   - Test with large documents and complex knowledge graphs

## Query Parameter Definition

```rust
/// Parameters for RAG queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryParams {
    /// Query mode (Naive, Local, Global, Hybrid, Mix)
    pub mode: QueryMode,
    
    /// Whether to stream the response
    pub stream: bool,
    
    /// Only return context without LLM processing
    pub only_need_context: bool,
    
    /// Only return the prompt without LLM processing
    pub only_need_prompt: bool,
    
    /// Response format specification (e.g., "Multiple Paragraphs", "Bullet Points")
    pub response_type: String,
    
    /// Number of top results to retrieve
    pub top_k: usize,
    
    /// Minimum similarity threshold (0.0-1.0)
    pub similarity_threshold: f32,
    
    /// Maximum tokens for context
    pub max_context_tokens: usize,
    
    /// Maximum tokens per text unit
    pub max_token_for_text_unit: usize,
    
    /// Maximum tokens for global context
    pub max_token_for_global_context: usize,
    
    /// Maximum tokens for local context
    pub max_token_for_local_context: usize,
    
    /// High-level keywords (optional)
    pub hl_keywords: Option<Vec<String>>,
    
    /// Low-level keywords (optional)
    pub ll_keywords: Option<Vec<String>>,
    
    /// Conversation history
    pub conversation_history: Option<Vec<ConversationTurn>>,
    
    /// Number of history turns to include
    pub history_turns: Option<usize>,
    
    /// Number of documents to retrieve
    pub num_docs: usize,
    
    /// Whether to include metadata in results
    pub include_metadata: bool,
    
    /// Maximum tokens in response
    pub max_tokens: Option<usize>,
    
    /// Temperature for LLM generation
    pub temperature: Option<f32>,
    
    /// Additional parameters
    pub extra_params: HashMap<String, String>,
}
```

## Implementation Timeline

1. **Week 1**: Implement basic query method and integration with processors
2. **Week 2**: Implement context retrieval and helper methods
3. **Week 3**: Implement streaming support and keyword-based querying
4. **Week 4**: Testing, optimization, and documentation

## Conclusion

This implementation plan provides a detailed roadmap for adding comprehensive query functionality to SuperLightRAG. By following this plan, we'll create a robust querying system that matches the capabilities of the Python implementation while leveraging Rust's performance and safety features. 