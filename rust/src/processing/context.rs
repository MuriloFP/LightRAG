use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde_json::Value;
use crate::types::llm::{LLMError, QueryMode, QueryParams};
use crate::storage::{KVStorage, VectorStorage, GraphStorage};
use crate::processing::keywords::ConversationTurn;

const CONTEXT_PROMPT_TEMPLATE: &str = r#"
Use the following context to answer the question:

{context}

Previous conversation:
{history}

Question: {query}

Please provide a {response_type} response based on the context above.
"#;

/// Context builder for LLM queries
pub struct ContextBuilder {
    /// KV storage for text chunks
    text_chunks: Arc<RwLock<dyn KVStorage>>,
    /// Vector storage for embeddings
    vector_storage: Arc<RwLock<dyn VectorStorage>>,
    /// Graph storage for relationships
    graph_storage: Arc<RwLock<dyn GraphStorage>>,
}

impl ContextBuilder {
    /// Create a new context builder
    pub fn new(
        text_chunks: Arc<RwLock<dyn KVStorage>>,
        vector_storage: Arc<RwLock<dyn VectorStorage>>,
        graph_storage: Arc<RwLock<dyn GraphStorage>>,
    ) -> Self {
        Self {
            text_chunks,
            vector_storage,
            graph_storage,
        }
    }

    /// Build context for a query based on query mode
    pub async fn build_context(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        let context = match params.mode {
            QueryMode::Local => self.build_local_context(query, params).await?,
            QueryMode::Global => self.build_global_context(query, params).await?,
            QueryMode::Hybrid => self.build_hybrid_context(query, params).await?,
            QueryMode::Naive => self.build_naive_context(query, params).await?,
            QueryMode::Mix => self.build_mix_context(query, params).await?,
        };

        if params.only_need_context {
            return Ok(context);
        }

        let history = self.format_conversation_history(params.conversation_history.as_deref(), params.history_turns);
        
        Ok(CONTEXT_PROMPT_TEMPLATE
            .replace("{context}", &context)
            .replace("{history}", &history)
            .replace("{query}", query)
            .replace("{response_type}", &params.response_type))
    }

    /// Extract content from chunk data
    fn extract_content(chunk_data: &HashMap<String, Value>) -> Option<String> {
        chunk_data.get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Build context using local text chunks
    async fn build_local_context(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        let vector_store = self.vector_storage.read().await;
        // Convert query to vector using embeddings
        let query_vector = self.get_query_vector(query).await?;
        let similar_chunks = (*vector_store).query(query_vector, params.top_k)
            .await
            .map_err(|e| LLMError::ConfigError(format!("Vector search failed: {}", e)))?;

        let text_store = self.text_chunks.read().await;
        let mut context = String::new();

        for result in similar_chunks {
            if let Ok(Some(chunk_data)) = (*text_store).get_by_id(&result.id).await {
                if let Some(content) = Self::extract_content(&chunk_data) {
                    context.push_str(&content);
                    context.push_str("\n\n");
                }
            }
        }

        Ok(context)
    }

    /// Get vector representation of a query
    async fn get_query_vector(&self, query: &str) -> Result<Vec<f32>, LLMError> {
        // TODO: Implement query vectorization using embeddings
        // For now, return a dummy vector
        Ok(vec![0.0; 384]) // Using 384 dimensions as an example
    }

    /// Build context using global knowledge graph
    async fn build_global_context(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        let mut context = String::new();

        // Use high-level keywords for graph traversal
        if let Some(keywords) = &params.hl_keywords {
            let graph = self.graph_storage.read().await;
            for keyword in keywords {
                if let Some(edges) = (*graph).get_node_edges(keyword).await {
                    for (_, target_id, _) in edges {
                        if let Ok(Some(chunk_data)) = (*self.text_chunks.read().await).get_by_id(&target_id).await {
                            if let Some(content) = Self::extract_content(&chunk_data) {
                                context.push_str(&content);
                                context.push_str("\n\n");
                            }
                        }
                    }
                }
            }
        }

        Ok(context)
    }

    /// Build context using high-level keywords
    async fn build_context_with_keywords(&self, params: &QueryParams) -> Result<String, LLMError> {
        // Use high-level keywords if available
        if let Some(keywords) = &params.hl_keywords {
            let mut context = String::new();
            
            // Get graph context for each keyword
            let graph = self.graph_storage.read().await;
            for keyword in keywords {
                let graph_context = (*graph).query_with_keywords(&[keyword.clone()]).await
                    .map_err(|e| LLMError::RequestFailed(format!("Graph query failed: {}", e)))?;
                
                context.push_str(&graph_context);
                context.push('\n');
            }
            
            Ok(context)
        } else {
            Ok(String::new())
        }
    }

    /// Build context using vector similarity
    async fn build_context_with_vectors(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        // Get query vector
        let query_vector = self.get_query_vector(query).await?;
        
        // Query vector store
        let vector_store = self.vector_storage.read().await;
        let results = (*vector_store).query(query_vector, params.top_k).await
            .map_err(|e| LLMError::RequestFailed(format!("Vector query failed: {}", e)))?;
        
        // Build context from results
        let mut context = String::new();
        let mut total_tokens = 0;
        
        for result in results {
            if result.distance >= params.similarity_threshold {
                if let Some(content) = result.metadata.get("content").and_then(|v| v.as_str()) {
                    let tokens = content.split_whitespace().count();
                    if total_tokens + tokens <= params.max_context_tokens {
                        context.push_str(content);
                        context.push('\n');
                        total_tokens += tokens;
                    }
                }
            }
        }
        
        Ok(context)
    }

    /// Build context using both keywords and vectors
    async fn build_hybrid_context(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        let mut context = String::new();
        
        // Get keyword context
        if params.hl_keywords.is_some() {
            let keyword_context = self.build_context_with_keywords(params).await?;
            context.push_str(&keyword_context);
        }
        
        // Get vector context
        let vector_context = self.build_context_with_vectors(query, params).await?;
        context.push_str(&vector_context);
        
        Ok(context)
    }

    /// Build context using naive text search
    async fn build_naive_context(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        self.build_local_context(query, params).await
    }

    /// Build context using mixed vector and graph context
    async fn build_mix_context(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        // Combine vector and graph context with weighted scoring
        let mut context = String::new();
        
        // Get vector context (60% weight)
        let vector_results = self.build_local_context(query, params).await?;
        context.push_str(&vector_results);
        
        // Get graph context (40% weight)
        let graph_results = self.build_global_context(query, params).await?;
        context.push_str("\n\nGraph Context:\n");
        context.push_str(&graph_results);
        
        Ok(context)
    }

    /// Format conversation history
    fn format_conversation_history(&self, history: Option<&[ConversationTurn]>, max_turns: Option<usize>) -> String {
        let mut formatted = String::new();

        if let Some(history) = history {
            let turns = max_turns.unwrap_or(history.len());
            for turn in history.iter().rev().take(turns) {
                formatted.push_str(&format!("{}: {}\n", turn.role, turn.content));
            }
        }

        formatted
    }

    async fn get_context(&self, query: &str, params: &QueryParams) -> Result<(Vec<String>, Vec<f32>), LLMError> {
        let vector = self.get_query_vector(query).await?;
        let vector_store = self.vector_storage.read().await;
        let results = (*vector_store).query(vector, params.top_k).await
            .map_err(|e| LLMError::RequestFailed(format!("Vector query failed: {}", e)))?;
        
        let mut filtered_chunks = Vec::new();
        let mut filtered_scores = Vec::new();
        
        for result in results {
            if result.distance >= params.similarity_threshold {
                if let Some(content) = result.metadata.get("content").and_then(|v| v.as_str()) {
                    filtered_chunks.push(content.to_string());
                    filtered_scores.push(result.distance);
                }
            }
        }
        
        // Add graph context if available
        if let Some(keywords) = &params.hl_keywords {
            let graph = self.graph_storage.read().await;
            if let Ok(graph_context) = (*graph).query_with_keywords(keywords).await {
                filtered_chunks.push(graph_context);
                filtered_scores.push(1.0); // Graph context gets max relevance score
            }
        }
        
        Ok((filtered_chunks, filtered_scores))
    }
} 