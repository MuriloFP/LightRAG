use std::sync::Arc;
use tokio::sync::RwLock;
use crate::types::llm::{LLMError, QueryMode, QueryParams};
use crate::storage::{
    kv::KVStorage,
    vector::VectorStorage,
    graph::GraphStorage,
};
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

    /// Build context using local text chunks
    async fn build_local_context(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        let vector_store = self.vector_storage.read().await;
        let similar_chunks = vector_store.search(query, params.top_k)
            .await
            .map_err(|e| LLMError::ConfigError(format!("Vector search failed: {}", e)))?;

        let text_store = self.text_chunks.read().await;
        let mut context = String::new();

        for chunk_id in similar_chunks {
            if let Ok(Some(chunk_data)) = text_store.get_by_id(&chunk_id).await {
                if let Some(content) = chunk_data.get("content") {
                    if let Some(content_str) = content.as_str() {
                        context.push_str(content_str);
                        context.push_str("\n\n");
                    }
                }
            }
        }

        Ok(context)
    }

    /// Build context using global knowledge graph
    async fn build_global_context(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        let mut context = String::new();

        // Use high-level keywords for graph traversal
        if let Some(keywords) = &params.high_level_keywords {
            let graph = self.graph_storage.read().await;
            for keyword in keywords {
                let related_nodes = graph.get_related_nodes(keyword)
                    .await
                    .map_err(|e| LLMError::ConfigError(format!("Graph traversal failed: {}", e)))?;

                for node_id in related_nodes {
                    if let Ok(Some(node_data)) = self.text_chunks.read().await.get_by_id(&node_id).await {
                        if let Some(content) = node_data.get("content") {
                            if let Some(content_str) = content.as_str() {
                                context.push_str(content_str);
                                context.push_str("\n\n");
                            }
                        }
                    }
                }
            }
        }

        Ok(context)
    }

    /// Build context using both local and global approaches
    async fn build_hybrid_context(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        let mut context = String::new();

        // Get local context
        let local_context = self.build_local_context(query, params).await?;
        context.push_str(&local_context);

        // Get global context if high-level keywords are available
        if params.high_level_keywords.is_some() {
            let global_context = self.build_global_context(query, params).await?;
            context.push_str(&global_context);
        }

        Ok(context)
    }

    /// Build context using naive text search
    async fn build_naive_context(&self, query: &str, params: &QueryParams) -> Result<String, LLMError> {
        self.build_local_context(query, params).await
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
} 