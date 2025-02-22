use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use async_trait::async_trait;
use tokio::sync::RwLock;
use std::sync::Arc;

use crate::{
    types::{
        Error,
        llm::{QueryMode, QueryParams},
    },
    storage::{
        GraphStorage,
        VectorStorage,
    },
    processing::keywords::{KeywordExtractor, ConversationTurn},
};

/// Result of query processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    /// Retrieved context chunks
    pub context_chunks: Vec<String>,
    
    /// Relevance scores for chunks
    pub relevance_scores: Vec<f32>,
    
    /// Extracted keywords
    pub keywords: Vec<String>,
    
    /// Graph context (if applicable)
    pub graph_context: Option<String>,
    
    /// Total tokens in context
    pub total_tokens: usize,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Trait for query processors
#[async_trait]
pub trait QueryProcessor: Send + Sync {
    /// Process a query and build context
    async fn process_query(
        &self,
        query: &str,
        params: &QueryParams,
        history: Option<&[ConversationTurn]>,
    ) -> Result<QueryResult, Error>;
    
    /// Get the current query mode
    fn get_mode(&self) -> QueryMode;
    
    /// Update query parameters
    fn update_params(&mut self, params: QueryParams);
}

/// Local context query processor
pub struct LocalQueryProcessor {
    vector_store: Arc<RwLock<dyn VectorStorage>>,
    keyword_extractor: Arc<dyn KeywordExtractor>,
    params: QueryParams,
}

/// Global context query processor
pub struct GlobalQueryProcessor {
    graph_store: Arc<RwLock<dyn GraphStorage>>,
    keyword_extractor: Arc<dyn KeywordExtractor>,
    params: QueryParams,
}

/// Hybrid context query processor
pub struct HybridQueryProcessor {
    vector_store: Arc<RwLock<dyn VectorStorage>>,
    graph_store: Arc<RwLock<dyn GraphStorage>>,
    keyword_extractor: Arc<dyn KeywordExtractor>,
    params: QueryParams,
}

/// Naive query processor
pub struct NaiveQueryProcessor {
    vector_store: Arc<RwLock<dyn VectorStorage>>,
    params: QueryParams,
}

/// Mix mode query processor that combines graph and vector retrieval with weighted scoring
pub struct MixQueryProcessor {
    vector_store: Arc<RwLock<dyn VectorStorage>>,
    graph_store: Arc<RwLock<dyn GraphStorage>>,
    keyword_extractor: Arc<dyn KeywordExtractor>,
    params: QueryParams,
}

impl LocalQueryProcessor {
    pub fn new(
        vector_store: Arc<RwLock<dyn VectorStorage>>,
        keyword_extractor: Arc<dyn KeywordExtractor>,
        params: QueryParams,
    ) -> Self {
        Self {
            vector_store,
            keyword_extractor,
            params,
        }
    }
}

impl GlobalQueryProcessor {
    pub fn new(
        graph_store: Arc<RwLock<dyn GraphStorage>>,
        keyword_extractor: Arc<dyn KeywordExtractor>,
        params: QueryParams,
    ) -> Self {
        Self {
            graph_store,
            keyword_extractor,
            params,
        }
    }
}

impl HybridQueryProcessor {
    pub fn new(
        vector_store: Arc<RwLock<dyn VectorStorage>>,
        graph_store: Arc<RwLock<dyn GraphStorage>>,
        keyword_extractor: Arc<dyn KeywordExtractor>,
        params: QueryParams,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            keyword_extractor,
            params,
        }
    }
}

impl NaiveQueryProcessor {
    pub fn new(
        vector_store: Arc<RwLock<dyn VectorStorage>>,
        params: QueryParams,
    ) -> Self {
        Self {
            vector_store,
            params,
        }
    }
}

impl MixQueryProcessor {
    pub fn new(
        vector_store: Arc<RwLock<dyn VectorStorage>>,
        graph_store: Arc<RwLock<dyn GraphStorage>>,
        keyword_extractor: Arc<dyn KeywordExtractor>,
        params: QueryParams,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            keyword_extractor,
            params,
        }
    }
}

#[async_trait]
impl QueryProcessor for LocalQueryProcessor {
    async fn process_query(
        &self,
        query: &str,
        params: &QueryParams,
        history: Option<&[ConversationTurn]>,
    ) -> Result<QueryResult, Error> {
        // Extract keywords from query and history
        let keywords = if let Some(history) = history {
            self.keyword_extractor.extract_keywords_with_history(query, history).await?
        } else {
            self.keyword_extractor.extract_keywords(query).await?
        };

        // Get vector store results
        let vector_store = self.vector_store.read().await;
        let results = vector_store.query(vec![], params.top_k).await?;

        // Build context from results
        let mut context_chunks = Vec::new();
        let mut relevance_scores = Vec::new();
        let mut total_tokens = 0;

        for result in results {
            if result.distance >= params.similarity_threshold {
                if let Some(content) = result.metadata.get("content").and_then(|v| v.as_str()) {
                    context_chunks.push(content.to_string());
                    relevance_scores.push(result.distance);
                    total_tokens += content.split_whitespace().count();
                }
            }
        }

        Ok(QueryResult {
            context_chunks,
            relevance_scores,
            keywords: keywords.high_level,
            graph_context: None,
            total_tokens,
            metadata: HashMap::new(),
        })
    }

    fn get_mode(&self) -> QueryMode {
        QueryMode::Local
    }

    fn update_params(&mut self, params: QueryParams) {
        self.params = params;
    }
}

#[async_trait]
impl QueryProcessor for GlobalQueryProcessor {
    async fn process_query(
        &self,
        query: &str,
        params: &QueryParams,
        history: Option<&[ConversationTurn]>,
    ) -> Result<QueryResult, Error> {
        // Extract keywords from query and history
        let keywords = if let Some(history) = history {
            self.keyword_extractor.extract_keywords_with_history(query, history).await?
        } else {
            self.keyword_extractor.extract_keywords(query).await?
        };

        // Get graph store results using keywords
        let graph_store = self.graph_store.read().await;
        let graph_context = (*graph_store).query_with_keywords(&keywords.high_level).await?;

        // Calculate total tokens
        let total_tokens = graph_context.split_whitespace().count();

        Ok(QueryResult {
            context_chunks: vec![],
            relevance_scores: vec![],
            keywords: keywords.high_level,
            graph_context: Some(graph_context),
            total_tokens,
            metadata: HashMap::new(),
        })
    }

    fn get_mode(&self) -> QueryMode {
        QueryMode::Global
    }

    fn update_params(&mut self, params: QueryParams) {
        self.params = params;
    }
}

#[async_trait]
impl QueryProcessor for HybridQueryProcessor {
    async fn process_query(
        &self,
        query: &str,
        params: &QueryParams,
        history: Option<&[ConversationTurn]>,
    ) -> Result<QueryResult, Error> {
        // Extract keywords from query and history
        let keywords = if let Some(history) = history {
            self.keyword_extractor.extract_keywords_with_history(query, history).await?
        } else {
            self.keyword_extractor.extract_keywords(query).await?
        };

        // Get vector store results
        let vector_store = self.vector_store.read().await;
        let results = (*vector_store).query(vec![], params.top_k).await?;

        // Build context from vector results
        let mut context_chunks = Vec::new();
        let mut relevance_scores = Vec::new();
        let mut total_tokens = 0;

        for result in results {
            if result.distance >= params.similarity_threshold {
                if let Some(content) = result.metadata.get("content").and_then(|v| v.as_str()) {
                    context_chunks.push(content.to_string());
                    relevance_scores.push(result.distance);
                    total_tokens += content.split_whitespace().count();
                }
            }
        }

        // Get graph store results using keywords
        let graph_store = self.graph_store.read().await;
        let graph_context = (*graph_store).query_with_keywords(&keywords.high_level).await?;
        total_tokens += graph_context.split_whitespace().count();

        Ok(QueryResult {
            context_chunks,
            relevance_scores,
            keywords: keywords.high_level,
            graph_context: Some(graph_context),
            total_tokens,
            metadata: HashMap::new(),
        })
    }

    fn get_mode(&self) -> QueryMode {
        QueryMode::Hybrid
    }

    fn update_params(&mut self, params: QueryParams) {
        self.params = params;
    }
}

#[async_trait]
impl QueryProcessor for NaiveQueryProcessor {
    async fn process_query(
        &self,
        _query: &str,
        params: &QueryParams,
        _history: Option<&[ConversationTurn]>,
    ) -> Result<QueryResult, Error> {
        // Simple vector similarity search without keyword extraction
        let vector_store = self.vector_store.read().await;
        let results = vector_store.query(vec![], params.top_k).await?;

        let mut context_chunks = Vec::new();
        let mut relevance_scores = Vec::new();
        let mut total_tokens = 0;

        for result in results {
            if result.distance >= params.similarity_threshold {
                if let Some(content) = result.metadata.get("content").and_then(|v| v.as_str()) {
                    context_chunks.push(content.to_string());
                    relevance_scores.push(result.distance);
                    total_tokens += content.split_whitespace().count();
                }
            }
        }

        Ok(QueryResult {
            context_chunks,
            relevance_scores,
            keywords: vec![],
            graph_context: None,
            total_tokens,
            metadata: HashMap::new(),
        })
    }

    fn get_mode(&self) -> QueryMode {
        QueryMode::Naive
    }

    fn update_params(&mut self, _params: QueryParams) {
        // Implementation not needed for now
    }
}

#[async_trait]
impl QueryProcessor for MixQueryProcessor {
    async fn process_query(
        &self,
        query: &str,
        params: &QueryParams,
        history: Option<&[ConversationTurn]>,
    ) -> Result<QueryResult, Error> {
        // Extract keywords from query and history
        let keywords = if let Some(history) = history {
            self.keyword_extractor.extract_keywords_with_history(query, history).await?
        } else {
            self.keyword_extractor.extract_keywords(query).await?
        };

        // Prepare augmented query with conversation history if available
        let augmented_query = if let Some(history) = history {
            let history_text = history.iter()
                .map(|turn| format!("{}: {}", turn.role, turn.content))
                .collect::<Vec<_>>()
                .join("\n");
            format!("{}\n{}", history_text, query)
        } else {
            query.to_string()
        };

        // Execute knowledge graph and vector searches in parallel
        let (kg_result, vector_result) = tokio::join!(
            self.get_kg_context(&keywords.high_level, &keywords.low_level, params),
            self.get_vector_context(&augmented_query, params)
        );

        // Combine results with weighted scoring
        let mut combined_chunks = Vec::new();
        let mut combined_scores = Vec::new();
        let mut combined_keywords = keywords.high_level;
        combined_keywords.extend(keywords.low_level);

        // Weight for vector results (0.6) and graph context (0.4)
        const VECTOR_WEIGHT: f32 = 0.6;
        const GRAPH_WEIGHT: f32 = 0.4;

        // Process vector results
        if let Ok(vector_chunks) = vector_result {
            for (chunk, score) in vector_chunks {
                combined_chunks.push(chunk);
                combined_scores.push(score * VECTOR_WEIGHT);
            }
        }

        // Process knowledge graph results
        if let Ok(kg_chunks) = kg_result {
            for (chunk, score) in kg_chunks {
                // Check if chunk already exists from vector search
                if let Some(pos) = combined_chunks.iter().position(|x| x == &chunk) {
                    // Update score with weighted graph score
                    combined_scores[pos] += score * GRAPH_WEIGHT;
                } else {
                    combined_chunks.push(chunk);
                    combined_scores.push(score * GRAPH_WEIGHT);
                }
            }
        }

        // Sort by combined relevance scores
        let mut chunk_scores: Vec<_> = combined_chunks.iter()
            .zip(combined_scores.iter())
            .collect();
        chunk_scores.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Take top_k results
        let top_k = params.top_k.min(chunk_scores.len());
        let (final_chunks, final_scores): (Vec<_>, Vec<_>) = chunk_scores
            .into_iter()
            .take(top_k)
            .map(|(chunk, score)| (chunk.clone(), *score))
            .unzip();

        let final_chunks_clone = final_chunks.clone();
        Ok(QueryResult {
            context_chunks: final_chunks,
            relevance_scores: final_scores,
            keywords: combined_keywords,
            graph_context: None, // Optional graph context can be added if needed
            total_tokens: final_chunks_clone.iter()
                .map(|chunk| chunk.split_whitespace().count())
                .sum(),
            metadata: HashMap::new(),
        })
    }

    fn get_mode(&self) -> QueryMode {
        QueryMode::Mix
    }

    fn update_params(&mut self, params: QueryParams) {
        self.params = params;
    }
}

// Helper methods for MixQueryProcessor
impl MixQueryProcessor {
    async fn get_kg_context(
        &self,
        hl_keywords: &[String],
        ll_keywords: &[String],
        params: &QueryParams,
    ) -> Result<Vec<(String, f32)>, Error> {
        let mut chunks = Vec::new();
        let mut scores = Vec::new();

        // Get graph store under read lock
        let graph_store = self.graph_store.read().await;

        // Process high-level keywords for global context
        for keyword in hl_keywords {
            if let Ok(results) = graph_store.query_with_keywords(&[keyword.clone()]).await {
                // Parse results and add to chunks
                chunks.push(results);
                scores.push(1.0); // Default score for now
            }
        }

        // Process low-level keywords for local context
        for keyword in ll_keywords {
            if let Some(edges) = graph_store.get_node_edges(keyword).await {
                for (src, tgt, edge_data) in edges {
                    if let Some(node) = graph_store.get_node(&tgt).await {
                        if let Some(content) = node.attributes.get("content").and_then(|v| v.as_str()) {
                            chunks.push(content.to_string());
                            scores.push(edge_data.weight as f32);
                        }
                    }
                }
            }
        }

        Ok(chunks.into_iter().zip(scores.into_iter()).collect())
    }

    async fn get_vector_context(
        &self,
        query: &str,
        params: &QueryParams,
    ) -> Result<Vec<(String, f32)>, Error> {
        // Get vector store under read lock
        let vector_store = self.vector_store.read().await;

        // Reduce top_k for vector search since we'll combine with graph results
        let vector_top_k = (params.top_k as f32 * 0.6).ceil() as usize;
        
        // Perform vector similarity search
        let results = vector_store.query(vec![], vector_top_k).await?;

        Ok(results.into_iter()
            .map(|result| (result.metadata.get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(), 
                result.distance))
            .filter(|(content, _)| !content.is_empty())
            .collect())
    }
}

/// Factory for creating query processors
pub struct QueryProcessorFactory {
    vector_store: Arc<RwLock<dyn VectorStorage>>,
    graph_store: Arc<RwLock<dyn GraphStorage>>,
    keyword_extractor: Arc<dyn KeywordExtractor>,
}

impl QueryProcessorFactory {
    pub fn new(
        vector_store: Arc<RwLock<dyn VectorStorage>>,
        graph_store: Arc<RwLock<dyn GraphStorage>>,
        keyword_extractor: Arc<dyn KeywordExtractor>,
    ) -> Self {
        Self {
            vector_store,
            graph_store,
            keyword_extractor,
        }
    }
    
    pub fn create_processor(&self, mode: QueryMode, params: QueryParams) -> Box<dyn QueryProcessor> {
        match mode {
            QueryMode::Local => Box::new(LocalQueryProcessor::new(
                self.vector_store.clone(),
                self.keyword_extractor.clone(),
                params,
            )),
            QueryMode::Global => Box::new(GlobalQueryProcessor::new(
                self.graph_store.clone(),
                self.keyword_extractor.clone(),
                params,
            )),
            QueryMode::Hybrid => Box::new(HybridQueryProcessor::new(
                self.vector_store.clone(),
                self.graph_store.clone(),
                self.keyword_extractor.clone(),
                params,
            )),
            QueryMode::Naive => Box::new(NaiveQueryProcessor::new(
                self.vector_store.clone(),
                params,
            )),
            QueryMode::Mix => Box::new(MixQueryProcessor::new(
                self.vector_store.clone(),
                self.graph_store.clone(),
                self.keyword_extractor.clone(),
                params,
            )),
        }
    }
} 