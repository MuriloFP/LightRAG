use std::sync::Arc;
use tokio::sync::RwLock;
use std::collections::HashMap;
use serde_json::Value;
use chrono::Utc;
use super_lightrag::{
    types::{
        Error,
        llm::{QueryMode, QueryParams},
    },
    processing::{
        query::{QueryProcessorFactory, LocalQueryProcessor, GlobalQueryProcessor, HybridQueryProcessor, NaiveQueryProcessor, QueryProcessor},
        keywords::{KeywordExtractor, ExtractedKeywords, ConversationTurn, KeywordConfig},
    },
    storage::{
        GraphStorage,
        VectorStorage,
        vector::{SearchResult, VectorData, UpsertResponse},
        graph::{NodeData, EdgeData, embeddings::EmbeddingAlgorithm},
    },
};

// Mock implementations for testing
#[derive(Clone)]
struct MockVectorStorage {
    vectors: HashMap<String, VectorData>,
}

impl MockVectorStorage {
    fn new() -> Self {
        Self {
            vectors: HashMap::new(),
        }
    }
}

#[async_trait::async_trait]
impl VectorStorage for MockVectorStorage {
    async fn initialize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    async fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    async fn query(&self, _query: Vec<f32>, _top_k: usize) -> Result<Vec<SearchResult>, Error> {
        Ok(vec![
            SearchResult {
                id: "1".to_string(),
                distance: 0.9,
                metadata: [("content".to_string(), Value::String("Test content 1".to_string()))]
                    .into_iter()
                    .collect(),
            },
            SearchResult {
                id: "2".to_string(),
                distance: 0.8,
                metadata: [("content".to_string(), Value::String("Test content 2".to_string()))]
                    .into_iter()
                    .collect(),
            },
        ])
    }

    async fn upsert(&mut self, data: Vec<VectorData>) -> Result<UpsertResponse, Error> {
        let mut inserted = Vec::new();
        let mut updated = Vec::new();
        
        for vector in data {
            if self.vectors.contains_key(&vector.id) {
                updated.push(vector.id.clone());
            } else {
                inserted.push(vector.id.clone());
            }
            self.vectors.insert(vector.id.clone(), vector);
        }
        
        Ok(UpsertResponse { inserted, updated })
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<(), Error> {
        for id in ids {
            self.vectors.remove(&id);
        }
        Ok(())
    }
}

#[derive(Clone)]
struct MockGraphStorage {
    nodes: HashMap<String, NodeData>,
    edges: HashMap<String, (String, String, EdgeData)>,
}

impl MockGraphStorage {
    fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }
}

#[async_trait::async_trait]
impl GraphStorage for MockGraphStorage {
    async fn initialize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    async fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    async fn has_node(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
    }

    async fn get_node(&self, id: &str) -> Option<NodeData> {
        self.nodes.get(id).cloned()
    }

    async fn upsert_node(&mut self, id: &str, data: HashMap<String, Value>) -> Result<(), Error> {
        self.nodes.insert(id.to_string(), NodeData { id: id.to_string(), attributes: data });
        Ok(())
    }

    async fn upsert_edge(&mut self, source: &str, target: &str, data: EdgeData) -> Result<(), Error> {
        let edge_id = format!("{}_{}", source, target);
        self.edges.insert(edge_id, (source.to_string(), target.to_string(), data));
        Ok(())
    }

    async fn get_edge(&self, source: &str, target: &str) -> Option<EdgeData> {
        let edge_id = format!("{}_{}", source, target);
        self.edges.get(&edge_id).map(|(_, _, data)| data.clone())
    }

    async fn has_edge(&self, source: &str, target: &str) -> bool {
        let edge_id = format!("{}_{}", source, target);
        self.edges.contains_key(&edge_id)
    }

    async fn get_node_edges(&self, node_id: &str) -> Option<Vec<(String, String, EdgeData)>> {
        let mut edges = Vec::new();
        for (_, (src, tgt, data)) in &self.edges {
            if src == node_id || tgt == node_id {
                edges.push((src.clone(), tgt.clone(), data.clone()));
            }
        }
        Some(edges)
    }

    async fn node_degree(&self, node_id: &str) -> Result<usize, Error> {
        let mut degree = 0;
        for (edge_id, _) in &self.edges {
            if edge_id.contains(node_id) {
                degree += 1;
            }
        }
        Ok(degree)
    }

    async fn edge_degree(&self, src_id: &str, tgt_id: &str) -> Result<usize, Error> {
        let src_degree = self.node_degree(src_id).await?;
        let tgt_degree = self.node_degree(tgt_id).await?;
        Ok(src_degree + tgt_degree)
    }

    async fn delete_node(&mut self, node_id: &str) -> Result<(), Error> {
        self.nodes.remove(node_id);
        self.edges.retain(|edge_id, _| !edge_id.contains(node_id));
        Ok(())
    }

    async fn delete_edge(&mut self, source_id: &str, target_id: &str) -> Result<(), Error> {
        let edge_id = format!("{}_{}", source_id, target_id);
        self.edges.remove(&edge_id);
        Ok(())
    }

    async fn upsert_nodes(&mut self, nodes: Vec<(String, HashMap<String, Value>)>) -> Result<(), Error> {
        for (id, attrs) in nodes {
            self.upsert_node(&id, attrs).await?;
        }
        Ok(())
    }

    async fn upsert_edges(&mut self, edges: Vec<(String, String, EdgeData)>) -> Result<(), Error> {
        for (src, tgt, data) in edges {
            self.upsert_edge(&src, &tgt, data).await?;
        }
        Ok(())
    }

    async fn remove_nodes(&mut self, node_ids: Vec<String>) -> Result<(), Error> {
        for id in node_ids {
            self.delete_node(&id).await?;
        }
        Ok(())
    }

    async fn remove_edges(&mut self, edges: Vec<(String, String)>) -> Result<(), Error> {
        for (src, tgt) in edges {
            self.delete_edge(&src, &tgt).await?;
        }
        Ok(())
    }

    async fn embed_nodes(&self, _algorithm: EmbeddingAlgorithm) -> Result<(Vec<f32>, Vec<String>), Error> {
        Ok((vec![], vec![])) // Mock implementation returns empty embeddings
    }

    async fn query_with_keywords(&self, keywords: &[String]) -> Result<String, Error> {
        let mut context = String::new();
        
        for keyword in keywords {
            // Search for nodes containing the keyword
            for (_, node_data) in &self.nodes {
                if let Some(content) = node_data.attributes.get("content").and_then(|v| v.as_str()) {
                    if content.to_lowercase().contains(&keyword.to_lowercase()) {
                        // Add node content to context
                        context.push_str(content);
                        context.push_str("\n");
                        
                        // Get connected nodes through edges
                        for (_, (src, tgt, _)) in &self.edges {
                            let neighbor_id = if src == &node_data.id { tgt } else if tgt == &node_data.id { src } else { continue };
                            if let Some(neighbor_data) = self.nodes.get(neighbor_id) {
                                if let Some(neighbor_content) = neighbor_data.attributes.get("content").and_then(|v| v.as_str()) {
                                    context.push_str(neighbor_content);
                                    context.push_str("\n");
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(context)
    }

    async fn get_all_labels(&self) -> Result<Vec<String>, Error> {
        // Return all node IDs as labels for the mock implementation
        Ok(self.nodes.keys().cloned().collect())
    }

    async fn get_knowledge_graph(&self, node_label: &str, max_depth: i32) -> Result<super_lightrag::types::KnowledgeGraph, Error> {
        use super_lightrag::types::{KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge};
        use std::collections::HashSet;

        let mut result = KnowledgeGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
        };

        let mut seen_nodes = HashSet::new();
        let mut seen_edges = HashSet::new();

        // Helper function to add a node and its edges recursively
        fn add_node_and_edges(
            storage: &MockGraphStorage,
            node_id: &str,
            depth: i32,
            max_depth: i32,
            result: &mut KnowledgeGraph,
            seen_nodes: &mut HashSet<String>,
            seen_edges: &mut HashSet<String>,
        ) {
            if depth > max_depth || seen_nodes.contains(node_id) {
                return;
            }

            // Add the node
            if let Some(node_data) = storage.nodes.get(node_id) {
                result.nodes.push(KnowledgeGraphNode {
                    id: node_id.to_string(),
                    labels: vec![node_id.to_string()],
                    properties: node_data.attributes.clone(),
                });
                seen_nodes.insert(node_id.to_string());

                // Add edges
                for (edge_id, (src, tgt, edge_data)) in &storage.edges {
                    if src == node_id || tgt == node_id {
                        if !seen_edges.contains(edge_id) {
                            result.edges.push(KnowledgeGraphEdge {
                                id: edge_id.clone(),
                                edge_type: None,
                                source: src.clone(),
                                target: tgt.clone(),
                                properties: {
                                    let mut props = HashMap::new();
                                    let weight_value = serde_json::Number::from_f64(edge_data.weight)
                                        .unwrap_or_else(|| serde_json::Number::from(0));
                                    props.insert("weight".to_string(), serde_json::Value::Number(weight_value));
                                    if let Some(desc) = &edge_data.description {
                                        props.insert("description".to_string(), serde_json::Value::String(desc.clone()));
                                    }
                                    if let Some(keywords) = &edge_data.keywords {
                                        props.insert("keywords".to_string(), serde_json::Value::Array(
                                            keywords.iter().map(|k| serde_json::Value::String(k.clone())).collect()
                                        ));
                                    }
                                    props
                                },
                            });
                            seen_edges.insert(edge_id.clone());

                            // Recursively process connected node
                            let next_node = if src == node_id { tgt } else { src };
                            add_node_and_edges(storage, next_node, depth + 1, max_depth, result, seen_nodes, seen_edges);
                        }
                    }
                }
            }
        }

        if node_label == "*" {
            // Add all nodes and their edges
            for node_id in self.nodes.keys() {
                add_node_and_edges(self, node_id, 0, max_depth, &mut result, &mut seen_nodes, &mut seen_edges);
            }
        } else {
            // Start from the specified node
            add_node_and_edges(self, node_label, 0, max_depth, &mut result, &mut seen_nodes, &mut seen_edges);
        }

        Ok(result)
    }
}

struct MockKeywordExtractor {
    config: KeywordConfig,
}

impl MockKeywordExtractor {
    fn new() -> Self {
        Self {
            config: KeywordConfig::default(),
        }
    }
}

#[async_trait::async_trait]
impl KeywordExtractor for MockKeywordExtractor {
    async fn extract_keywords(&self, _text: &str) -> Result<ExtractedKeywords, Error> {
        Ok(ExtractedKeywords {
            high_level: vec!["test".to_string(), "keyword".to_string()],
            low_level: vec!["detail".to_string()],
            metadata: HashMap::new(),
        })
    }

    async fn extract_keywords_with_history(&self, _text: &str, _history: &[ConversationTurn]) -> Result<ExtractedKeywords, Error> {
        Ok(ExtractedKeywords {
            high_level: vec!["test".to_string(), "keyword".to_string(), "history".to_string()],
            low_level: vec!["detail".to_string()],
            metadata: HashMap::new(),
        })
    }

    fn get_config(&self) -> &KeywordConfig {
        &self.config
    }
}

#[tokio::test]
async fn test_local_query_processing() -> Result<(), Error> {
    let vector_store = Arc::new(RwLock::new(MockVectorStorage::new()));
    let keyword_extractor = Arc::new(MockKeywordExtractor::new());
    let params = QueryParams::default();

    let processor = LocalQueryProcessor::new(
        vector_store,
        keyword_extractor,
        params,
    );

    let query = "test query";
    let history = vec![
        ConversationTurn {
            role: "user".to_string(),
            content: "previous query".to_string(),
            timestamp: Some(Utc::now()),
        },
    ];

    let result = processor.process_query(query, &QueryParams::default(), Some(&history)).await?;
    assert!(!result.context_chunks.is_empty());
    assert!(!result.relevance_scores.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_global_query_processing() -> Result<(), Error> {
    let graph_store = Arc::new(RwLock::new(MockGraphStorage::new()));
    let keyword_extractor = Arc::new(MockKeywordExtractor::new());
    let params = QueryParams::default();

    let processor = GlobalQueryProcessor::new(
        graph_store,
        keyword_extractor,
        params,
    );

    let query = "test query";
    let history = vec![
        ConversationTurn {
            role: "user".to_string(),
            content: "previous query".to_string(),
            timestamp: Some(Utc::now()),
        },
    ];

    let result = processor.process_query(query, &QueryParams::default(), Some(&history)).await?;
    assert!(result.graph_context.is_some());

    Ok(())
}

#[tokio::test]
async fn test_hybrid_query_processing() -> Result<(), Error> {
    let vector_store = Arc::new(RwLock::new(MockVectorStorage::new()));
    let graph_store = Arc::new(RwLock::new(MockGraphStorage::new()));
    let keyword_extractor = Arc::new(MockKeywordExtractor::new());
    let params = QueryParams::default();

    let processor = HybridQueryProcessor::new(
        vector_store,
        graph_store,
        keyword_extractor,
        params,
    );

    let query = "test query";
    let history = vec![
        ConversationTurn {
            role: "user".to_string(),
            content: "previous query".to_string(),
            timestamp: Some(Utc::now()),
        },
    ];

    let result = processor.process_query(query, &QueryParams::default(), Some(&history)).await?;
    assert!(!result.context_chunks.is_empty());
    assert!(!result.relevance_scores.is_empty());
    assert!(result.graph_context.is_some());

    Ok(())
}

#[tokio::test]
async fn test_naive_query_processing() -> Result<(), Error> {
    let vector_store = Arc::new(RwLock::new(MockVectorStorage::new()));
    let params = QueryParams::default();

    let processor = NaiveQueryProcessor::new(
        vector_store,
        params,
    );

    let query = "test query";
    let result = processor.process_query(query, &QueryParams::default(), None).await?;
    assert!(!result.context_chunks.is_empty());
    assert!(!result.relevance_scores.is_empty());

    Ok(())
}

#[tokio::test]
async fn test_query_processor_factory() -> Result<(), Error> {
    let vector_store = Arc::new(RwLock::new(MockVectorStorage::new()));
    let graph_store = Arc::new(RwLock::new(MockGraphStorage::new()));
    let keyword_extractor = Arc::new(MockKeywordExtractor::new());

    let factory = QueryProcessorFactory::new(
        vector_store,
        graph_store,
        keyword_extractor,
    );

    let params = QueryParams::default();

    let local_processor = factory.create_processor(QueryMode::Local, params.clone());
    let global_processor = factory.create_processor(QueryMode::Global, params.clone());
    let hybrid_processor = factory.create_processor(QueryMode::Hybrid, params.clone());
    let naive_processor = factory.create_processor(QueryMode::Naive, params);

    assert_eq!(local_processor.get_mode(), QueryMode::Local);
    assert_eq!(global_processor.get_mode(), QueryMode::Global);
    assert_eq!(hybrid_processor.get_mode(), QueryMode::Hybrid);
    assert_eq!(naive_processor.get_mode(), QueryMode::Naive);

    Ok(())
} 