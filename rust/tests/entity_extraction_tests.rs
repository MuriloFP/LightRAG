use std::sync::Arc;
use std::collections::HashMap;
use chrono::Utc;
use async_trait::async_trait;
use tokio::sync::RwLock;
use serde_json::Value;
use tokio::sync::Mutex;
use futures::Stream;
use std::pin::Pin;

use super_lightrag::{
    types::{
        llm::{LLMParams, LLMResponse, LLMError, StreamingResponse},
        Error,
        KnowledgeGraph,
    },
    storage::{
        GraphStorage, VectorStorage, 
        graph::{EdgeData, NodeData},
        vector::{VectorData, SearchResult, UpsertResponse},
    },
    processing::{
        entity::{
            EntityExtractor, EntityExtractionConfig, LLMEntityExtractor,
            EntityType,
            Entity,
        },
        keywords::ConversationTurn,
    },
    llm::{Provider, ProviderConfig},
};

/// Mock graph storage for testing
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

#[async_trait]
impl GraphStorage for MockGraphStorage {
    async fn query_with_keywords(&self, _keywords: &[String]) -> Result<String, Error> {
        Ok("".to_string())
    }

    async fn initialize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    async fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    async fn has_node(&self, node_id: &str) -> bool {
        self.nodes.contains_key(node_id)
    }

    async fn has_edge(&self, source_id: &str, target_id: &str) -> bool {
        let edge_id = format!("{}_{}", source_id, target_id);
        self.edges.contains_key(&edge_id)
    }

    async fn get_node(&self, node_id: &str) -> Option<NodeData> {
        self.nodes.get(node_id).cloned()
    }

    async fn get_edge(&self, source_id: &str, target_id: &str) -> Option<EdgeData> {
        let edge_id = format!("{}_{}", source_id, target_id);
        self.edges.get(&edge_id).map(|(_, _, data)| data.clone())
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

    async fn upsert_node(&mut self, node_id: &str, attributes: HashMap<String, Value>) -> Result<(), Error> {
        self.nodes.insert(node_id.to_string(), NodeData { id: node_id.to_string(), attributes });
        Ok(())
    }

    async fn upsert_edge(&mut self, source_id: &str, target_id: &str, data: EdgeData) -> Result<(), Error> {
        let edge_id = format!("{}_{}", source_id, target_id);
        self.edges.insert(edge_id, (source_id.to_string(), target_id.to_string(), data));
        Ok(())
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

    async fn embed_nodes(&self, _algorithm: super_lightrag::storage::graph::embeddings::EmbeddingAlgorithm) -> Result<(Vec<f32>, Vec<String>), Error> {
        Ok((vec![], vec![]))
    }

    async fn get_all_labels(&self) -> Result<Vec<String>, Error> {
        Ok(self.nodes.keys().cloned().collect())
    }

    async fn get_knowledge_graph(&self, _node_label: &str, _max_depth: i32) -> Result<KnowledgeGraph, Error> {
        Ok(KnowledgeGraph {
            nodes: vec![],
            edges: vec![],
        })
    }
}

/// Mock vector storage for testing
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

#[async_trait]
impl VectorStorage for MockVectorStorage {
    async fn initialize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    async fn finalize(&mut self) -> Result<(), Error> {
        Ok(())
    }

    async fn query(&self, _query: Vec<f32>, top_k: usize) -> Result<Vec<SearchResult>, Error> {
        let mut results = Vec::new();
        for (id, data) in self.vectors.iter().take(top_k) {
            results.push(SearchResult {
                id: id.clone(),
                distance: 1.0,
                metadata: data.metadata.clone(),
            });
        }
        Ok(results)
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

/// Mock LLM provider for testing
struct MockProvider {
    responses: Vec<String>,
    current: std::sync::atomic::AtomicUsize,
}

impl MockProvider {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses,
            current: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

#[async_trait::async_trait]
impl Provider for MockProvider {
    async fn initialize(&mut self) -> std::result::Result<(), LLMError> {
        Ok(())
    }

    async fn complete(&self, _prompt: &str, _params: &LLMParams) -> std::result::Result<LLMResponse, LLMError> {
        let idx = self.current.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if idx < self.responses.len() {
            Ok(LLMResponse {
                text: self.responses[idx].clone(),
                tokens_used: 100,
                model: "mock".to_string(),
                cached: false,
                context: None,
                metadata: HashMap::new(),
            })
        } else {
            Err(LLMError::RequestFailed("No more mock responses".to_string()))
        }
    }

    async fn complete_stream(
        &self,
        _prompt: &str,
        _params: &LLMParams,
    ) -> std::result::Result<Pin<Box<dyn Stream<Item = std::result::Result<StreamingResponse, LLMError>> + Send>>, LLMError> {
        Err(LLMError::RequestFailed("Streaming not needed for tests".to_string()))
    }

    async fn complete_batch(
        &self,
        _prompts: &[String],
        _params: &LLMParams,
    ) -> std::result::Result<Vec<LLMResponse>, LLMError> {
        Err(LLMError::RequestFailed("Batch completion not needed for tests".to_string()))
    }

    fn get_config(&self) -> &ProviderConfig {
        panic!("Config not needed for tests")
    }

    fn update_config(&mut self, _config: ProviderConfig) -> std::result::Result<(), LLMError> {
        Err(LLMError::RequestFailed("Config update not needed for tests".to_string()))
    }
}

#[tokio::test]
async fn test_basic_entity_extraction() {
    // Mock response in the expected format from LLM
    let mock_response = r#"{type: person, name: JOHN DOE, description: A software engineer at Acme Corp}
{type: organization, name: ACME CORP, description: A technology company}
[src: JOHN DOE, tgt: ACME CORP, type: employment, description: Works as an engineer]"#.to_string();

    let provider = Arc::new(Box::new(MockProvider::new(vec![mock_response])) as Box<dyn Provider>);
    
    let config = EntityExtractionConfig {
        max_gleaning_attempts: 1,
        language: "English".to_string(),
        entity_types: vec![
            EntityType::Person,
            EntityType::Organization,
        ],
        use_cache: false,
        cache_similarity_threshold: 0.95,
        extra_params: HashMap::new(),
    };

    let extractor = LLMEntityExtractor::new(provider, config);

    let text = "John Doe is a software engineer working at Acme Corp.";
    let (entities, relationships) = extractor.extract_entities(text).await.unwrap();

    // Verify entities
    assert_eq!(entities.len(), 2);
    let john = entities.iter().find(|e| e.name == "JOHN DOE").unwrap();
    assert_eq!(john.entity_type, EntityType::Person);
    assert_eq!(john.description, "A software engineer at Acme Corp");

    let acme = entities.iter().find(|e| e.name == "ACME CORP").unwrap();
    assert_eq!(acme.entity_type, EntityType::Organization);
    assert_eq!(acme.description, "A technology company");

    // Verify relationships
    assert_eq!(relationships.len(), 1);
    let rel = &relationships[0];
    assert_eq!(rel.src_id, "JOHN DOE");
    assert_eq!(rel.tgt_id, "ACME CORP");
    assert_eq!(rel.description, "Works as an engineer");
    assert_eq!(rel.keywords, "employment");
    assert_eq!(rel.weight, 0.9);
}

#[tokio::test]
async fn test_empty_content() {
    let provider = Arc::new(Box::new(MockProvider::new(vec![])) as Box<dyn Provider>);
    
    let config = EntityExtractionConfig::default();
    let extractor = LLMEntityExtractor::new(provider, config);

    let result = extractor.extract_entities("").await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_multiple_entity_types() {
    let mock_response = r#"{type: geo, name: NEW YORK, description: A major city in the United States}
{type: event, name: TECH EXPO 2024, description: Annual technology conference}
{type: category, name: AI SYSTEMS, description: Advanced computing technologies}
[src: TECH EXPO 2024, tgt: NEW YORK, type: location, description: Takes place in]
[src: TECH EXPO 2024, tgt: AI SYSTEMS, type: topic, description: Features presentations on]"#.to_string();

    let provider = Arc::new(Box::new(MockProvider::new(vec![mock_response])) as Box<dyn Provider>);
    
    let config = EntityExtractionConfig {
        max_gleaning_attempts: 1,
        language: "English".to_string(),
        entity_types: vec![
            EntityType::Geo,
            EntityType::Event,
            EntityType::Category,
        ],
        use_cache: false,
        cache_similarity_threshold: 0.95,
        extra_params: HashMap::new(),
    };

    let extractor = LLMEntityExtractor::new(provider, config);

    let text = "The Tech Expo 2024 in New York will showcase the latest AI Systems.";
    let (entities, relationships) = extractor.extract_entities(text).await.unwrap();

    // Verify entities
    assert_eq!(entities.len(), 3);
    assert!(entities.iter().any(|e| e.name == "NEW YORK" && e.entity_type == EntityType::Geo));
    assert!(entities.iter().any(|e| e.name == "TECH EXPO 2024" && e.entity_type == EntityType::Event));
    assert!(entities.iter().any(|e| e.name == "AI SYSTEMS" && e.entity_type == EntityType::Category));

    // Verify relationships
    assert_eq!(relationships.len(), 2);
    assert!(relationships.iter().any(|r| r.src_id == "TECH EXPO 2024" && r.tgt_id == "NEW YORK"));
    assert!(relationships.iter().any(|r| r.src_id == "TECH EXPO 2024" && r.tgt_id == "AI SYSTEMS"));
}

#[tokio::test]
async fn test_entity_extraction_with_gleaning() {
    let mock_responses = vec![
        r#"{type: person, name: ALICE SMITH, description: CEO of Tech Corp}"#.to_string(),
        r#"{type: organization, name: TECH CORP, description: Technology company}"#.to_string(),
        r#""#.to_string()
    ];

    let provider = Arc::new(Box::new(MockProvider::new(mock_responses)) as Box<dyn Provider>);
    
    let config = EntityExtractionConfig {
        max_gleaning_attempts: 2,
        language: "English".to_string(),
        entity_types: vec![EntityType::Person, EntityType::Organization],
        use_cache: false,
        cache_similarity_threshold: 0.95,
        extra_params: HashMap::new(),
    };

    let extractor = LLMEntityExtractor::new(provider, config);

    let text = "Alice Smith leads Tech Corp as its CEO.";
    let (entities, _) = extractor.extract_entities(text).await.unwrap();

    assert_eq!(entities.len(), 2);
    assert!(entities.iter().any(|e| e.name == "ALICE SMITH" && e.entity_type == EntityType::Person));
    assert!(entities.iter().any(|e| e.name == "TECH CORP" && e.entity_type == EntityType::Organization));
} 