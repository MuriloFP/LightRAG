use std::collections::HashMap;
use std::sync::Arc;
use std::time::SystemTime;
use std::default::Default;
use tokio::sync::RwLock;
use async_trait::async_trait;
use regex::Regex;
use lazy_static::lazy_static;
use serde_json::Value;
use futures::TryFutureExt;
use crate::{
    types::{Error, llm::{LLMClient, LLMParams}},
    storage::{
        GraphStorage, VectorStorage,
        graph::EdgeData,
        vector::VectorData,
    },
};
use super::{EntityExtractor, EntityExtractionConfig, Entity, EntityType, Relationship, EntityError};

const TUPLE_DELIMITER: &str = "<|>";
const RECORD_DELIMITER: &str = "##";
const COMPLETION_DELIMITER: &str = "<|COMPLETE|>";

lazy_static! {
    static ref ENTITY_REGEX: Regex = Regex::new(r#"\("entity"<\|>([^<\|>]+)<\|>([^<\|>]+)<\|>([^)]+)\)"#).unwrap();
    static ref RELATIONSHIP_REGEX: Regex = Regex::new(r#"\("relationship"<\|>([^<\|>]+)<\|>([^<\|>]+)<\|>([^<\|>]+)<\|>([^<\|>]+)<\|>([^)]+)\)"#).unwrap();
}

/// Cache entry for LLM responses
#[derive(Debug, Clone)]
struct CacheEntry {
    response: String,
    timestamp: SystemTime,
}

/// LLM-based implementation of entity extraction
pub struct LLMEntityExtractor {
    /// Configuration
    config: EntityExtractionConfig,
    
    /// LLM client
    llm_client: Arc<dyn LLMClient>,
    
    /// Knowledge graph storage
    knowledge_graph: Arc<RwLock<dyn GraphStorage>>,
    
    /// Entity vector storage
    entity_vector_store: Arc<RwLock<dyn VectorStorage>>,
    
    /// Relationship vector storage
    relationship_vector_store: Arc<RwLock<dyn VectorStorage>>,
    
    /// Response cache
    cache: Option<Arc<RwLock<HashMap<String, CacheEntry>>>>,
}

impl LLMEntityExtractor {
    /// Create a new LLMEntityExtractor
    pub fn new(
        config: EntityExtractionConfig,
        llm_client: Arc<dyn LLMClient>,
        knowledge_graph: Arc<RwLock<dyn GraphStorage>>,
        entity_vector_store: Arc<RwLock<dyn VectorStorage>>,
        relationship_vector_store: Arc<RwLock<dyn VectorStorage>>,
    ) -> Self {
        let cache = if config.use_cache {
            Some(Arc::new(RwLock::new(HashMap::new())))
        } else {
            None
        };
        
        Self {
            config,
            llm_client,
            knowledge_graph,
            entity_vector_store,
            relationship_vector_store,
            cache,
        }
    }
    
    /// Generate the entity extraction prompt
    fn generate_entity_extraction_prompt(&self, content: &str) -> String {
        let entity_types = self.config.entity_types
            .iter()
            .map(|t| t.as_str())
            .collect::<Vec<_>>()
            .join(",");
            
        format!(
            r#"---Goal---
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.
Use {} as output language.

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If English, capitalized the name.
- entity_type: One of the following types: [{}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"{}{{NAME}}{}{{TYPE}}{}{{DESCRIPTION}})

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship
Format each relationship as ("relationship"{}{{SOURCE}}{}{{TARGET}}{}{{DESCRIPTION}}{}{{KEYWORDS}}{}{{STRENGTH}})

Text: {}"#,
            self.config.language,      // 1
            entity_types,              // 2
            TUPLE_DELIMITER,           // 3 for entity formatting
            TUPLE_DELIMITER,           // 4
            TUPLE_DELIMITER,           // 5
            TUPLE_DELIMITER,           // 6 for relationship formatting
            TUPLE_DELIMITER,           // 7
            TUPLE_DELIMITER,           // 8
            TUPLE_DELIMITER,           // 9
            TUPLE_DELIMITER,           // 10
            content                    // 11
        )
    }
    
    /// Generate the continue extraction prompt
    fn generate_continue_prompt(&self, previous_result: &str) -> String {
        format!(
            "Based on the previous extraction:\n{}\n\nAre there any additional entities or relationships that could be extracted? If yes, please extract them in the same format.",
            previous_result
        )
    }
    
    /// Check if we should continue extraction
    async fn should_continue_extraction(&self, result: &str) -> Result<bool, Error> {
        let prompt = "Based on the previous extraction, should we continue looking for more entities and relationships? Answer with just 'yes' or 'no'.";
        
        let response = self.llm_client.generate(
            prompt,
            &LLMParams {
                max_tokens: 10,
                temperature: 0.1,
                ..Default::default()
            },
        ).await.map_err(|e| EntityError::LLMError(e.to_string()))?;
        
        Ok(response.text.to_lowercase().contains("yes"))
    }
    
    /// Extract entities and relationships using LLM
    async fn extract_with_llm(&self, prompt: &str) -> Result<String, Error> {
        // Check cache first if enabled
        if let Some(cache) = &self.cache {
            if let Some(cached_response) = cache.read().await.get(prompt) {
                return Ok(cached_response.response.clone());
            }
        }
        
        // Get response from LLM
        let response = self.llm_client.generate(
            prompt,
            &LLMParams {
                max_tokens: 2000,
                temperature: 0.7,
                ..Default::default()
            },
        ).await.map_err(|e| EntityError::LLMError(e.to_string()))?;
        
        // Cache the response if enabled
        if let Some(cache) = &self.cache {
            let mut cache = cache.write().await;
            cache.insert(prompt.to_string(), CacheEntry {
                response: response.text.clone(),
                timestamp: SystemTime::now(),
            });
        }
        
        Ok(response.text)
    }
    
    /// Process the extraction result
    fn process_extraction_result(&self, result: &str) -> Result<(Vec<Entity>, Vec<Relationship>), Error> {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();
        
        // Extract entities
        for cap in ENTITY_REGEX.captures_iter(result) {
            let name = cap[1].trim().to_string();
            let entity_type = match cap[2].trim().to_lowercase().as_str() {
                "organization" => EntityType::Organization,
                "person" => EntityType::Person,
                "geo" => EntityType::Geo,
                "event" => EntityType::Event,
                "category" => EntityType::Category,
                _ => EntityType::Unknown,
            };
            let description = cap[3].trim().to_string();
            
            entities.push(Entity {
                name,
                entity_type,
                description,
                source_id: String::new(), // Will be set by caller
                metadata: HashMap::new(),
            });
        }
        
        // Extract relationships
        for cap in RELATIONSHIP_REGEX.captures_iter(result) {
            let src_id = cap[1].trim().to_string();
            let tgt_id = cap[2].trim().to_string();
            let description = cap[3].trim().to_string();
            let keywords = cap[4].trim().to_string();
            let weight = cap[5].trim().parse::<f32>().unwrap_or(1.0);
            
            relationships.push(Relationship {
                src_id,
                tgt_id,
                description,
                keywords,
                weight,
                source_id: String::new(), // Will be set by caller
                metadata: HashMap::new(),
            });
        }
        
        Ok((entities, relationships))
    }
    
    /// Store extracted entities
    async fn store_entities(&self, entities: &[Entity]) -> Result<(), Error> {
        for entity in entities {
            self.store_entity(entity).await?;
        }
        Ok(())
    }
    
    /// Store extracted relationships
    async fn store_relationships(&self, relationships: &[Relationship]) -> Result<(), Error> {
        for rel in relationships {
            self.store_relationship(rel).await?;
        }
        Ok(())
    }

    async fn store_entity(&self, entity: &Entity) -> Result<(), EntityError> {
        // Convert entity.metadata (HashMap<String, String>) to serde_json::Map<String, Value>
        let metadata_map: serde_json::Map<String, Value> = entity.metadata.iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        
        let mut attributes = HashMap::new();
        attributes.insert("entity_type".to_string(), Value::String(entity.entity_type.as_str().to_string()));
        attributes.insert("description".to_string(), Value::String(entity.description.clone()));
        attributes.insert("source_id".to_string(), Value::String(entity.source_id.clone()));
        attributes.insert("metadata".to_string(), Value::Object(metadata_map.clone()));
        
        // Store in graph
        self.knowledge_graph.write().await
            .upsert_node(&entity.name, attributes)
            .await
            .map_err(|e| EntityError::StorageError(e.to_string()))?;
        
        // Convert entity.metadata again for vector storage
        let mut vec_metadata: HashMap<String, Value> = entity.metadata.iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        vec_metadata.insert("content".to_string(), Value::String(format!("{} {}", entity.name, entity.description)));
        vec_metadata.insert("entity_name".to_string(), Value::String(entity.name.clone()));
        
        let vector_data = VectorData {
            id: format!("ent-{}", entity.name),
            vector: vec![0.0; 384], // Default vector, will be computed by the vector store
            metadata: vec_metadata,
            created_at: SystemTime::now(),
        };
        
        self.entity_vector_store.write().await
            .upsert(vec![vector_data])
            .await
            .map_err(|e| EntityError::StorageError(e.to_string()))?;
        
        Ok(())
    }

    async fn store_relationship(&self, rel: &Relationship) -> Result<(), EntityError> {
        // Construct EdgeData using only supported fields: weight, description, keywords.
        // Cast weight from f32 to f64, set description as Some(...), keywords as Some(vec![...]).
        let edge_data = EdgeData {
            weight: rel.weight as f64,
            description: Some(rel.description.clone()),
            keywords: Some(vec![rel.keywords.clone()]),
        };
        
        // Store in graph
        self.knowledge_graph.write().await
            .upsert_edge(&rel.src_id, &rel.tgt_id, edge_data)
            .await
            .map_err(|e| EntityError::StorageError(e.to_string()))?;
        
        // Convert relationship.metadata for vector storage
        let mut vec_metadata: HashMap<String, Value> = rel.metadata.iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        vec_metadata.insert("content".to_string(), Value::String(format!("{} {} {} {}",
            rel.keywords, rel.src_id, rel.tgt_id, rel.description)));
        vec_metadata.insert("src_id".to_string(), Value::String(rel.src_id.clone()));
        vec_metadata.insert("tgt_id".to_string(), Value::String(rel.tgt_id.clone()));
        
        let vector_data = VectorData {
            id: format!("rel-{}-{}", rel.src_id, rel.tgt_id),
            vector: vec![0.0; 384],
            metadata: vec_metadata,
            created_at: SystemTime::now(),
        };
        
        self.relationship_vector_store.write().await
            .upsert(vec![vector_data])
            .await
            .map_err(|e| EntityError::StorageError(e.to_string()))?;
        
        Ok(())
    }

    fn get_entity_prompt(&self, content: &str) -> String {
        format!(
            "Extract entities and relationships from the following text. \
            For each entity, provide: name, type (person, organization, location, date, etc), and description. \
            For each relationship between entities, provide: source entity, target entity, description, and keywords. \
            Format each entity as: (\"entity\"<|>NAME<|>TYPE<|>DESCRIPTION) \
            Format each relationship as: (\"relationship\"<|>SRC_NAME<|>TGT_NAME<|>DESCRIPTION<|>KEYWORDS<|>STRENGTH) \
            Use ## to separate multiple entities/relationships. \
            Text: {}", 
            content
        )
    }
}

#[async_trait]
impl EntityExtractor for LLMEntityExtractor {
    async fn extract_entities(&self, content: &str) -> Result<(Vec<Entity>, Vec<Relationship>), Error> {
        if content.is_empty() {
            return Err(EntityError::EmptyContent.into());
        }
        
        // Generate initial prompt and get extraction
        let prompt = self.get_entity_prompt(content);
        let mut final_result = self.extract_with_llm(&prompt).await?;
        
        // Multiple extraction attempts if configured
        for _ in 0..self.config.max_gleaning_attempts {
            let continue_prompt = self.generate_continue_prompt(&final_result);
            let additional_result = self.extract_with_llm(&continue_prompt).await?;
            
            if additional_result.is_empty() {
                break;
            }
            
            final_result.push_str(&additional_result);
            
            // Check if we should continue
            if !self.should_continue_extraction(&final_result).await? {
                break;
            }
        }
        
        // Process the final result
        let (mut entities, mut relationships) = self.process_extraction_result(&final_result)?;
        
        // Set source IDs using hexadecimal formatting for md5
        let source_id = format!("doc-{:x}", md5::compute(content.as_bytes()));
        for entity in &mut entities {
            entity.source_id = source_id.clone();
        }
        for rel in &mut relationships {
            rel.source_id = source_id.clone();
        }
        
        // Store the extracted data
        self.store_entities(&entities).await?;
        self.store_relationships(&relationships).await?;
        
        Ok((entities, relationships))
    }
    
    fn get_config(&self) -> &EntityExtractionConfig {
        &self.config
    }
} 