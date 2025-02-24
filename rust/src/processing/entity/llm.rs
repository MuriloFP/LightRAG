use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use std::time::{SystemTime, Duration};
use serde_json::Value;
use lazy_static::lazy_static;

use crate::types::Error;
use crate::types::llm::{LLMParams, LLMResponse};
use crate::llm::Provider;
use crate::llm::cache::entry::CacheEntry;
use crate::llm::cache::types::CacheType;
use crate::processing::entity::{Entity, EntityType, Relationship, EntityExtractor, EntityExtractionConfig};
use regex::Regex;

lazy_static! {
    static ref ENTITY_REGEX: Regex = Regex::new(r"(?s)\{(.*?)\}").unwrap();
    static ref RELATIONSHIP_REGEX: Regex = Regex::new(r"(?s)\[(.*?)\]").unwrap();
}

fn str_to_entity_type(s: &str) -> EntityType {
    match s.to_lowercase().as_str() {
        "organization" => EntityType::Organization,
        "person" => EntityType::Person,
        "geo" => EntityType::Geo,
        "event" => EntityType::Event,
        "category" => EntityType::Category,
        _ => EntityType::Unknown
    }
}

pub struct LLMEntityExtractor {
    provider: Arc<Box<dyn Provider>>,
    config: EntityExtractionConfig,
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
}

impl LLMEntityExtractor {
    pub fn new(provider: Arc<Box<dyn Provider>>, config: EntityExtractionConfig) -> Self {
        Self {
            provider,
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn generate_entity_extraction_prompt(&self, text: &str) -> String {
        format!(
            "Extract entities and relationships from the following text. \
            Return entities in {{type: entity_type, name: entity_name, description: description}} format \
            and relationships in [src: src_entity, tgt: tgt_entity, type: relationship_type, description: description] format.\n\n\
            Text: {}\n\n\
            Entities and Relationships:",
            text
        )
    }

    fn parse_entities(&self, response: &str) -> Result<(Vec<Entity>, Vec<Relationship>), Error> {
        let mut entities = Vec::new();
        let mut relationships = Vec::new();

        // Parse entities (in curly braces)
        for cap in ENTITY_REGEX.captures_iter(response) {
            let entity_text = cap[1].trim();
            let mut fields = entity_text.split(',').map(|s| s.trim());
            let mut entity_type = EntityType::Unknown;
            let mut name = String::new();
            let mut description = String::new();
            
            while let Some(field) = fields.next() {
                let mut kv = field.splitn(2, ':').map(|s| s.trim());
                if let (Some(key), Some(value)) = (kv.next(), kv.next()) {
                    match key.to_lowercase().as_str() {
                        "type" => entity_type = str_to_entity_type(value),
                        "name" => name = value.to_string(),
                        "description" => description = value.to_string(),
                        _ => {}
                    }
                }
            }
            
            if !name.is_empty() {
                entities.push(Entity {
                    entity_type,
                    name: name.to_uppercase(),
                    description,
                    source_id: uuid::Uuid::new_v4().to_string(),
                    metadata: std::collections::HashMap::new(),
                });
            }
        }

        // Parse relationships (in square brackets)
        for cap in RELATIONSHIP_REGEX.captures_iter(response) {
            let rel_text = cap[1].trim();
            let mut fields = rel_text.split(',').map(|s| s.trim());
            let mut src_id = String::new();
            let mut tgt_id = String::new();
            let mut description = String::new();
            let mut keywords = String::new();
            
            while let Some(field) = fields.next() {
                let mut kv = field.splitn(2, ':').map(|s| s.trim());
                if let (Some(key), Some(value)) = (kv.next(), kv.next()) {
                    match key.to_lowercase().as_str() {
                        "src" => src_id = value.to_string(),
                        "tgt" => tgt_id = value.to_string(),
                        "type" => keywords = value.to_string(),
                        "description" => description = value.to_string(),
                        _ => {}
                    }
                }
            }
            
            if !src_id.is_empty() && !tgt_id.is_empty() {
                relationships.push(Relationship {
                    src_id: src_id.to_uppercase(),
                    tgt_id: tgt_id.to_uppercase(),
                    description,
                    keywords,
                    weight: 0.9, // Default weight for now
                    source_id: uuid::Uuid::new_v4().to_string(),
                    metadata: std::collections::HashMap::new(),
                });
            }
        }

        Ok((entities, relationships))
    }

    fn parse_relationships(&self, response: &str) -> Vec<Relationship> {
        if response.contains("\"relationships\"") {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(response) {
                if let Some(val) = json.get("relationships") {
                    if let Ok(rels) = serde_json::from_value(val.clone()) {
                        return rels;
                    }
                }
            }
            return Vec::new();
        }
        let mut relationships = Vec::new();
        let overall_source_id = uuid::Uuid::new_v4().to_string();

        for cap in RELATIONSHIP_REGEX.captures_iter(response) {
            let rel_text = cap[1].trim();
            let count_src = rel_text.matches("src:").count();
            if count_src > 1 {
                // Multiple 'src:' occurrences in one bracket, split on "src:" and process each
                for sub in rel_text.split("src:").skip(1) {
                    let sub_rel_text = format!("src:{}", sub).trim().to_string();
                    if let Some(rel) = self.parse_single_relationship(&sub_rel_text, &overall_source_id) {
                        relationships.push(rel);
                    }
                }
            } else if rel_text.contains(";") || rel_text.contains("\n") {
                // Relationships delimited by semicolon or newline
                let delimiter = if rel_text.contains(";") { ";" } else { "\n" };
                for sub in rel_text.split(delimiter) {
                    let sub = sub.trim();
                    if !sub.is_empty() {
                        let formatted = if sub.to_lowercase().starts_with("src:") {
                            sub.to_string()
                        } else {
                            format!("src:{}", sub)
                        };
                        if let Some(rel) = self.parse_single_relationship(&formatted, &overall_source_id) {
                            relationships.push(rel);
                        }
                    }
                }
            } else {
                if let Some(rel) = self.parse_single_relationship(rel_text, &overall_source_id) {
                    relationships.push(rel);
                }
            }
        }
        relationships
    }

    fn parse_single_relationship(&self, rel_text: &str, source_id: &str) -> Option<Relationship> {
        let mut parts = rel_text.split(',').map(|s| s.trim());
        let mut src = String::new();
        let mut tgt = String::new();
        let mut rel_type = String::new();
        let mut description = String::new();
        let mut weight = 0.9; // default weight set to 0.9 per test expectation

        while let Some(part) = parts.next() {
            let mut kv = part.splitn(2, ':').map(|s| s.trim());
            if let (Some(key), Some(value)) = (kv.next(), kv.next()) {
                match key.to_lowercase().as_str() {
                    "src" => src = value.to_string(),
                    "tgt" => tgt = value.to_string(),
                    "type" => rel_type = value.to_string(),
                    "description" => description = value.to_string(),
                    "weight" => weight = value.parse().unwrap_or(0.9),
                    _ => {}
                }
            }
        }

        if !src.is_empty() && !tgt.is_empty() {
            Some(Relationship {
                src_id: src,
                tgt_id: tgt,
                description,
                keywords: rel_type,
                weight,
                source_id: source_id.to_string(),
                metadata: HashMap::new(),
            })
        } else {
            None
        }
    }

    async fn check_cache(&self, prompt: &str) -> Option<CacheEntry> {
        let cache = self.cache.read().await;
        if let Some(entry) = cache.get(prompt) {
            if !entry.is_expired() {
                Some(entry.clone())
            } else {
                None
            }
        } else {
            None
        }
    }

    async fn update_cache(&self, prompt: &str, response: LLMResponse) {
        let mut cache = self.cache.write().await;
        let entry = CacheEntry::new(
            response,
            Some(Duration::from_secs(3600)), // 1 hour TTL
            Some(CacheType::Memory)
        );
        cache.insert(prompt.to_string(), entry);
    }

    async fn extract_entities_impl(&self, text: &str) -> Result<(Vec<Entity>, Vec<Relationship>), Error> {
        if text.trim().is_empty() {
            return Err(Error::InvalidInput("Empty content".to_string()));
        }

        // Check cache first
        if let Some(cached_entry) = self.check_cache(text).await {
            let (entities, relationships) = self.parse_entities(&cached_entry.response.text)?;
            return Ok((entities, relationships));
        }

        // Generate prompt
        let prompt = self.generate_entity_extraction_prompt(text);

        // Initialize LLM parameters
        let params = LLMParams {
            model: self.get_config().extra_params.get("model")
                .map(|s| s.to_string())
                .unwrap_or_else(|| "gpt-4".to_string()),
            max_tokens: 1000,
            temperature: 0.1,
            top_p: 1.0,
            stream: false,
            system_prompt: None,
            query_params: None,
            extra_params: std::collections::HashMap::new(),
        };

        // Call LLM for initial extraction
        let mut response = self.provider.complete(&prompt, &params)
            .await
            .map_err(|e| Error::Api(format!("LLM error: {}", e)))?;

        let mut combined_text = response.text.clone();

        // Gleaning loop: if max_gleaning_attempts > 1, perform additional LLM calls
        for _ in 1..self.config.max_gleaning_attempts {
            // Use a simple glean prompt
            let glean_prompt = "Add missing entities";
            let glean_response = self.provider.complete(glean_prompt, &params)
                .await
                .map_err(|e| Error::Api(format!("LLM error on gleaning: {}", e)))?;
            if glean_response.text.trim().is_empty() {
                break;
            }
            combined_text.push_str("\n");
            combined_text.push_str(&glean_response.text);
        }

        // Update cache with combined response
        let combined_response = LLMResponse {
            text: combined_text.clone(),
            tokens_used: 0,
            model: String::new(),
            cached: false,
            context: None,
            metadata: std::collections::HashMap::new(),
        };
        self.update_cache(text, combined_response.clone()).await;

        // Parse combined response
        let (entities, relationships) = self.parse_entities(&combined_text)?;
        Ok((entities, relationships))
    }
}

#[async_trait::async_trait]
impl EntityExtractor for LLMEntityExtractor {
    async fn extract_entities(&self, text: &str) -> Result<(Vec<Entity>, Vec<Relationship>), Error> {
        self.extract_entities_impl(text).await
    }

    fn get_config(&self) -> &EntityExtractionConfig {
        &self.config
    }
} 