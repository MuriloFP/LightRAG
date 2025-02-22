use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use async_trait::async_trait;

use crate::types::Error;
use crate::processing::keywords::ConversationTurn;

/// Errors that can occur during entity extraction
#[derive(thiserror::Error, Debug)]
pub enum EntityError {
    /// Error when content is empty
    #[error("Empty content")]
    EmptyContent,
    
    /// Error during LLM processing
    #[error("LLM error: {0}")]
    LLMError(String),
    
    /// Error during extraction
    #[error("Extraction error: {0}")]
    ExtractionError(String),
    
    /// Error during storage operations
    #[error("Storage error: {0}")]
    StorageError(String),
}

impl From<EntityError> for Error {
    fn from(err: EntityError) -> Self {
        Error::Storage(err.to_string())
    }
}

/// Entity types supported by the system
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EntityType {
    Organization,
    Person,
    Geo,
    Event,
    Category,
    Unknown,
}

impl EntityType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Organization => "organization",
            Self::Person => "person",
            Self::Geo => "geo",
            Self::Event => "event",
            Self::Category => "category",
            Self::Unknown => "unknown",
        }
    }
}

/// Represents an extracted entity with its properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Name of the entity (unique identifier)
    pub name: String,
    
    /// Type of the entity
    pub entity_type: EntityType,
    
    /// Description of the entity
    pub description: String,
    
    /// Source document/chunk ID
    pub source_id: String,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Represents a relationship between two entities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    /// Source entity ID
    pub src_id: String,
    
    /// Target entity ID
    pub tgt_id: String,
    
    /// Description of the relationship
    pub description: String,
    
    /// Keywords characterizing the relationship
    pub keywords: String,
    
    /// Relationship strength (0.0 to 1.0)
    pub weight: f32,
    
    /// Source document/chunk ID
    pub source_id: String,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Configuration for entity extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionConfig {
    /// Maximum number of extraction attempts for ambiguous content
    pub max_gleaning_attempts: usize,
    
    /// Language for extraction
    pub language: String,
    
    /// Types of entities to extract
    pub entity_types: Vec<EntityType>,
    
    /// Whether to use caching
    pub use_cache: bool,
    
    /// Threshold for cache similarity matching
    pub cache_similarity_threshold: f32,
    
    /// Additional parameters
    pub extra_params: HashMap<String, String>,
}

impl Default for EntityExtractionConfig {
    fn default() -> Self {
        Self {
            max_gleaning_attempts: 1,
            language: "English".to_string(),
            entity_types: vec![
                EntityType::Organization,
                EntityType::Person,
                EntityType::Geo,
                EntityType::Event,
                EntityType::Category,
            ],
            use_cache: true,
            cache_similarity_threshold: 0.95,
            extra_params: HashMap::new(),
        }
    }
}

/// Trait for entity extraction implementations
#[async_trait]
pub trait EntityExtractor: Send + Sync {
    /// Extract entities and relationships from content
    async fn extract_entities(&self, content: &str) -> Result<(Vec<Entity>, Vec<Relationship>), Error>;
    
    /// Extract entities and relationships with conversation history
    async fn extract_with_history(
        &self,
        content: &str,
        history: &[ConversationTurn],
    ) -> Result<(Vec<Entity>, Vec<Relationship>), Error> {
        // Default implementation falls back to regular extraction
        self.extract_entities(content).await
    }
    
    /// Get the current configuration
    fn get_config(&self) -> &EntityExtractionConfig;
}

// Re-export implementations
mod llm;
pub use llm::LLMEntityExtractor; 