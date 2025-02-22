use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CacheType {
    /// For query results
    Query,
    /// For entity extraction results
    Extract,
    /// For keyword extraction results
    Keywords,
    /// For custom cache types
    Custom(String),
}

impl CacheType {
    pub fn as_str(&self) -> String {
        match self {
            CacheType::Query => "query".to_string(),
            CacheType::Extract => "extract".to_string(),
            CacheType::Keywords => "keywords".to_string(),
            CacheType::Custom(s) => s.clone(),
        }
    }
}

impl Default for CacheType {
    fn default() -> Self {
        CacheType::Query
    }
} 