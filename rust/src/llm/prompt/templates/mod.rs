pub mod rag;
pub mod entity;
pub mod summary;

pub use rag::*;
pub use entity::*;
pub use summary::*;

use super::{PromptTemplate, VariableDefinition, VariableType};

/// Create a standard set of templates
pub fn create_default_templates() -> Vec<PromptTemplate> {
    vec![
        rag::create_rag_template(),
        rag::create_naive_rag_template(),
        rag::create_mix_rag_template(),
        entity::create_entity_extraction_template(),
        summary::create_summary_template(),
    ]
} 