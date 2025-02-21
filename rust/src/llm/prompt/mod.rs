use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur during prompt operations
#[derive(Error, Debug)]
pub enum PromptError {
    /// Template not found
    #[error("Template not found: {0}")]
    TemplateNotFound(String),
    
    /// Variable not found
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    
    /// Invalid template format
    #[error("Invalid template format: {0}")]
    InvalidFormat(String),
    
    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// Template variable types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VariableType {
    /// String value
    String,
    /// Integer value
    Integer,
    /// Float value
    Float,
    /// Boolean value
    Boolean,
    /// Array of strings
    StringArray,
    /// Object (key-value pairs)
    Object,
}

/// Template variable definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariableDefinition {
    /// Variable name
    pub name: String,
    /// Variable type
    pub var_type: VariableType,
    /// Description of the variable
    pub description: String,
    /// Whether the variable is required
    pub required: bool,
    /// Default value if any (as JSON string)
    pub default: Option<String>,
}

/// Prompt template definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptTemplate {
    /// Template identifier
    pub id: String,
    /// Template description
    pub description: String,
    /// Template content with placeholders
    pub content: String,
    /// Variable definitions
    pub variables: Vec<VariableDefinition>,
    /// Default language
    pub default_language: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl PromptTemplate {
    /// Create a new prompt template
    pub fn new(
        id: impl Into<String>,
        description: impl Into<String>,
        content: impl Into<String>,
        variables: Vec<VariableDefinition>,
    ) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            content: content.into(),
            variables,
            default_language: "English".to_string(),
            metadata: HashMap::new(),
        }
    }
    
    /// Validate template format
    pub fn validate(&self) -> Result<(), PromptError> {
        // Check for empty content
        if self.content.is_empty() {
            return Err(PromptError::ValidationError("Empty template content".to_string()));
        }
        
        // Check variable placeholders
        let mut found_vars = Vec::new();
        let mut pos = 0;
        while let Some(start) = self.content[pos..].find('{') {
            if let Some(end) = self.content[pos + start..].find('}') {
                let var_name = &self.content[pos + start + 1..pos + start + end];
                found_vars.push(var_name.to_string());
            }
            pos += start + 1;
        }
        
        // Verify all required variables have placeholders
        for var in &self.variables {
            if var.required && !found_vars.contains(&var.name) {
                return Err(PromptError::ValidationError(
                    format!("Missing placeholder for required variable: {}", var.name)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Format template with variables
    pub fn format(&self, variables: &HashMap<String, String>) -> Result<String, PromptError> {
        let mut result = self.content.clone();
        
        // Check required variables
        for var in &self.variables {
            if var.required {
                if !variables.contains_key(&var.name) {
                    if let Some(default) = &var.default {
                        // Use default value
                        result = result.replace(&format!("{{{}}}", var.name), default);
                    } else {
                        return Err(PromptError::VariableNotFound(var.name.clone()));
                    }
                }
            }
        }
        
        // Replace variables
        for (name, value) in variables {
            result = result.replace(&format!("{{{}}}", name), value);
        }
        
        Ok(result)
    }
}

// Re-export submodules
pub mod config;
pub mod template;
pub mod templates;
pub mod utils;

// Re-export common types
pub use config::PromptConfig;
pub use template::TemplateManager;
pub use templates::*; 