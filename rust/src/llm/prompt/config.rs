use std::collections::HashMap;
use std::path::PathBuf;
use serde::{Deserialize, Serialize};

/// Configuration for prompt template management
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptConfig {
    /// Default language for templates
    pub default_language: String,
    
    /// Path to template directory
    pub template_dir: Option<PathBuf>,
    
    /// Whether to validate templates on load
    pub validate_on_load: bool,
    
    /// Whether to cache formatted templates
    pub use_template_cache: bool,
    
    /// Maximum cache size
    pub max_cache_size: Option<usize>,
    
    /// Default variable values
    pub default_variables: HashMap<String, String>,
    
    /// Additional configuration parameters
    pub extra_config: HashMap<String, String>,
}

impl Default for PromptConfig {
    fn default() -> Self {
        Self {
            default_language: "English".to_string(),
            template_dir: None,
            validate_on_load: true,
            use_template_cache: true,
            max_cache_size: Some(1000),
            default_variables: HashMap::new(),
            extra_config: HashMap::new(),
        }
    }
}

impl PromptConfig {
    /// Create a new configuration with custom settings
    pub fn new(
        default_language: impl Into<String>,
        template_dir: Option<PathBuf>,
        validate_on_load: bool,
    ) -> Self {
        Self {
            default_language: default_language.into(),
            template_dir,
            validate_on_load,
            ..Default::default()
        }
    }
    
    /// Set a default variable value
    pub fn set_default_variable(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.default_variables.insert(name.into(), value.into());
    }
    
    /// Get a default variable value
    pub fn get_default_variable(&self, name: &str) -> Option<&String> {
        self.default_variables.get(name)
    }
    
    /// Set an extra configuration value
    pub fn set_extra_config(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.extra_config.insert(key.into(), value.into());
    }
    
    /// Get an extra configuration value
    pub fn get_extra_config(&self, key: &str) -> Option<&String> {
        self.extra_config.get(key)
    }
} 