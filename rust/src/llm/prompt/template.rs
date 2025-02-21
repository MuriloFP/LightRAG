use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use std::path::Path;
use tokio::fs;
use serde_json;

use super::{PromptTemplate, PromptError, PromptConfig};

/// Manager for prompt templates
pub struct TemplateManager {
    /// Template storage
    templates: Arc<RwLock<HashMap<String, PromptTemplate>>>,
    
    /// Template cache
    formatted_cache: Arc<RwLock<HashMap<String, String>>>,
    
    /// Configuration
    config: PromptConfig,
}

impl TemplateManager {
    /// Create a new template manager
    pub fn new(config: PromptConfig) -> Self {
        Self {
            templates: Arc::new(RwLock::new(HashMap::new())),
            formatted_cache: Arc::new(RwLock::new(HashMap::new())),
            config,
        }
    }
    
    /// Load templates from a directory
    pub async fn load_templates(&self, dir: impl AsRef<Path>) -> Result<(), PromptError> {
        let dir = dir.as_ref();
        if !dir.exists() {
            return Err(PromptError::ValidationError(format!("Directory not found: {}", dir.display())));
        }
        
        let mut entries = fs::read_dir(dir).await
            .map_err(|e| PromptError::ValidationError(format!("Failed to read directory: {}", e)))?;
            
        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                let content = fs::read_to_string(&path).await
                    .map_err(|e| PromptError::ValidationError(format!("Failed to read file: {}", e)))?;
                    
                let template: PromptTemplate = serde_json::from_str(&content)
                    .map_err(|e| PromptError::InvalidFormat(format!("Invalid template format: {}", e)))?;
                    
                if self.config.validate_on_load {
                    template.validate()?;
                }
                
                let mut templates = self.templates.write().await;
                templates.insert(template.id.clone(), template);
            }
        }
        
        Ok(())
    }
    
    /// Add a template
    pub async fn add_template(&self, template: PromptTemplate) -> Result<(), PromptError> {
        if self.config.validate_on_load {
            template.validate()?;
        }
        
        let mut templates = self.templates.write().await;
        templates.insert(template.id.clone(), template);
        Ok(())
    }
    
    /// Get a template by ID
    pub async fn get_template(&self, id: &str) -> Option<PromptTemplate> {
        let templates = self.templates.read().await;
        templates.get(id).cloned()
    }
    
    /// Format a template with variables
    pub async fn format_template(
        &self,
        id: &str,
        variables: &HashMap<String, String>
    ) -> Result<String, PromptError> {
        // Check cache first
        if self.config.use_template_cache {
            let cache_key = format!("{}:{}", id, serde_json::to_string(variables).unwrap_or_default());
            let cache = self.formatted_cache.read().await;
            if let Some(cached) = cache.get(&cache_key) {
                return Ok(cached.clone());
            }
        }
        
        // Get and format template
        let template = self.get_template(id).await
            .ok_or_else(|| PromptError::TemplateNotFound(id.to_string()))?;
            
        // Merge default variables with provided ones
        let mut merged_vars = self.config.default_variables.clone();
        merged_vars.extend(variables.clone());
        
        let formatted = template.format(&merged_vars)?;
        
        // Cache the result
        if self.config.use_template_cache {
            let cache_key = format!("{}:{}", id, serde_json::to_string(variables).unwrap_or_default());
            let mut cache = self.formatted_cache.write().await;
            
            // Check cache size limit
            if let Some(max_size) = self.config.max_cache_size {
                if cache.len() >= max_size {
                    // Remove oldest entry (first key)
                    if let Some(first_key) = cache.keys().next().cloned() {
                        cache.remove(&first_key);
                    }
                }
            }
            
            cache.insert(cache_key, formatted.clone());
        }
        
        Ok(formatted)
    }
    
    /// Remove a template
    pub async fn remove_template(&self, id: &str) -> Option<PromptTemplate> {
        let mut templates = self.templates.write().await;
        templates.remove(id)
    }
    
    /// Clear all templates
    pub async fn clear_templates(&self) {
        let mut templates = self.templates.write().await;
        templates.clear();
    }
    
    /// Clear the format cache
    pub async fn clear_cache(&self) {
        let mut cache = self.formatted_cache.write().await;
        cache.clear();
    }
    
    /// Get the current configuration
    pub fn get_config(&self) -> &PromptConfig {
        &self.config
    }
    
    /// Update the configuration
    pub fn update_config(&mut self, config: PromptConfig) {
        self.config = config;
    }
} 