use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use serde_json::Value;

/// Errors that can occur during content validation
#[derive(Error, Debug)]
pub enum ValidationError {
    /// Error when content length exceeds maximum
    #[error("Content length {length} exceeds maximum {max}")]
    ContentTooLong {
        /// Actual length of the content
        length: usize,
        /// Maximum allowed length
        max: usize,
    },
    
    /// Error when content length is below minimum
    #[error("Content length {length} below minimum {min}")]
    ContentTooShort {
        /// Actual length of the content
        length: usize,
        /// Minimum required length
        min: usize,
    },
    
    /// Error when content has invalid UTF-8 encoding
    #[error("Invalid UTF-8 encoding: {0}")]
    InvalidEncoding(String),
    
    /// Error when content is malformed
    #[error("Malformed content: {0}")]
    MalformedContent(String),
    
    /// Error when content validation fails
    #[error("Validation failed: {0}")]
    ValidationFailed(String),
}

/// Result of content validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether the content is valid
    pub is_valid: bool,
    /// List of validation issues found
    pub issues: Vec<ValidationIssue>,
    /// Additional metadata about the validation
    pub metadata: HashMap<String, Value>,
}

/// Individual validation issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationIssue {
    /// Type of validation issue
    pub issue_type: ValidationIssueType,
    /// Description of the issue
    pub description: String,
    /// Severity level of the issue
    pub severity: ValidationSeverity,
    /// Location or context of the issue
    pub location: Option<String>,
}

/// Types of validation issues
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ValidationIssueType {
    /// Content length issues
    Length,
    /// Character encoding issues
    Encoding,
    /// Content structure issues
    Structure,
    /// Content format issues
    Format,
    /// Custom validation issues
    Custom(String),
}

/// Severity levels for validation issues
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationSeverity {
    /// Information only, not an error
    Info,
    /// Warning, but content can still be processed
    Warning,
    /// Error that should be addressed
    Error,
    /// Critical error that prevents processing
    Critical,
}

/// Configuration for content validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Minimum allowed content length
    pub min_length: usize,
    /// Maximum allowed content length
    pub max_length: usize,
    /// Whether to enforce strict UTF-8 encoding
    pub strict_encoding: bool,
    /// Custom validation rules
    pub custom_rules: HashMap<String, Value>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            min_length: 1,
            max_length: 1_000_000, // 1MB default
            strict_encoding: true,
            custom_rules: HashMap::new(),
        }
    }
}

/// Trait for content validators
pub trait ContentValidator: Send + Sync {
    /// Validate content according to validator's rules
    fn validate(&self, content: &str) -> Result<ValidationResult, ValidationError>;
    
    /// Get validator's configuration
    fn get_config(&self) -> &ValidationConfig;
}

/// Basic length validator
pub struct LengthValidator {
    config: ValidationConfig,
}

impl LengthValidator {
    /// Create a new length validator with given configuration
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }
}

impl ContentValidator for LengthValidator {
    fn validate(&self, content: &str) -> Result<ValidationResult, ValidationError> {
        let length = content.len();
        let mut issues = Vec::new();
        
        if length < self.config.min_length {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::Length,
                description: format!("Content length {} is below minimum {}", length, self.config.min_length),
                severity: ValidationSeverity::Error,
                location: None,
            });
        }
        
        if length > self.config.max_length {
            issues.push(ValidationIssue {
                issue_type: ValidationIssueType::Length,
                description: format!("Content length {} exceeds maximum {}", length, self.config.max_length),
                severity: ValidationSeverity::Error,
                location: None,
            });
        }
        
        let is_valid = issues.is_empty();
        let mut metadata = HashMap::new();
        metadata.insert("content_length".to_string(), Value::Number(length.into()));
        
        Ok(ValidationResult {
            is_valid,
            issues,
            metadata,
        })
    }
    
    fn get_config(&self) -> &ValidationConfig {
        &self.config
    }
}

/// UTF-8 encoding validator
pub struct EncodingValidator {
    config: ValidationConfig,
}

impl EncodingValidator {
    /// Create a new encoding validator with given configuration
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }
    
    /// Check if a string is valid UTF-8 and find invalid sequences
    fn find_invalid_utf8(&self, content: &str) -> Option<(usize, Vec<u8>)> {
        // Since the input is already a &str, it's guaranteed to be valid UTF-8
        // This method is for finding specific encoding issues or non-standard characters
        let bytes = content.as_bytes();
        for (i, &byte) in bytes.iter().enumerate() {
            // Check for control characters (except whitespace)
            if byte < 32 && ![9, 10, 13].contains(&byte) {
                return Some((i, vec![byte]));
            }
        }
        None
    }
}

impl ContentValidator for EncodingValidator {
    fn validate(&self, content: &str) -> Result<ValidationResult, ValidationError> {
        let mut issues = Vec::new();
        
        if self.config.strict_encoding {
            if let Some((pos, bytes)) = self.find_invalid_utf8(content) {
                issues.push(ValidationIssue {
                    issue_type: ValidationIssueType::Encoding,
                    description: format!("Invalid character sequence at position {}: {:?}", pos, bytes),
                    severity: ValidationSeverity::Error,
                    location: Some(format!("position {}", pos)),
                });
            }
        }
        
        let is_valid = issues.is_empty();
        let mut metadata = HashMap::new();
        metadata.insert("encoding".to_string(), Value::String("UTF-8".to_string()));
        
        Ok(ValidationResult {
            is_valid,
            issues,
            metadata,
        })
    }
    
    fn get_config(&self) -> &ValidationConfig {
        &self.config
    }
}

/// Composite validator that combines multiple validators
pub struct CompositeValidator {
    validators: Vec<Box<dyn ContentValidator>>,
    config: ValidationConfig,
}

impl CompositeValidator {
    /// Create a new composite validator with given validators
    pub fn new(validators: Vec<Box<dyn ContentValidator>>, config: ValidationConfig) -> Self {
        Self { validators, config }
    }
}

impl ContentValidator for CompositeValidator {
    fn validate(&self, content: &str) -> Result<ValidationResult, ValidationError> {
        let mut all_issues = Vec::new();
        let mut combined_metadata = HashMap::new();
        
        for validator in &self.validators {
            match validator.validate(content) {
                Ok(result) => {
                    all_issues.extend(result.issues);
                    combined_metadata.extend(result.metadata);
                }
                Err(e) => {
                    return Err(ValidationError::ValidationFailed(format!(
                        "Validator failed: {}",
                        e
                    )));
                }
            }
        }
        
        // Consider valid if no critical or error issues
        let is_valid = !all_issues.iter().any(|issue| {
            matches!(
                issue.severity,
                ValidationSeverity::Critical | ValidationSeverity::Error
            )
        });
        
        Ok(ValidationResult {
            is_valid,
            issues: all_issues,
            metadata: combined_metadata,
        })
    }
    
    fn get_config(&self) -> &ValidationConfig {
        &self.config
    }
} 