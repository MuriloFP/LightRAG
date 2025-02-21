use super_lightrag::processing::{
    ValidationConfig,
    ValidationSeverity,
    ValidationIssueType,
    ContentValidator,
    LengthValidator,
    EncodingValidator,
    CompositeValidator,
};

#[test]
fn test_length_validator() {
    let config = ValidationConfig {
        min_length: 10,
        max_length: 100,
        ..Default::default()
    };
    let validator = LengthValidator::new(config);

    // Test content within limits
    let content = "This is a valid length text";
    let result = validator.validate(content).unwrap();
    assert!(result.is_valid);
    assert!(result.issues.is_empty());
    assert_eq!(result.metadata["content_length"].as_u64().unwrap(), content.len() as u64);

    // Test content too short
    let content = "Too short";
    let result = validator.validate(content).unwrap();
    assert!(!result.is_valid);
    assert_eq!(result.issues.len(), 1);
    assert_eq!(result.issues[0].issue_type, ValidationIssueType::Length);
    assert_eq!(result.issues[0].severity, ValidationSeverity::Error);

    // Test content too long
    let content = &"x".repeat(101);
    let result = validator.validate(content).unwrap();
    assert!(!result.is_valid);
    assert_eq!(result.issues.len(), 1);
    assert_eq!(result.issues[0].issue_type, ValidationIssueType::Length);
    assert_eq!(result.issues[0].severity, ValidationSeverity::Error);
}

#[test]
fn test_encoding_validator() {
    let config = ValidationConfig {
        strict_encoding: true,
        ..Default::default()
    };
    let validator = EncodingValidator::new(config);

    // Test valid UTF-8 content
    let content = "Hello, 世界! 🌍";
    let result = validator.validate(content).unwrap();
    assert!(result.is_valid);
    assert!(result.issues.is_empty());
    assert_eq!(result.metadata["encoding"].as_str().unwrap(), "UTF-8");

    // Test content with control characters
    let content = "Hello\x00World";
    let result = validator.validate(content).unwrap();
    assert!(!result.is_valid);
    assert_eq!(result.issues.len(), 1);
    assert_eq!(result.issues[0].issue_type, ValidationIssueType::Encoding);
    assert_eq!(result.issues[0].severity, ValidationSeverity::Error);

    // Test with strict encoding disabled
    let config = ValidationConfig {
        strict_encoding: false,
        ..Default::default()
    };
    let validator = EncodingValidator::new(config);
    let result = validator.validate(content).unwrap();
    assert!(result.is_valid);
    assert!(result.issues.is_empty());
}

#[test]
fn test_composite_validator() {
    let length_config = ValidationConfig {
        min_length: 10,
        max_length: 100,
        ..Default::default()
    };
    let encoding_config = ValidationConfig {
        strict_encoding: true,
        ..Default::default()
    };

    let validators: Vec<Box<dyn ContentValidator>> = vec![
        Box::new(LengthValidator::new(length_config)),
        Box::new(EncodingValidator::new(encoding_config)),
    ];

    let composite = CompositeValidator::new(validators, ValidationConfig::default());

    // Test valid content
    let content = "This is a valid text with proper encoding";
    let result = composite.validate(content).unwrap();
    assert!(result.is_valid);
    assert!(result.issues.is_empty());
    assert!(result.metadata.contains_key("content_length"));
    assert!(result.metadata.contains_key("encoding"));

    // Test content with multiple issues
    let content = "x\x00"; // Too short and has control character
    let result = composite.validate(content).unwrap();
    assert!(!result.is_valid);
    assert_eq!(result.issues.len(), 2);
    
    // Verify both length and encoding issues are present
    let has_length_issue = result.issues.iter()
        .any(|issue| issue.issue_type == ValidationIssueType::Length);
    let has_encoding_issue = result.issues.iter()
        .any(|issue| issue.issue_type == ValidationIssueType::Encoding);
    assert!(has_length_issue);
    assert!(has_encoding_issue);
}

#[test]
fn test_validation_config_defaults() {
    let config = ValidationConfig::default();
    assert_eq!(config.min_length, 1);
    assert_eq!(config.max_length, 1_000_000);
    assert!(config.strict_encoding);
    assert!(config.custom_rules.is_empty());
}

#[test]
fn test_validation_severity_ordering() {
    assert!(ValidationSeverity::Critical > ValidationSeverity::Error);
    assert!(ValidationSeverity::Error > ValidationSeverity::Warning);
    assert!(ValidationSeverity::Warning > ValidationSeverity::Info);
} 