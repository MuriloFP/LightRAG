use html_escape;
use regex::Regex;
use thiserror::Error;

/// Errors that can occur during content cleaning operations
#[derive(Error, Debug)]
pub enum CleaningError {
    /// Error when invalid UTF-8 sequence is detected in the input
    #[error("Invalid UTF-8 sequence detected")]
    InvalidUtf8,
    /// Error when regex compilation or execution fails
    #[error("Regex error: {0}")]
    RegexError(#[from] regex::Error),
    /// Other general cleaning errors
    #[error("Content cleaning error: {0}")]
    Other(String),
}

/// Configuration for content cleaning operations
#[derive(Debug, Clone)]
pub struct CleaningConfig {
    /// Whether to remove HTML entities
    pub unescape_html: bool,
    /// Whether to remove control characters
    pub remove_control_chars: bool,
    /// Whether to normalize whitespace
    pub normalize_whitespace: bool,
}

impl Default for CleaningConfig {
    fn default() -> Self {
        Self {
            unescape_html: true,
            remove_control_chars: true,
            normalize_whitespace: true,
        }
    }
}

/// Cleans text content according to the provided configuration
///
/// # Arguments
/// * `content` - The text content to clean
/// * `config` - Configuration specifying which cleaning operations to perform
///
/// # Returns
/// * `Result<String, CleaningError>` - The cleaned text or an error
pub fn clean_text(content: &str, config: &CleaningConfig) -> Result<String, CleaningError> {
    let mut cleaned = content.to_string();

    if config.unescape_html {
        cleaned = html_escape::decode_html_entities(&cleaned).to_string();
    }

    if config.remove_control_chars {
        // Remove control characters except for newline (0x0A), tab (0x09), and carriage return (0x0D)
        let control_chars_regex = Regex::new(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")?;
        cleaned = control_chars_regex.replace_all(&cleaned, "").to_string();
    }

    if config.normalize_whitespace {
        // Normalize whitespace by splitting on any whitespace and rejoining with a single space
        cleaned = cleaned.split_whitespace().collect::<Vec<&str>>().join(" ");
    }

    Ok(cleaned)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_text_with_default_config() {
        let config = CleaningConfig::default();
        let input = "  Hello &amp; World\n\t with\r\nspaces  ";
        let result = clean_text(input, &config).unwrap();
        assert_eq!(result, "Hello & World with spaces");
    }

    #[test]
    fn test_clean_text_with_html_entities() {
        let config = CleaningConfig::default();
        let input = "&lt;div&gt;Test&lt;/div&gt;";
        let result = clean_text(input, &config).unwrap();
        assert_eq!(result, "<div>Test</div>");
    }

    #[test]
    fn test_clean_text_with_control_chars() {
        let config = CleaningConfig::default();
        let input = "Test\x00with\x1Fcontrol\x7Fchars";
        let result = clean_text(input, &config).unwrap();
        assert_eq!(result, "Testwithcontrolchars");
    }

    #[test]
    fn test_clean_text_with_custom_config() {
        let config = CleaningConfig {
            unescape_html: false,
            remove_control_chars: true,
            normalize_whitespace: true,
        };
        let input = "  &amp; Test\n\twith\r\nspaces  ";
        let result = clean_text(input, &config).unwrap();
        assert_eq!(result, "&amp; Test with spaces");
    }
} 