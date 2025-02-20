use std::path::Path;
use thiserror::Error;
use tracing::error;
use docx_rs::read_docx;
use pdf_extract::extract_text as extract_pdf_text;
use serde_json;
use std::fs;

/// Supported document formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    /// Plain text files (.txt)
    PlainText,
    /// Markdown files (.md)
    Markdown,
    /// PDF files (.pdf)
    Pdf,
    /// Word documents (.docx)
    Word,
}

/// Errors that can occur during format handling
#[derive(Error, Debug)]
pub enum FormatError {
    /// Error when file format is not supported
    #[error("Unsupported file format: {0}")]
    UnsupportedFormat(String),
    
    /// Error when reading file
    #[error("File read error: {0}")]
    FileReadError(#[from] std::io::Error),
    
    /// Error during format-specific processing
    #[error("Processing error: {0}")]
    ProcessingError(String),
    
    /// Error when file extension is missing
    #[error("Missing file extension")]
    MissingExtension,
}

/// Trait for format-specific document handlers
#[async_trait::async_trait]
pub trait FormatHandler: Send + Sync {
    /// Extract text content from a file
    async fn extract_text(&self, file_path: &Path) -> Result<String, FormatError>;
    
    /// Get supported file extensions
    fn supported_extensions(&self) -> Vec<&'static str>;
}

/// Detect document format from file extension
pub fn detect_format(file_path: &Path) -> Result<DocumentFormat, FormatError> {
    let extension = file_path
        .extension()
        .and_then(|ext| ext.to_str())
        .ok_or(FormatError::MissingExtension)?;

    match extension.to_lowercase().as_str() {
        "txt" => Ok(DocumentFormat::PlainText),
        "md" | "markdown" => Ok(DocumentFormat::Markdown),
        "pdf" => Ok(DocumentFormat::Pdf),
        "docx" => Ok(DocumentFormat::Word),
        _ => Err(FormatError::UnsupportedFormat(extension.to_string())),
    }
}

/// Get appropriate format handler for document type
pub fn get_format_handler(format: DocumentFormat) -> Box<dyn FormatHandler> {
    match format {
        DocumentFormat::PlainText | DocumentFormat::Markdown => Box::new(TextHandler),
        DocumentFormat::Pdf => Box::new(PdfHandler),
        DocumentFormat::Word => Box::new(WordHandler),
    }
}

/// Handler for plain text and markdown files
pub struct TextHandler;

#[async_trait::async_trait]
impl FormatHandler for TextHandler {
    async fn extract_text(&self, file_path: &Path) -> Result<String, FormatError> {
        fs::read_to_string(file_path)
            .map_err(|e| FormatError::FileReadError(e))
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["txt", "md", "markdown"]
    }
}

/// Handler for PDF files
pub struct PdfHandler;

#[async_trait::async_trait]
impl FormatHandler for PdfHandler {
    async fn extract_text(&self, file_path: &Path) -> Result<String, FormatError> {
        extract_pdf_text(file_path)
            .map_err(|e| FormatError::ProcessingError(format!("PDF extraction error: {}", e)))
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["pdf"]
    }
}

/// Handler for Word documents
pub struct WordHandler;

#[async_trait::async_trait]
impl FormatHandler for WordHandler {
    async fn extract_text(&self, file_path: &Path) -> Result<String, FormatError> {
        // Read the file into a byte vector
        let content = std::fs::read(file_path)
            .map_err(|e| FormatError::FileReadError(e))?;
        
        // Parse the document and convert to JSON
        let docx = read_docx(&content)
            .map_err(|e| FormatError::ProcessingError(format!("Word document parsing error: {}", e)))?;
        
        // Convert to JSON and parse it
        let json = docx.json();
        let json_value: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| FormatError::ProcessingError(format!("JSON parsing error: {}", e)))?;
            
        // Extract text from paragraphs
        let mut text = String::new();
        
        // Navigate to document.children array
        if let Some(document) = json_value.get("document") {
            if let Some(children) = document.get("children").and_then(|v| v.as_array()) {
                for paragraph in children {
                    if let Some(para_data) = paragraph.get("data") {
                        if let Some(para_children) = para_data.get("children").and_then(|v| v.as_array()) {
                            for run in para_children {
                                if let Some(run_data) = run.get("data") {
                                    if let Some(run_children) = run_data.get("children").and_then(|v| v.as_array()) {
                                        for text_elem in run_children {
                                            if let Some(text_data) = text_elem.get("data") {
                                                if let Some(content) = text_data.get("text").and_then(|v| v.as_str()) {
                                                    text.push_str(content);
                                                    text.push(' ');
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            text.push('\n');
                        }
                    }
                }
            }
        }
        
        Ok(text)
    }

    fn supported_extensions(&self) -> Vec<&'static str> {
        vec!["docx"]
    }
} 