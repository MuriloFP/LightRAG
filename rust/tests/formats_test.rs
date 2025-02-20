use std::path::Path;
use tempfile::tempdir;
use tokio_test::block_on;

use super_lightrag::processing::{
    DocumentFormat,
    FormatHandler,
    detect_format,
    get_format_handler,
    formats::{TextHandler, PdfHandler, WordHandler},
};

#[test]
fn test_detect_format() {
    let txt_path = Path::new("test.txt");
    let md_path = Path::new("test.md");
    let pdf_path = Path::new("test.pdf");
    let docx_path = Path::new("test.docx");
    let unknown_path = Path::new("test.xyz");

    assert_eq!(detect_format(txt_path).unwrap(), DocumentFormat::PlainText);
    assert_eq!(detect_format(md_path).unwrap(), DocumentFormat::Markdown);
    assert_eq!(detect_format(pdf_path).unwrap(), DocumentFormat::Pdf);
    assert_eq!(detect_format(docx_path).unwrap(), DocumentFormat::Word);
    assert!(detect_format(unknown_path).is_err());
}

#[test]
fn test_text_handler() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.txt");
    let content = "Hello, World!\nThis is a test.";

    // Create test file
    std::fs::write(&file_path, content).unwrap();

    let handler = TextHandler;
    let result = block_on(handler.extract_text(&file_path)).unwrap();
    assert_eq!(result, content);
}

#[test]
fn test_markdown_handler() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("test.md");
    let content = "# Title\n\nThis is a *markdown* test.";

    // Create test file
    std::fs::write(&file_path, content).unwrap();

    let handler = TextHandler;
    let result = block_on(handler.extract_text(&file_path)).unwrap();
    assert_eq!(result, content);
}

#[test]
fn test_pdf_handler() {
    let file_path = Path::new("tests/resources/test.pdf");
    let handler = PdfHandler;
    let result = block_on(handler.extract_text(&file_path)).unwrap();
    
    // Verify the extracted text contains expected content
    assert!(result.contains("Test PDF Document"));
    assert!(result.contains("Multiple paragraphs"));
    assert!(result.contains("Special characters"));
    assert!(result.contains("Numbers"));
}

#[test]
fn test_word_handler() {
    let file_path = Path::new("tests/resources/test.docx");
    let handler = WordHandler;
    let result = block_on(handler.extract_text(&file_path)).unwrap();
    
    // Print the extracted content for debugging
    println!("Extracted content:\n{}", result);
    
    // Verify the extracted text contains expected content
    assert!(result.contains("Test Word Document"));
    assert!(result.contains("multiple paragraphs"));
    assert!(result.contains("formatting"));
    assert!(result.contains("Bullet points"));
}

#[test]
fn test_get_format_handler() {
    let txt_handler = get_format_handler(DocumentFormat::PlainText);
    let md_handler = get_format_handler(DocumentFormat::Markdown);
    let pdf_handler = get_format_handler(DocumentFormat::Pdf);
    let word_handler = get_format_handler(DocumentFormat::Word);

    assert!(txt_handler.supported_extensions().contains(&"txt"));
    assert!(md_handler.supported_extensions().contains(&"md"));
    assert!(pdf_handler.supported_extensions().contains(&"pdf"));
    assert!(word_handler.supported_extensions().contains(&"docx"));
} 