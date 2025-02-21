use super::super::{PromptTemplate, VariableDefinition, VariableType};

/// Create the standard RAG template
pub fn create_rag_template() -> PromptTemplate {
    PromptTemplate::new(
        "rag_response",
        "Template for generating responses using RAG with local context",
        r#"---Role---

You are a helpful assistant responding to user query about Knowledge Base provided below.

---Goal---

Generate a concise response based on Knowledge Base and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Knowledge Base, and incorporating general knowledge relevant to the Knowledge Base. Do not include information not provided by Knowledge Base.

When handling relationships with timestamps:
1. Each relationship has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting relationships, consider both the semantic content and the timestamp
3. Don't automatically prefer the most recently created relationships - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Knowledge Base---
{context_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not make anything up. Do not include information not provided by the Knowledge Base."#,
        vec![
            VariableDefinition {
                name: "history".to_string(),
                var_type: VariableType::String,
                description: "Conversation history".to_string(),
                required: true,
                default: Some("No previous conversation".to_string()),
            },
            VariableDefinition {
                name: "context_data".to_string(),
                var_type: VariableType::String,
                description: "Retrieved context data".to_string(),
                required: true,
                default: None,
            },
            VariableDefinition {
                name: "response_type".to_string(),
                var_type: VariableType::String,
                description: "Desired response format".to_string(),
                required: true,
                default: Some("Multiple Paragraphs".to_string()),
            },
        ],
    )
}

/// Create the naive RAG template
pub fn create_naive_rag_template() -> PromptTemplate {
    PromptTemplate::new(
        "naive_rag_response",
        "Template for generating responses using naive RAG with direct document chunks",
        r#"---Role---

You are a helpful assistant responding to user query about Document Chunks provided below.

---Goal---

Generate a concise response based on Document Chunks and follow Response Rules, considering both the conversation history and the current query. Summarize all information in the provided Document Chunks, and incorporating general knowledge relevant to the Document Chunks. Do not include information not provided by Document Chunks.

When handling content with timestamps:
1. Each piece of content has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content and the timestamp
3. Don't automatically prefer the most recent content - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Document Chunks---
{content_data}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- If you don't know the answer, just say so.
- Do not include information not provided by the Document Chunks."#,
        vec![
            VariableDefinition {
                name: "history".to_string(),
                var_type: VariableType::String,
                description: "Conversation history".to_string(),
                required: true,
                default: Some("No previous conversation".to_string()),
            },
            VariableDefinition {
                name: "content_data".to_string(),
                var_type: VariableType::String,
                description: "Document chunks".to_string(),
                required: true,
                default: None,
            },
            VariableDefinition {
                name: "response_type".to_string(),
                var_type: VariableType::String,
                description: "Desired response format".to_string(),
                required: true,
                default: Some("Multiple Paragraphs".to_string()),
            },
        ],
    )
}

/// Create the mix RAG template
pub fn create_mix_rag_template() -> PromptTemplate {
    PromptTemplate::new(
        "mix_rag_response",
        "Template for generating responses using mixed RAG with both KG and vector data",
        r#"---Role---

You are a helpful assistant responding to user query about Data Sources provided below.

---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both the conversation history and the current query. Data sources contain two parts: Knowledge Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources, and incorporating general knowledge relevant to the Data Sources. Do not include information not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on the context
4. For time-specific queries, prioritize temporal information in the content before considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

2. From Document Chunks(DC):
{vector_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sections focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" section. Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC), in the following format: [KG/DC] Source content
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."#,
        vec![
            VariableDefinition {
                name: "history".to_string(),
                var_type: VariableType::String,
                description: "Conversation history".to_string(),
                required: true,
                default: Some("No previous conversation".to_string()),
            },
            VariableDefinition {
                name: "kg_context".to_string(),
                var_type: VariableType::String,
                description: "Knowledge graph context".to_string(),
                required: true,
                default: None,
            },
            VariableDefinition {
                name: "vector_context".to_string(),
                var_type: VariableType::String,
                description: "Vector search context".to_string(),
                required: true,
                default: None,
            },
            VariableDefinition {
                name: "response_type".to_string(),
                var_type: VariableType::String,
                description: "Desired response format".to_string(),
                required: true,
                default: Some("Multiple Paragraphs".to_string()),
            },
        ],
    )
} 