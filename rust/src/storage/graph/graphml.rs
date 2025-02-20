use crate::types::{Result, Error};
use petgraph::graph::Graph;
use petgraph::visit::EdgeRef;
use quick_xml::Reader;
use quick_xml::events::Event;
use std::path::Path;
use std::collections::HashMap;
use super::{NodeData, EdgeData};

/// Handler for GraphML format reading and writing
pub struct GraphMlHandler {
    /// The graph being handled
    pub graph: Graph<NodeData, EdgeData>,
    /// Whether to use pretty printing
    pretty_print: bool,
}

impl GraphMlHandler {
    /// Creates a new GraphML handler for the given graph
    pub fn new(graph: Graph<NodeData, EdgeData>) -> Self {
        let mut node_map = HashMap::new();
        for node_idx in graph.node_indices() {
            if let Some(node_data) = graph.node_weight(node_idx) {
                node_map.insert(node_data.id.clone(), node_idx);
            }
        }
        Self { 
            graph,
            pretty_print: false,
        }
    }

    /// Enable or disable pretty printing
    pub fn pretty_print(mut self, enabled: bool) -> Self {
        self.pretty_print = enabled;
        self
    }

    /// Reads a graph from a GraphML file
    pub fn read_graphml(path: &Path) -> Result<Graph<NodeData, EdgeData>> {
        if !path.exists() {
            return Err(Error::Storage(format!("File not found: {}", path.display())));
        }
        let content = std::fs::read_to_string(path)?;
        if content.trim().is_empty() {
            return Err(Error::Storage("Empty GraphML file".to_string()));
        }
        if !content.contains("<graphml") {
            return Err(Error::Storage("Invalid GraphML file".to_string()));
        }
        let mut reader = Reader::from_str(&content);
        reader.trim_text(true);

        let mut graph = Graph::new();
        let mut _node_map = HashMap::new();
        let mut buf = Vec::new();

        // Parse GraphML content
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) => {
                    match e.name().as_ref() {
                        b"node" => {
                            if let Some(node_data) = Self::parse_node(&mut reader, e)? {
                                let idx = graph.add_node(node_data.clone());
                                _node_map.insert(node_data.id.clone(), idx);
                            }
                        }
                        b"edge" => {
                            if let Some((source, target, edge_data)) = Self::parse_edge(&mut reader, e)? {
                                // Note: For edges, we assume nodes are already added
                                // so we add edge only if both source and target exist
                                // Here, we scan the current nodes in the graph
                                let src_opt = graph.node_indices().find(|&i| graph[i].id == source);
                                let tgt_opt = graph.node_indices().find(|&i| graph[i].id == target);
                                if let (Some(src_idx), Some(tgt_idx)) = (src_opt, tgt_opt) {
                                    graph.add_edge(src_idx, tgt_idx, edge_data);
                                }
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(Error::Storage(format!("Error parsing GraphML: {}", e))),
                _ => {}
            }
            buf.clear();
        }

        Ok(graph)
    }

    /// Writes the graph to a GraphML file
    pub fn write_graphml(&self, path: &Path) -> Result<()> {
        let mut writer = if self.pretty_print {
            quick_xml::Writer::new_with_indent(Vec::new(), b' ', 2)
        } else {
            quick_xml::Writer::new(Vec::new())
        };

        // Write GraphML header with XML namespace
        let mut root = quick_xml::events::BytesStart::new("graphml");
        root.push_attribute(("xmlns", "http://graphml.graphdrawing.org/xmlns"));
        root.push_attribute(("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance"));
        root.push_attribute(("xsi:schemaLocation", 
            "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd"));
        writer.write_event(Event::Start(root))?;

        // Write key definitions
        self.write_node_keys(&mut writer)?;
        self.write_edge_keys(&mut writer)?;

        // Start graph element with directionality
        let mut graph_elem = quick_xml::events::BytesStart::new("graph");
        graph_elem.push_attribute(("id", "G"));
        graph_elem.push_attribute(("edgedefault", "directed"));
        writer.write_event(Event::Start(graph_elem))?;

        // Write nodes and edges
        for node_idx in self.graph.node_indices() {
            if let Some(node_data) = self.graph.node_weight(node_idx) {
                self.write_node(&mut writer, node_data)?;
            }
        }

        for edge_ref in self.graph.edge_references() {
            let source = &self.graph[edge_ref.source()].id;
            let target = &self.graph[edge_ref.target()].id;
            self.write_edge(&mut writer, source, target, edge_ref.weight())?;
        }

        // Close elements
        writer.write_event(Event::End(quick_xml::events::BytesEnd::new("graph")))?;
        writer.write_event(Event::End(quick_xml::events::BytesEnd::new("graphml")))?;

        // Write to file with proper UTF-8 handling
        std::fs::write(path, writer.into_inner())?;
        Ok(())
    }

    // Helper methods for parsing nodes and edges
    fn parse_node(reader: &mut Reader<&[u8]>, start_event: &quick_xml::events::BytesStart) -> Result<Option<NodeData>> {
        let mut id = None;
        let mut attributes = HashMap::new();
        let mut buf = Vec::new();

        // Get node ID from attributes and unescape it
        for attr in start_event.attributes() {
            let attr = attr.map_err(|e| Error::Storage(format!("Failed to read node attribute: {}", e)))?;
            if attr.key.as_ref() == b"id" {
                let raw = String::from_utf8_lossy(&attr.value);
                let unescaped = quick_xml::escape::unescape(raw.as_ref())
                    .unwrap_or_else(|_| std::borrow::Cow::Borrowed(raw.as_ref()))
                    .into_owned();
                let final_id = unescaped.replace("&amp;", "&")
                                      .replace("&lt;", "<")
                                      .replace("&gt;", ">")
                                      .replace("&quot;", "\"")
                                      .replace("&apos;", "'");
                id = Some(final_id);
            }
        }

        // Parse all <data> elements and store them in attributes
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) if e.name().as_ref() == b"data" => {
                    let mut key_attr = None;
                    for a in e.attributes() {
                        let a = a.map_err(|e| Error::Storage(format!("Failed to read data key: {}", e)))?;
                        if a.key.as_ref() == b"key" {
                            key_attr = Some(String::from_utf8_lossy(&a.value).to_string());
                        }
                    }
                    let mut value_str = String::new();
                    loop {
                        match reader.read_event_into(&mut buf) {
                            Ok(Event::Text(e)) => {
                                let raw_text = String::from_utf8_lossy(e.as_ref()).to_string();
                                let unescaped_text = quick_xml::escape::unescape(raw_text.as_str())
                                    .unwrap_or_else(|_| std::borrow::Cow::Borrowed(raw_text.as_str()))
                                    .into_owned();
                                let final_text = unescaped_text.replace("&quot;", "\"").replace("&apos;", "'");
                                value_str = final_text;
                            },
                            Ok(Event::End(ref e)) if e.name().as_ref() == b"data" => break,
                            Ok(Event::Eof) => return Err(Error::Storage("Unexpected EOF in node data".to_string())),
                            Err(e) => return Err(Error::Storage(format!("Error parsing node data: {}", e))),
                            _ => {}
                        }
                        buf.clear();
                    }
                    if let Some(key) = key_attr {
                        // If key starts with "attr_", remove the prefix to get the original attribute name
                        let attr_name = if key.starts_with("attr_") { key.trim_start_matches("attr_").to_string() } else { key };
                        // Try parsing the value as JSON, fallback to string if parsing fails
                        let parsed_value = serde_json::from_str(&value_str).unwrap_or(serde_json::Value::String(value_str.clone()));
                        attributes.insert(attr_name, parsed_value);
                    }
                },
                Ok(Event::End(ref e)) if e.name().as_ref() == b"node" => break,
                Ok(Event::Eof) => return Err(Error::Storage("Unexpected EOF in node".to_string())),
                Err(e) => return Err(Error::Storage(format!("Error parsing node: {}", e))),
                _ => {}
            }
            buf.clear();
        }

        if let Some(id) = id {
            Ok(Some(NodeData { id, attributes }))
        } else {
            Ok(None)
        }
    }

    fn parse_edge(reader: &mut Reader<&[u8]>, start_event: &quick_xml::events::BytesStart) -> Result<Option<(String, String, EdgeData)>> {
        let mut source = None;
        let mut target = None;
        let mut weight = None;
        let mut description = None;
        let mut keywords = None;
        let mut buf = Vec::new();

        for attr in start_event.attributes() {
            let attr = attr.map_err(|e| Error::Storage(format!("Failed to read edge attribute: {}", e)))?;
            match attr.key.as_ref() {
                b"source" => {
                    let raw = String::from_utf8_lossy(&attr.value);
                    let unescaped = quick_xml::escape::unescape(raw.as_ref()).unwrap_or_else(|_| std::borrow::Cow::Borrowed(raw.as_ref())).into_owned();
                    source = Some(unescaped);
                },
                b"target" => {
                    let raw = String::from_utf8_lossy(&attr.value);
                    let unescaped = quick_xml::escape::unescape(raw.as_ref()).unwrap_or_else(|_| std::borrow::Cow::Borrowed(raw.as_ref())).into_owned();
                    target = Some(unescaped);
                },
                _ => {}
            }
        }

        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(ref e)) if e.name().as_ref() == b"data" => {
                    let mut key = None;
                    for attr in e.attributes() {
                        let attr = attr.map_err(|e| Error::Storage(format!("Failed to read data key: {}", e)))?;
                        if attr.key.as_ref() == b"key" {
                            key = Some(String::from_utf8_lossy(&attr.value).to_string());
                        }
                    }

                    let mut value = String::new();
                    loop {
                        match reader.read_event_into(&mut buf) {
                            Ok(Event::Text(e)) => {
                                value = String::from_utf8_lossy(e.as_ref()).to_string();
                            },
                            Ok(Event::End(ref e)) if e.name().as_ref() == b"data" => break,
                            Ok(Event::Eof) => return Err(Error::Storage("Unexpected EOF in edge data".to_string())),
                            Err(e) => return Err(Error::Storage(format!("Error parsing edge data: {}", e))),
                            _ => {}
                        }
                        buf.clear();
                    }

                    match key.as_deref() {
                        Some("d3") => weight = Some(value.parse::<f64>().unwrap_or(0.0)),
                        Some("d4") => description = Some(value),
                        Some("d5") => keywords = Some(value.split(',').map(|s| s.to_string()).collect()),
                        _ => {}
                    }
                },
                Ok(Event::End(ref e)) if e.name().as_ref() == b"edge" => break,
                Ok(Event::Eof) => return Err(Error::Storage("Unexpected EOF in edge".to_string())),
                Err(e) => return Err(Error::Storage(format!("Error parsing edge: {}", e))),
                _ => {}
            }
            buf.clear();
        }

        if let (Some(source), Some(target)) = (source, target) {
            let edge_data = EdgeData {
                weight: weight.unwrap_or(0.0),
                description,
                keywords,
            };
            Ok(Some((source, target, edge_data)))
        } else {
            Ok(None)
        }
    }

    // Helper methods for writing nodes and edges
    fn write_node_keys(&self, writer: &mut quick_xml::Writer<Vec<u8>>) -> Result<()> {
        let mut keys_set = std::collections::HashSet::new();
        for node_idx in self.graph.node_indices() {
            if let Some(node) = self.graph.node_weight(node_idx) {
                for key in node.attributes.keys() {
                    keys_set.insert(key);
                }
            }
        }
        for key in keys_set {
            let key_id = format!("attr_{}", key);
            let mut elem = quick_xml::events::BytesStart::new("key");
            elem.push_attribute(("id", key_id.as_str()));
            elem.push_attribute(("for", "node"));
            elem.push_attribute(("attr.name", key.as_str()));
            elem.push_attribute(("attr.type", "string"));
            writer.write_event(Event::Empty(elem))?;
        }
        Ok(())
    }

    fn write_edge_keys(&self, writer: &mut quick_xml::Writer<Vec<u8>>) -> Result<()> {
        // Write key definitions for edge attributes
        let keys = [
            ("d3", "weight", "double"),
            ("d4", "description", "string"),
            ("d5", "keywords", "string"),
        ];

        for (id, attr, attr_type) in keys {
            let mut elem = quick_xml::events::BytesStart::new("key");
            elem.push_attribute(("id", id));
            elem.push_attribute(("for", "edge"));
            elem.push_attribute(("attr.name", attr));
            elem.push_attribute(("attr.type", attr_type));
            writer.write_event(Event::Empty(elem))?;
        }

        Ok(())
    }

    /// Escapes special characters in XML attribute values
    fn escape_xml_attr(s: &str) -> String {
        s.replace('&', "&amp;")
         .replace('<', "&lt;")
         .replace('>', "&gt;")
         .replace('"', "&quot;")
         .replace('\'', "&apos;")
    }

    fn write_node(&self, writer: &mut quick_xml::Writer<Vec<u8>>, node_data: &NodeData) -> Result<()> {
        let mut node_elem = quick_xml::events::BytesStart::new("node");
        node_elem.push_attribute(("id", Self::escape_xml_attr(&node_data.id).as_str()));
        writer.write_event(Event::Start(node_elem))?;

        for (key, value) in &node_data.attributes {
            let key_id = format!("attr_{}", key);
            let mut data_elem = quick_xml::events::BytesStart::new("data");
            data_elem.push_attribute(("key", key_id.as_str()));
            writer.write_event(Event::Start(data_elem))?;
            let value_str = if let serde_json::Value::String(s) = value {
                        s.clone()
                     } else {
                        serde_json::to_string(value).unwrap_or_default()
                     };
            writer.write_event(Event::Text(quick_xml::events::BytesText::new(&value_str)))?;
            writer.write_event(Event::End(quick_xml::events::BytesEnd::new("data")))?;
        }

        writer.write_event(Event::End(quick_xml::events::BytesEnd::new("node")))?;
        Ok(())
    }

    fn write_edge(&self, writer: &mut quick_xml::Writer<Vec<u8>>, source: &str, target: &str, edge_data: &EdgeData) -> Result<()> {
        // Start edge element
        let mut edge_elem = quick_xml::events::BytesStart::new("edge");
        edge_elem.push_attribute(("source", Self::escape_xml_attr(source).as_str()));
        edge_elem.push_attribute(("target", Self::escape_xml_attr(target).as_str()));
        writer.write_event(Event::Start(edge_elem))?;

        // Write weight
        let mut weight_elem = quick_xml::events::BytesStart::new("data");
        weight_elem.push_attribute(("key", "d3"));
        writer.write_event(Event::Start(weight_elem))?;
        writer.write_event(Event::Text(quick_xml::events::BytesText::new(&edge_data.weight.to_string())))?;
        writer.write_event(Event::End(quick_xml::events::BytesEnd::new("data")))?;

        // Write description if present
        if let Some(ref desc) = edge_data.description {
            let mut desc_elem = quick_xml::events::BytesStart::new("data");
            desc_elem.push_attribute(("key", "d4"));
            writer.write_event(Event::Start(desc_elem))?;
            writer.write_event(Event::Text(quick_xml::events::BytesText::new(desc)))?;
            writer.write_event(Event::End(quick_xml::events::BytesEnd::new("data")))?;
        }

        // Write keywords if present
        if let Some(ref keywords) = edge_data.keywords {
            let mut keywords_elem = quick_xml::events::BytesStart::new("data");
            keywords_elem.push_attribute(("key", "d5"));
            writer.write_event(Event::Start(keywords_elem))?;
            let keywords_str = keywords.join(",");
            writer.write_event(Event::Text(quick_xml::events::BytesText::new(&keywords_str)))?;
            writer.write_event(Event::End(quick_xml::events::BytesEnd::new("data")))?;
        }

        // End edge element
        writer.write_event(Event::End(quick_xml::events::BytesEnd::new("edge")))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_graphml_roundtrip() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let file_path = temp_dir.path().join("test.graphml");

        // Create a test graph
        let mut graph = Graph::new();
        let mut node_data = NodeData {
            id: "node1".to_string(),
            attributes: HashMap::new(),
        };
        node_data.attributes.insert("type".to_string(), serde_json::Value::String("test".to_string()));
        let n1 = graph.add_node(node_data);

        let mut node_data = NodeData {
            id: "node2".to_string(),
            attributes: HashMap::new(),
        };
        node_data.attributes.insert("type".to_string(), serde_json::Value::String("test".to_string()));
        let n2 = graph.add_node(node_data);

        let edge_data = EdgeData {
            weight: 1.0,
            description: Some("test edge".to_string()),
            keywords: Some(vec!["test".to_string()]),
        };
        graph.add_edge(n1, n2, edge_data);

        // Write to GraphML
        let handler = GraphMlHandler::new(graph);
        handler.write_graphml(&file_path)?;

        // Read back
        let loaded_graph = GraphMlHandler::read_graphml(&file_path)?;

        // Verify
        assert_eq!(loaded_graph.node_count(), 2);
        assert_eq!(loaded_graph.edge_count(), 1);

        Ok(())
    }
} 