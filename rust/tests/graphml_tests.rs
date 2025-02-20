use super_lightrag::storage::graph::{NodeData, EdgeData};
use super_lightrag::storage::graph::graphml::GraphMlHandler;
use super_lightrag::types::Result;
use petgraph::graph::Graph;
use std::collections::HashMap;
use serde_json::json;
use tempfile::TempDir;

#[test]
fn test_basic_graphml_roundtrip() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("test_graph.graphml");

    // Create a simple test graph
    let mut graph = Graph::new();
    
    // Add nodes with attributes
    let mut node1_attrs = HashMap::new();
    node1_attrs.insert("entity_type".to_string(), json!("person"));
    node1_attrs.insert("description".to_string(), json!("Test Person 1"));
    let node1 = NodeData { id: "n1".to_string(), attributes: node1_attrs };
    
    let mut node2_attrs = HashMap::new();
    node2_attrs.insert("entity_type".to_string(), json!("person"));
    node2_attrs.insert("description".to_string(), json!("Test Person 2"));
    let node2 = NodeData { id: "n2".to_string(), attributes: node2_attrs };
    
    let n1_idx = graph.add_node(node1);
    let n2_idx = graph.add_node(node2);

    // Add edge with attributes
    let edge = EdgeData {
        weight: 1.0,
        description: Some("test relationship".to_string()),
        keywords: Some(vec!["friend".to_string(), "colleague".to_string()]),
    };
    graph.add_edge(n1_idx, n2_idx, edge);

    // Write graph to file
    let handler = GraphMlHandler::new(graph);
    handler.write_graphml(&file_path)?;

    // Read graph back
    let loaded_graph = GraphMlHandler::read_graphml(&file_path)?;

    // Verify structure
    assert_eq!(loaded_graph.node_count(), 2);
    assert_eq!(loaded_graph.edge_count(), 1);

    // Verify node attributes
    let mut found_n1 = false;
    let mut found_n2 = false;
    for node_idx in loaded_graph.node_indices() {
        let node = &loaded_graph[node_idx];
        match node.id.as_str() {
            "n1" => {
                found_n1 = true;
                assert_eq!(node.attributes.get("entity_type").unwrap(), &json!("person"));
                assert_eq!(node.attributes.get("description").unwrap(), &json!("Test Person 1"));
            },
            "n2" => {
                found_n2 = true;
                assert_eq!(node.attributes.get("entity_type").unwrap(), &json!("person"));
                assert_eq!(node.attributes.get("description").unwrap(), &json!("Test Person 2"));
            },
            _ => panic!("Unexpected node ID"),
        }
    }
    assert!(found_n1 && found_n2);

    // Verify edge attributes
    let mut found_edge = false;
    for edge_ref in loaded_graph.edge_references() {
        let edge = edge_ref.weight();
        assert_eq!(edge.weight, 1.0);
        assert_eq!(edge.description.as_ref().unwrap(), "test relationship");
        assert_eq!(edge.keywords.as_ref().unwrap(), &vec!["friend".to_string(), "colleague".to_string()]);
        found_edge = true;
    }
    assert!(found_edge);

    Ok(())
}

#[test]
fn test_pretty_print() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let pretty_file = temp_dir.path().join("pretty.graphml");
    let compact_file = temp_dir.path().join("compact.graphml");

    // Create a simple graph
    let mut graph = Graph::new();
    let node = NodeData { 
        id: "n1".to_string(), 
        attributes: {
            let mut attrs = HashMap::new();
            attrs.insert("test".to_string(), json!("value"));
            attrs
        }
    };
    graph.add_node(node);

    // Write with pretty printing
    let handler = GraphMlHandler::new(graph.clone()).pretty_print(true);
    handler.write_graphml(&pretty_file)?;

    // Write without pretty printing
    let handler = GraphMlHandler::new(graph).pretty_print(false);
    handler.write_graphml(&compact_file)?;

    // Verify files are different but both valid
    let pretty_content = std::fs::read_to_string(&pretty_file)?;
    let compact_content = std::fs::read_to_string(&compact_file)?;
    
    assert!(pretty_content.len() > compact_content.len());
    assert!(pretty_content.contains("\n"));
    assert!(!compact_content.contains("\n"));

    // Verify both can be read back correctly
    let graph1 = GraphMlHandler::read_graphml(&pretty_file)?;
    let graph2 = GraphMlHandler::read_graphml(&compact_file)?;
    
    assert_eq!(graph1.node_count(), 1);
    assert_eq!(graph2.node_count(), 1);

    Ok(())
}

#[test]
fn test_complex_attributes() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("complex.graphml");

    let mut graph = Graph::new();
    
    // Node with complex attributes
    let mut node_attrs = HashMap::new();
    node_attrs.insert("entity_type".to_string(), json!("document"));
    node_attrs.insert("metadata".to_string(), json!({
        "title": "Test Document",
        "tags": ["important", "urgent"],
        "version": 1.0
    }));
    let node = NodeData { id: "doc1".to_string(), attributes: node_attrs };
    graph.add_node(node);

    // Write and read back
    let handler = GraphMlHandler::new(graph);
    handler.write_graphml(&file_path)?;
    let loaded_graph = GraphMlHandler::read_graphml(&file_path)?;

    // Verify complex attributes
    let node = &loaded_graph[loaded_graph.node_indices().next().unwrap()];
    let metadata = node.attributes.get("metadata").unwrap();
    assert_eq!(metadata["title"], "Test Document");
    assert_eq!(metadata["tags"][0], "important");
    assert_eq!(metadata["version"], 1.0);

    Ok(())
}

#[test]
fn test_error_handling() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("nonexistent.graphml");

    // Test reading non-existent file
    let result = GraphMlHandler::read_graphml(&file_path);
    assert!(result.is_err());

    // Test reading invalid XML
    std::fs::write(&file_path, "invalid xml content")?;
    let result = GraphMlHandler::read_graphml(&file_path);
    assert!(result.is_err());

    // Test reading valid XML but invalid GraphML
    std::fs::write(&file_path, "<valid>xml but not graphml</valid>")?;
    let result = GraphMlHandler::read_graphml(&file_path);
    assert!(result.is_err());

    Ok(())
}

#[test]
fn test_large_graph() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("large.graphml");

    let mut graph = Graph::new();
    let mut node_indices = Vec::new();

    // Create 1000 nodes
    for i in 0..1000 {
        let mut attrs = HashMap::new();
        attrs.insert("index".to_string(), json!(i));
        let node = NodeData { 
            id: format!("n{}", i), 
            attributes: attrs 
        };
        node_indices.push(graph.add_node(node));
    }

    // Create 2000 edges
    for i in 0..1000 {
        let edge = EdgeData {
            weight: i as f64,
            description: Some(format!("edge {}", i)),
            keywords: Some(vec![format!("key{}", i)]),
        };
        // Create a ring of nodes
        graph.add_edge(node_indices[i], node_indices[(i + 1) % 1000], edge.clone());
        // Create some cross edges
        if i < 1000 {
            graph.add_edge(node_indices[i], node_indices[(i + 100) % 1000], edge);
        }
    }

    // Write and read back
    let handler = GraphMlHandler::new(graph);
    handler.write_graphml(&file_path)?;
    let loaded_graph = GraphMlHandler::read_graphml(&file_path)?;

    // Verify structure
    assert_eq!(loaded_graph.node_count(), 1000);
    assert_eq!(loaded_graph.edge_count(), 2000);

    Ok(())
}

#[test]
fn test_special_characters() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("special.graphml");

    let mut graph = Graph::new();
    
    // Create nodes with special characters
    let special_chars = vec![
        ("node&1", "value & with ampersand"),
        ("node<2", "value < with less than"),
        ("node>3", "value > with greater than"),
        ("node\"4", "value \" with quotes"),
        ("node'5", "value ' with apostrophe"),
        ("nodeé6", "value é with accent"),
        ("node🦀7", "value 🦀 with emoji"),
    ];

    // Create nodes with special characters
    for &(id, desc) in &special_chars {
        let mut attrs = HashMap::new();
        attrs.insert("description".to_string(), json!(desc));
        let node = NodeData { id: id.to_string(), attributes: attrs };
        graph.add_node(node);
    }

    // Write and read back
    let handler = GraphMlHandler::new(graph);
    handler.write_graphml(&file_path)?;
    let loaded_graph = GraphMlHandler::read_graphml(&file_path)?;

    // Verify all special characters were preserved
    assert_eq!(loaded_graph.node_count(), special_chars.len());
    for &(id, desc) in &special_chars {
        let mut found = false;
        for node_idx in loaded_graph.node_indices() {
            let node = &loaded_graph[node_idx];
            if node.id == id {
                assert_eq!(node.attributes.get("description").unwrap(), &json!(desc));
                found = true;
                break;
            }
        }
        assert!(found, "Node with id {} not found", id);
    }

    Ok(())
}

#[test]
fn test_networkx_compatibility() -> Result<()> {
    let temp_dir = TempDir::new().expect("Failed to create temp dir");
    let file_path = temp_dir.path().join("networkx_compat.graphml");

    // Create a graph matching NetworkX format
    let mut graph = Graph::new();
    
    // Add nodes with attributes matching NetworkX style
    let mut node1_attrs = HashMap::new();
    node1_attrs.insert("label".to_string(), json!("Node 1"));
    node1_attrs.insert("type".to_string(), json!("entity"));
    let node1 = NodeData { id: "1".to_string(), attributes: node1_attrs };
    
    let mut node2_attrs = HashMap::new();
    node2_attrs.insert("label".to_string(), json!("Node 2"));
    node2_attrs.insert("type".to_string(), json!("entity"));
    let node2 = NodeData { id: "2".to_string(), attributes: node2_attrs };
    
    let n1_idx = graph.add_node(node1);
    let n2_idx = graph.add_node(node2);

    // Add edge with NetworkX-style attributes
    let edge = EdgeData {
        weight: 1.0,
        description: Some("NetworkX Edge".to_string()),
        keywords: Some(vec!["relationship".to_string()]),
    };
    graph.add_edge(n1_idx, n2_idx, edge);

    // Write with pretty printing for NetworkX compatibility
    let handler = GraphMlHandler::new(graph).pretty_print(true);
    handler.write_graphml(&file_path)?;

    // Verify the file contains expected NetworkX-compatible elements
    let content = std::fs::read_to_string(&file_path)?;
    assert!(content.contains("xmlns=\"http://graphml.graphdrawing.org/xmlns\""));
    assert!(content.contains("edgedefault=\"directed\""));
    assert!(content.contains("<key id="));
    assert!(content.contains("<node id="));
    assert!(content.contains("<edge source="));

    Ok(())
} 