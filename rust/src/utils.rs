use md5::Context as Md5;

/// Computes an MD5 hash ID with a prefix for a given string.
/// This matches LightRAG's compute_mdhash_id function.
/// 
/// # Arguments
/// * `content` - The string to hash
/// * `prefix` - Optional prefix to add to the hash (e.g., "ent-", "rel-", "chunk-")
/// 
/// # Returns
/// A string containing the prefixed MD5 hash
pub fn compute_mdhash_id(content: &str, prefix: &str) -> String {
    let mut hasher = Md5::new();
    hasher.consume(content.as_bytes());
    format!("{}{:x}", prefix, hasher.compute())
} 