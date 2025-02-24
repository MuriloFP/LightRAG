use flate2::{write::ZlibEncoder, read::ZlibDecoder, Compression};
use std::io::{Write, Read};

/// Compresses data using Zlib and prepends the original length as a 4-byte big-endian integer.
pub fn compress_prepend_size(data: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data).expect("Failed to write data");
    let compressed = encoder.finish().expect("Failed to finish compression");
    let mut result = Vec::with_capacity(4 + compressed.len());
    result.extend(&(data.len() as u32).to_be_bytes());
    result.extend(compressed);
    result
}

/// Decompresses data that has its original length prepended as a 4-byte big-endian integer.
pub fn decompress_size_prepended(data: &[u8]) -> Result<Vec<u8>, std::io::Error> {
    if data.len() < 4 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Data too short"));
    }
    let (len_bytes, compressed) = data.split_at(4);
    let _expected_len = u32::from_be_bytes([len_bytes[0], len_bytes[1], len_bytes[2], len_bytes[3]]) as usize;
    let mut decoder = ZlibDecoder::new(compressed);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
} 