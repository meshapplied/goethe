// Safetensors header scan for Marlin dual-layout VRAM estimate.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

pub fn marlin_extra_bytes_in_header(header: &[u8]) -> Result<u64, String> {
    let map: HashMap<String, serde_json::Value> =
        serde_json::from_slice(header).map_err(|e| format!("header json: {e}"))?;
    let mut total = 0u64;
    for (name, value) in map {
        if name == "__metadata__" {
            continue;
        }
        let Some(obj) = value.as_object() else {
            continue;
        };
        let Some(offsets) = obj.get("data_offsets").and_then(|v| v.as_array()) else {
            continue;
        };
        if offsets.len() < 2 {
            continue;
        }
        if !tensor_counts_marlin_extra(&name) {
            continue;
        }
        let start = offsets[0].as_i64().unwrap_or(0) as u64;
        let end = offsets[1].as_i64().unwrap_or(0) as u64;
        if end > start {
            total += end - start;
        }
    }
    Ok(total)
}

pub fn marlin_extra_bytes_in_file(path: &Path) -> Result<u64, String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| format!("read header len: {e}"))?;
    let n = u64::from_le_bytes(len_buf) as usize;
    if n > 64 * 1024 * 1024 {
        return Err("safetensors header too large".to_string());
    }
    let mut header = vec![0u8; n];
    file.read_exact(&mut header)
        .map_err(|e| format!("read header: {e}"))?;
    marlin_extra_bytes_in_header(&header)
}

pub fn marlin_extra_bytes_in_dir(dir: &Path) -> Result<u64, String> {
    let mut total = 0u64;
    for entry in std::fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.path().extension().and_then(|e| e.to_str()) != Some("safetensors") {
            continue;
        }
        total += marlin_extra_bytes_in_file(&entry.path())?;
    }
    Ok(total)
}

pub fn is_nvfp4_compressed_model(config: &serde_json::Value) -> bool {
    if config
        .pointer("/quantization_config/quant_method")
        .and_then(|v| v.as_str())
        .map(|s| s.contains("compressed"))
        .unwrap_or(false)
    {
        return true;
    }
    if config
        .get("quantization_config")
        .and_then(|q| q.get("config_groups"))
        .is_some()
    {
        return true;
    }
    config
        .pointer("/compression_config/format")
        .and_then(|v| v.as_str())
        == Some("nvfp4")
}

fn tensor_counts_marlin_extra(name: &str) -> bool {
    if name.contains(".experts.") {
        return false;
    }
    let lower = name.to_ascii_lowercase();
    lower.contains("weight_packed") || lower.contains("weight_scale")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_scan_synthetic() {
        let header = br#"{
            "__metadata__": {"format": "pt"},
            "layer.0.weight_packed": {"dtype": "U8", "data_offsets": [0, 1000]},
            "layer.0.experts.weight_packed": {"dtype": "U8", "data_offsets": [1000, 5000]},
            "layer.0.weight_scale": {"dtype": "F8", "data_offsets": [5000, 5100]},
            "layer.1.bias": {"dtype": "F32", "data_offsets": [5100, 5200]}
        }"#;
        let n = marlin_extra_bytes_in_header(header).expect("scan");
        assert_eq!(n, 1100);
    }
}
