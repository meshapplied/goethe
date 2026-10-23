// Safetensors header scans: Marlin dual-layout extra and host-offloaded tensor bytes.

use crate::model_facts::weight_files;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

pub fn header_tensor_bytes(header: &[u8], pred: &dyn Fn(&str) -> bool) -> Result<u64, String> {
    // Sum data sizes of tensors whose names match pred.
    let map: HashMap<String, serde_json::Value> =
        serde_json::from_slice(header).map_err(|e| format!("header json: {e}"))?;
    let mut total = 0u64;
    for (name, value) in map {
        if name == "__metadata__" || !pred(&name) {
            continue;
        }
        let Some(offsets) = value.get("data_offsets").and_then(|v| v.as_array()) else {
            continue;
        };
        if offsets.len() < 2 {
            continue;
        }
        let start = offsets[0].as_u64().unwrap_or(0);
        let end = offsets[1].as_u64().unwrap_or(0);
        total += end.saturating_sub(start);
    }
    Ok(total)
}

pub fn tensor_bytes_in_file(path: &Path, pred: &dyn Fn(&str) -> bool) -> Result<u64, String> {
    // Read only the JSON header of one safetensors file.
    let mut file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| format!("{}: read header len: {e}", path.display()))?;
    let n = u64::from_le_bytes(len_buf) as usize;
    if n > 64 * 1024 * 1024 {
        return Err(format!("{}: safetensors header too large", path.display()));
    }
    let mut header = vec![0u8; n];
    file.read_exact(&mut header)
        .map_err(|e| format!("{}: read header: {e}", path.display()))?;
    header_tensor_bytes(&header, pred)
}

pub fn tensor_bytes_in_dir(dir: &Path, pred: &dyn Fn(&str) -> bool) -> Result<u64, String> {
    let mut total = 0u64;
    for f in weight_files(dir)? {
        total += tensor_bytes_in_file(&f, pred)?;
    }
    Ok(total)
}

pub fn marlin_extra_bytes_in_dir(dir: &Path) -> Result<u64, String> {
    tensor_bytes_in_dir(dir, &tensor_counts_marlin_extra)
}

pub fn is_ple_embedding(name: &str) -> bool {
    name.contains("ple_embedding")
}

pub fn is_nvfp4_compressed_model(config: &serde_json::Value) -> bool {
    // compressed-tensors checkpoints whose linear layers Marlin may repack.
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
            "layer.1.bias": {"dtype": "F32", "data_offsets": [5100, 5200]},
            "model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight": {"dtype": "BF16", "data_offsets": [5200, 9200]}
        }"#;
        assert_eq!(header_tensor_bytes(header, &tensor_counts_marlin_extra).unwrap(), 1100);
        assert_eq!(header_tensor_bytes(header, &is_ple_embedding).unwrap(), 4000);
    }
}
