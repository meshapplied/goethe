// Optional runtimes (ternary kernels) under MESH_HOME.

use std::path::{Path, PathBuf};

pub fn ternary_kernels_dir(home: &Path) -> PathBuf {
    home.join("goethe/ternary-kernels")
}

pub fn require_ternary_runtime(home: &Path) -> Result<(), String> {
    if ternary_kernels_dir(home).is_dir() {
        return Ok(());
    }
    Err(
        "ternary runtime not installed: run `mesh update --ternary` (needs MESH_PKG_TOKEN for private packages)"
            .to_string(),
    )
}

pub fn mesh_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
