// TEMPORARY STUB (replaced by the pyenv branch at merge).
use std::path::Path;
pub enum Layer { Base, Engine, Sglang }
pub const ENGINE_VERSION: &str = "0.1.0";
pub fn has(home: &Path, layer: Layer) -> bool {
    let _ = (home, layer); true
}
pub fn ensure(home: &Path, layer: Layer) -> Result<(), String> {
    let _ = (home, layer); Ok(())
}
pub fn sglang_support(home: &Path) -> Option<crate::scan::SglangInfo> { crate::scan::sglang_info(home) }
