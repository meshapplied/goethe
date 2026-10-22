// MESH_HOME and related path resolution.

use std::path::{Path, PathBuf};

pub fn mesh_home() -> PathBuf {
    if let Ok(v) = std::env::var("MESH_HOME") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    let workspace = Path::new("/workspace");
    if workspace.is_dir() && is_writable(workspace) {
        return workspace.join(".mesh");
    }
    let varlib = Path::new("/var/lib/mesh");
    if varlib.is_dir() && is_writable(varlib) {
        return varlib.to_path_buf();
    }
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".mesh")
}

pub fn mesh_mux_port() -> u16 {
    let home = mesh_home();
    if let Ok(v) = std::env::var("MESH_MUX_PORT") {
        let t = v.trim();
        if let Ok(p) = t.parse::<u16>() {
            let _ = std::fs::write(home.join("nginx/mux_port"), t);
            return p;
        }
    }
    if let Ok(v) = std::fs::read_to_string(home.join("nginx/mux_port")) {
        if let Ok(p) = v.trim().parse::<u16>() {
            return p;
        }
    }
    30000
}

pub fn mesh_mux_addr() -> String {
    format!("127.0.0.1:{}", mesh_mux_port())
}

pub fn mesh_sock() -> PathBuf {
    if let Ok(v) = std::env::var("MESH_SOCK") {
        if !v.is_empty() {
            return PathBuf::from(v);
        }
    }
    let euid = unsafe { libc::geteuid() };
    if euid == 0 {
        PathBuf::from("/var/run/mesh.sock")
    } else {
        mesh_home().join("mesh.sock")
    }
}

pub fn ensure_layout(home: &Path) -> std::io::Result<()> {
    for sub in ["bin", "models", "logs", "nginx", "run"] {
        std::fs::create_dir_all(home.join(sub))?;
    }
    Ok(())
}

pub fn sanitize_model_id(model_id: &str) -> String {
    model_id.replace('/', "__")
}

pub fn model_dir(home: &Path, model_id: &str) -> PathBuf {
    home.join("models").join(sanitize_model_id(model_id))
}

fn is_writable(path: &Path) -> bool {
    let probe = path.join(format!(".mesh-write-probe-{}", std::process::id()));
    match std::fs::OpenOptions::new().write(true).create_new(true).open(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(probe);
            true
        }
        Err(_) => false,
    }
}
