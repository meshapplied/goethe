// Embed git commit and build time for mesh --version.

use std::process::Command;
use std::time::SystemTime;

fn main() {
    let commit = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let date = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    println!("cargo:rustc-env=MESH_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=MESH_BUILD_DATE={date}");
    println!("cargo:rerun-if-changed=../../.git/HEAD");
}
