// Compile-time version metadata for the mesh CLI.

pub fn version_line() -> String {
    let ver = env!("CARGO_PKG_VERSION");
    let commit = option_env!("MESH_GIT_COMMIT").unwrap_or("unknown");
    let date_raw = option_env!("MESH_BUILD_DATE").unwrap_or("unknown");
    let date = if let Ok(secs) = date_raw.parse::<i64>() {
        chrono::DateTime::from_timestamp(secs, 0)
            .map(|d| d.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| date_raw.to_string())
    } else {
        date_raw.to_string()
    };
    format!("mesh {ver} (commit {commit}, built {date})")
}
