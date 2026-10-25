// Compile-time version metadata for the goethe CLI.

pub fn version_line() -> String {
    format!("goethe {}", env!("CARGO_PKG_VERSION"))
}

pub fn build_line() -> String {
    let ver = env!("CARGO_PKG_VERSION");
    let commit = option_env!("GOETHE_GIT_COMMIT").unwrap_or("unknown");
    let date_raw = option_env!("GOETHE_BUILD_DATE").unwrap_or("unknown");
    let date = if let Ok(secs) = date_raw.parse::<i64>() {
        chrono::DateTime::from_timestamp(secs, 0)
            .map(|d| d.format("%Y-%m-%d %H:%M:%S UTC").to_string())
            .unwrap_or_else(|| date_raw.to_string())
    } else {
        date_raw.to_string()
    };
    format!("goethe {ver} (commit {commit}, built {date})")
}
