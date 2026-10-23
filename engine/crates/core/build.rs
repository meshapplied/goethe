// Embed git commit and build time for goethe --version (GOETHE_GIT_COMMIT/GOETHE_BUILD_DATE win when set).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

fn git(args: &[&str]) -> Option<String> {
    // Run git and return its trimmed stdout when it succeeds.
    Command::new("git")
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

fn main() {
    // Watch HEAD, the checked-out branch ref, and packed-refs; worktrees keep them in different dirs.
    let given = |key: &str| std::env::var(key).ok().filter(|v| !v.trim().is_empty());
    let commit = given("GOETHE_GIT_COMMIT")
        .or_else(|| git(&["rev-parse", "--short", "HEAD"]))
        .unwrap_or_else(|| "unknown".to_string());
    let date = given("GOETHE_BUILD_DATE").unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "unknown".to_string())
    });
    println!("cargo:rustc-env=GOETHE_GIT_COMMIT={commit}");
    println!("cargo:rustc-env=GOETHE_BUILD_DATE={date}");
    let mut watch: Vec<PathBuf> = Vec::new();
    if let Some(dir) = git(&["rev-parse", "--absolute-git-dir"]) {
        watch.push(Path::new(&dir).join("HEAD"));
    }
    if let (Some(common), Some(branch)) = (
        git(&["rev-parse", "--path-format=absolute", "--git-common-dir"]),
        git(&["symbolic-ref", "-q", "HEAD"]),
    ) {
        watch.push(Path::new(&common).join(&branch));
        watch.push(Path::new(&common).join("packed-refs"));
    }
    for p in watch.iter().filter(|p| p.exists()) {
        println!("cargo:rerun-if-changed={}", p.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=GOETHE_GIT_COMMIT");
    println!("cargo:rerun-if-env-changed=GOETHE_BUILD_DATE");
}
