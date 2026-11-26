// Maintainer tool, never shipped: CI builds the registry from registry.src.json with it; `validate` checks one against a local SGLang.

use goethe_core::registry::{run_build, run_validate, ValidateOpts};
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let a: Vec<&str> = args.iter().map(String::as_str).collect();
    let done = match a[..] {
        ["build", src, out] => run_build(Path::new(src), Path::new(out)),
        ["validate", source, ref flags @ ..] => {
            let on = |f: &str| flags.contains(&f);
            run_validate(source, &ValidateOpts { online: on("--online"), allow_individuals: on("--allow-individuals"), check_sglang: on("--check-sglang") })
        }
        _ => Err("usage: registry build SRC OUT | registry validate SOURCE [--online] [--check-sglang] [--allow-individuals]".into()),
    };
    if let Err(e) = done {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
