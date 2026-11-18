// Sentences more than one place says: each lives here once, and a test fails on a second copy anywhere else.

pub const GATED: &str = "is gated or private on Hugging Face";
pub const NOT_FOUND: &str = "not found on Hugging Face, or private";
pub const FALLBACK_HINT: &str = "GOETHE_ALLOW_FALLBACK=0 is set; pass --fallback (or unset it) to serve it with the plain runtime instead";
pub const NOTHING_DOWNLOADED: &str = "nothing was downloaded";
pub const ONLY_PACK: &str = "runs only with its kernel pack";
pub const LOGIN_FIRST: &str = "run goethe login first";

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::{Path, PathBuf};

    const ALL: [(&str, &str); 6] = [
        ("GATED", GATED),
        ("NOT_FOUND", NOT_FOUND),
        ("FALLBACK_HINT", FALLBACK_HINT),
        ("NOTHING_DOWNLOADED", NOTHING_DOWNLOADED),
        ("ONLY_PACK", ONLY_PACK),
        ("LOGIN_FIRST", LOGIN_FIRST),
    ];

    fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
        for e in std::fs::read_dir(dir).unwrap().flatten() {
            let p = e.path();
            if p.is_dir() {
                sources(&p, out);
            } else if p.extension().is_some_and(|x| x == "rs") {
                out.push(p);
            }
        }
    }

    #[test]
    fn a_message_lives_in_one_constant_and_no_other_file_repeats_it() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files = Vec::new();
        sources(&root, &mut files);
        assert!(files.len() > 20, "the sources were found");
        for file in files {
            let text = std::fs::read_to_string(&file).unwrap();
            let code = text.split("#[cfg(test)]").next().unwrap_or_default();
            for (name, phrase) in ALL {
                let copies = code.matches(phrase).count();
                let allowed = usize::from(file.ends_with("messages.rs"));
                assert_eq!(copies, allowed, "{name} (\"{phrase}\") appears {copies} time(s) in {}: use messages::{name}", file.display());
            }
        }
    }
}
