// llama.cpp: llama-server on one GGUF, one GPU, OpenAI-compatible; the universal engine for GGUF models.

use super::{card_env, kept_env, placement_env, EngineSpec};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const PIN_FILE: &str = include_str!("../../../../llama-cpp/PIN");

pub const PIN: &str = crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_COMMIT");
const ASSET: (&str, &str) = (crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_ASSET_URL"), crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_ASSET_SHA256"));

type Get<'a> = &'a dyn Fn(&str) -> Result<Box<dyn Read>, String>;

pub const ENGINE: super::Engine = super::Engine {
    id: "llama.cpp",
    layer: None,
    params: Some(&[
        "temperature", "top_p", "top_k", "min_p", "max_tokens", "max_completion_tokens", "n", "stop", "seed",
        "presence_penalty", "frequency_penalty", "repeat_penalty", "logit_bias", "logprobs", "top_logprobs",
        "response_format", "tools", "tool_choice", "parallel_tool_calls", "ignore_eos", "json_schema", "grammar",
        "n_probs", "chat_template_kwargs", "reasoning_format", "encoding_format", "embd_normalize",
    ]),
    command: llama_command,
};

pub fn server_bin(home: &Path) -> Result<PathBuf, String> {
    // $LLAMA_SERVER, else the one under GOETHE_HOME; no PATH lookup, so the build is the one installed for goethe.
    let p = std::env::var_os("LLAMA_SERVER").map(PathBuf::from).unwrap_or_else(|| home.join("engines/llama.cpp/bin/llama-server"));
    if p.is_file() {
        Ok(p)
    } else {
        Err(format!("llama-server not found at {}: build llama.cpp {PIN} there or set LLAMA_SERVER", p.display()))
    }
}

pub fn pinned<'a>(asset: (&'a str, &'a str), arch: &str) -> Option<(&'a str, &'a str)> {
    // The prebuilt llama-server archive of the pin: Linux x86_64 only, and only once a release carries it.
    let (url, sha) = asset;
    (arch == "x86_64" && url.starts_with("https://") && sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit())).then_some(asset)
}

pub fn server_ready(home: &Path) -> Result<(), String> {
    // Installed, or downloadable on first use: what an install plan needs, without touching the network.
    match server_bin(home) {
        Ok(_) => Ok(()),
        Err(why) if std::env::var_os("LLAMA_SERVER").is_some() => Err(why),
        Err(why) => pinned(ASSET, &crate::update::normalize_arch()).map(|_| ()).ok_or(why),
    }
}

pub fn ensure_server(home: &Path) -> Result<PathBuf, String> {
    // The installed llama-server, else the pinned prebuilt one downloaded once; a set LLAMA_SERVER is never replaced.
    match server_bin(home) {
        Ok(p) => Ok(p),
        Err(why) if std::env::var_os("LLAMA_SERVER").is_some() => Err(why),
        Err(why) => match pinned(ASSET, &crate::update::normalize_arch()) {
            Some(asset) => provision(home, asset, &http_get),
            None => Err(why),
        },
    }
}

fn http_get(url: &str) -> Result<Box<dyn Read>, String> {
    let agent = crate::net::agent_builder().timeout_read(std::time::Duration::from_secs(60)).build();
    let r = agent.get(url).call().map_err(|e| e.to_string())?;
    Ok(Box::new(r.into_reader()))
}

fn provision(home: &Path, (url, sha): (&str, &str), get: Get) -> Result<PathBuf, String> {
    // One line, then the archive at its pinned sha256, unpacked aside and moved into engines/llama.cpp: a failure leaves nothing.
    let dest = home.join("engines/llama.cpp");
    let part = home.join("engines/.llama.cpp.part");
    println!("llama-server is not installed: downloading llama.cpp {} (CUDA, once) from {url}", &PIN[..12]);
    let _ = std::fs::remove_dir_all(&part);
    let done = unpack(&part, &dest, url, sha, get);
    let _ = std::fs::remove_dir_all(&part);
    done.map(|()| dest.join("bin/llama-server"))
}

fn unpack(part: &Path, dest: &Path, url: &str, sha: &str, get: Get) -> Result<(), String> {
    let (tgz, x) = (part.join("llama.cpp.tar.gz"), part.join("x"));
    std::fs::create_dir_all(&x).map_err(|e| format!("{}: {e}", x.display()))?;
    let offline = |e: String| {
        format!("cannot download llama-server from {url}: {e}; nothing was installed. Build llama.cpp {PIN} into {}/bin or set LLAMA_SERVER", dest.display())
    };
    let mut src = get(url).map_err(offline)?;
    let mut f = std::fs::File::create(&tgz).map_err(|e| format!("{}: {e}", tgz.display()))?;
    std::io::copy(&mut src, &mut f).map_err(|e| offline(e.to_string()))?;
    let got = crate::update::sha256_file(&tgz)?;
    if !got.eq_ignore_ascii_case(sha) {
        return Err(format!("llama-server download from {url} has sha256 {got}, the pin says {sha}: refused, nothing was installed"));
    }
    let ok = Command::new("tar").arg("-xzf").arg(&tgz).arg("-C").arg(&x).arg("--no-same-owner").status().map_err(|e| format!("tar: {e}"))?.success();
    if !ok || !x.join("bin/llama-server").is_file() {
        return Err(format!("{url}: not a llama-server archive (no bin/llama-server): refused, nothing was installed"));
    }
    std::fs::create_dir_all(dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    for e in std::fs::read_dir(&x).map_err(|e| e.to_string())?.flatten() {
        let to = dest.join(e.file_name());
        std::fs::rename(e.path(), &to).map_err(|err| format!("cannot place {}: {err}; move it away or set LLAMA_SERVER", to.display()))?;
    }
    Ok(())
}

pub fn gguf_paths(dir: &Path) -> Vec<PathBuf> {
    // GGUF files in dir and one subdirectory level down (repos keep each quant in its own folder), sorted.
    let list = |d: &Path| std::fs::read_dir(d).into_iter().flatten().flatten().map(|e| e.path());
    let mut out: Vec<PathBuf> =
        list(dir).flat_map(|p| if p.is_dir() { list(&p).collect() } else { vec![p] }).filter(|p| p.extension().is_some_and(|x| x == "gguf")).collect();
    out.sort();
    out
}

fn is_model_gguf(name: &str) -> bool {
    // A model file or the first shard of a split one; mmproj files (named mmproj-X or X.mmproj-f16) are vision projectors.
    let base = name.rsplit('/').next().unwrap_or(name);
    base.ends_with(".gguf") && !base.to_ascii_lowercase().contains("mmproj") && (!base.contains("-of-") || base.contains("-00001-of-"))
}

pub fn gguf_in(dir: &Path, card: &toml::Value, quant: Option<&str>) -> Result<PathBuf, String> {
    // The card's gguf file, else the directory's only model GGUF; a quant picks its file whatever the card names.
    if let Some(f) = card.get("gguf").and_then(|v| v.as_str()).filter(|_| quant.is_none()) {
        return Ok(dir.join(f));
    }
    let names: Vec<String> = gguf_paths(dir)
        .iter()
        .filter_map(|p| Some(p.strip_prefix(dir).ok()?.to_string_lossy().into_owned()))
        .filter(|n| is_model_gguf(n))
        .collect();
    if let Some(q) = quant {
        return Ok(dir.join(&pick_gguf(&names.iter().map(String::as_str).collect::<Vec<_>>(), Some(q))?[0]));
    }
    match names.as_slice() {
        [one] => Ok(dir.join(one)),
        [] => Err(format!("{}: no GGUF file for llama.cpp", dir.display())),
        many => Err(format!("{}: several GGUF files ({}); set gguf = \"<file>\" in goethe_card.toml", dir.display(), many.join(", "))),
    }
}

pub fn pick_gguf(names: &[&str], quant: Option<&str>) -> Result<Vec<String>, String> {
    // Of a repo's files: the first model GGUF by name whose name holds `quant` (any case), else Q4_K_M, else Q8_0, else the first; with every shard of a split.
    let mut models: Vec<&str> = names.iter().copied().filter(|n| is_model_gguf(n)).collect();
    models.sort();
    let has = |n: &&&str, q: &str| n.to_ascii_uppercase().contains(&q.to_ascii_uppercase());
    let pick = match quant {
        Some(q) => models.iter().find(|n| has(n, q)).ok_or_else(|| format!("no GGUF file matches {q} (available: {})", quants(&models)))?,
        None => match ["Q4_K_M", "Q8_0"].iter().find_map(|q| models.iter().find(|n| has(n, q))).or(models.first()) {
            Some(p) => p,
            None => return Ok(Vec::new()),
        },
    };
    let stem = pick.split("-00001-of-").next().unwrap_or(pick);
    let mut out: Vec<String> = names.iter().filter(|n| *n == pick || (n.starts_with(&format!("{stem}-000")) && n.ends_with(".gguf"))).map(|n| n.to_string()).collect();
    out.sort();
    Ok(out)
}

fn quants(models: &[&str]) -> String {
    // The quant tag of each model GGUF (the last '-' or '.' part of its name, shard suffix aside), for a refusal.
    let mut q: Vec<&str> = models
        .iter()
        .map(|&n| n.rsplit('/').next().unwrap_or(n).trim_end_matches(".gguf").split("-00001-of-").next().unwrap_or(n).rsplit(['-', '.']).next().unwrap_or(n))
        .collect();
    q.sort();
    q.dedup();
    match q.is_empty() {
        true => "none".into(),
        false => q.join(", "),
    }
}

pub fn is_embedding(gguf: &Path) -> bool {
    // An embedding model: its header names pooling 1 (mean), 2 (cls) or 3 (last); 0 (none) and 4 (rank) are not served as embeddings.
    header(gguf).is_ok_and(|m| m.iter().any(|(k, v)| k.ends_with(".pooling_type") && v.0.first().is_some_and(|p| (1..=3).contains(p))))
}

pub fn facts(dir: &Path, gguf: &Path) -> Result<crate::model_facts::ModelFacts, String> {
    // Shape from the GGUF header; every layer's KV counted at full context (sliding windows overcounted, never under).
    let m = header(gguf).map_err(|e| format!("{}: {e}", gguf.display()))?;
    let arch = m.get("general.architecture").map(|v| v.1.clone()).filter(|a| !a.is_empty()).ok_or_else(|| format!("{}: no general.architecture", gguf.display()))?;
    let nums = |k: &str| m.get(&format!("{arch}.{k}")).map(|v| v.0.clone()).unwrap_or_default();
    let max = |k: &str| nums(k).into_iter().max().unwrap_or(0);
    let (layers, heads) = (max("block_count"), max("attention.head_count"));
    let mut kv = nums("attention.head_count_kv");
    if kv.len() < 2 {
        kv = vec![kv.first().copied().unwrap_or(heads); layers as usize];
    }
    let dim = |k: &str| Some(max(k)).filter(|d| *d > 0).unwrap_or(max("embedding_length") / heads.max(1));
    let weights = gguf_paths(dir).into_iter().filter(|p| !p.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains("mmproj")));
    let bytes: u64 = weights.filter_map(|p| p.metadata().ok()).map(|m| m.len()).sum();
    Ok(crate::model_facts::ModelFacts {
        architecture: if max("expert_count") > 0 { "moe" } else { "dense" }.into(),
        static_vram_gb: bytes as f64 / crate::model_facts::GIB,
        kv_elems_per_token: kv.iter().sum::<u64>() * (dim("attention.key_length") + dim("attention.value_length")),
        max_ctx: max("context_length"),
        num_attention_heads: heads,
        num_kv_heads: kv.iter().copied().max().unwrap_or(0),
        attention: "gqa".into(),
        swa_kv_elems_per_request: 0,
        linear_state_bytes_per_request: 0,
    })
}

fn le(r: &mut impl Read, n: usize) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b[..n])?;
    Ok(u64::from_le_bytes(b))
}

fn text(r: &mut impl Read) -> std::io::Result<String> {
    let mut b = vec![0u8; le(r, 8)? as usize];
    r.read_exact(&mut b)?;
    Ok(String::from_utf8_lossy(&b).into_owned())
}

fn value(r: &mut impl Read, t: u64, nums: &mut Vec<u64>, s: &mut String) -> std::io::Result<()> {
    // One GGUF value: integers (floats as raw bits, never read as shape), a string, or an array of either.
    match t {
        0 | 1 | 7 => nums.push(le(r, 1)?),
        2 | 3 => nums.push(le(r, 2)?),
        4..=6 => nums.push(le(r, 4)?),
        10..=12 => nums.push(le(r, 8)?),
        8 => *s = text(r)?,
        9 => {
            let (t, n) = (le(r, 4)?, le(r, 8)?);
            for _ in 0..n {
                value(r, t, nums, s)?;
            }
        }
        _ => return Err(std::io::Error::other(format!("GGUF value type {t}"))),
    }
    Ok(())
}

fn header(path: &Path) -> std::io::Result<HashMap<String, (Vec<u64>, String)>> {
    // The key-value section of a GGUF v2/v3 file.
    let mut r = std::io::BufReader::new(std::fs::File::open(path)?);
    if le(&mut r, 4)? != u64::from(u32::from_le_bytes(*b"GGUF")) || le(&mut r, 4)? < 2 {
        return Err(std::io::Error::other("not a GGUF v2+ file"));
    }
    let (_tensors, n) = (le(&mut r, 8)?, le(&mut r, 8)?);
    let mut m = HashMap::new();
    for _ in 0..n {
        let (k, t) = (text(&mut r)?, le(&mut r, 4)?);
        let (mut nums, mut s) = (Vec::new(), String::new());
        value(&mut r, t, &mut nums, &mut s)?;
        m.insert(k, (nums, s));
    }
    Ok(m)
}

fn llama_command(spec: &EngineSpec) -> Result<Command, String> {
    // One slot per parallel request in one KV pool of the size goethe budgeted, each capped at the context.
    if spec.gpus.len() != 1 || spec.tp != 1 {
        return Err(format!("llama.cpp runs on one GPU in goethe; this load asks for {} (tp {})", spec.gpus.len(), spec.tp));
    }
    let mut cmd = Command::new(server_bin(spec.home)?);
    cmd.arg("-m")
        .arg(gguf_in(spec.model_path, spec.card, None)?)
        .args(["--host", "127.0.0.1", "--port"])
        .arg(spec.port.to_string())
        .arg("--api-key")
        .arg(spec.engine_key)
        .arg("--alias")
        .arg(spec.served_name)
        .arg("-c")
        .arg(crate::chooser::kv_tokens(spec.ctx, spec.max_conc).to_string())
        .arg("-np")
        .arg(spec.max_conc.to_string())
        .arg("-kvu")
        .arg("--kv-unified-per-slot")
        .arg(spec.ctx.to_string())
        .args(["-ngl", "all", "--no-webui"]);
    if spec.card.get("task").and_then(|v| v.as_str()) == Some("embed") {
        cmd.arg("--embeddings");
    }
    cmd.env_clear()
        .envs(kept_env())
        .envs(card_env(spec))
        .envs(placement_env(spec));
    Ok(cmd)
}

#[cfg(test)]
pub fn write_test_gguf(path: &Path) {
    write_test_gguf_with(path, &[]);
}

#[cfg(test)]
pub fn write_test_gguf_with(path: &Path, extra: &[(&str, u32)]) {
    // A llama header: 2 layers, 8 heads of 64, per-layer KV heads 2 and 4, context 4096, a token list to skip, and extra u32 keys.
    let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
    let arr = |t: u32, items: Vec<Vec<u8>>| [t.to_le_bytes().to_vec(), (items.len() as u64).to_le_bytes().to_vec(), items.concat()].concat();
    let mut kvs = vec![("general.architecture".to_string(), 8u32, s("llama")), ("tokenizer.ggml.tokens".into(), 9, arr(8, vec![s("a"), s("bc")]))];
    for (k, v) in [("block_count", 2u32), ("attention.head_count", 8), ("embedding_length", 512), ("context_length", 4096)] {
        kvs.push((format!("llama.{k}"), 4, v.to_le_bytes().to_vec()));
    }
    kvs.push(("llama.attention.head_count_kv".into(), 9, arr(4, vec![2u32.to_le_bytes().to_vec(), 4u32.to_le_bytes().to_vec()])));
    kvs.extend(extra.iter().map(|(k, v)| (k.to_string(), 4u32, v.to_le_bytes().to_vec())));
    let mut g = [b"GGUF".to_vec(), 3u32.to_le_bytes().to_vec(), 0u64.to_le_bytes().to_vec(), (kvs.len() as u64).to_le_bytes().to_vec()].concat();
    for (k, t, v) in kvs {
        g.extend([s(&k), t.to_le_bytes().to_vec(), v].concat());
    }
    std::fs::write(path, g).unwrap();
}

#[cfg(test)]
mod tests {
    use super::super::{check_params, command, test_spec};
    use super::*;
    use serde_json::json;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn launches_llama_server_on_the_gguf_with_goethe_budget_and_one_gpu() {
        let d = tempfile::tempdir().unwrap();
        let bin = d.path().join("engines/llama.cpp/bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("llama-server"), b"").unwrap();
        for f in ["m-Q4_K_M.gguf", "mmproj-m.gguf", "config.json"] {
            std::fs::write(d.path().join(f), b"x").unwrap();
        }
        let card: toml::Value = toml::from_str("runtime = \"llama.cpp\"").unwrap();
        let c = command(&test_spec(d.path(), &card, &[2])).unwrap();
        let a: Vec<String> = c.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        let at = |f: &str| a[a.iter().position(|x| x == f).unwrap() + 1].clone();
        assert!(at("-m").ends_with("m-Q4_K_M.gguf"));
        assert_eq!((at("--host").as_str(), at("--alias").as_str(), at("-np").as_str()), ("127.0.0.1", "org/m", "4"));
        assert_eq!((at("-c"), at("--kv-unified-per-slot")), (crate::chooser::kv_tokens(4096, 4).to_string(), "4096".to_string()));
        assert!(!a.iter().any(|x| x == "--embeddings"));
        let env = |k: &str| c.get_envs().find(|(n, _)| *n == k).and_then(|(_, v)| v).map(|v| v.to_string_lossy().into_owned());
        assert_eq!(env("CUDA_VISIBLE_DEVICES").as_deref(), Some("2"));
        let emb: toml::Value = toml::from_str("runtime = \"llama.cpp\"\ntask = \"embed\"").unwrap();
        assert!(command(&test_spec(d.path(), &emb, &[2])).unwrap().get_args().any(|x| x == "--embeddings"));
        assert!(command(&test_spec(d.path(), &card, &[2, 3])).unwrap_err().contains("one GPU"));
        std::fs::write(d.path().join("m-Q8_0.gguf"), b"x").unwrap();
        assert!(command(&test_spec(d.path(), &card, &[2])).unwrap_err().contains("several GGUF"));
        let pick: toml::Value = toml::from_str("runtime = \"llama.cpp\"\ngguf = \"m-Q8_0.gguf\"").unwrap();
        assert!(command(&test_spec(d.path(), &pick, &[2])).is_ok());
    }

    #[test]
    fn a_missing_server_names_the_pin() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("m.gguf"), b"x").unwrap();
        let card: toml::Value = toml::from_str("runtime = \"llama.cpp\"").unwrap();
        if std::env::var_os("LLAMA_SERVER").is_none() {
            assert!(command(&test_spec(d.path(), &card, &[0])).unwrap_err().contains(PIN));
        }
    }

    fn archive(files: &[&str]) -> Vec<u8> {
        // A tar.gz holding the given files (bin/llama-server executable), as the release workflow packs it.
        let d = tempfile::tempdir().unwrap();
        for f in files {
            let p = d.path().join("in").join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b"#!/bin/sh\n").unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let out = d.path().join("a.tar.gz");
        assert!(Command::new("tar").arg("-czf").arg(&out).arg("-C").arg(d.path().join("in")).arg(".").status().unwrap().success());
        std::fs::read(out).unwrap()
    }

    fn serve(bytes: Vec<u8>) -> impl Fn(&str) -> Result<Box<dyn Read>, String> {
        move |_: &str| Ok(Box::new(std::io::Cursor::new(bytes.clone())) as Box<dyn Read>)
    }

    fn sha(bytes: &[u8]) -> String {
        use sha2::Digest;
        format!("{:x}", sha2::Sha256::digest(bytes))
    }

    const URL: &str = "https://example.invalid/llama-server.tar.gz";

    #[test]
    fn the_pinned_archive_is_verified_unpacked_and_moved_in_whole() {
        let home = tempfile::tempdir().unwrap();
        let a = archive(&["bin/llama-server", "lib/libcudart.so.12"]);
        let p = provision(home.path(), (URL, &sha(&a)), &serve(a.clone())).unwrap();
        assert_eq!(p, home.path().join("engines/llama.cpp/bin/llama-server"));
        assert!(p.is_file() && home.path().join("engines/llama.cpp/lib/libcudart.so.12").is_file());
        assert!(std::fs::metadata(&p).unwrap().permissions().mode() & 0o111 != 0, "the server stays executable");
        assert!(!home.path().join("engines/.llama.cpp.part").exists());
        if std::env::var_os("LLAMA_SERVER").is_none() {
            assert_eq!(ensure_server(home.path()).unwrap(), p, "installed: never downloaded again");
        }
    }

    #[test]
    fn a_wrong_hash_a_refused_download_or_a_foreign_archive_install_nothing() {
        let home = tempfile::tempdir().unwrap();
        let a = archive(&["bin/llama-server"]);
        let err = provision(home.path(), (URL, &"0".repeat(64)), &serve(a)).unwrap_err();
        assert!(err.contains("sha256") && err.contains("refused"), "{err}");
        let offline = |_: &str| -> Result<Box<dyn Read>, String> { Err("connection refused".into()) };
        let err = provision(home.path(), (URL, &"0".repeat(64)), &offline).unwrap_err();
        assert!(err.contains("cannot download llama-server") && err.contains("connection refused") && err.contains("LLAMA_SERVER"), "{err}");
        let other = archive(&["bin/something-else"]);
        let err = provision(home.path(), (URL, &sha(&other)), &serve(other)).unwrap_err();
        assert!(err.contains("no bin/llama-server"), "{err}");
        assert!(!home.path().join("engines/llama.cpp").exists() && !home.path().join("engines/.llama.cpp.part").exists());
    }

    #[test]
    fn only_a_released_x86_64_archive_is_pinned() {
        let sha = "ab".repeat(32);
        assert_eq!(pinned((URL, &sha), "x86_64"), Some((URL, sha.as_str())));
        assert_eq!(pinned(("", ""), "x86_64"), None, "no asset yet: the clear error and LLAMA_SERVER stay");
        assert_eq!(pinned((URL, &sha), "aarch64"), None);
        assert_eq!(pinned((URL, "abc"), "x86_64"), None);
        assert_eq!(pinned(("http://plain.invalid/a.tar.gz", &sha), "x86_64"), None);
        assert_eq!(PIN.len(), 40, "the commit comes from llama-cpp/PIN");
    }

    #[test]
    fn a_gguf_with_a_pooling_type_is_an_embedding_model() {
        let d = tempfile::tempdir().unwrap();
        write_test_gguf(&d.path().join("g.gguf"));
        let is = |name: &str, pooling: &[(&str, u32)]| {
            write_test_gguf_with(&d.path().join(name), pooling);
            is_embedding(&d.path().join(name))
        };
        assert!(!is_embedding(&d.path().join("g.gguf")));
        assert_eq!([0, 1, 2, 3, 4].map(|p| is(&format!("p{p}.gguf"), &[("llama.pooling_type", p)])), [false, true, true, true, false]);
    }

    #[test]
    fn a_split_gguf_is_served_from_its_first_shard() {
        let d = tempfile::tempdir().unwrap();
        for f in ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"] {
            std::fs::write(d.path().join(f), b"x").unwrap();
        }
        assert!(gguf_in(d.path(), &toml::Value::Table(Default::default()), None).unwrap().ends_with("m-00001-of-00002.gguf"));
        std::fs::write(d.path().join("m-Q8_0.gguf"), b"x").unwrap();
        let card: toml::Value = toml::from_str("gguf = \"m-00001-of-00002.gguf\"").unwrap();
        assert!(gguf_in(d.path(), &card, Some("q8_0")).unwrap().ends_with("m-Q8_0.gguf"), "a quant overrides the card's file");
    }

    #[test]
    fn conformance_the_same_request_is_served_or_refused_by_name() {
        // Request, then the parameter each engine refuses (None: passed on; goethe-engine checks its pack's io.params itself).
        let ge = &super::super::goethe_engine::ENGINE;
        let cases = [
            (json!({"model": "m", "messages": [], "temperature": 0, "max_tokens": 8, "stop": ["x"], "seed": 1, "stream": true}), None),
            (json!({"model": "m", "prompt": "hi", "top_p": 0.9, "top_k": 40, "n": 1, "logprobs": 2}), None),
            (json!({"model": "m", "messages": [], "structured": {"kind": "regex", "spec": "a"}}), Some("structured")),
            (json!({"model": "m", "input": ["a"], "encoding_format": "float"}), None),
            (json!({"model": "m", "input": ["a"], "dimensions": 64}), Some("dimensions")),
            (json!({"model": "m", "input": ["a"], "normalize": null}), None),
        ];
        for (body, refused) in cases {
            assert_eq!(check_params(&ENGINE, body.to_string().as_bytes()).err().as_deref(), refused, "{body}");
            assert!(check_params(ge, body.to_string().as_bytes()).is_ok(), "goethe-engine refuses in the engine");
        }
    }

    #[test]
    fn facts_come_from_the_gguf_header_and_the_install_picks_one_quant() {
        let d = tempfile::tempdir().unwrap();
        std::fs::create_dir(d.path().join("Q4_K_M")).unwrap();
        write_test_gguf(&d.path().join("Q4_K_M/m.gguf"));
        let f = facts(d.path(), &gguf_in(d.path(), &toml::Value::Table(Default::default()), None).unwrap()).unwrap();
        assert_eq!((f.kv_elems_per_token, f.max_ctx, f.num_attention_heads, f.num_kv_heads), ((2 + 4) * 128, 4096, 8, 4));
        assert_eq!(f.architecture, "dense");
        let names = ["m-Q8_0.gguf", "Q4_K_M/m-Q4_K_M-00002-of-00002.gguf", "Q4_K_M/m-Q4_K_M-00001-of-00002.gguf", "mmproj-m.gguf", "m.mmproj-f16.gguf", "README.md"];
        let split = ["Q4_K_M/m-Q4_K_M-00001-of-00002.gguf", "Q4_K_M/m-Q4_K_M-00002-of-00002.gguf"];
        assert_eq!(pick_gguf(&names, None).unwrap(), split);
        assert_eq!(pick_gguf(&["b-IQ2.gguf", "a-IQ3.gguf"], None).unwrap(), ["a-IQ3.gguf"]);
        assert_eq!(pick_gguf(&names, Some("q8_0")).unwrap(), ["m-Q8_0.gguf"]);
        assert_eq!(pick_gguf(&names, Some("Q4_K_M")).unwrap(), split, "every shard of a split");
        assert_eq!(pick_gguf(&names, Some("IQ2_XS")).unwrap_err(), "no GGUF file matches IQ2_XS (available: Q4_K_M, Q8_0)");
        assert_eq!(pick_gguf(&["README.md"], Some("Q8_0")).unwrap_err(), "no GGUF file matches Q8_0 (available: none)");
    }
}
