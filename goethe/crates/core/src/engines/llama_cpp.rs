// llama.cpp: llama-server on one GGUF, one GPU, OpenAI-compatible; the universal engine for GGUF models.

use super::{card_env, engine_key_env, kept_env, placement_env, EngineSpec};
use std::collections::HashMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;

const PIN_FILE: &str = include_str!("../../../../llama-cpp/PIN");
const ARCHS: &str = include_str!("../../../../llama-cpp/ARCHS");

pub fn serves_arch(arch: &str) -> bool {
    // A GGUF general.architecture the pinned llama.cpp build has a model for (llama-cpp/ARCHS, written by update_archs.sh).
    ARCHS.lines().filter(|l| !l.starts_with('#')).any(|l| l == arch)
}

pub const PIN: &str = crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_COMMIT");
pub const TAG: &str = crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_TAG");
const MAIN: (&str, &str) = (crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_URL"), crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_SHA256"));
const CUDART: (&str, &str) = (crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_CUDART_URL"), crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_CUDART_SHA256"));
const MACOS: (&str, &str) = (crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_MACOS_URL"), crate::pyenv::pin_value(PIN_FILE, "LLAMA_CPP_MACOS_SHA256"));

type Get<'a> = &'a dyn Fn(&str) -> Result<Box<dyn Read>, String>;

pub const ENGINE: super::Engine = super::Engine {
    id: "llama.cpp",
    overhead_gb: crate::calc::METAL_OVERHEAD_GB,
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
    // $LLAMA_SERVER, else the one under GOETHE_HOME (the flat upstream layout, or bin/ for a hand-installed build); no PATH lookup.
    let dir = home.join("engines/llama.cpp");
    let p = std::env::var_os("LLAMA_SERVER").map(PathBuf::from).unwrap_or_else(|| {
        [dir.join("llama-server"), dir.join("bin/llama-server")].into_iter().find(|p| p.is_file()).unwrap_or_else(|| dir.join("llama-server"))
    });
    if p.is_file() {
        Ok(p)
    } else {
        Err(format!("llama-server not found at {}: install llama.cpp {TAG} ({PIN}) there or set LLAMA_SERVER", p.display()))
    }
}

pub fn pinned<'a>(asset: (&'a str, &'a str), arch: &str) -> Option<(&'a str, &'a str)> {
    // The upstream release archive of the pin: Linux x86_64 only, with an https URL and a sha256.
    let (url, sha) = asset;
    (arch == "x86_64" && url.starts_with("https://") && sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit())).then_some(asset)
}

pub fn pinned_macos<'a>(asset: (&'a str, &'a str), arch: &str) -> Option<(&'a str, &'a str)> {
    // The upstream macOS arm64 build (Metal) of the pin, for Apple Silicon.
    let (url, sha) = asset;
    (arch == "aarch64" && url.starts_with("https://") && sha.len() == 64 && sha.bytes().all(|b| b.is_ascii_hexdigit())).then_some(asset)
}

fn platform_asset() -> Option<(&'static str, &'static str)> {
    // The archive for this machine: the Metal build on a Mac, the CUDA build on Linux x86_64.
    let arch = crate::update::normalize_arch();
    if cfg!(target_os = "macos") { pinned_macos(MACOS, &arch) } else { pinned(MAIN, &arch) }
}

pub fn server_ready(home: &Path) -> Result<(), String> {
    // Installed, or downloadable on first use: what an install plan needs, without touching the network.
    match server_bin(home) {
        Ok(_) => Ok(()),
        Err(why) if std::env::var_os("LLAMA_SERVER").is_some() => Err(why),
        Err(why) => platform_asset().map(|_| ()).ok_or(why),
    }
}

pub fn ensure_server(home: &Path) -> Result<PathBuf, String> {
    // The installed llama-server, else the pinned upstream release downloaded once; a set LLAMA_SERVER is never replaced.
    match server_bin(home) {
        Ok(p) => Ok(p),
        Err(why) if std::env::var_os("LLAMA_SERVER").is_some() => Err(why),
        Err(why) if cfg!(target_os = "macos") => match platform_asset() {
            Some(main) => provision(home, main, ("", ""), &http_get, &|_| false),
            None => Err(why),
        },
        Err(why) => match (pinned(MAIN, &crate::update::normalize_arch()), pinned(CUDART, &crate::update::normalize_arch())) {
            (Some(main), Some(cudart)) => provision(home, main, cudart, &http_get, &cuda_libs_missing),
            _ => Err(why),
        },
    }
}

fn http_get(url: &str) -> Result<Box<dyn Read>, String> {
    let agent = crate::net::agent_builder().timeout_read(std::time::Duration::from_secs(60)).build();
    let r = agent.get(url).call().map_err(|e| e.to_string())?;
    Ok(Box::new(r.into_reader()))
}

fn cuda_libs_missing(dir: &Path) -> bool {
    // ldd of the CUDA backend as unpacked (RUNPATH $ORIGIN included): one of the three CUDA 12 runtime libraries it needs is not found on this host.
    let Ok(out) = Command::new("ldd").arg(dir.join("libggml-cuda.so")).output() else { return false };
    let text = String::from_utf8_lossy(&out.stdout);
    ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"].iter().any(|lib| text.lines().any(|l| l.contains(lib) && l.contains("not found")))
}

fn provision(home: &Path, main: (&str, &str), cudart: (&str, &str), get: Get, missing: &dyn Fn(&Path) -> bool) -> Result<PathBuf, String> {
    // The release archive at its pinned sha256, plus the CUDA runtime archive only when this host lacks the libraries, unpacked aside and moved flat into engines/llama.cpp: a failure leaves nothing.
    let dest = home.join("engines/llama.cpp");
    let part = home.join("engines/.llama.cpp.part");
    let build = if cfg!(target_os = "macos") { "Metal build" } else { "CUDA build" };
    println!("llama-server is not installed: downloading llama.cpp {TAG} ({build}, once) from {}", main.0);
    let _ = std::fs::remove_dir_all(&part);
    let done = unpack(&part, &dest, main, cudart, get, missing);
    let _ = std::fs::remove_dir_all(&part);
    done.map(|()| dest.join("llama-server"))
}

fn fetch_into(part: &Path, x: &Path, (url, sha): (&str, &str), get: Get, name: &str) -> Result<(), String> {
    // One archive downloaded, checked against its pinned sha256 and unpacked into x without its top folder.
    let offline = |e: String| format!("cannot download {name} from {url}: {e}; nothing was installed. Install llama.cpp {TAG} into engines/llama.cpp or set LLAMA_SERVER");
    let tgz = part.join(format!("{name}.tar.gz"));
    let mut src = get(url).map_err(offline)?;
    let mut f = std::fs::File::create(&tgz).map_err(|e| format!("{}: {e}", tgz.display()))?;
    std::io::copy(&mut src, &mut f).map_err(|e| offline(e.to_string()))?;
    let got = crate::update::sha256_file(&tgz)?;
    if !got.eq_ignore_ascii_case(sha) {
        return Err(format!("{name} download from {url} has sha256 {got}, the pin says {sha}: refused, nothing was installed"));
    }
    let ok = Command::new("tar").arg("-xzf").arg(&tgz).arg("-C").arg(x).arg("--strip-components=1").arg("--no-same-owner").status().map_err(|e| format!("tar: {e}"))?.success();
    if ok {
        Ok(())
    } else {
        Err(format!("{url}: not a tar archive: refused, nothing was installed"))
    }
}

fn unpack(part: &Path, dest: &Path, main: (&str, &str), cudart: (&str, &str), get: Get, missing: &dyn Fn(&Path) -> bool) -> Result<(), String> {
    let x = part.join("x");
    std::fs::create_dir_all(&x).map_err(|e| format!("{}: {e}", x.display()))?;
    fetch_into(part, &x, main, get, "llama-server")?;
    if !x.join("llama-server").is_file() {
        return Err(format!("{}: not a llama-server archive (no llama-server): refused, nothing was installed", main.0));
    }
    if missing(&x) {
        println!("the CUDA 12 runtime libraries are not on this host: downloading them beside llama-server (once) from {}", cudart.0);
        fetch_into(part, &x, cudart, get, "cudart")?;
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

pub fn is_model_gguf(name: &str) -> bool {
    // A model file or a split's first shard; mmproj files are vision projectors, mtp-, dflash- and eagle3- files drafters.
    let base = name.rsplit('/').next().unwrap_or(name);
    let lower = base.to_ascii_lowercase();
    let drafter = ["mtp-", "dflash-", "eagle3-", "eagle-", "draft-"].iter().any(|p| lower.starts_with(p));
    base.ends_with(".gguf") && !lower.contains("mmproj") && !drafter && (!base.contains("-of-") || base.contains("-00001-of-"))
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

pub fn architecture(gguf: &Path) -> Option<String> {
    // general.architecture of a GGUF file, from its header.
    header(gguf).ok()?.get("general.architecture").map(|v| v.1.clone()).filter(|a| !a.is_empty())
}

pub fn is_embedding(gguf: &Path) -> bool {
    // An embedding model: its header names pooling 1 (mean), 2 (cls) or 3 (last); 0 (none) and 4 (rank) are not served as embeddings.
    header(gguf).is_ok_and(|m| m.iter().any(|(k, v)| k.ends_with(".pooling_type") && v.0.first().is_some_and(|p| (1..=3).contains(p))))
}

pub fn mmproj_in(dir: &Path) -> Option<PathBuf> {
    // The multimodal projector GGUF beside the model (named mmproj-X or X.mmproj-f16), the first by name.
    gguf_paths(dir).into_iter().find(|p| p.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains("mmproj")))
}

pub fn has_audio_encoder(mmproj: &Path) -> bool {
    // A projector whose header says it carries an audio encoder (clip.has_audio_encoder, or any clip.audio.* key): the model transcribes speech.
    header(mmproj).is_ok_and(|m| m.iter().any(|(k, v)| (k == "clip.has_audio_encoder" && v.0.first() == Some(&1)) || k.starts_with("clip.audio.")))
}

const SWA_BATCH: u64 = 512;

pub fn sliding_layers(arch: &str, layers: usize, window: u64, pattern: &[u64]) -> Vec<bool> {
    // Which layers attend over a sliding window, as llama.cpp reads them: a per-layer pattern, a period (every n-th layer is full), or the period it builds in for its architecture.
    let period = |n: usize| (0..layers).map(|i| (i + 1) % n != 0).collect();
    match (window, pattern) {
        (0, _) => vec![false; layers],
        (_, p) if p.len() == layers => p.iter().map(|v| *v != 0).collect(),
        (_, [n]) if *n > 1 => period(*n as usize),
        _ => match arch {
            "gemma3" => period(6),
            "gemma2" => period(2),
            "cohere2" => period(4),
            _ => vec![false; layers],
        },
    }
}

pub fn with_header_kv(mut input: crate::chooser::ChooserInput, dir: &Path, card: &toml::Value) -> crate::chooser::ChooserInput {
    // A llama.cpp model's KV from its GGUF header, read at every plan so a model installed before the header was read this way gets it too (KV in f16).
    if card.get("runtime").and_then(|v| v.as_str()) != Some(ENGINE.id) {
        return input;
    }
    if let Some(f) = gguf_in(dir, card, None).ok().and_then(|g| facts(dir, &g).ok()) {
        input.kv_bytes_per_token = f.kv_elems_per_token * 2;
        input.window_gb_per_request = (f.swa_kv_elems_per_request * 2) as f64 / crate::model_facts::GIB;
    }
    input
}

pub fn facts(dir: &Path, gguf: &Path) -> Result<crate::model_facts::ModelFacts, String> {
    // Shape from the GGUF header: full-attention layers hold KV for the whole context, sliding-window layers for their window (plus a batch) per request, and layers that share an earlier layer's KV hold none.
    let m = header(gguf).map_err(|e| format!("{}: {e}", gguf.display()))?;
    let arch = m.get("general.architecture").map(|v| v.1.clone()).filter(|a| !a.is_empty()).ok_or_else(|| format!("{}: no general.architecture", gguf.display()))?;
    let nums = |k: &str| m.get(&format!("{arch}.{k}")).map(|v| v.0.clone()).unwrap_or_default();
    let max = |k: &str| nums(k).into_iter().max().unwrap_or(0);
    let (layers, heads) = (max("block_count"), max("attention.head_count"));
    if layers > MAX_LAYERS {
        return Err(format!("{}: {arch}.block_count {layers} is not a model", gguf.display()));
    }
    let mut kv = nums("attention.head_count_kv");
    if kv.len() < 2 {
        kv = vec![kv.first().copied().unwrap_or(heads); layers as usize];
    }
    let dim = |k: &str| Some(max(k)).filter(|d| *d > 0).unwrap_or(max("embedding_length") / heads.max(1));
    let (k, v) = (dim("attention.key_length"), dim("attention.value_length"));
    let (ks, vs) = (Some(max("attention.key_length_swa")).filter(|d| *d > 0).unwrap_or(k), Some(max("attention.value_length_swa")).filter(|d| *d > 0).unwrap_or(v));
    let n = layers as usize;
    let window = max("attention.sliding_window");
    let sliding = sliding_layers(&arch, n, window, &nums("attention.sliding_window_pattern"));
    let own = n.saturating_sub(max("attention.shared_kv_layers") as usize);
    let heads_at = |i: usize| kv.get(i).copied().unwrap_or(0);
    let full: u64 = (0..own).filter(|&i| !sliding[i]).map(|i| heads_at(i) * (k + v)).sum();
    let windowed: u64 = (0..own).filter(|&i| sliding[i]).map(|i| heads_at(i) * (ks + vs)).sum();
    let weights = gguf_paths(dir).into_iter().filter(|p| !p.file_name().is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains("mmproj")));
    let bytes: u64 = weights.filter_map(|p| p.metadata().ok()).map(|m| m.len()).sum();
    Ok(crate::model_facts::ModelFacts {
        architecture: if max("expert_count") > 0 { "moe" } else { "dense" }.into(),
        static_vram_gb: bytes as f64 / crate::model_facts::GIB,
        kv_elems_per_token: full.max(1),
        max_ctx: max("context_length"),
        num_attention_heads: heads,
        num_kv_heads: kv.iter().copied().max().unwrap_or(0),
        attention: "gqa".into(),
        swa_kv_elems_per_request: windowed * (window.min(max("context_length").max(1)) + SWA_BATCH),
        linear_state_bytes_per_request: 0,
    })
}

fn le(r: &mut impl Read, n: usize) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b[..n])?;
    Ok(u64::from_le_bytes(b))
}

fn text(r: &mut impl Read) -> std::io::Result<String> {
    // A length the file cannot back is read as far as the file goes, never allocated up front.
    let n = le(r, 8)?;
    if n > MAX_TEXT {
        return Err(std::io::Error::other(format!("GGUF string of {n} bytes")));
    }
    let mut b = Vec::new();
    r.take(n).read_to_end(&mut b)?;
    if b.len() as u64 != n {
        return Err(std::io::ErrorKind::UnexpectedEof.into());
    }
    Ok(String::from_utf8_lossy(&b).into_owned())
}

const MAX_TEXT: u64 = 1 << 24;
const MAX_ARRAY_DEPTH: u32 = 4;
const MAX_LAYERS: u64 = 1 << 16;

fn value(r: &mut impl Read, t: u64, depth: u32, nums: &mut Vec<u64>, s: &mut String) -> std::io::Result<()> {
    // One GGUF value: integers (floats as raw bits, never read as shape), a string, or an array of either.
    match t {
        0 | 1 | 7 => nums.push(le(r, 1)?),
        2 | 3 => nums.push(le(r, 2)?),
        4..=6 => nums.push(le(r, 4)?),
        10..=12 => nums.push(le(r, 8)?),
        8 => *s = text(r)?,
        9 if depth >= MAX_ARRAY_DEPTH => return Err(std::io::Error::other("GGUF arrays nested too deep")),
        9 => {
            let (t, n) = (le(r, 4)?, le(r, 8)?);
            for _ in 0..n {
                value(r, t, depth + 1, nums, s)?;
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
        value(&mut r, t, 0, &mut nums, &mut s)?;
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
        .arg("--alias")
        .arg(spec.served_name)
        .arg("-c")
        .arg(crate::chooser::kv_tokens(spec.ctx, spec.max_conc, crate::chooser::card_floor(spec.card)).to_string())
        .arg("-np")
        .arg(spec.max_conc.to_string())
        .arg("-kvu")
        .arg("--kv-unified-per-slot")
        .arg(spec.ctx.to_string())
        .args(["-ngl", "all", "--no-webui"]);
    if spec.card.get("task").and_then(|v| v.as_str()) == Some("embed") {
        cmd.arg("--embeddings");
    }
    if let Some(m) = spec.card.get("mmproj").and_then(|v| v.as_str()) {
        cmd.arg("--mmproj").arg(spec.model_path.join(m));
    }
    cmd.env_clear()
        .envs(kept_env())
        .envs(card_env(spec))
        .envs(placement_env(spec));
    engine_key_env(&mut cmd, spec.engine_key);
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
pub fn write_gguf(path: &Path, arch: &str, nums: &[(&str, Vec<u32>)], bools: &[(&str, Vec<bool>)]) {
    // A GGUF header of `arch` with u32 keys (an array when several values) and bool arrays.
    let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
    let arr = |t: u32, items: Vec<Vec<u8>>| [t.to_le_bytes().to_vec(), (items.len() as u64).to_le_bytes().to_vec(), items.concat()].concat();
    let mut kvs = vec![("general.architecture".to_string(), 8u32, s(arch))];
    for (k, v) in nums {
        let key = format!("{arch}.{k}");
        match v.as_slice() {
            [one] => kvs.push((key, 4, one.to_le_bytes().to_vec())),
            many => kvs.push((key, 9, arr(4, many.iter().map(|x| x.to_le_bytes().to_vec()).collect()))),
        }
    }
    for (k, v) in bools {
        kvs.push((format!("{arch}.{k}"), 9, arr(7, v.iter().map(|b| vec![u8::from(*b)]).collect())));
    }
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
    fn the_architecture_list_is_the_pinned_builds() {
        assert!(ARCHS.lines().next().unwrap().contains(&format!("llama.cpp {TAG} ")), "rerun llama-cpp/update_archs.sh after a pin change");
        assert!(serves_arch("gemma-embedding") && serves_arch("gemma4") && serves_arch("llama"));
        assert!(!serves_arch("clip") && !serves_arch("not-a-model") && !serves_arch(""));
    }

    #[test]
    fn sliding_and_shared_layers_hold_no_full_context_kv() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("m.gguf");
        let pattern: Vec<bool> = (0..10).map(|i| i % 5 != 4).collect();
        let nums = vec![
            ("block_count", vec![10]), ("attention.head_count", vec![8]), ("attention.head_count_kv", vec![1]), ("context_length", vec![131072]),
            ("attention.key_length", vec![512]), ("attention.value_length", vec![512]), ("attention.key_length_swa", vec![256]),
            ("attention.value_length_swa", vec![256]), ("attention.sliding_window", vec![512]), ("attention.shared_kv_layers", vec![5]),
        ];
        write_gguf(&p, "gemma4", &nums, &[("attention.sliding_window_pattern", pattern)]);
        let f = facts(d.path(), &p).unwrap();
        assert_eq!(f.kv_elems_per_token, 1024, "one full layer of the five that own KV");
        assert_eq!(f.swa_kv_elems_per_request, 4 * 512 * (512 + 512), "four sliding layers over their window and a batch");
        assert_eq!(sliding_layers("gemma3", 6, 1024, &[]), vec![true, true, true, true, true, false]);
        assert_eq!(sliding_layers("llama", 3, 4096, &[]), vec![false; 3], "an unknown pattern counts every layer in full");
        assert_eq!(sliding_layers("x", 4, 0, &[1, 1, 1, 1]), vec![false; 4], "no window, no sliding");
    }

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
        assert_eq!((at("-c"), at("--kv-unified-per-slot")), (crate::chooser::kv_tokens(4096, 4, crate::chooser::card_floor(&card)).to_string(), "4096".to_string()));
        assert!(!a.iter().any(|x| x == "--embeddings"));
        assert!(!a.iter().any(|x| x == "--api-key"), "key stays off argv: {a:?}");
        let env = |k: &str| c.get_envs().find(|(n, _)| *n == k).and_then(|(_, v)| v).map(|v| v.to_string_lossy().into_owned());
        assert_eq!(env("CUDA_VISIBLE_DEVICES").as_deref(), Some("2"));
        assert_eq!(env("LLAMA_API_KEY").as_deref().map(|s| s.len()), Some(32));
        assert_eq!(env(super::super::ENGINE_KEY_ENV).as_deref().map(|s| s.len()), Some(32));
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

    fn archive(top: &str, files: &[&str]) -> Vec<u8> {
        // A tar.gz holding the given files (executable scripts) under one top folder, as upstream's release archives are.
        let d = tempfile::tempdir().unwrap();
        for f in files {
            let p = d.path().join("in").join(top).join(f);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(&p, b"#!/bin/sh\n").unwrap();
            std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let out = d.path().join("a.tar.gz");
        assert!(Command::new("tar").arg("-czf").arg(&out).arg("-C").arg(d.path().join("in")).arg(top).status().unwrap().success());
        std::fs::read(out).unwrap()
    }

    fn serve(parts: Vec<(&'static str, Vec<u8>)>) -> impl Fn(&str) -> Result<Box<dyn Read>, String> {
        move |url: &str| match parts.iter().find(|(u, _)| *u == url) {
            Some((_, bytes)) => Ok(Box::new(std::io::Cursor::new(bytes.clone())) as Box<dyn Read>),
            None => Err(format!("unexpected download {url}")),
        }
    }

    fn sha(bytes: &[u8]) -> String {
        crate::sha::digest_hex(bytes)
    }

    const URL: &str = "https://example.invalid/llama-b1-bin.tar.gz";
    const CUDA_URL: &str = "https://example.invalid/cudart-llama-b1-bin.tar.gz";

    #[test]
    fn the_release_archive_is_verified_and_unpacked_flat_without_cuda_libraries_when_the_host_has_them() {
        let home = tempfile::tempdir().unwrap();
        let a = archive("llama-b1", &["llama-server", "libggml-cuda.so"]);
        let get = serve(vec![(URL, a.clone())]);
        let p = provision(home.path(), (URL, &sha(&a)), (CUDA_URL, &"0".repeat(64)), &get, &|_| false).unwrap();
        assert_eq!(p, home.path().join("engines/llama.cpp/llama-server"));
        assert!(home.path().join("engines/llama.cpp/libggml-cuda.so").is_file() && !home.path().join("engines/llama.cpp/llama-b1").exists());
        assert!(std::fs::metadata(&p).unwrap().permissions().mode() & 0o111 != 0, "the server stays executable");
        assert!(!home.path().join("engines/.llama.cpp.part").exists());
        if std::env::var_os("LLAMA_SERVER").is_none() {
            assert_eq!(server_bin(home.path()).unwrap(), p, "installed: found without a download");
            assert_eq!(ensure_server(home.path()).unwrap(), p);
        }
    }

    #[test]
    fn the_cuda_runtime_archive_is_fetched_only_when_the_host_lacks_the_libraries() {
        let home = tempfile::tempdir().unwrap();
        let a = archive("llama-b1", &["llama-server", "libggml-cuda.so"]);
        let c = archive("cudart-llama-b1", &["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"]);
        let get = serve(vec![(URL, a.clone()), (CUDA_URL, c.clone())]);
        let p = provision(home.path(), (URL, &sha(&a)), (CUDA_URL, &sha(&c)), &get, &|_| true).unwrap();
        let dir = home.path().join("engines/llama.cpp");
        assert!(p.is_file() && ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"].iter().all(|l| dir.join(l).is_file()), "beside llama-server");
        let other = tempfile::tempdir().unwrap();
        let bad = provision(other.path(), (URL, &sha(&a)), (CUDA_URL, &"0".repeat(64)), &get, &|_| true).unwrap_err();
        assert!(bad.contains("cudart download") && bad.contains("refused"), "{bad}");
        assert!(!other.path().join("engines/llama.cpp").exists(), "a bad runtime archive installs nothing, not even the server");
    }

    #[test]
    fn a_wrong_hash_a_refused_download_or_a_foreign_archive_install_nothing() {
        let home = tempfile::tempdir().unwrap();
        let a = archive("llama-b1", &["llama-server"]);
        let get = serve(vec![(URL, a)]);
        let err = provision(home.path(), (URL, &"0".repeat(64)), (CUDA_URL, ""), &get, &|_| false).unwrap_err();
        assert!(err.contains("sha256") && err.contains("refused"), "{err}");
        let offline = |_: &str| -> Result<Box<dyn Read>, String> { Err("connection refused".into()) };
        let err = provision(home.path(), (URL, &"0".repeat(64)), (CUDA_URL, ""), &offline, &|_| false).unwrap_err();
        assert!(err.contains("cannot download llama-server") && err.contains("connection refused") && err.contains("LLAMA_SERVER"), "{err}");
        let other = archive("llama-b1", &["something-else"]);
        let get = serve(vec![(URL, other.clone())]);
        let err = provision(home.path(), (URL, &sha(&other)), (CUDA_URL, ""), &get, &|_| false).unwrap_err();
        assert!(err.contains("no llama-server"), "{err}");
        assert!(!home.path().join("engines/llama.cpp").exists() && !home.path().join("engines/.llama.cpp.part").exists());
    }

    #[test]
    fn a_hand_installed_build_under_bin_is_still_found() {
        let home = tempfile::tempdir().unwrap();
        let bin = home.path().join("engines/llama.cpp/bin");
        std::fs::create_dir_all(&bin).unwrap();
        std::fs::write(bin.join("llama-server"), b"").unwrap();
        if std::env::var_os("LLAMA_SERVER").is_none() {
            assert_eq!(server_bin(home.path()).unwrap(), bin.join("llama-server"));
        }
    }

    #[test]
    fn the_macos_build_is_pinned_for_apple_silicon_only() {
        let sha = "a".repeat(64);
        assert_eq!(pinned_macos((URL, &sha), "aarch64"), Some((URL, sha.as_str())));
        assert_eq!(pinned_macos((URL, &sha), "x86_64"), None, "Intel Macs are not supported");
        assert!(pinned_macos(MACOS, "aarch64").is_some() && MACOS.0.contains(TAG) && MACOS.0.contains("macos-arm64"));
    }

    #[test]
    fn only_a_https_x86_64_archive_with_a_sha256_is_pinned_and_the_pin_names_the_release() {
        let sha = "ab".repeat(32);
        assert_eq!(pinned((URL, &sha), "x86_64"), Some((URL, sha.as_str())));
        assert_eq!(pinned(("", ""), "x86_64"), None, "no asset: the clear error and LLAMA_SERVER stay");
        assert_eq!(pinned((URL, &sha), "aarch64"), None);
        assert_eq!(pinned((URL, "abc"), "x86_64"), None);
        assert_eq!(pinned(("http://plain.invalid/a.tar.gz", &sha), "x86_64"), None);
        assert_eq!(PIN.len(), 40, "the commit comes from llama-cpp/PIN");
        assert!(pinned(MAIN, "x86_64").is_some() && pinned(CUDART, "x86_64").is_some() && MAIN.0.contains(TAG) && CUDART.0.contains(TAG));
    }

    fn hostile_gguf(kvs: &[(&str, u32, Vec<u8>)]) -> Vec<u8> {
        let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
        let mut g = [b"GGUF".to_vec(), 3u32.to_le_bytes().to_vec(), 0u64.to_le_bytes().to_vec(), (kvs.len() as u64).to_le_bytes().to_vec()].concat();
        for (k, t, v) in kvs {
            g.extend([s(k), t.to_le_bytes().to_vec(), v.clone()].concat());
        }
        g
    }

    #[test]
    fn a_hostile_gguf_header_is_refused_without_allocating_what_it_claims() {
        let d = tempfile::tempdir().unwrap();
        let s = |v: &str| [(v.len() as u64).to_le_bytes().to_vec(), v.as_bytes().to_vec()].concat();
        let cases = [
            ("a string that claims 2^62 bytes", hostile_gguf(&[("general.architecture", 8, (1u64 << 62).to_le_bytes().to_vec())])),
            ("4e9 layers", hostile_gguf(&[("general.architecture", 8, s("llama")), ("llama.block_count", 4, 4_000_000_000u32.to_le_bytes().to_vec())])),
            ("arrays nested 300000 deep", hostile_gguf(&[("general.architecture", 8, s("llama")), ("x", 9, [9u32.to_le_bytes().to_vec(), 1u64.to_le_bytes().to_vec()].concat().repeat(300_000))])),
        ];
        for (what, bytes) in cases {
            let p = d.path().join("m.gguf");
            std::fs::write(&p, bytes).unwrap();
            assert!(facts(d.path(), &p).is_err(), "{what}");
        }
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
    fn a_projector_with_an_audio_encoder_marks_a_speech_model() {
        let d = tempfile::tempdir().unwrap();
        write_test_gguf(&d.path().join("m.Q4_K_M.gguf"));
        assert_eq!(mmproj_in(d.path()), None);
        write_test_gguf_with(&d.path().join("m.mmproj-f16.gguf"), &[("clip.vision.image_size", 448)]);
        let vision = mmproj_in(d.path()).unwrap();
        assert!(vision.ends_with("m.mmproj-f16.gguf") && !has_audio_encoder(&vision));
        write_test_gguf_with(&vision, &[("clip.has_audio_encoder", 1)]);
        assert!(has_audio_encoder(&vision));
        write_test_gguf_with(&vision, &[("clip.has_audio_encoder", 0)]);
        assert!(!has_audio_encoder(&vision));
        write_test_gguf_with(&vision, &[("clip.audio.num_mel_bins", 128)]);
        assert!(has_audio_encoder(&vision));
        std::fs::create_dir_all(d.path().join("engines/llama.cpp")).unwrap();
        std::fs::write(d.path().join("engines/llama.cpp/llama-server"), b"").unwrap();
        let card: toml::Value = toml::from_str("runtime = \"llama.cpp\"\ngguf = \"m.Q4_K_M.gguf\"\nmmproj = \"m.mmproj-f16.gguf\"").unwrap();
        let c = command(&test_spec(d.path(), &card, &[0])).unwrap();
        let a: Vec<String> = c.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        let at = a.iter().position(|x| x == "--mmproj").expect("--mmproj");
        assert!(a[at + 1].ends_with("m.mmproj-f16.gguf"));
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
