// Before download: which engine serves this model (SGLang vs llama.cpp GGUF).

use crate::hf;
use crate::registry::RegistryModel;
use std::path::Path;

const GIB_MB: f64 = 1024.0;

pub fn gguf_need_mb(bytes: u64) -> u64 {
    // Weights plus room for a modest context and llama.cpp's buffers: what a GGUF needs to load on the GPU.
    let mb = bytes / (1024 * 1024);
    mb + mb / 7 + 512
}

pub fn pick_fitting(sizes: &[(String, u64)], quant: Option<&str>, limit_mb: u64) -> Result<Vec<String>, String> {
    // The quant asked for, else the first of Q4_K_M, Q4_0, MXFP4, Q8_0 that fits, else the largest that fits (splits count together).
    let names: Vec<&str> = sizes.iter().map(|(n, _)| n.as_str()).collect();
    let bytes = |files: &[String]| files.iter().filter_map(|f| sizes.iter().find(|(n, _)| n == f)).map(|(_, b)| *b).sum::<u64>();
    let fits = |files: &[String]| gguf_need_mb(bytes(files)) <= limit_mb;
    let gb = |mb: u64| mb as f64 / GIB_MB;
    if let Some(q) = quant {
        let files = crate::engines::llama_cpp::pick_gguf(&names, Some(q))?;
        if !fits(&files) {
            return Err(format!("{q} needs about {:.1} GB and this Mac lets the GPU use {:.1} GB; pick a smaller quant", gb(gguf_need_mb(bytes(&files))), gb(limit_mb)));
        }
        return Ok(files);
    }
    for q in ["Q4_K_M", "Q4_0", "MXFP4", "Q8_0"] {
        if let Ok(files) = crate::engines::llama_cpp::pick_gguf(&names, Some(q)) {
            if !files.is_empty() && fits(&files) {
                return Ok(files);
            }
        }
    }
    let mut options: Vec<Vec<String>> = Vec::new();
    for n in names.iter().filter(|n| crate::engines::llama_cpp::is_model_gguf(n) && !n.contains("-0000") || n.contains("-00001-of-")) {
        if let Ok(files) = crate::engines::llama_cpp::pick_gguf(&names, Some(n.trim_end_matches(".gguf"))) {
            if !options.contains(&files) {
                options.push(files);
            }
        }
    }
    options.sort_by_key(|f| std::cmp::Reverse(bytes(f)));
    if let Some(best) = options.iter().find(|f| fits(f)) {
        return Ok(best.clone());
    }
    let smallest = options.iter().map(|f| gguf_need_mb(bytes(f))).min().unwrap_or(0);
    Err(format!("its smallest GGUF needs about {:.1} GB and this Mac lets the GPU use {:.1} GB", gb(smallest), gb(limit_mb)))
}

pub enum Choice {
    Sglang,
    Gguf(Vec<String>),
    Build { repo: String, why: String },
}

pub(crate) fn gguf_build(model_id: &str, pre: &hf::ModelPreflight) -> Option<String> {
    // A GGUF build llama.cpp serves: the publisher's own <id>-GGUF repo, else the most downloaded GGUF build Hugging Face lists for the model or a model it derives from.
    let serves = |repo: &str| {
        hf::preflight_model(repo, None).is_ok_and(|p| {
            p.siblings.iter().any(|s| crate::engines::llama_cpp::is_model_gguf(&s.rfilename)) && p.gguf_arch.as_deref().is_none_or(crate::engines::llama_cpp::serves_arch)
        })
    };
    let own = format!("{model_id}-GGUF");
    if serves(&own) {
        return Some(own);
    }
    let mut found: Vec<(String, u64)> = std::iter::once(model_id.to_string()).chain(pre.base_models.iter().cloned()).flat_map(|b| hf::gguf_builds(&b)).collect();
    found.sort_by_key(|(_, downloads)| std::cmp::Reverse(*downloads));
    found.dedup_by(|a, b| a.0.eq_ignore_ascii_case(&b.0));
    found.into_iter().take(4).map(|(repo, _)| repo).find(|repo| serves(repo))
}

pub(crate) fn llama_cpp_refusal(pre: &hf::ModelPreflight) -> Option<String> {
    // A GGUF whose architecture the pinned llama.cpp does not know, named before any download.
    pre.gguf_arch.as_deref().filter(|a| !crate::engines::llama_cpp::serves_arch(a)).map(|a| format!("llama.cpp {} does not serve its architecture {a}", crate::engines::llama_cpp::TAG))
}

pub(crate) fn mac_gguf(home: &Path, model_id: &str, revision: &str, quant: Option<&str>) -> Result<Choice, String> {
    // On a Mac, before any download: GGUF files in the repo (or the GGUF build Hugging Face lists for it), a llama.cpp architecture, and a quant that fits the unified memory macOS lets the GPU use.
    if hf::offline_mode() {
        return Err(format!("{model_id}: HF_HUB_OFFLINE=1, so goethe cannot see which GGUF files it has"));
    }
    let pre = hf::preflight_model(model_id, Some(revision))?;
    let sizes: Vec<(String, u64)> = pre.siblings.iter().filter(|s| s.rfilename.ends_with(".gguf")).map(|s| (s.rfilename.clone(), s.size)).collect();
    if !sizes.iter().any(|(n, _)| crate::engines::llama_cpp::is_model_gguf(n)) {
        return match gguf_build(model_id, &pre) {
            Some(repo) => Ok(Choice::Build { repo, why: "it has no GGUF files, and on a Mac goethe serves GGUF models with llama.cpp".to_string() }),
            None => Err(format!("{model_id} is not supported: it has no GGUF files, and Hugging Face lists no GGUF build of it that llama.cpp serves. On a Mac goethe serves GGUF models with llama.cpp.")),
        };
    }
    if let Some(why) = llama_cpp_refusal(&pre) {
        return Err(format!("{model_id} is not supported: {why}"));
    }
    let limit = crate::hw::profile(home).ok().and_then(|h| h.gpus.first().map(|g| g.vram_total_mb)).unwrap_or(u64::MAX);
    let files = pick_fitting(&sizes, quant, limit).map_err(|e| format!("{model_id} does not fit: {e}"))?;
    crate::engines::llama_cpp::server_ready(home).map_err(|why| format!("{model_id}: {why}"))?;
    Ok(Choice::Gguf(files))
}

pub(crate) fn engine_check(
    home: &Path,
    model_id: &str,
    revision: Option<&str>,
    entry: Option<&RegistryModel>,
    (declined, no_pack_gpu): (bool, Option<&str>),
    quant: Option<&str>,
) -> Result<Choice, String> {
    // Before any download, in order: SGLang when it serves the architecture and quantization (catalog entries are vetted), else llama.cpp on the repo's GGUF files or the GGUF build Hugging Face lists for it, else refused with each engine's reason. A model with no plain runtime and no pack for this GPU is refused with the GPUs its packs are built for.
    if cfg!(target_os = "macos") {
        return mac_gguf(home, model_id, revision.unwrap_or("main"), quant);
    }
    if (entry.is_some() && quant.is_none()) || hf::offline_mode() {
        return Ok(Choice::Sglang);
    }
    let rev = revision.unwrap_or("main");
    let pre = hf::preflight_model(model_id, Some(rev))?;
    let hint = match (declined, no_pack_gpu) {
        (true, _) => " Optimised kernels exist for it on this GPU: run goethe login, then goethe install again.".to_string(),
        (_, Some(n)) => format!(" {n}."),
        _ => String::new(),
    };
    let names: Vec<&str> = pre.siblings.iter().map(|s| s.rfilename.as_str()).collect();
    let config = names.contains(&"config.json");
    let has_gguf = names.iter().any(|n| crate::engines::llama_cpp::is_model_gguf(n));
    let llama = |why_sglang: &str| -> Result<Choice, String> {
        if let Some(why) = llama_cpp_refusal(&pre) {
            return Err(format!("{model_id} is not supported: {why_sglang}{why}.{hint}"));
        }
        let files = crate::engines::llama_cpp::pick_gguf(&names, quant).map_err(|e| format!("{model_id}: {e}"))?;
        crate::engines::llama_cpp::server_ready(home).map_err(|why| format!("{model_id} is not supported: {why_sglang}{why}.{hint}"))?;
        Ok(Choice::Gguf(files))
    };
    if quant.is_some() || (!config && has_gguf) {
        return llama(if config { "" } else { "it ships GGUF files only, which SGLang does not serve; " });
    }
    if !config {
        return match no_pack_gpu {
            Some(n) => Err(format!("{model_id} is not supported: it is not a plain checkpoint (no config.json, no GGUF). {n}.")),
            None => Ok(Choice::Sglang),
        };
    }
    let cfg = hf::fetch_config_json(model_id, rev)?;
    let Err(why) = crate::scan::sglang_serves(&crate::scan::sglang_support(home), &cfg) else {
        return Ok(Choice::Sglang);
    };
    if has_gguf {
        return llama(&format!("{why}; "));
    }
    match gguf_build(model_id, &pre) {
        Some(repo) => Ok(Choice::Build { repo, why }),
        None => Err(format!("{model_id} is not supported: {why}, and Hugging Face lists no GGUF build of it that llama.cpp serves.{hint}")),
    }
}
