// Pack install-time fit: Marlin bytes, ChooserInput, and the pre-download fit line.

use super::{vram_estimate, InstallOpts};
use crate::hf;
use crate::messages::NOTHING_DOWNLOADED;
use crate::model_facts::{self, GIB};
use crate::registry::RegistryModel;
use crate::safetensors;
use serde_json::Value;
use std::path::Path;

pub(super) fn pack_marlin_gb(cfg: &Value, repo: &str, rev: &str, files: &[String]) -> f64 {
    // The Marlin copy load budgets for a compressed NVFP4 checkpoint, from the safetensors headers of the files about to be downloaded (two small ranged reads each, no weights); unreadable says so and counts none.
    if !safetensors::is_nvfp4_compressed_model(cfg) {
        return 0.0;
    }
    let mut bytes = 0u64;
    for f in files.iter().filter(|f| f.ends_with(".safetensors")) {
        match hf::safetensors_header(repo, rev, f).and_then(|h| safetensors::header_tensor_bytes(&h, &safetensors::tensor_counts_marlin_extra)) {
            Ok(n) => bytes += n,
            Err(e) => {
                crate::step::note(&format!("note: the Marlin copy is not counted in the fit: {e}"));
                return 0.0;
            }
        }
    }
    bytes as f64 / GIB
}

pub(super) struct PackWeights {
    pub(super) bytes: u64,
    pub(super) marlin_gb: f64,
    pub(super) ctx_cap: Option<u64>,
    pub(super) lanes: Option<u64>,
}

pub(super) fn pack_input(cfg: &Value, entry: Option<&RegistryModel>, opts: &InstallOpts, arch: &str, w: &PackWeights, dir: &Path) -> Result<(crate::chooser::ChooserInput, Option<usize>), String> {
    // The budget input goethe load will build for this pack model: the card install would write (kernel-pack runtime, its context capped by the pack, the Marlin copy, the compute capabilities of the arch) through the same constructor and memory spec the load uses.
    let facts = model_facts::facts_from_config(cfg, w.bytes as f64 / GIB, entry.map(|e| e.architecture.as_str()), &opts.model_id)?;
    let vram = vram_estimate(dir, cfg, &facts, &[], "auto");
    let max_ctx = super::ctx_limit(&facts, entry, opts.ctx)?;
    let ctx = opts.ctx.unwrap_or_else(|| entry.and_then(|e| e.effective_max_ctx()).unwrap_or(max_ctx)).min(max_ctx).min(w.ctx_cap.unwrap_or(u64::MAX));
    let mut card = toml::Table::new();
    card.insert("runtime".into(), toml::Value::String(crate::engines::goethe_engine::ENGINE.id.into()));
    card.insert("kv_cache_dtype".into(), toml::Value::String("auto".into()));
    card.insert("supported_sm".into(), toml::Value::Array(crate::packs::arch_ccs(arch).into_iter().map(toml::Value::String).collect()));
    if w.marlin_gb > 0.0 {
        card.insert("marlin_extra_gb".into(), toml::Value::Float(w.marlin_gb));
    }
    let card = toml::Value::Table(card);
    let installed = crate::chooser::CardFacts {
        static_vram_gb: vram.weights_gb + vram.state_gb,
        kv_bytes_per_token: vram.kv_bytes_per_token,
        num_attention_heads: facts.num_attention_heads,
        num_kv_heads: facts.num_kv_heads,
        attention: facts.attention.clone(),
        max_ctx,
    };
    let max_conc = opts.max_conc.or(w.lanes).unwrap_or(crate::chooser::DEFAULT_CONC);
    let input = crate::chooser::ChooserInput::from_card(installed, &card, crate::memory_spec::installed_from_config(cfg, &card, &[]), ctx, max_conc);
    Ok((input, opts.tp.or(entry.and_then(|e| e.default_tp)).filter(|t| *t > 0).map(|t| t as usize)))
}

pub(super) fn pack_fit(view: &crate::recommend::View, cfg: &Value, entry: Option<&RegistryModel>, opts: &InstallOpts, arch: &str, w: PackWeights, dir: &Path) -> Result<String, String> {
    // The per-GPU budget goethe load will compute for a kernel-pack model (the pack runtime's own learned rows), from its config and the listed size of its weights, before a weight byte is fetched: refused with what is missing when no GPU set fits, else the fit line with its headroom.
    let id = &opts.model_id;
    let (input, pinned) = pack_input(cfg, entry, opts, arch, &w, dir)?;
    let (loading, memory) = (opts.load, if opts.load { "free now" } else { "in total" });
    let fit = view.fit_as(id, &input, pinned, crate::engine_overhead::PACK_RUNTIME).map_err(|why| {
        let cards: Vec<String> = view.gpus.iter().map(|g| format!("gpu{} {} {:.1} GB {memory}", g.index, view.hw.gpus.iter().find(|h| h.index == g.index).map_or("", |h| h.name.as_str()), g.free_vram_mb as f64 / 1024.0)).collect();
        if loading {
            format!("{id} does not fit the allowed GPUs now ({}), so {NOTHING_DOWNLOADED}: {why}. Stop what is using the GPU (goethe gpu shows what), or install without --load to download only; --ctx with a smaller value or goethe gpu deny or allow also change the fit", cards.join(", "))
        } else {
            format!("{id} does not fit the allowed GPUs even when they are empty ({}), so {NOTHING_DOWNLOADED}: {why}. Options: --ctx with a smaller value, or more cards with goethe gpu allow", cards.join(", "))
        }
    })?;
    let free = fit.gpus.iter().filter_map(|g| view.gpus.iter().find(|x| x.index == *g)).map(|g| g.free_vram_mb as f64 / 1024.0).fold(f64::INFINITY, f64::min);
    let total = fit.need.total();
    Ok(format!("fit: {}; {total:.1} of {free:.1} GB {}, {:.1} GB spare", fit.describe(&input), if loading { "free" } else { "on the card" }, free - total))
}
