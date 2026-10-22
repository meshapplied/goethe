// Model install: HF download, mesh_card.toml, and state.db registration.

use crate::db;
use crate::hf::{self, DownloadOpts};
use crate::model_facts;
use crate::paths::{ensure_layout, mesh_home, model_dir};
use crate::safetensors::{is_nvfp4_compressed_model, marlin_extra_bytes_in_dir};
use crate::registry::{fetch, find_model, registry_url};
use crate::sglang_infer;
use std::os::unix::fs::symlink;
use serde::{Deserialize, Serialize};
use std::io::IsTerminal;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshCard {
    pub model_id: String,
    pub context_length: u64,
    pub max_running_requests: u64,
    pub tp_size: u64,
    pub ep_size: u64,
    pub kv_cache_dtype: String,
    pub preferred_gpus: Vec<u32>,
    pub extra_sglang_args: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pack_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub base_model_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env: Option<toml::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pythonpath: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub marlin_extra_gb: Option<f64>,
}

pub struct InstallOpts {
    pub model_id: String,
    pub ctx: Option<u64>,
    pub max_conc: Option<u64>,
    pub tp: Option<u64>,
    pub gpus: Option<Vec<u32>>,
    pub load: bool,
    pub force: bool,
    pub no_verify: bool,
    pub name: Option<String>,
    pub copy_local: bool,
    pub sglang_args: Vec<String>,
    pub no_wait: bool,
}

pub fn run(opts: InstallOpts) -> Result<(), String> {
    let home = mesh_home();
    ensure_layout(&home).map_err(|e| e.to_string())?;
    if Path::new(&opts.model_id).is_dir() {
        return run_local_install(&home, opts);
    }
    let reg = fetch(&registry_url()).ok();
    let entry = reg.as_ref().and_then(|r| find_model(r, &opts.model_id));
    let ternary = entry.as_ref().and_then(|e| e.runtime.as_deref()) == Some("ternary");
    let local = if ternary {
        let pack = entry
            .as_ref()
            .and_then(|e| e.pack_id.as_deref())
            .ok_or_else(|| "ternary model missing pack_id".to_string())?;
        ternary_asset_dir(&home, pack, "pack")
    } else {
        model_dir(&home, &opts.model_id)
    };
    let revision = entry.as_ref().and_then(|e| e.revision.clone());
    let dl = |hf_id: &str, dir: &Path| {
        hf::ensure_model(
            &home,
            hf_id,
            dir,
            &DownloadOpts {
                revision: revision.clone(),
                force: opts.force,
                no_verify: opts.no_verify,
            },
        )
    };
    if ternary {
        crate::runtime::require_ternary_runtime(&home)?;
        let pack = entry
            .as_ref()
            .and_then(|e| e.pack_id.as_deref())
            .ok_or_else(|| "ternary model missing pack_id".to_string())?;
        dl(pack, &local)?;
        if let Some(base) = entry.as_ref().and_then(|e| e.base_model_id.as_deref()) {
            let base_dir = if base == pack {
                ternary_asset_dir(&home, base, "base")
            } else {
                model_dir(&home, base)
            };
            dl(base, &base_dir)?;
        }
    } else {
        dl(&opts.model_id, &local)?;
    }
    let cfg = read_config(&local)?;
    let marlin_extra_gb = if is_nvfp4_compressed_model(&cfg) {
        match marlin_extra_bytes_in_dir(&local) {
            Ok(n) => Some(n as f64 / (1024.0_f64.powi(3))),
            Err(e) => {
                eprintln!("warning: marlin_extra_gb scan skipped: {e}");
                None
            }
        }
    } else {
        None
    };
    let facts = model_facts::facts_from_install_dir(
        &local,
        entry.map(|e| e.architecture.as_str()),
    )?;
    let kv_dtype = entry
        .and_then(|e| e.kv_cache_dtype.clone())
        .unwrap_or_else(|| "auto".to_string());
    let kv_bytes = facts.kv_elems_per_token * kv_element_bytes(&kv_dtype, &cfg);
    let (inferred, _) = sglang_infer::infer_extra_sglang_args(&cfg);
    let mut extra = entry
        .map(|e| e.extra_sglang_args.clone())
        .unwrap_or_default();
    extra = sglang_infer::merge_extra_args(&inferred, &extra);
    extra = sglang_infer::merge_extra_args(&extra, &opts.sglang_args);
    let load_ctx_default = facts.max_ctx.min(131072);
    let ctx = opts.ctx.unwrap_or(load_ctx_default);
    let max_conc = opts.max_conc.unwrap_or(8);
    let tp = opts.tp.unwrap_or(0);
    let gpus_for_load = opts.gpus.clone();
    let preferred = opts.gpus.unwrap_or_default();
    let card = MeshCard {
        model_id: opts.model_id.clone(),
        context_length: ctx,
        max_running_requests: max_conc,
        tp_size: tp,
        ep_size: 1,
        kv_cache_dtype: kv_dtype,
        preferred_gpus: preferred,
        extra_sglang_args: extra,
        runtime: entry.and_then(|e| e.runtime.clone()),
        pack_id: entry.and_then(|e| e.pack_id.clone()),
        base_model_id: entry.and_then(|e| e.base_model_id.clone()),
        env: entry.and_then(|e| e.env.clone()),
        pythonpath: entry.and_then(|e| e.pythonpath.clone()),
        marlin_extra_gb,
    };
    write_card(&local, &card)?;
    let rel = local
        .strip_prefix(&home)
        .map(|p| p.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| {
            format!(
                "models/{}",
                crate::paths::sanitize_model_id(&opts.model_id)
            )
        });
    let con = db::open(&home).map_err(|e| e.to_string())?;
    con.execute(
        "INSERT OR REPLACE INTO installed_models
        (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token,
         default_ctx, default_max_conc, default_tp, kv_elems_per_token, max_ctx,
         num_attention_heads, num_kv_heads, attention, installed_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, datetime('now'))",
        rusqlite::params![
            opts.model_id,
            rel,
            facts.architecture,
            facts.static_vram_gb,
            kv_bytes as i64,
            ctx as i64,
            max_conc as i64,
            tp as i64,
            facts.kv_elems_per_token as i64,
            facts.max_ctx as i64,
            facts.num_attention_heads as i64,
            facts.num_kv_heads as i64,
            facts.attention,
        ],
    )
    .map_err(|e| e.to_string())?;
    if opts.load {
        let gpus_csv = gpus_for_load
            .as_ref()
            .map(|v| v.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(","));
        let reply = crate::rpc::call(
            "load",
            serde_json::json!({
                "model_id": opts.model_id,
                "gpus": gpus_csv,
                "ctx": opts.ctx,
                "max_conc": opts.max_conc,
                "tp": opts.tp,
                "force": false,
                "yes": false,
                "is_tty": std::io::stdin().is_terminal(),
                "extra_sglang_args": opts.sglang_args,
                "async_finish": true,
            }),
        )
        .map_err(crate::rpc::map_daemon_err)?;
        if reply.code != 0 {
            eprintln!("{}", reply.stderr);
            std::process::exit(reply.code as i32);
        }
        if !reply.stdout.is_empty() {
            print!("{}", reply.stdout);
        }
        crate::load_wait::wait_after_load_stdout(&reply.stdout, opts.no_wait)?;
    }
    Ok(())
}

fn run_local_install(home: &Path, opts: InstallOpts) -> Result<(), String> {
    let src = PathBuf::from(&opts.model_id);
    if !src.join("config.json").is_file() {
        return Err("local install requires config.json".to_string());
    }
    let has_weights = std::fs::read_dir(&src)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok())
        .any(|e| {
            e.path()
                .extension()
                .and_then(|x| x.to_str())
                .is_some_and(|x| x == "safetensors")
        });
    if !has_weights {
        return Err("local install requires at least one .safetensors file".to_string());
    }
    let dir_name = src
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| "invalid local path".to_string())?;
    let model_id = opts
        .name
        .clone()
        .unwrap_or_else(|| format!("local/{dir_name}"));
    let link_dir = model_dir(home, &model_id);
    if link_dir.exists() {
        if !opts.force {
            return Err(format!("model {model_id} already installed; use --force"));
        }
        fs::remove_dir_all(&link_dir).ok();
    }
    std::fs::create_dir_all(link_dir.parent().unwrap()).map_err(|e| e.to_string())?;
    if opts.copy_local {
        copy_dir_all(&src, &link_dir)?;
    } else {
        symlink(&src, &link_dir).map_err(|e| e.to_string())?;
    }
    let local = if opts.copy_local { link_dir } else { src.clone() };
    register_installed(home, &opts, &model_id, &local, None)
}

fn register_installed(
    home: &Path,
    opts: &InstallOpts,
    model_id: &str,
    local: &Path,
    entry: Option<&crate::registry::RegistryModel>,
) -> Result<(), String> {
    let cfg = read_config(local)?;
    let facts = model_facts::facts_from_install_dir(local, entry.map(|e| e.architecture.as_str()))?;
    let kv_dtype = entry
        .and_then(|e| e.kv_cache_dtype.clone())
        .unwrap_or_else(|| "auto".to_string());
    let kv_bytes = facts.kv_elems_per_token * kv_element_bytes(&kv_dtype, &cfg);
    let (inferred, _) = sglang_infer::infer_extra_sglang_args(&cfg);
    let mut extra = entry
        .map(|e| e.extra_sglang_args.clone())
        .unwrap_or_default();
    extra = sglang_infer::merge_extra_args(&inferred, &extra);
    extra = sglang_infer::merge_extra_args(&extra, &opts.sglang_args);
    let ctx = opts.ctx.unwrap_or(facts.max_ctx.min(131072));
    let max_conc = opts.max_conc.unwrap_or(8);
    let tp = opts.tp.unwrap_or(0);
    let card = MeshCard {
        model_id: model_id.to_string(),
        context_length: ctx,
        max_running_requests: max_conc,
        tp_size: tp,
        ep_size: 1,
        kv_cache_dtype: kv_dtype,
        preferred_gpus: opts.gpus.clone().unwrap_or_default(),
        extra_sglang_args: extra,
        runtime: entry.and_then(|e| e.runtime.clone()),
        pack_id: entry.and_then(|e| e.pack_id.clone()),
        base_model_id: entry.and_then(|e| e.base_model_id.clone()),
        env: entry.and_then(|e| e.env.clone()),
        pythonpath: entry.and_then(|e| e.pythonpath.clone()),
        marlin_extra_gb: None,
    };
    write_card(local, &card)?;
    let rel = if local.starts_with(home) {
        local
            .strip_prefix(home)
            .map(|p| p.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|_| format!("models/{}", crate::paths::sanitize_model_id(model_id)))
    } else {
        format!("models/{}", crate::paths::sanitize_model_id(model_id))
    };
    let con = db::open(home).map_err(|e| e.to_string())?;
    con.execute(
        "INSERT OR REPLACE INTO installed_models
        (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token,
         default_ctx, default_max_conc, default_tp, kv_elems_per_token, max_ctx,
         num_attention_heads, num_kv_heads, attention, installed_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, datetime('now'))",
        rusqlite::params![
            model_id,
            rel,
            facts.architecture,
            facts.static_vram_gb,
            kv_bytes as i64,
            ctx as i64,
            max_conc as i64,
            tp as i64,
            facts.kv_elems_per_token as i64,
            facts.max_ctx as i64,
            facts.num_attention_heads as i64,
            facts.num_kv_heads as i64,
            facts.attention,
        ],
    )
    .map_err(|e| e.to_string())?;
    if opts.load {
        let gpus_csv = opts
            .gpus
            .as_ref()
            .map(|v| v.iter().map(|g| g.to_string()).collect::<Vec<_>>().join(","));
        let reply = crate::rpc::call(
            "load",
            serde_json::json!({
                "model_id": model_id,
                "gpus": gpus_csv,
                "ctx": opts.ctx,
                "max_conc": opts.max_conc,
                "tp": opts.tp,
                "force": false,
                "yes": false,
                "is_tty": std::io::stdin().is_terminal(),
                "extra_sglang_args": opts.sglang_args,
                "async_finish": true,
            }),
        )
        .map_err(crate::rpc::map_daemon_err)?;
        if reply.code != 0 {
            eprintln!("{}", reply.stderr);
            std::process::exit(reply.code as i32);
        }
        if !reply.stdout.is_empty() {
            print!("{}", reply.stdout);
        }
        crate::load_wait::wait_after_load_stdout(&reply.stdout, opts.no_wait)?;
    }
    Ok(())
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dst).map_err(|e| e.to_string())?;
    for ent in std::fs::read_dir(src).map_err(|e| e.to_string())? {
        let ent = ent.map_err(|e| e.to_string())?;
        let to = dst.join(ent.file_name());
        if ent.file_type().map_err(|e| e.to_string())?.is_dir() {
            copy_dir_all(&ent.path(), &to)?;
        } else {
            std::fs::copy(ent.path(), to).map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn ternary_asset_dir(home: &Path, hf_id: &str, role: &str) -> PathBuf {
    model_dir(home, &format!("{hf_id}/{role}"))
}

fn read_config(dir: &Path) -> Result<serde_json::Value, String> {
    let path = dir.join("config.json");
    let text = std::fs::read_to_string(&path).map_err(|e| format!("config.json: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("config.json parse: {e}"))
}

pub fn kv_bytes_from_config(cfg: &serde_json::Value, kv_cache_dtype: &str) -> u64 {
    let tc = cfg.get("text_config").unwrap_or(cfg);
    let layers = if let Some(lt) = tc.get("layer_types").and_then(|v| v.as_array()) {
        lt.iter()
            .filter(|t| {
                t.as_str()
                    .map(|s| s.contains("full_attention") || s == "attention")
                    .unwrap_or(false)
            })
            .count()
    } else {
        tc.get("num_hidden_layers")
            .or_else(|| tc.get("n_layer"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1) as usize
    };
    let kv_heads = tc
        .get("num_key_value_heads")
        .or_else(|| tc.get("n_head"))
        .and_then(|v| v.as_u64())
        .unwrap_or(1);
    let hidden = tc
        .get("hidden_size")
        .and_then(|v| v.as_u64())
        .unwrap_or(64);
    let head_dim = tc
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| hidden / kv_heads.max(1));
    let width = kv_element_bytes(kv_cache_dtype, cfg);
    2 * layers as u64 * kv_heads * head_dim * width
}

fn kv_element_bytes(kv_cache_dtype: &str, cfg: &serde_json::Value) -> u64 {
    let k = kv_cache_dtype.to_ascii_lowercase();
    if k.contains("fp8") {
        return 1;
    }
    if k.contains("fp16") || k.contains("bf16") || k.contains("float16") || k.contains("bfloat16") {
        return 2;
    }
    if k == "auto" {
        let tc = cfg.get("text_config").unwrap_or(cfg);
        let dtype = tc
            .get("torch_dtype")
            .or_else(|| tc.get("dtype"))
            .and_then(|v| v.as_str())
            .unwrap_or("float16");
        if dtype.contains("16") || dtype.contains("bf16") {
            2
        } else {
            4
        }
    } else {
        2
    }
}

fn write_card(dir: &Path, card: &MeshCard) -> Result<(), String> {
    let text = toml::to_string_pretty(card).map_err(|e| e.to_string())?;
    std::fs::write(dir.join("mesh_card.toml"), text).map_err(|e| e.to_string())
}

pub fn write_card_public(dir: &Path, card: &MeshCard) -> Result<(), String> {
    write_card(dir, card)
}

pub fn read_card(dir: &Path) -> Result<MeshCard, String> {
    let text = std::fs::read_to_string(dir.join("mesh_card.toml")).map_err(|e| e.to_string())?;
    toml::from_str(&text).map_err(|e| e.to_string())
}

#[cfg(test)]
mod kv_tests {
    use super::*;

    #[test]
    fn qwen38_nvfp4_kv_bytes_fp8() {
        let cfg: serde_json::Value = serde_json::json!({
            "text_config": {
                "head_dim": 256,
                "num_key_value_heads": 4,
                "layer_types": [
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention",
                    "linear_attention","linear_attention","linear_attention","full_attention"
                ]
            }
        });
        assert_eq!(kv_bytes_from_config(&cfg, "fp8_e4m3"), 32768);
    }
}
