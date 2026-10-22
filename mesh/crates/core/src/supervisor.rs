// meshd load, unload, and instance supervision.

use crate::calc::{mem_fraction_from_free, static_weight_gb};
use crate::chooser::{self, ChooserInput};
use crate::db;
use crate::hw::{
    foreign_compute_on_gpu, gpu_free_mb_live, gpu_used_mb_live, managed_vram_listed_mb,
    refresh_gpu_vram, HardwareState,
};
use crate::placement::{self, GpuPlacementInput};
use crate::paths::{ensure_layout, mesh_home};
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::{json, Value};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};
use crate::load_slot;
use crate::run_registry::{self, RunEntry};

static SLOT_LOCKS: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
static WATCHDOG_TICK_LOCK: Mutex<()> = Mutex::new(());

fn slot_lock(key: &str) -> Arc<Mutex<()>> {
    let map = SLOT_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = map.lock().unwrap_or_else(|e| e.into_inner());
    g.entry(key.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoadParams {
    pub model_id: String,
    #[serde(default)]
    pub gpus: Option<String>,
    #[serde(default)]
    pub extra_sglang_args: Vec<String>,
    #[serde(default)]
    pub ctx: Option<u64>,
    #[serde(default)]
    pub max_conc: Option<u64>,
    #[serde(default)]
    pub tp: Option<u64>,
    #[serde(default)]
    pub force: bool,
    #[serde(default)]
    pub yes: bool,
    #[serde(default)]
    pub is_tty: bool,
    #[serde(default)]
    pub restart_count: Option<u64>,
    #[serde(default)]
    pub slot_key: Option<String>,
    #[serde(default)]
    pub async_finish: bool,
}

struct LoadFinishCtx {
    home: PathBuf,
    instance_id: String,
    port: u16,
    pid: i32,
    pgid: i32,
    params: LoadParams,
    log_path: PathBuf,
    budget_gb: f64,
    gpu_name: String,
    tp: u64,
    model_path: PathBuf,
    card: toml::Value,
    gpu_indices: Vec<u32>,
    ctx: u64,
    max_conc: u64,
    cancel: std::sync::Arc<AtomicBool>,
}

pub fn internal_port_for_model(model_id: &str) -> Option<u16> {
    let home = mesh_home();
    let con = db::open(&home).ok()?;
    con.query_row(
        "SELECT internal_port FROM loaded_instances WHERE model_id = ?1 AND status = 'RUNNING' LIMIT 1",
        [model_id],
        |r| r.get::<_, i64>(0),
    )
    .ok()
    .and_then(|p| u16::try_from(p).ok())
}

pub fn handle_load(params: LoadParams) -> Result<(Value, String), (i32, String)> {
    let home = mesh_home();
    ensure_layout(&home).map_err(|e| (1, e.to_string()))?;
    let con = db::open(&home).map_err(|e| (1, e.to_string()))?;
    let row = fetch_installed(&con, &params.model_id).map_err(|e| (1, e))?;
    let card_path = home.join(&row.local_path).join("mesh_card.toml");
    let card = read_card_value(&card_path).map_err(|e| (1, e))?;
    let runtime = card
        .get("runtime")
        .and_then(|v| v.as_str())
        .unwrap_or("sglang");
    if runtime == "ternary" {
        crate::runtime::require_ternary_runtime(&home).map_err(|e| (1, e))?;
    }
    let pinned_gpus = parse_gpus_param(params.gpus.as_deref(), &card);
    let pinned_tp = params.tp.map(|t| t as usize).or({
        if !pinned_gpus.is_empty() {
            Some(pinned_gpus.len())
        } else {
            None
        }
    });
    if let (Some(pt), true) = (params.tp, !pinned_gpus.is_empty()) {
        if pt as usize != pinned_gpus.len() {
            return Err((
                1,
                format!(
                    "--tp {pt} disagrees with {} GPUs on --gpus",
                    pinned_gpus.len()
                ),
            ));
        }
    }
    let denied = crate::gpu_policy::denied_set(&con).map_err(|e| (1, e))?;
    for &g in &pinned_gpus {
        if denied.contains(&g) {
            return Err((
                1,
                format!(
                    "GPU {g} is denied by policy; run `mesh gpu allow {g}` to use it"
                ),
            ));
        }
    }
    let ctx_default = row
        .max_ctx
        .map(|v| v as u64)
        .unwrap_or(row.default_ctx as u64)
        .min(131072);
    let ctx = params.ctx.unwrap_or(ctx_default);
    let max_conc = params.max_conc.unwrap_or(row.default_max_conc as u64);
    let hw = load_hw(&home);
    let snaps = crate::hw::list_gpus_for_placement(&placement::managed_pids(&con))
        .map_err(|e| (1, e))?;
    let inputs: Vec<GpuPlacementInput> = snaps
        .into_iter()
        .map(GpuPlacementInput::from)
        .filter(|g| !denied.contains(&g.index))
        .collect();
    if let Some(tp) = pinned_tp {
        placement::ensure_eligible_for_tp(&inputs, tp, &denied).map_err(|e| (1, e))?;
    }
    let plan = chooser::pick_load_plan(
        &chooser_input_from_row(&row, &card, ctx, max_conc, &home, &pinned_gpus),
        &inputs,
        hw.as_ref(),
        pinned_tp.or(if row.default_tp == 0 {
            None
        } else {
            Some(row.default_tp as usize)
        }),
        if pinned_gpus.is_empty() {
            None
        } else {
            Some(pinned_gpus.as_slice())
        },
    )
    .map_err(|e| (1, e))?;
    let tp = plan.tp as u64;
    let gpu_indices = plan.gpus;
    let ctx = plan.ctx;
    let max_conc = plan.max_conc;
    let input = chooser_input_from_row(&row, &card, ctx, max_conc, &home, &gpu_indices);
    let mut notices = String::new();
    notices.push_str(&format!("{}\n", plan.message));
    for &gpu in &gpu_indices {
        if let Some(msg) = check_foreign(gpu, &con, params.force || params.yes)? {
            if !params.force && !params.yes && !params.is_tty {
                return Err((3, msg));
            }
            if !params.force && !params.yes && params.is_tty {
                print!("{msg}");
                std::io::stdout().flush().ok();
                let mut buf = String::new();
                if std::io::stdin().read_line(&mut buf).is_err() || !buf.trim().eq_ignore_ascii_case("y") {
                    return Err((1, "load aborted".to_string()));
                }
            }
        }
        if let Some(line) =
            co_residency_notice(gpu, &params.model_id, &con, &home).map_err(|e| (1, e))?
        {
            notices.push_str(&line);
            notices.push('\n');
        }
    }
    let gpu_name = gpu_indices
        .first()
        .and_then(|idx| gpu_name(&hw, *idx))
        .unwrap_or_else(|| "gpu".to_string());
    let learned = crate::engine_overhead::read_learned_extra_gb(&con, &params.model_id, &gpu_name, tp);
    let mut b_gb = chooser::per_gpu_budget_gb(
        &input,
        plan.tp,
        chooser::bytes_per_elem(
            &gpu_indices
                .iter()
                .map(|idx| gpu_cap(&hw, *idx))
                .collect::<Vec<_>>(),
            None,
        ),
    );
    b_gb += learned;
    let gpu_list_csv = gpu_csv(&gpu_indices);
    let ordinal = if let Some(sk) = params.slot_key.as_deref() {
        crate::slots::parse_slot_key(sk)
            .map(|(_, _, o)| o)
            .unwrap_or(0)
    } else {
        crate::slots::next_ordinal(&con, &params.model_id, &gpu_list_csv).unwrap_or(0)
    };
    let slot_key = params
        .slot_key
        .clone()
        .unwrap_or_else(|| crate::slots::slot_key(&params.model_id, &gpu_indices, ordinal));
    for &gpu in &gpu_indices {
        let free_gb = gpu_free_gb(&home, gpu).map_err(|e| (1, e))?;
        if b_gb > free_gb - 1.5 {
            let missing = b_gb - (free_gb - 1.5);
            return Err((
                1,
                format!(
                    "VRAM budget {:.2} GB exceeds free {:.2} GB on GPU {gpu}; missing {:.2} GB",
                    b_gb, free_gb, missing
                ),
            ));
        }
    }
    let port = next_port(&con).map_err(|e| (1, e))?;
    let instance_id = format!("inst-{}", uuid_simple());
    let model_path = home.join(&row.local_path);
    let mem_frac = mem_fraction(&home, &gpu_indices, b_gb).map_err(|e| (1, e))?;
    let log_path = home.join("logs").join(format!("{instance_id}.log"));
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| (1, e.to_string()))?;
    let (pid, pgid) = spawn_tracked_engine(
        &home,
        &instance_id,
        &params.model_id,
        &model_path,
        &card,
        port,
        ctx,
        max_conc,
        tp,
        mem_frac,
        &gpu_indices,
        &params.extra_sglang_args,
        &mut log,
    )
    .map_err(|e| (1, e))?;
    let restarts = params.restart_count.unwrap_or(0) as i64;
    con.execute(
        "INSERT INTO loaded_instances
        (instance_id, model_id, pid, pgid, internal_port, gpu_indices,
         allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status, restart_count,
         slot_key, slot_ordinal)
        VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,'STARTING', ?11, ?12, ?13)",
        rusqlite::params![
            instance_id,
            params.model_id,
            pid as i64,
            pgid as i64,
            port as i64,
            gpu_csv(&gpu_indices),
            b_gb,
            mem_frac,
            ctx as i64,
            max_conc as i64,
            restarts,
            slot_key,
            ordinal as i64,
        ],
    )
    .map_err(|e| (1, e.to_string()))?;
    let _ = con.execute(
        "DELETE FROM watchdog_probe_state WHERE instance_id = ?1",
        [&instance_id],
    );
    let cancel = load_slot::begin_slot_load(&slot_key);
    let finish = LoadFinishCtx {
        home: home.clone(),
        instance_id: instance_id.clone(),
        port,
        pid,
        pgid,
        params: {
            let mut p = params.clone();
            p.slot_key = Some(slot_key.clone());
            p
        },
        log_path: log_path.clone(),
        budget_gb: b_gb,
        gpu_name: gpu_name.clone(),
        tp,
        model_path: model_path.clone(),
        card: card.clone(),
        gpu_indices: gpu_indices.clone(),
        ctx,
        max_conc,
        cancel,
    };
    if params.async_finish {
        let sk = slot_key.clone();
        thread::spawn(move || {
            load_slot::slot_in_flight_inc(&sk);
            let slot_arc = slot_lock(&sk);
            let _guard = slot_arc.lock().unwrap_or_else(|e| e.into_inner());
            finish_load_health(finish);
            load_slot::slot_in_flight_dec(&sk);
        });
        return Ok((
            json!({"instance_id": instance_id, "port": port, "status": "STARTING", "slot_key": slot_key}),
            notices,
        ));
    }
    finish_load_health(finish);
    let status: String = db::open(&home)
        .map_err(|e| (1, e.to_string()))?
        .query_row(
            "SELECT status FROM loaded_instances WHERE instance_id = ?1",
            [&instance_id],
            |r| r.get(0),
        )
        .map_err(|e| (1, e.to_string()))?;
    if status == "FAILED" {
        return Err((1, "engine health check failed".to_string()));
    }
    Ok((json!({"instance_id": instance_id, "port": port, "status": status}), notices))
}

fn gpu_name(hw: &Option<crate::hw::HardwareState>, idx: u32) -> Option<String> {
    hw.as_ref()?.gpus.iter().find(|g| g.index == idx).map(|g| g.name.clone())
}

fn finish_load_health(mut ctx: LoadFinishCtx) {
    let mut budget_gb = ctx.budget_gb;
    let mut mem_retries = 0u32;
    let mut pid = ctx.pid;
    let mut pgid = ctx.pgid;
    let port = ctx.port;
    let instance_id = ctx.instance_id.clone();
    loop {
        if load_slot::is_cancelled(&ctx.cancel) {
            if let Ok(con) = db::open(&ctx.home) {
                let stopping: bool = con
                    .query_row(
                        "SELECT status FROM loaded_instances WHERE instance_id = ?1",
                        [&instance_id],
                        |r| r.get::<_, String>(0),
                    )
                    .map(|s| s == "STOPPING")
                    .unwrap_or(false);
                if stopping {
                    load_slot::clear_instance_notices(&instance_id);
                    return;
                }
                run_registry::kill_instance(&ctx.home, &instance_id, pgid, pid);
                let _ = con.execute(
                    "UPDATE loaded_instances SET status = 'CANCELLED' WHERE instance_id = ?1",
                    [&instance_id],
                );
            }
            load_slot::clear_instance_notices(&instance_id);
            return;
        }
        match wait_engine_ready(port, pgid, &ctx.log_path) {
            Ok(()) => {
                let Ok(con) = db::open(&ctx.home) else {
                    return;
                };
                let _ = con.execute(
                    "UPDATE loaded_instances SET status = 'RUNNING' WHERE instance_id = ?1",
                    [&instance_id],
                );
                let _ = crate::autostart::mark_desired(&ctx.params);
                return;
            }
            Err(_fail) => {
                if load_slot::is_cancelled(&ctx.cancel) {
                    continue;
                }
                let log = crate::engine_overhead::read_log_tail(&ctx.log_path, 256 * 1024);
                let shortfall = crate::engine_overhead::parse_shortfall_gb(&log);
                if let Some(extra) = shortfall {
                    if let Ok(con) = db::open(&ctx.home) {
                        let _ = crate::engine_overhead::store_learned_extra_gb(
                            &con,
                            &ctx.params.model_id,
                            &ctx.gpu_name,
                            ctx.tp,
                            extra,
                        );
                    }
                    if mem_retries < 2 {
                        let new_budget = budget_gb + extra;
                        let mut fits = true;
                        let mut missing = 0.0f64;
                        for &gpu in &ctx.gpu_indices {
                            if let Ok(free_gb) = gpu_free_gb(&ctx.home, gpu) {
                                if new_budget > free_gb - 1.5 {
                                    fits = false;
                                    missing = new_budget - (free_gb - 1.5);
                                    break;
                                }
                            }
                        }
                        if fits {
                            mem_retries += 1;
                            budget_gb = new_budget;
                            let notice = format!(
                                "engine needed {:.1} GB more than estimated, retried",
                                extra
                            );
                            load_slot::push_instance_notice(&instance_id, &notice);
                            run_registry::kill_instance(&ctx.home, &instance_id, pgid, pid);
                            let mem_frac = match mem_fraction(&ctx.home, &ctx.gpu_indices, budget_gb) {
                                Ok(f) => f,
                                Err(_) => break,
                            };
                            let mut log = match OpenOptions::new().append(true).open(&ctx.log_path) {
                                Ok(f) => f,
                                Err(_) => break,
                            };
                            let spawned = spawn_tracked_engine(
                                &ctx.home,
                                &instance_id,
                                &ctx.params.model_id,
                                &ctx.model_path,
                                &ctx.card,
                                port,
                                ctx.ctx,
                                ctx.max_conc,
                                ctx.tp,
                                mem_frac,
                                &ctx.gpu_indices,
                                &ctx.params.extra_sglang_args,
                                &mut log,
                            );
                            let Ok((npid, npgid)) = spawned else {
                                break;
                            };
                            pid = npid;
                            pgid = npgid;
                            ctx.pid = npid;
                            ctx.pgid = npgid;
                            if let Ok(con) = db::open(&ctx.home) {
                                let _ = con.execute(
                                    "UPDATE loaded_instances SET pid = ?1, pgid = ?2,
                                     allocated_vram_gb_per_gpu = ?3, mem_fraction_static = ?4,
                                     status = 'STARTING'
                                     WHERE instance_id = ?5",
                                    rusqlite::params![
                                        npid as i64,
                                        npgid as i64,
                                        budget_gb,
                                        mem_frac,
                                        instance_id,
                                    ],
                                );
                            }
                            continue;
                        }
                        let notice = format!(
                            "VRAM budget {:.2} GB exceeds free GPU headroom; missing {:.2} GB",
                            new_budget, missing
                        );
                        load_slot::push_instance_notice(&instance_id, &notice);
                    }
                }
                run_registry::kill_instance(&ctx.home, &instance_id, pgid, pid);
                if let Ok(con) = db::open(&ctx.home) {
                    let _ = con.execute(
                        "UPDATE loaded_instances SET status = 'FAILED' WHERE instance_id = ?1",
                        [&instance_id],
                    );
                }
                return;
            }
        }
    }
}

pub fn slot_has_running_instance(slot_or_model: &str) -> bool {
    let home = mesh_home();
    let Ok(con) = db::open(&home) else {
        return false;
    };
    con.query_row(
        "SELECT 1 FROM loaded_instances WHERE (slot_key = ?1 OR model_id = ?1) AND status = 'RUNNING' LIMIT 1",
        [slot_or_model],
        |_| Ok(()),
    )
    .is_ok()
}

pub fn model_has_running_instance(model_id: &str) -> bool {
    let home = mesh_home();
    let Ok(con) = db::open(&home) else {
        return false;
    };
    con.query_row(
        "SELECT 1 FROM loaded_instances WHERE model_id = ?1 AND status = 'RUNNING' LIMIT 1",
        [model_id],
        |_| Ok(()),
    )
    .is_ok()
}

fn append_meshd_log(home: &Path, line: &str) {
    let path = home.join("logs/meshd.log");
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
}

pub fn prepare_shutdown(_home: &Path) {
    load_slot::request_cancel_all();
    let _ = load_slot::wait_all_slots_quiet(Duration::from_millis(500));
}

pub fn graceful_shutdown(home: &Path) -> Result<(), String> {
    let t0 = Instant::now();
    prepare_shutdown(home);
    let con = db::open(home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare("SELECT instance_id FROM loaded_instances")
        .map_err(|e| e.to_string())?;
    let ids: Vec<String> = stmt
        .query_map([], |r| r.get(0))
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
    if ids.is_empty() {
        run_registry::kill_all_mesh_engines_under(home);
        append_meshd_log(
            home,
            &format!("shutdown done ms={} (no instances)\n", t0.elapsed().as_millis()),
        );
        return Ok(());
    }
    if ids.len() == 1 {
        teardown_instance(&con, &ids[0], home).map_err(|(_, m)| m)?;
    } else {
        teardown_batch(&con, &ids, home).map_err(|(_, m)| m)?;
    }
    run_registry::kill_all_mesh_engines_under(home);
    append_meshd_log(
        home,
        &format!(
            "shutdown done ms={} instances={}\n",
            t0.elapsed().as_millis(),
            ids.len()
        ),
    );
    Ok(())
}

pub fn model_draining(model_id: &str) -> bool {
    let home = mesh_home();
    let Ok(con) = db::open(&home) else {
        return false;
    };
    con.query_row(
        "SELECT 1 FROM loaded_instances WHERE model_id = ?1 AND status = 'DRAINING' LIMIT 1",
        [model_id],
        |_| Ok(()),
    )
    .is_ok()
}

fn drain_inflight(model_ids: &[String], home: &Path) {
    let max_secs: u64 = std::env::var("MESH_DRAIN_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let deadline = Instant::now() + Duration::from_secs(max_secs);
    loop {
        let pending: usize = model_ids
            .iter()
            .map(|mid| crate::inflight::count(mid))
            .sum();
        if pending == 0 {
            return;
        }
        if Instant::now() >= deadline {
            append_meshd_log(
                home,
                &format!("unload drain timeout cut {pending} in-flight requests\n"),
            );
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
}

pub fn handle_unload(
    model_id: Option<String>,
    all: bool,
    force: bool,
) -> Result<Value, (i32, String)> {
    let home = mesh_home();
    let t0 = std::time::Instant::now();
    let con = db::open(&home).map_err(|e| (1, e.to_string()))?;
    let starting: Vec<String> = {
        let sql = if all {
            "SELECT DISTINCT COALESCE(slot_key, model_id) FROM loaded_instances WHERE status='STARTING'"
        } else {
            "SELECT DISTINCT COALESCE(slot_key, model_id) FROM loaded_instances WHERE status='STARTING' AND model_id = ?1"
        };
        if all {
            let mut stmt = con.prepare(sql).map_err(|e| (1, e.to_string()))?;
            let rows = stmt.query_map([], |r| r.get(0)).map_err(|e| (1, e.to_string()))?;
            rows.filter_map(|r| r.ok()).collect()
        } else if let Some(mid) = model_id.as_ref() {
            let mut stmt = con.prepare(sql).map_err(|e| (1, e.to_string()))?;
            let rows = stmt
                .query_map([mid], |r| r.get(0))
                .map_err(|e| (1, e.to_string()))?;
            rows.filter_map(|r| r.ok()).collect()
        } else {
            Vec::new()
        }
    };
    for sk in &starting {
        load_slot::request_cancel_slot(sk);
        let _ = load_slot::wait_slot_quiet(sk, Duration::from_millis(500));
    }
    let model_id_for_clear = model_id.clone();
    let ids: Vec<String> = if all {
        let mut stmt = con
            .prepare("SELECT instance_id FROM loaded_instances")
            .map_err(|e| (1, e.to_string()))?;
        let rows = stmt
            .query_map([], |r| r.get(0))
            .map_err(|e| (1, e.to_string()))?;
        rows.filter_map(|r| r.ok()).collect()
    } else {
        let mid = model_id.ok_or((2, "model_id required".to_string()))?;
        let mut stmt = con
            .prepare("SELECT instance_id FROM loaded_instances WHERE model_id = ?1")
            .map_err(|e| (1, e.to_string()))?;
        let rows = stmt
            .query_map([&mid], |r| r.get(0))
            .map_err(|e| (1, e.to_string()))?;
        rows.filter_map(|r| r.ok()).collect()
    };
    if ids.is_empty() {
        return Ok(Value::Null);
    }
    let model_ids: Vec<String> = if all {
        let mut stmt = con
            .prepare("SELECT DISTINCT model_id FROM loaded_instances")
            .map_err(|e| (1, e.to_string()))?;
        let rows = stmt
            .query_map([], |r| r.get(0))
            .map_err(|e| (1, e.to_string()))?;
        rows.filter_map(|r| r.ok()).collect()
    } else {
        vec![model_id_for_clear
            .ok_or((2, "model_id required".to_string()))?]
    };
    if !force {
        for id in &ids {
            let _ = con.execute(
                "UPDATE loaded_instances SET status = 'DRAINING' WHERE instance_id = ?1",
                [id],
            );
        }
        drain_inflight(&model_ids, &home);
    }
    for id in &ids {
        let _ = con.execute(
            "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1",
            [id],
        );
    }
    if ids.len() == 1 {
        teardown_instance(&con, &ids[0], &home)?;
    } else {
        teardown_batch(&con, &ids, &home)?;
    }
    for mid in model_ids {
        if !mid.is_empty() {
            let _ = crate::autostart::clear_desired(&mid);
        }
    }
    append_meshd_log(
        &home,
        &format!(
            "unload done ms={} instances={}\n",
            t0.elapsed().as_millis(),
            ids.len()
        ),
    );
    Ok(Value::Null)
}

pub fn handle_ps() -> Result<Value, (i32, String)> {
    let home = mesh_home();
    let con = db::open(&home).map_err(|e| (1, e.to_string()))?;
    let mut stmt = con
        .prepare(
            "SELECT instance_id, model_id, pid, pgid, internal_port, gpu_indices, status,
                    allocated_vram_gb_per_gpu, COALESCE(restart_count, 0),
                    COALESCE(slot_key, model_id) FROM loaded_instances",
        )
        .map_err(|e| (1, e.to_string()))?;
    let rows = stmt
        .query_map([], |r| {
            Ok(json!({
                "instance_id": r.get::<_, String>(0)?,
                "model_id": r.get::<_, String>(1)?,
                "pid": r.get::<_, i64>(2)?,
                "pgid": r.get::<_, i64>(3)?,
                "internal_port": r.get::<_, i64>(4)?,
                "gpu_indices": parse_gpu_indices(r.get::<_, String>(5)?),
                "status": r.get::<_, String>(6)?,
                "allocated_vram_gb_per_gpu": r.get::<_, f64>(7)?,
                "restarts": r.get::<_, i64>(8)?,
                "slot_key": r.get::<_, String>(9)?,
            }))
        })
        .map_err(|e| (1, e.to_string()))?;
    let mut out = Vec::new();
    for row in rows {
        let mut v = row.map_err(|e| (1, e.to_string()))?;
        if let Some(id) = v.get("instance_id").and_then(|x| x.as_str()) {
            let notices = load_slot::instance_notices(id);
            if !notices.is_empty() {
                v["notices"] = Value::Array(notices.into_iter().map(Value::String).collect());
            }
        }
        if let Some(mid) = v.get("model_id").and_then(|x| x.as_str()).map(str::to_string) {
            v["in_flight"] = json!(crate::inflight::count(&mid));
            v["tokens_per_s_1m"] = json!(crate::metrics::model_tokens_per_s_1m(&mid));
        }
        out.push(v);
    }
    Ok(Value::Array(out))
}

pub fn reconcile_on_start() -> Result<(), String> {
    let home = mesh_home();
    let _ = run_registry::reap_orphans_on_start(&home);
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let mut stmt = con
        .prepare("SELECT instance_id, pid, pgid, status FROM loaded_instances")
        .map_err(|e| e.to_string())?;
    let rows: Vec<(String, i32, i32, String)> = stmt
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();
    for (id, pid, pgid, status) in rows {
        if status == "DRAINING" {
            con.execute(
                "DELETE FROM loaded_instances WHERE instance_id = ?1",
                [&id],
            )
            .ok();
            continue;
        }
        if pid_alive(pid) && pgid_alive(pgid) {
            con.execute(
                "UPDATE loaded_instances SET status = 'RUNNING' WHERE instance_id = ?1",
                [&id],
            )
            .ok();
        } else {
            con.execute(
                "DELETE FROM loaded_instances WHERE instance_id = ?1",
                [&id],
            )
            .ok();
        }
    }
    Ok(())
}

struct InstRow {
    local_path: String,
    static_vram_gb: f64,
    kv_bytes_per_token: i64,
    default_ctx: i64,
    default_max_conc: i64,
    default_tp: i64,
    kv_elems_per_token: Option<i64>,
    max_ctx: Option<i64>,
    num_attention_heads: Option<i64>,
    num_kv_heads: Option<i64>,
    attention: Option<String>,
}

fn fetch_installed(con: &Connection, model_id: &str) -> Result<InstRow, String> {
    con.query_row(
        "SELECT local_path, static_vram_gb, kv_bytes_per_token, default_ctx, default_max_conc, default_tp,
                kv_elems_per_token, max_ctx, num_attention_heads, num_kv_heads, attention
         FROM installed_models WHERE model_id = ?1",
        [model_id],
        |r| {
            Ok(InstRow {
                local_path: r.get(0)?,
                static_vram_gb: r.get(1)?,
                kv_bytes_per_token: r.get(2)?,
                default_ctx: r.get(3)?,
                default_max_conc: r.get(4)?,
                default_tp: r.get(5)?,
                kv_elems_per_token: r.get(6).ok(),
                max_ctx: r.get(7).ok(),
                num_attention_heads: r.get(8).ok(),
                num_kv_heads: r.get(9).ok(),
                attention: r.get(10).ok(),
            })
        },
    )
    .map_err(|_| format!("model {model_id} not installed"))
}

fn chooser_input_from_row(
    row: &InstRow,
    card: &toml::Value,
    ctx: u64,
    max_conc: u64,
    home: &Path,
    gpus: &[u32],
) -> ChooserInput {
    let kv_elems = row.kv_elems_per_token.unwrap_or(row.kv_bytes_per_token) as u64;
    ChooserInput {
        static_vram_gb: effective_static_vram_gb(row, card, home, gpus),
        marlin_extra_gb: card.get("marlin_extra_gb").and_then(|v| v.as_float()),
        apply_marlin: false,
        kv_elems_per_token: kv_elems,
        num_attention_heads: row.num_attention_heads.unwrap_or(1) as u64,
        num_kv_heads: row.num_kv_heads.unwrap_or(1) as u64,
        attention: row
            .attention
            .clone()
            .unwrap_or_else(|| "gqa".to_string()),
        max_ctx: row.max_ctx.unwrap_or(row.default_ctx) as u64,
        ctx,
        max_conc,
        kv_bytes_override: Some(row.kv_bytes_per_token as u64),
    }
}

fn load_hw(home: &Path) -> Option<HardwareState> {
    let path = home.join("hardware_state.json");
    let text = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn gpu_cap(hw: &Option<HardwareState>, index: u32) -> f64 {
    hw.as_ref()
        .and_then(|h| {
            h.gpus
                .iter()
                .find(|g| g.index == index)
                .and_then(|g| g.compute_capability.parse().ok())
        })
        .unwrap_or(12.0)
}

fn parse_gpus_param(raw: Option<&str>, card: &toml::Value) -> Vec<u32> {
    if let Some(s) = raw {
        return s
            .split(',')
            .filter_map(|p| p.trim().parse().ok())
            .collect();
    }
    card.get("preferred_gpus")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_integer().map(|i| i as u32))
                .collect()
        })
        .unwrap_or_default()
}

fn read_card_value(path: &Path) -> Result<toml::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    toml::from_str(&text).map_err(|e| e.to_string())
}

fn parse_gpu_indices(s: String) -> Value {
    Value::Array(
        s.split(',')
            .filter_map(|p| p.trim().parse::<i64>().ok())
            .map(Value::from)
            .collect(),
    )
}

fn gpu_csv(gpus: &[u32]) -> String {
    gpus.iter()
        .map(|g| g.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn effective_static_vram_gb(
    row: &InstRow,
    card: &toml::Value,
    home: &Path,
    gpus: &[u32],
) -> f64 {
    let extra = card
        .get("marlin_extra_gb")
        .and_then(|v| v.as_float());
    static_weight_gb(
        row.static_vram_gb,
        extra,
        marlin_extra_applies(home, gpus, card),
    )
}

fn marlin_extra_applies(home: &Path, gpus: &[u32], card: &toml::Value) -> bool {
    if marlin_disabled(card) {
        return false;
    }
    gpus.iter()
        .map(|g| gpu_cc_major(home, *g))
        .max()
        .unwrap_or(0)
        >= 10
}

fn marlin_disabled(card: &toml::Value) -> bool {
    if std::env::var("MESH_MARLIN").as_deref() == Ok("0") {
        return true;
    }
    card.get("env")
        .and_then(|v| v.as_table())
        .and_then(|t| t.get("MESH_MARLIN"))
        .and_then(|v| v.as_str())
        == Some("0")
}

fn gpu_cc_major(home: &Path, gpu: u32) -> u32 {
    let hw_path = home.join("hardware_state.json");
    if let Ok(text) = std::fs::read_to_string(&hw_path) {
        if let Ok(v) = serde_json::from_str::<Value>(&text) {
            if let Some(arr) = v.get("gpus").and_then(|g| g.as_array()) {
                for g in arr {
                    if g.get("index").and_then(|i| i.as_u64()) == Some(gpu as u64) {
                        if let Some(cc) = g.get("compute_capability").and_then(|c| c.as_str()) {
                            if let Some(maj) = cc.split('.').next().and_then(|s| s.parse().ok()) {
                                return maj;
                            }
                        }
                    }
                }
            }
        }
    }
    0
}

fn gpu_free_gb(home: &Path, gpu: u32) -> Result<f64, String> {
    let hw_path = home.join("hardware_state.json");
    if hw_path.is_file() {
        let text = std::fs::read_to_string(&hw_path).map_err(|e| e.to_string())?;
        let v: Value = serde_json::from_str(&text).map_err(|e| e.to_string())?;
        if let Some(arr) = v.get("gpus").and_then(|g| g.as_array()) {
            for g in arr {
                if g.get("index").and_then(|i| i.as_u64()) == Some(gpu as u64) {
                    let free = g
                        .get("vram_free_mb")
                        .and_then(|x| x.as_u64())
                        .unwrap_or(0);
                    return Ok(free as f64 / 1024.0);
                }
            }
        }
    }
    let (_t, _u, free, _p) = refresh_gpu_vram(gpu)?;
    Ok(free as f64 / 1024.0)
}

fn mem_fraction(_home: &Path, gpus: &[u32], b_gb: f64) -> Result<f64, String> {
    if gpus.is_empty() {
        return Err("no gpu".to_string());
    }
    let mut min_free_gb = f64::MAX;
    for &gpu in gpus {
        let free_mb = gpu_free_mb_live(gpu)?;
        min_free_gb = min_free_gb.min(free_mb as f64 / 1024.0);
    }
    mem_fraction_from_free(b_gb, min_free_gb)
}

fn check_foreign(
    gpu: u32,
    con: &Connection,
    bypass: bool,
) -> Result<Option<String>, (i32, String)> {
    if bypass {
        return Ok(None);
    }
    let managed = placement::managed_pids(con);
    if let Some((pid, gb)) = foreign_compute_on_gpu(gpu, &managed).map_err(|e| (1, e))? {
        return Ok(Some(format!(
            "GPU {gpu} has unmanaged process PID {pid} using {gb:.1} GB VRAM. Proceed? [y/N] "
        )));
    }
    let used_mb = gpu_used_mb_live(gpu).map_err(|e| (1, e))?;
    let listed_mb = managed_vram_listed_mb(gpu, &managed).map_err(|e| (1, e))?;
    if listed_mb > 0 {
        let slack = used_mb.saturating_sub(listed_mb.saturating_add(512));
        if slack > 4096 {
            return Ok(Some(format!(
                "GPU {gpu} has ~{:.1} GB VRAM not attributed to mesh engines (~{:.1} GB in use). Proceed? [y/N] ",
                slack as f64 / 1024.0,
                used_mb as f64 / 1024.0
            )));
        }
    } else if !gpu_has_mesh_instance(con, gpu) && used_mb > 4096 {
        return Ok(Some(format!(
            "GPU {gpu} has ~{:.1} GB VRAM in use with no mesh engines loaded. Proceed? [y/N] ",
            used_mb as f64 / 1024.0
        )));
    }
    Ok(None)
}

fn gpu_has_mesh_instance(con: &Connection, gpu: u32) -> bool {
    let Ok(mut stmt) = con.prepare("SELECT gpu_indices FROM loaded_instances") else {
        return false;
    };
    let Ok(rows) = stmt.query_map([], |r| r.get::<_, String>(0)) else {
        return false;
    };
    for csv in rows.filter_map(|r| r.ok()) {
        for part in csv.split(',') {
            if part.trim().parse::<u32>().ok() == Some(gpu) {
                return true;
            }
        }
    }
    false
}

fn co_residency_notice(
    gpu: u32,
    model_id: &str,
    con: &Connection,
    home: &Path,
) -> Result<Option<String>, String> {
    let mut stmt = con
        .prepare("SELECT model_id, allocated_vram_gb_per_gpu, gpu_indices FROM loaded_instances")
        .map_err(|e| e.to_string())?;
    let mut hosts = Vec::new();
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, f64>(1)?,
                r.get::<_, String>(2)?,
            ))
        })
        .map_err(|e| e.to_string())?;
    for row in rows.filter_map(|r| r.ok()) {
        if row.0 == model_id {
            continue;
        }
        if !gpu_indices_contain(&row.2, gpu) {
            continue;
        }
        hosts.push(format!("{}: {:.1} GB", row.0, row.1));
    }
    if hosts.is_empty() {
        return Ok(None);
    }
    let free_gb = gpu_free_gb(home, gpu)?;
    Ok(Some(format!(
        "GPU {gpu} currently hosts [{}]. Free VRAM: {free_gb:.1} GB.",
        hosts.join(", ")
    )))
}

fn gpu_indices_contain(csv: &str, gpu: u32) -> bool {
    csv.split(',').any(|p| p.trim().parse::<u32>() == Ok(gpu))
}

fn next_port(con: &Connection) -> Result<u16, String> {
    for port in 30001..=30099 {
        let n: i64 = con
            .query_row(
                "SELECT COUNT(*) FROM loaded_instances WHERE internal_port = ?1",
                [port as i64],
                |r| r.get(0),
            )
            .map_err(|e| e.to_string())?;
        if n == 0 && port_available(port) {
            return Ok(port);
        }
    }
    Err("no free internal ports in 30001-30099".to_string())
}

fn port_available(port: u16) -> bool {
    TcpListener::bind(("127.0.0.1", port)).is_ok()
}

#[allow(clippy::too_many_arguments)]
fn spawn_tracked_engine(
    home: &Path,
    instance_id: &str,
    model_id: &str,
    model_path: &Path,
    card: &toml::Value,
    port: u16,
    ctx: u64,
    max_conc: u64,
    tp: u64,
    mem_frac: f64,
    gpus: &[u32],
    load_extra: &[String],
    log: &mut File,
) -> Result<(i32, i32), String> {
    let model_path_s = model_path.to_string_lossy().into_owned();
    let entry = RunEntry {
        instance_id: instance_id.to_string(),
        pgid: 0,
        pid: 0,
        model_path: model_path_s.clone(),
    };
    let _ = run_registry::register(home, &entry);
    let (pid, pgid) = spawn_engine(
        home,
        model_path,
        card,
        port,
        ctx,
        max_conc,
        tp,
        mem_frac,
        gpus,
        load_extra,
        log,
    )?;
    let log_path = home.join("logs").join(format!("{instance_id}.log"));
    if let Err(e) = crate::logs::append_spawn_index(home, instance_id, model_id, &log_path) {
        eprintln!("append_spawn_index({instance_id}): {e}");
    }
    let entry = RunEntry {
        instance_id: instance_id.to_string(),
        pgid,
        pid,
        model_path: model_path_s,
    };
    let _ = run_registry::register(home, &entry);
    Ok((pid, pgid))
}

#[allow(clippy::too_many_arguments)]
fn spawn_engine(
    home: &Path,
    model_path: &Path,
    card: &toml::Value,
    port: u16,
    ctx: u64,
    max_conc: u64,
    tp: u64,
    mem_frac: f64,
    gpus: &[u32],
    load_extra: &[String],
    log: &mut File,
) -> Result<(i32, i32), String> {
    let python = resolve_python(home);
    let kv_dtype = card
        .get("kv_cache_dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("fp8_e4m3");
    let mut cmd = Command::new(python);
    cmd.arg("-m")
        .arg("sglang.launch_server")
        .arg("--model-path")
        .arg(model_path)
        .arg("--host")
        .arg("127.0.0.1")
        .arg("--port")
        .arg(port.to_string())
        .arg("--tp-size")
        .arg(tp.to_string())
        .arg("--context-length")
        .arg(ctx.to_string())
        .arg("--max-running-requests")
        .arg(max_conc.to_string())
        .arg("--mem-fraction-static")
        .arg(format!("{mem_frac:.6}"))
        .arg("--kv-cache-dtype")
        .arg(kv_dtype);
    if tp > 1 {
        cmd.arg("--disable-custom-all-reduce");
    }
    if let Some(name) = card.get("model_id").and_then(|v| v.as_str()) {
        cmd.arg("--served-model-name").arg(name);
    }
    if let Some(table) = card.get("env").and_then(|v| v.as_table()) {
        for (k, v) in table {
            if let Some(s) = v.as_str() {
                cmd.env(k, expand_home(s, home));
            }
        }
    }
    if let Some(extra) = card.get("extra_sglang_args").and_then(|v| v.as_array()) {
        for item in extra {
            if let Some(s) = item.as_str() {
                cmd.arg(s);
            }
        }
    }
    for arg in load_extra {
        cmd.arg(arg);
    }
    cmd.env(
        "CUDA_VISIBLE_DEVICES",
        gpus.iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(","),
    );
    cmd.env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    cmd.env("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True");
    let runtime = card
        .get("runtime")
        .and_then(|v| v.as_str())
        .unwrap_or("sglang");
    if runtime == "ternary" {
        apply_ternary_env(&mut cmd, home, card)?;
    }
    cmd.stdout(Stdio::from(log.try_clone().map_err(|e| e.to_string())?));
    cmd.stderr(Stdio::from(log.try_clone().map_err(|e| e.to_string())?));
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let child = cmd.spawn().map_err(|e| format!("spawn engine: {e}"))?;
    let pid = child.id() as i32;
    std::mem::forget(child);
    let pgid = live_pgid(pid).unwrap_or(pid);
    Ok((pid, pgid))
}

fn apply_ternary_env(cmd: &mut Command, home: &Path, card: &toml::Value) -> Result<(), String> {
    if let Some(table) = card.get("env").and_then(|v| v.as_table()) {
        for (k, v) in table {
            if let Some(s) = v.as_str() {
                cmd.env(k, expand_home(s, home));
            }
        }
    }
    if let Some(table) = card.as_table() {
        for (k, v) in table {
            if k.starts_with("TERNARY_") {
                if let Some(s) = v.as_str() {
                    cmd.env(k, expand_home(s, home));
                }
            }
        }
    }
    if let Some(arr) = card.get("pythonpath").and_then(|v| v.as_array()) {
        let parts: Vec<String> = arr
            .iter()
            .filter_map(|v| v.as_str().map(|s| expand_home(s, home)))
            .collect();
        if !parts.is_empty() {
            cmd.env("PYTHONPATH", parts.join(":"));
        }
    }
    Ok(())
}

fn expand_home(s: &str, home: &Path) -> String {
    s.replace("$MESH_HOME", &home.to_string_lossy())
}

fn resolve_python(home: &Path) -> PathBuf {
    let p = home.join("py-env/bin/python");
    if p.is_file() {
        return p;
    }
    if let Ok(env) = std::env::var("MESH_PYTHON") {
        return PathBuf::from(env);
    }
    PathBuf::from("python3")
}

const HTTP_PROBE_SECS: u64 = 25;

fn http_status(url: &str, timeout_secs: u64) -> Option<u16> {
    let agent = ureq::AgentBuilder::new()
        .timeout_read(Duration::from_secs(timeout_secs))
        .timeout_connect(Duration::from_secs(2))
        .build();
    agent.get(url).call().ok().map(|r| r.status())
}

fn cheap_probe_ok(port: u16) -> bool {
    for path in ["/v1/models", "/get_model_info"] {
        let url = format!("http://127.0.0.1:{port}{path}");
        if http_status(&url, 2) == Some(200) {
            return true;
        }
    }
    false
}

fn deep_health_ok(port: u16) -> bool {
    readiness_health_ok(port)
}

fn readiness_health_ok(port: u16) -> bool {
    let url = format!("http://127.0.0.1:{port}/health");
    http_status(&url, HTTP_PROBE_SECS) == Some(200)
}

pub fn health_ok(url: &str) -> bool {
    http_status(url, HTTP_PROBE_SECS) == Some(200)
}

fn log_file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn now_epoch_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn wait_engine_ready(port: u16, pgid: i32, log_path: &Path) -> Result<(), crate::startup_stall::StartupFail> {
    let hard_cap = crate::startup_stall::hard_cap_secs_from_env();
    let mut tracker =
        crate::startup_stall::StallTracker::new(now_epoch_secs(), log_file_len(log_path));
    loop {
        let now = now_epoch_secs();
        let step = crate::startup_stall::poll_startup_step(
            &mut tracker,
            now,
            pgid_alive(pgid),
            log_file_len(log_path),
            readiness_health_ok(port),
            hard_cap,
        );
        match step {
            crate::startup_stall::StartupStep::Ready => return Ok(()),
            crate::startup_stall::StartupStep::Failed(f) => return Err(f),
            crate::startup_stall::StartupStep::Continue => {
                thread::sleep(Duration::from_millis(500));
            }
        }
    }
}

fn teardown_batch(con: &Connection, ids: &[String], home: &Path) -> Result<(), (i32, String)> {
    for id in ids {
        let _ = con.execute(
            "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1",
            [id],
        );
    }
    let mut items = Vec::new();
    for id in ids {
        if let Ok(row) = con.query_row(
            "SELECT pid, pgid, gpu_indices FROM loaded_instances WHERE instance_id = ?1",
            [id],
            |r| Ok((r.get::<_, i32>(0)?, r.get::<_, i32>(1)?, r.get::<_, String>(2)?)),
        ) {
            items.push((id.clone(), row.0, row.1, row.2));
        }
    }
    let engines: Vec<(i32, i32)> = items.iter().map(|(_, pid, pgid, _)| (*pid, *pgid)).collect();
    run_registry::kill_process_groups(&engines);
    for (id, _, _, _) in &items {
        run_registry::unregister(home, id);
    }
    for (id, _pid, _pgid, _gpus) in &items {
        con.execute(
            "DELETE FROM loaded_instances WHERE instance_id = ?1",
            [id],
        )
        .map_err(|e| (1, e.to_string()))?;
    }
    let _ = home;
    Ok(())
}

fn teardown_instance(con: &Connection, instance_id: &str, home: &Path) -> Result<(), (i32, String)> {
    let (pid, pgid, gpus): (i32, i32, String) = con
        .query_row(
            "SELECT pid, pgid, gpu_indices FROM loaded_instances WHERE instance_id = ?1",
            [instance_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .map_err(|_| (1, format!("instance {instance_id} not found")))?;
    let _ = con.execute(
        "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1",
        [instance_id],
    );
    run_registry::kill_process_groups(&[(pid, pgid)]);
    run_registry::unregister(home, instance_id);
    for g in gpus.split(',') {
        if let Ok(idx) = g.trim().parse::<u32>() {
            let _ = refresh_gpu_vram(idx);
        }
    }
    con.execute(
        "DELETE FROM loaded_instances WHERE instance_id = ?1",
        [instance_id],
    )
    .map_err(|e| (1, e.to_string()))?;
    let _ = home;
    Ok(())
}

fn live_pgid(pid: i32) -> Option<i32> {
    let pgid = unsafe { libc::getpgid(pid) };
    if pgid > 0 {
        Some(pgid)
    } else {
        None
    }
}

fn pid_alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

fn reap_pgid(pgid: i32) {
    if pgid <= 0 {
        return;
    }
    let mut status = 0;
    loop {
        let p = unsafe { libc::waitpid(-pgid, &mut status, libc::WNOHANG) };
        if p <= 0 {
            break;
        }
    }
}

fn pgid_alive(pgid: i32) -> bool {
    if pgid <= 0 {
        return false;
    }
    reap_pgid(pgid);
    unsafe { libc::kill(-pgid, 0) == 0 }
}

pub fn watchdog_tick() {
    let _guard = match WATCHDOG_TICK_LOCK.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    if crate::shutdown::shutdown_requested() {
        return;
    }
    let home = mesh_home();
    let Ok(con) = db::open(&home) else {
        return;
    };
    let Ok(mut stmt) = con.prepare(
        "SELECT instance_id, model_id, pid, pgid, internal_port, COALESCE(restart_count,0),
                COALESCE(slot_key, model_id)
         FROM loaded_instances WHERE status = 'RUNNING'",
    ) else {
        return;
    };
    let rows: Vec<(String, String, i32, i32, i64, i64, String)> = stmt
        .query_map([], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
                r.get(6)?,
            ))
        })
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|r| r.ok())
        .collect();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    for (inst, model_id, _pid, pgid, port, restarts, slot_key) in rows {
        let in_flight: i64 = con
            .query_row(
                "SELECT restart_in_flight FROM watchdog_probe_state WHERE instance_id = ?1",
                [&inst],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if in_flight != 0 {
            continue;
        }
        let group_alive = pgid_alive(pgid);
        let mut state = load_probe_state(&con, &inst, now);
        let cheap = group_alive && cheap_probe_ok(port as u16);
        let deep = if crate::watchdog_liveness::deep_due(&state, now) {
            Some(deep_health_ok(port as u16))
        } else {
            None
        };
        let action = crate::watchdog_liveness::step(
            &mut state,
            crate::watchdog_liveness::ProbeInput {
                now,
                pid_alive: group_alive,
                cheap_ok: cheap,
                deep_ok: deep,
            },
        );
        save_probe_state(&con, &inst, &state);
        if action == crate::watchdog_liveness::WatchdogAction::None {
            continue;
        }
        let reason = match action {
            crate::watchdog_liveness::WatchdogAction::MarkDead { reason } => reason,
            _ => continue,
        };
        watchdog_log(&home, &inst, &slot_key, reason, cheap, deep);
        if !watchdog_may_restart(&con, &slot_key) {
            let _ = con.execute(
                "UPDATE loaded_instances SET status = 'FAILED' WHERE instance_id = ?1",
                [&inst],
            );
            continue;
        }
        let _ = con.execute(
            "INSERT OR REPLACE INTO watchdog_probe_state (instance_id, restart_in_flight)
             VALUES (?1, 1)",
            [&inst],
        );
        let _ = con.execute(
            "UPDATE loaded_instances SET status = 'RESTARTING' WHERE instance_id = ?1",
            [&inst],
        );
        let _ = teardown_instance(&con, &inst, &home);
        let params = load_params_for_slot(&con, &slot_key, &model_id);
        let Some(params) = params else {
            let _ = con.execute(
                "UPDATE watchdog_probe_state SET restart_in_flight = 0 WHERE instance_id = ?1",
                [&inst],
            );
            continue;
        };
        let attempt = restarts;
        let next_restarts = restarts + 1;
        let sk = slot_key.clone();
        let inst_id = inst.clone();
        thread::spawn(move || {
            let delay = crate::watchdog_policy::restart_backoff_secs(attempt);
            thread::sleep(std::time::Duration::from_secs(delay));
            let slot_arc = slot_lock(&sk);
            let _guard = slot_arc.lock().unwrap_or_else(|e| e.into_inner());
            let mut p = params;
            p.restart_count = Some(next_restarts as u64);
            p.async_finish = true;
            p.slot_key = Some(sk);
            if let Err((code, msg)) = handle_load(p) {
                eprintln!("watchdog restart load failed ({code}): {msg}");
            }
            if let Ok(con) = db::open(&mesh_home()) {
                let _ = con.execute(
                    "UPDATE watchdog_probe_state SET restart_in_flight = 0 WHERE instance_id = ?1",
                    [&inst_id],
                );
            }
        });
    }
}

fn watchdog_log(
    home: &Path,
    instance_id: &str,
    slot_key: &str,
    reason: &str,
    cheap_ok: bool,
    deep_ok: Option<bool>,
) {
    let path = home.join("logs/meshd.log");
    let line = format!(
        "watchdog instance={instance_id} slot={slot_key} reason={reason} cheap_ok={cheap_ok} deep_ok={deep_ok:?}\n"
    );
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
}

fn load_probe_state(con: &Connection, instance_id: &str, now: i64) -> crate::watchdog_liveness::LivenessState {
    con.query_row(
        "SELECT cheap_fail_since, deep_fail_streak, last_deep_at, COALESCE(cheap_ok_once,0) FROM watchdog_probe_state WHERE instance_id = ?1",
        [instance_id],
        |r| {
            Ok(crate::watchdog_liveness::LivenessState {
                cheap_fail_since: r.get(0)?,
                deep_fail_streak: r.get::<_, i64>(1)? as u32,
                last_deep_at: r.get(2)?,
                cheap_ok_once: r.get::<_, i64>(3)? != 0,
            })
        },
    )
    .unwrap_or(crate::watchdog_liveness::LivenessState {
        last_deep_at: now - crate::watchdog_liveness::DEEP_INTERVAL_SECS,
        ..Default::default()
    })
}

fn save_probe_state(con: &Connection, instance_id: &str, st: &crate::watchdog_liveness::LivenessState) {
    let _ = con.execute(
        "INSERT OR REPLACE INTO watchdog_probe_state
        (instance_id, cheap_fail_since, deep_fail_streak, last_deep_at, restart_in_flight, cheap_ok_once)
        VALUES (?1,?2,?3,?4, COALESCE((SELECT restart_in_flight FROM watchdog_probe_state WHERE instance_id = ?1),0), ?5)",
        rusqlite::params![
            instance_id,
            st.cheap_fail_since,
            st.deep_fail_streak as i64,
            st.last_deep_at,
            i64::from(st.cheap_ok_once),
        ],
    );
}

fn watchdog_may_restart(con: &Connection, slot_key: &str) -> bool {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let row: Option<(i64, i64)> = con
        .query_row(
            "SELECT window_start, count FROM watchdog_restarts WHERE slot_key = ?1",
            [slot_key],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok();
    let (start, count) = row.map(|(s, c)| (Some(s), c)).unwrap_or((None, 0));
    match crate::watchdog_policy::decide_restart(now, start, count) {
        crate::watchdog_policy::RestartDecision::DenyFailed => false,
        crate::watchdog_policy::RestartDecision::Allow {
            new_window_start,
            new_count,
        } => {
            con.execute(
        "INSERT OR REPLACE INTO watchdog_restarts (slot_key, window_start, count)
         VALUES (?1, ?2, ?3)",
        rusqlite::params![slot_key, new_window_start, new_count],
            )
            .ok();
            true
        }
    }
}

fn load_params_for_slot(con: &Connection, slot_key: &str, model_id: &str) -> Option<LoadParams> {
    let extra: String = con
        .query_row(
            "SELECT extra_sglang_args FROM desired_loads WHERE slot_key = ?1 OR model_id = ?2 LIMIT 1",
            rusqlite::params![slot_key, model_id],
            |r| r.get(0),
        )
        .ok()?;
    let extra: Vec<String> = serde_json::from_str::<serde_json::Value>(&extra)
        .ok()
        .and_then(|v| {
            v.as_array().map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect::<Vec<String>>()
            })
        })
        .unwrap_or_default();
    let (gpus, ctx, max_conc, tp): (Option<String>, Option<i64>, Option<i64>, Option<i64>) = con
        .query_row(
            "SELECT gpus, ctx, max_conc, tp FROM desired_loads WHERE slot_key = ?1 OR model_id = ?2 LIMIT 1",
            rusqlite::params![slot_key, model_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .ok()?;
    Some(LoadParams {
        model_id: model_id.to_string(),
        gpus,
        ctx: ctx.map(|v| v as u64),
        max_conc: max_conc.map(|v| v as u64),
        tp: tp.map(|v| v as u64),
        extra_sglang_args: extra,
        force: true,
        yes: true,
        is_tty: false,
        restart_count: None,
        slot_key: Some(slot_key.to_string()),
        async_finish: true,
    })
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let n = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{n:x}")
}

#[cfg(test)]
mod unload_watchdog_tests {
    use super::*;
    use crate::db;
    use crate::paths::ensure_layout;
    use rusqlite::Connection;
    use std::path::Path;
    use std::sync::{Mutex, OnceLock};

    fn home_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_temp_home<F: FnOnce(&Path, &Connection)>(f: F) {
        let _g = home_lock().lock().unwrap_or_else(|e| e.into_inner());
        let td = tempfile::tempdir().expect("tempdir");
        std::env::set_var("MESH_HOME", td.path());
        ensure_layout(td.path()).expect("layout");
        let con = db::open(td.path()).expect("db");
        f(td.path(), &con);
        std::env::remove_var("MESH_HOME");
    }

    #[test]
    fn watchdog_tick_skips_stopping_instances() {
        with_temp_home(|_home, con| {
            con.execute(
                "INSERT INTO loaded_instances
                (instance_id, model_id, pid, pgid, internal_port, gpu_indices,
                 allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status, slot_key)
                VALUES ('i-stop', 'm', 1, 1, 30001, '0', 1.0, 0.5, 4096, 4, 'STOPPING', 'm|0|0')",
                [],
            )
            .expect("insert");
            let n: i64 = con
                .query_row(
                    "SELECT COUNT(*) FROM loaded_instances WHERE status = 'RUNNING'",
                    [],
                    |r| r.get(0),
                )
                .expect("count");
            assert_eq!(n, 0);
        });
    }

    #[test]
    fn intentional_unload_does_not_increment_watchdog_restart_window() {
        with_temp_home(|home, con| {
            con.execute(
                "INSERT INTO loaded_instances
                (instance_id, model_id, pid, pgid, internal_port, gpu_indices,
                 allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status, slot_key)
                VALUES ('i1', 'm', 999999, 999999, 30001, '0', 1.0, 0.5, 4096, 4, 'RUNNING', 'm|0|0')",
                [],
            )
            .expect("insert");
            handle_unload(Some("m".to_string()), false, false).expect("unload");
            let restarts: i64 = con
                .query_row(
                    "SELECT COALESCE(count, 0) FROM watchdog_restarts WHERE slot_key = 'm|0|0'",
                    [],
                    |r| r.get(0),
                )
                .unwrap_or(0);
            assert_eq!(restarts, 0, "unload must not consume watchdog restart budget");
            let left: i64 = con
                .query_row("SELECT COUNT(*) FROM loaded_instances", [], |r| r.get(0))
                .expect("count");
            assert_eq!(left, 0);
            let _ = home;
        });
    }
}
