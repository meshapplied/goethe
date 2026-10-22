// goethed load, unload, adoption and watchdog supervision of SGLang engines.

use crate::autostart;
use crate::calc::{mem_fraction_from_free, missing_gb, static_weight_gb};
use crate::chooser::{self, ChooserInput, LoadPlan};
use crate::coordinator;
use crate::db;
use crate::engine_cmd::{self, EngineSpec};
use crate::foreign::{self, Foreign};
use crate::hw::HardwareState;
use crate::load_slot;
use crate::paths::{ensure_layout, goethe_home};
use crate::placement;
use crate::proc_ident::{self, Identity};
use crate::run_registry;
use crate::shutdown::shutdown_requested;
use crate::watchdog_liveness::{self, LivenessState, ProbeInput, WatchdogAction};
use rusqlite::{Connection, ToSql, TransactionBehavior};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::OpenOptions;
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

static SLOT_LOCKS: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
static WATCHDOG_TICK_LOCK: Mutex<()> = Mutex::new(());

const LIVE: &str = "('STARTING', 'RUNNING', 'RESTARTING', 'DRAINING', 'STOPPING')";
const GPU_RELEASE_WAIT: Duration = Duration::from_secs(2);
const HTTP_PROBE_SECS: u64 = 25;

pub type LoadErr = (i32, String, Value);

fn slot_lock(key: &str) -> Arc<Mutex<()>> {
    let map = SLOT_LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = map.lock().unwrap_or_else(|e| e.into_inner());
    g.entry(key.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

fn err1(e: impl ToString) -> LoadErr {
    (1, e.to_string(), Value::Null)
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
    pub restart_count: Option<u64>,
    #[serde(default)]
    pub slot_key: Option<String>,
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

struct LoadReq {
    home: PathBuf,
    params: LoadParams,
    row: InstRow,
    model_path: PathBuf,
    card: toml::Value,
    pinned: Vec<u32>,
    pinned_tp: Option<usize>,
    default_tp: Option<usize>,
    ctx: u64,
    max_conc: u64,
    denied: HashSet<u32>,
    hw: Option<HardwareState>,
}

struct Starting {
    home: PathBuf,
    params: LoadParams,
    instance_id: String,
    engine_key: String,
    identity: Identity,
    port: u16,
    log_path: PathBuf,
    log_offset: u64,
    budget_gb: f64,
    gpu_name: String,
    tp: u64,
    model_path: PathBuf,
    card: toml::Value,
    gpus: Vec<u32>,
    ctx: u64,
    max_conc: u64,
    cancel: Arc<AtomicBool>,
}

struct Inst {
    instance_id: String,
    model_id: String,
    slot_key: String,
    status: String,
    identity: Option<Identity>,
    gpus: Vec<u32>,
    port: u16,
    engine_key: String,
    restarts: i64,
}

enum Step {
    Started(Box<Starting>, String),
    Wait,
}

enum Startup {
    Ready,
    Cancelled,
    Failed,
}

fn insts(con: &Connection, filter: &str, args: &[&dyn ToSql]) -> Vec<Inst> {
    // Instance rows matching a WHERE clause, with identity, GPUs and engine key parsed.
    let sql = format!(
        "SELECT instance_id, model_id, COALESCE(slot_key, model_id), status, pid, boot_id, start_ticks,
                gpu_indices, internal_port, COALESCE(engine_key, ''), COALESCE(restart_count, 0)
         FROM loaded_instances WHERE {filter} ORDER BY rowid"
    );
    let Ok(mut stmt) = con.prepare(&sql) else {
        return Vec::new();
    };
    stmt.query_map(args, |r| {
        let instance_id: String = r.get(0)?;
        let identity = Identity::from_row(&instance_id, r.get(4)?, r.get(5)?, r.get(6)?);
        Ok(Inst {
            identity,
            model_id: r.get(1)?,
            slot_key: r.get(2)?,
            status: r.get(3)?,
            gpus: parse_gpu_csv(&r.get::<_, String>(7)?),
            port: u16::try_from(r.get::<_, i64>(8)?).unwrap_or(0),
            engine_key: r.get(9)?,
            restarts: r.get(10)?,
            instance_id,
        })
    })
    .map(|rows| rows.filter_map(|r| r.ok()).collect())
    .unwrap_or_default()
}

pub fn internal_port_for_model(model_id: &str) -> Option<u16> {
    let home = goethe_home();
    let con = db::open(&home).ok()?;
    con.query_row(
        "SELECT internal_port FROM loaded_instances WHERE model_id = ?1 AND status = 'RUNNING' LIMIT 1",
        [model_id],
        |r| r.get::<_, i64>(0),
    )
    .ok()
    .and_then(|p| u16::try_from(p).ok())
}

pub fn handle_load(params: LoadParams) -> Result<(Value, String), LoadErr> {
    // Validate, then plan, gate, spawn and record one engine; the health wait continues on a background thread.
    let home = goethe_home();
    ensure_layout(&home).map_err(err1)?;
    let con = db::open(&home).map_err(err1)?;
    let row = fetch_installed(&con, &params.model_id).map_err(err1)?;
    let model_path = home.join(&row.local_path);
    let card = read_card_value(&model_path.join("goethe_card.toml")).map_err(err1)?;
    if card.get("runtime").and_then(|v| v.as_str()) == Some("ternary") {
        crate::runtime::require_ternary_runtime(&home).map_err(err1)?;
    }
    for extra in [engine_cmd::card_extra_args(&card), params.extra_sglang_args.clone()] {
        engine_cmd::check_extra_args(&extra).map_err(|e| (2, e, Value::Null))?;
    }
    let pinned = pinned_gpus(params.gpus.as_deref(), &card, params.tp).map_err(|e| (2, e, Value::Null))?;
    let denied = crate::gpu_policy::denied_set(&con).map_err(err1)?;
    if let Some(g) = pinned.iter().find(|g| denied.contains(g)) {
        return Err(err1(format!(
            "GPU {g} is denied by policy; run `goethe gpu allow {g}` to use it"
        )));
    }
    let req = LoadReq {
        pinned_tp: params
            .tp
            .map(|t| t as usize)
            .or((!pinned.is_empty()).then_some(pinned.len())),
        default_tp: (row.default_tp > 0).then_some(row.default_tp as usize),
        ctx: params.ctx.unwrap_or_else(|| default_ctx(&row)),
        max_conc: params.max_conc.unwrap_or(row.default_max_conc as u64),
        hw: placement::load_hw(&home),
        home,
        params,
        row,
        model_path,
        card,
        pinned,
        denied,
    };
    let ticket = coordinator::enqueue(&req.params.model_id);
    loop {
        if ticket.cancelled() || shutdown_requested() {
            return Err(err1("load cancelled"));
        }
        let step = {
            let _plan = coordinator::plan_lock();
            if ticket.cancelled() {
                return Err(err1("load cancelled"));
            }
            plan_and_start(&req, &con)?
        };
        match step {
            Step::Started(starting, notices) => return Ok(launch_finish(*starting, notices)),
            Step::Wait => coordinator::wait_change(Duration::from_secs(1)),
        }
    }
}

fn plan_and_start(req: &LoadReq, con: &Connection) -> Result<Step, LoadErr> {
    // Under the planning lock: live GPU view minus the ledger, plan, fit and foreign checks, then the per-GPU gate.
    let snap = foreign::snapshot().map_err(err1)?;
    if let Some(g) = req.pinned.iter().find(|g| !snap.iter().any(|s| s.index == **g)) {
        return Err(err1(format!(
            "GPU {g} does not exist; this host has {} GPUs",
            snap.len()
        )));
    }
    let managed = placement::managed_pids(con);
    let live = placement::live_instance_ids(con);
    let inputs = placement::placement_inputs(&snap, &managed, &live, &req.denied, coordinator::reserved_gb);
    if let Some(tp) = req.pinned_tp {
        placement::ensure_eligible_for_tp(&inputs, tp, &req.denied).map_err(err1)?;
    }
    let base = chooser_input(req, &req.pinned, req.ctx, req.max_conc);
    let pinned = (!req.pinned.is_empty()).then_some(req.pinned.as_slice());
    let plan = chooser::pick_load_plan(&base, &inputs, req.hw.as_ref(), req.pinned_tp.or(req.default_tp), pinned)
        .map_err(|e| missing_error(e.message, &foreign::foreign_on(&snap, &e.gpus, &managed, &live)))?;
    let free: HashMap<u32, f64> = plan
        .gpus
        .iter()
        .map(|g| {
            let mb = inputs.iter().find(|i| i.index == *g).map(|i| i.free_vram_mb).unwrap_or(0);
            (*g, mb as f64 / 1024.0)
        })
        .collect();
    let gpu_name = gpu_name(req.hw.as_ref(), plan.gpus[0]);
    let input = chooser_input(req, &plan.gpus, plan.ctx, plan.max_conc);
    let budget = chooser::per_gpu_budget_gb(&input, plan.tp)
        + crate::engine_overhead::read_learned_extra_gb(con, &req.params.model_id, &gpu_name, plan.tp as u64);
    let found = foreign::foreign_on(&snap, &plan.gpus, &managed, &live);
    if let Some((g, f)) = plan.gpus.iter().map(|g| (*g, free[g])).find(|(_, f)| missing_gb(budget, *f) > 0.0) {
        return Err(missing_error(
            format!(
                "VRAM budget {budget:.2} GB exceeds free {f:.2} GB on GPU {g}; missing {:.2} GB",
                missing_gb(budget, f)
            ),
            &found,
        ));
    }
    if !found.is_empty() && !(req.params.force || req.params.yes) {
        return Err((3, foreign::notice(&found, &free, budget), json!({ "foreign": found })));
    }
    if coordinator::busy(&plan.gpus) {
        return Ok(Step::Wait);
    }
    if let Some(slot) = req.params.slot_key.as_deref() {
        if !autostart::desired_exists(con, slot) {
            return Err(err1(format!("slot {slot} is no longer desired; not starting it")));
        }
    }
    let mut notices = format!("{}\n", plan.message);
    for &g in &plan.gpus {
        if let Some(line) = co_residency_notice(con, g, &req.params.model_id, free[&g]) {
            notices.push_str(&line);
            notices.push('\n');
        }
    }
    let min_free = free.values().copied().fold(f64::INFINITY, f64::min);
    let starting = start_engine(req, con, &plan, budget, min_free, gpu_name)?;
    Ok(Step::Started(Box::new(starting), notices))
}

fn missing_error(message: String, holders: &[Foreign]) -> LoadErr {
    let mut lines = vec![message];
    lines.extend(foreign::holder_lines(holders));
    (1, lines.join("\n"), Value::Null)
}

fn start_engine(
    req: &LoadReq,
    con: &Connection,
    plan: &LoadPlan,
    budget: f64,
    min_free: f64,
    gpu_name: String,
) -> Result<Starting, LoadErr> {
    // Port and slot, spawn, then the STARTING row and ledger entry; any failure after spawn stops the engine.
    let (lo, hi) = crate::ports::engine_port_range().map_err(err1)?;
    let port = next_port(con, lo, hi).map_err(err1)?;
    let model_id = &req.params.model_id;
    let ordinal = match req.params.slot_key.as_deref() {
        Some(sk) => crate::slots::parse_slot_key(sk).map(|(_, _, o)| o).unwrap_or(0),
        None => crate::slots::next_ordinal(con, model_id, &plan.gpus).map_err(err1)?,
    };
    let slot_key = req
        .params
        .slot_key
        .clone()
        .unwrap_or_else(|| crate::slots::slot_key(model_id, &plan.gpus, ordinal));
    let instance_id = format!("inst-{}", uuid_simple());
    let engine_key = engine_cmd::new_engine_key().map_err(err1)?;
    let mem_frac = mem_fraction_from_free(budget, min_free).map_err(err1)?;
    let log_path = req.home.join("logs").join(format!("{instance_id}.log"));
    let tp = plan.tp as u64;
    let spec = EngineSpec {
        home: &req.home,
        instance_id: &instance_id,
        engine_key: &engine_key,
        model_path: &req.model_path,
        card: &req.card,
        port,
        ctx: plan.ctx,
        max_conc: plan.max_conc,
        tp,
        mem_frac,
        gpus: &plan.gpus,
        load_extra: &req.params.extra_sglang_args,
    };
    let (identity, log_offset) = spawn_engine(&spec, &log_path).map_err(err1)?;
    let mut params = req.params.clone();
    params.slot_key = Some(slot_key.clone());
    let recorded = con
        .execute(
            "DELETE FROM loaded_instances WHERE slot_key = ?1 AND status IN ('FAILED', 'CANCELLED')",
            [&slot_key],
        )
        .and_then(|_| {
            con.execute(
                "INSERT INTO loaded_instances
                (instance_id, model_id, pid, pgid, internal_port, gpu_indices, allocated_vram_gb_per_gpu,
                 mem_fraction_static, ctx_len, max_conc, status, restart_count, slot_key, slot_ordinal,
                 extra_sglang_args, boot_id, start_ticks, engine_key)
                VALUES (?1, ?2, ?3, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'STARTING', ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
                rusqlite::params![
                    instance_id,
                    model_id,
                    identity.pid as i64,
                    port as i64,
                    gpu_csv(&plan.gpus),
                    budget,
                    mem_frac,
                    plan.ctx as i64,
                    plan.max_conc as i64,
                    params.restart_count.unwrap_or(0) as i64,
                    slot_key,
                    ordinal as i64,
                    serde_json::to_string(&params.extra_sglang_args).unwrap_or_default(),
                    identity.boot_id,
                    identity.start_ticks as i64,
                    engine_key,
                ],
            )
        });
    if let Err(e) = recorded {
        proc_ident::terminate(std::slice::from_ref(&identity));
        run_registry::unregister(&req.home, &instance_id);
        return Err(err1(format!("record instance: {e}; engine stopped")));
    }
    coordinator::reserve(&instance_id, &plan.gpus, budget);
    if let Err(e) = crate::logs::append_spawn_index(&req.home, &instance_id, model_id, &log_path) {
        eprintln!("append_spawn_index({instance_id}): {e}");
    }
    Ok(Starting {
        home: req.home.clone(),
        cancel: load_slot::begin_slot_load(&slot_key),
        params,
        instance_id,
        engine_key,
        identity,
        port,
        log_path,
        log_offset,
        budget_gb: budget,
        gpu_name,
        tp,
        model_path: req.model_path.clone(),
        card: req.card.clone(),
        gpus: plan.gpus.clone(),
        ctx: plan.ctx,
        max_conc: plan.max_conc,
    })
}

fn spawn_engine(spec: &EngineSpec, log_path: &Path) -> Result<(Identity, u64), String> {
    // Start the engine in its own session, capture its identity and register its run entry before anything else.
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| e.to_string())?;
    let offset = log.metadata().map(|m| m.len()).unwrap_or(0);
    let mut cmd = engine_cmd::command(spec);
    cmd.stdin(Stdio::null())
        .stdout(log.try_clone().map_err(|e| e.to_string())?)
        .stderr(log);
    unsafe {
        cmd.pre_exec(|| {
            libc::setsid();
            Ok(())
        });
    }
    let mut child = cmd.spawn().map_err(|e| format!("spawn engine: {e}"))?;
    let Some(identity) = proc_ident::capture(spec.instance_id, child.id() as i32) else {
        let _ = child.kill();
        let _ = child.wait();
        return Err("cannot read the engine identity from /proc; engine stopped".to_string());
    };
    if let Err(e) = run_registry::register(spec.home, &identity) {
        proc_ident::terminate(std::slice::from_ref(&identity));
        return Err(format!("run registry: {e}; engine stopped"));
    }
    drop(child);
    Ok((identity, offset))
}

fn launch_finish(starting: Starting, notices: String) -> (Value, String) {
    // Hand the STARTING engine to a slot-serialized thread that waits for health.
    let result = json!({
        "instance_id": starting.instance_id,
        "port": starting.port,
        "status": "STARTING",
        "slot_key": starting.params.slot_key,
    });
    let sk = starting.params.slot_key.clone().unwrap_or_default();
    load_slot::slot_in_flight_inc(&sk);
    thread::spawn(move || {
        let slot = slot_lock(&sk);
        let _guard = slot.lock().unwrap_or_else(|e| e.into_inner());
        finish_load_health(starting);
        load_slot::slot_in_flight_dec(&sk);
    });
    (result, notices)
}

fn finish_load_health(mut s: Starting) {
    // Drive one STARTING engine to RUNNING, CANCELLED or FAILED, then free its ledger reservation.
    startup(&mut s);
    coordinator::release(&s.instance_id);
}

fn startup(s: &mut Starting) {
    // Wait for health and retry twice with a larger budget on a VRAM shortfall; every exit leaves a final status.
    let mut mem_retries = 0;
    loop {
        match wait_engine_ready(s) {
            Startup::Ready => return mark_running(s),
            Startup::Cancelled => return cancel_engine(s),
            Startup::Failed => {}
        }
        if load_slot::is_cancelled(&s.cancel) {
            return cancel_engine(s);
        }
        let log = crate::engine_overhead::read_log_since(&s.log_path, s.log_offset);
        if let Some(extra) = crate::engine_overhead::parse_shortfall_gb(&log) {
            if let Ok(con) = db::open(&s.home) {
                let model = &s.params.model_id;
                let _ = crate::engine_overhead::store_learned_extra_gb(&con, model, &s.gpu_name, s.tp, extra);
            }
            if mem_retries < 2 {
                match retry_with_budget(s, extra) {
                    Ok(()) => {
                        mem_retries += 1;
                        continue;
                    }
                    Err(notice) => load_slot::push_instance_notice(&s.instance_id, &notice),
                }
            }
        }
        return fail_engine(s);
    }
}

fn wait_engine_ready(s: &Starting) -> Startup {
    // Poll health every 500 ms; a gone leader, a stalled log or the hard cap fail the start.
    let hard_cap = crate::startup_stall::hard_cap_secs_from_env();
    let mut tracker = crate::startup_stall::StallTracker::new(now_epoch_secs(), log_file_len(&s.log_path));
    loop {
        if load_slot::is_cancelled(&s.cancel) {
            return Startup::Cancelled;
        }
        let alive = proc_ident::leader_alive(&s.identity);
        let step = crate::startup_stall::poll_startup_step(
            &mut tracker,
            now_epoch_secs(),
            alive,
            log_file_len(&s.log_path),
            alive && health_ok(s.port, &s.engine_key),
            hard_cap,
        );
        match step {
            crate::startup_stall::StartupStep::Ready => return Startup::Ready,
            crate::startup_stall::StartupStep::Failed(_) => return Startup::Failed,
            crate::startup_stall::StartupStep::Continue => thread::sleep(Duration::from_millis(500)),
        }
    }
}

fn mark_running(s: &Starting) {
    // STARTING to RUNNING and the desired slot in one transaction, so a concurrent unload can never resurrect it.
    let Ok(mut con) = db::open(&s.home) else {
        return;
    };
    let Ok(tx) = con.transaction_with_behavior(TransactionBehavior::Immediate) else {
        return;
    };
    let changed = tx
        .execute(
            "UPDATE loaded_instances SET status = 'RUNNING' WHERE instance_id = ?1 AND status = 'STARTING'",
            [&s.instance_id],
        )
        .unwrap_or(0);
    if changed == 1 {
        let _ = autostart::mark_desired(&tx, &s.params);
    }
    let _ = tx.commit();
}

fn cancel_engine(s: &Starting) {
    // A cancelled start nobody else is stopping: stop it (verified) and leave a CANCELLED tombstone.
    let Ok(con) = db::open(&s.home) else {
        return;
    };
    let status: Option<String> = con
        .query_row(
            "SELECT status FROM loaded_instances WHERE instance_id = ?1",
            [&s.instance_id],
            |r| r.get(0),
        )
        .ok();
    if status.as_deref() != Some("STARTING") {
        return;
    }
    proc_ident::terminate(std::slice::from_ref(&s.identity));
    run_registry::unregister(&s.home, &s.instance_id);
    let _ = con.execute(
        "UPDATE loaded_instances SET status = 'CANCELLED', pid = 0, pgid = 0
         WHERE instance_id = ?1 AND status = 'STARTING'",
        [&s.instance_id],
    );
    load_slot::clear_instance_notices(&s.instance_id);
}

fn fail_engine(s: &Starting) {
    // A start that cannot succeed: stop the engine (verified) and leave a FAILED tombstone for ps.
    proc_ident::terminate(std::slice::from_ref(&s.identity));
    run_registry::unregister(&s.home, &s.instance_id);
    if let Ok(con) = db::open(&s.home) {
        let _ = con.execute(
            "UPDATE loaded_instances SET status = 'FAILED', pid = 0, pgid = 0
             WHERE instance_id = ?1 AND status = 'STARTING'",
            [&s.instance_id],
        );
        let _ = con.execute("DELETE FROM watchdog_probe_state WHERE instance_id = ?1", [&s.instance_id]);
    }
}

fn retry_with_budget(s: &mut Starting, extra: f64) -> Result<(), String> {
    // Stop the short engine, re-check live VRAM for the larger budget and respawn under the same instance id.
    let budget = s.budget_gb + extra;
    let pids = proc_ident::terminate(std::slice::from_ref(&s.identity));
    foreign::wait_released(&s.gpus, &pids, GPU_RELEASE_WAIT);
    let snap = foreign::snapshot()?;
    let free = s
        .gpus
        .iter()
        .map(|g| snap.iter().find(|x| x.index == *g).map(|x| x.free_mb as f64 / 1024.0).unwrap_or(0.0))
        .fold(f64::INFINITY, f64::min);
    let missing = missing_gb(budget, free);
    if missing > 0.0 {
        return Err(format!(
            "VRAM budget {budget:.2} GB exceeds free GPU headroom ({free:.2} GB free); missing {missing:.2} GB"
        ));
    }
    load_slot::push_instance_notice(
        &s.instance_id,
        &format!("engine needed {extra:.1} GB more than estimated, retried"),
    );
    let spec = EngineSpec {
        home: &s.home,
        instance_id: &s.instance_id,
        engine_key: &s.engine_key,
        model_path: &s.model_path,
        card: &s.card,
        port: s.port,
        ctx: s.ctx,
        max_conc: s.max_conc,
        tp: s.tp,
        mem_frac: budget / free,
        gpus: &s.gpus,
        load_extra: &s.params.extra_sglang_args,
    };
    let (identity, offset) = spawn_engine(&spec, &s.log_path)?;
    let changed = db::open(&s.home)
        .and_then(|con| {
            con.execute(
                "UPDATE loaded_instances SET pid = ?1, pgid = ?1, boot_id = ?2, start_ticks = ?3,
                 allocated_vram_gb_per_gpu = ?4, mem_fraction_static = ?5
                 WHERE instance_id = ?6 AND status = 'STARTING'",
                rusqlite::params![
                    identity.pid as i64,
                    identity.boot_id,
                    identity.start_ticks as i64,
                    budget,
                    budget / free,
                    s.instance_id,
                ],
            )
        })
        .unwrap_or(0);
    if changed != 1 {
        proc_ident::terminate(std::slice::from_ref(&identity));
        run_registry::unregister(&s.home, &s.instance_id);
        return Err("instance left STARTING during the retry; new engine stopped".to_string());
    }
    s.identity = identity;
    s.log_offset = offset;
    s.budget_gb = budget;
    coordinator::reserve(&s.instance_id, &s.gpus, budget);
    if let Err(e) = crate::logs::append_spawn_index(&s.home, &s.instance_id, &s.params.model_id, &s.log_path) {
        eprintln!("append_spawn_index({}): {e}", s.instance_id);
    }
    Ok(())
}

pub fn slot_has_running_instance(slot_or_model: &str) -> bool {
    let home = goethe_home();
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

fn append_goethed_log(home: &Path, line: &str) {
    let path = home.join("logs/goethed.log");
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
}

fn stop_processes(home: &Path, list: &[Inst]) {
    // Stop instances (verified, all at once), wait for their pids to leave NVML, then drop their run entries.
    let ids: Vec<Identity> = list.iter().filter_map(|i| i.identity.clone()).collect();
    let pids = proc_ident::terminate(&ids);
    let gpus: Vec<u32> = list
        .iter()
        .flat_map(|i| i.gpus.iter().copied())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    if !foreign::wait_released(&gpus, &pids, GPU_RELEASE_WAIT) {
        append_goethed_log(home, &format!("teardown: pids {pids:?} still listed on GPUs {gpus:?} after 2s\n"));
    }
    for i in list {
        run_registry::unregister(home, &i.instance_id);
    }
}

fn teardown(con: &Connection, home: &Path, list: &[Inst]) {
    // Verified stop of instances, then their rows, probe state and notices go.
    stop_processes(home, list);
    for i in list {
        let _ = con.execute("DELETE FROM loaded_instances WHERE instance_id = ?1", [&i.instance_id]);
        let _ = con.execute("DELETE FROM watchdog_probe_state WHERE instance_id = ?1", [&i.instance_id]);
        load_slot::clear_instance_notices(&i.instance_id);
    }
}

pub fn graceful_shutdown(home: &Path) -> Result<(), String> {
    // Cancel pending loads and stop every live engine (verified); desired slots stay for the next start.
    let t0 = Instant::now();
    coordinator::cancel_queued(None);
    let con = db::open(home).map_err(|e| e.to_string())?;
    let list = {
        let _plan = coordinator::plan_lock();
        let found = insts(&con, &format!("status IN {LIVE}"), &[]);
        let _ = con.execute(
            &format!("UPDATE loaded_instances SET status = 'STOPPING' WHERE status IN {LIVE}"),
            [],
        );
        found
    };
    load_slot::request_cancel_all();
    let _ = load_slot::wait_all_slots_quiet(Duration::from_millis(500));
    teardown(&con, home, &list);
    append_goethed_log(
        home,
        &format!("shutdown done ms={} instances={}\n", t0.elapsed().as_millis(), list.len()),
    );
    Ok(())
}

pub fn model_draining(model_id: &str) -> bool {
    let home = goethe_home();
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
    // Wait for in-flight mux requests of these models to finish, bounded by GOETHE_DRAIN_SECS.
    let max_secs: u64 = std::env::var("GOETHE_DRAIN_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let deadline = Instant::now() + Duration::from_secs(max_secs);
    loop {
        let pending: usize = model_ids.iter().map(|mid| crate::inflight::count(mid)).sum();
        if pending == 0 {
            return;
        }
        if Instant::now() >= deadline {
            append_goethed_log(
                home,
                &format!("unload drain timeout cut {pending} in-flight requests\n"),
            );
            return;
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn claim_for_unload(con: &mut Connection, model: Option<&str>, next: &str) -> rusqlite::Result<Vec<Inst>> {
    // One immediate transaction: drop desired slots and tombstones, collect live rows, mark them DRAINING/STOPPING.
    let tx = con.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let only = "(?1 IS NULL OR model_id = ?1)";
    tx.execute(&format!("DELETE FROM desired_loads WHERE {only}"), [model])?;
    tx.execute(
        &format!("DELETE FROM loaded_instances WHERE status IN ('FAILED', 'CANCELLED') AND {only}"),
        [model],
    )?;
    let list = insts(&tx, &format!("status IN {LIVE} AND {only}"), &[&model]);
    tx.execute(
        &format!(
            "UPDATE loaded_instances SET status = ?2 WHERE status IN ('STARTING', 'RUNNING', 'RESTARTING') AND {only}"
        ),
        rusqlite::params![model, next],
    )?;
    tx.commit()?;
    Ok(list)
}

pub fn handle_unload(
    model_id: Option<String>,
    all: bool,
    force: bool,
) -> Result<Value, (i32, String)> {
    // Clear desired slots first (even with no instances), then drain unless forced and stop every match (verified).
    let model = match (all, model_id.as_deref()) {
        (true, _) => None,
        (false, Some(m)) => Some(m),
        (false, None) => return Err((2, "model_id required".to_string())),
    };
    let home = goethe_home();
    let t0 = Instant::now();
    coordinator::cancel_queued(model);
    let mut con = db::open(&home).map_err(|e| (1, e.to_string()))?;
    let list = {
        let _plan = coordinator::plan_lock();
        claim_for_unload(&mut con, model, if force { "STOPPING" } else { "DRAINING" })
            .map_err(|e| (1, e.to_string()))?
    };
    for i in list.iter().filter(|i| i.status == "STARTING") {
        load_slot::request_cancel_slot(&i.slot_key);
        let _ = load_slot::wait_slot_quiet(&i.slot_key, Duration::from_millis(500));
    }
    if list.is_empty() {
        return Ok(Value::Null);
    }
    if !force {
        let models: Vec<String> = list
            .iter()
            .map(|i| i.model_id.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        drain_inflight(&models, &home);
        for i in &list {
            let _ = con.execute(
                "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1 AND status = 'DRAINING'",
                [&i.instance_id],
            );
        }
    }
    teardown(&con, &home, &list);
    append_goethed_log(
        &home,
        &format!("unload done ms={} instances={}\n", t0.elapsed().as_millis(), list.len()),
    );
    Ok(Value::Null)
}

pub fn handle_ps() -> Result<Value, (i32, String)> {
    // Every instance row (tombstones included) with notices, in-flight count and token rate.
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| (1, e.to_string()))?;
    let mut stmt = con
        .prepare(
            "SELECT instance_id, model_id, pid, pgid, internal_port, gpu_indices, status,
                    allocated_vram_gb_per_gpu, COALESCE(restart_count, 0),
                    COALESCE(slot_key, model_id) FROM loaded_instances ORDER BY rowid",
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
                "gpu_indices": parse_gpu_csv(&r.get::<_, String>(5)?),
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
    // Adopt verified, answering RUNNING engines; clear other rows without signaling; reap verified orphans.
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let _ = con.execute(
        "UPDATE loaded_instances SET pid = 0, pgid = 0 WHERE status IN ('FAILED', 'CANCELLED')",
        [],
    );
    let mut adopted = HashSet::new();
    for i in insts(&con, &format!("status IN {LIVE}"), &[]) {
        let verified = i.identity.as_ref().filter(|id| proc_ident::leader_alive(id));
        let _ = con.execute("DELETE FROM watchdog_probe_state WHERE instance_id = ?1", [&i.instance_id]);
        match verified {
            Some(id) if i.status == "RUNNING" && cheap_probe_ok(i.port, &i.engine_key) => {
                let _ = run_registry::register(&home, id);
                append_goethed_log(&home, &format!("adopted instance {} pid {}\n", i.instance_id, id.pid));
                adopted.insert(i.instance_id);
            }
            _ => {
                let _ = con.execute("DELETE FROM loaded_instances WHERE instance_id = ?1", [&i.instance_id]);
                append_goethed_log(
                    &home,
                    &format!("cleared instance {} ({}) without signaling: not adoptable\n", i.instance_id, i.status),
                );
            }
        }
    }
    run_registry::reap_orphans(&home, &adopted);
    Ok(())
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

fn default_ctx(row: &InstRow) -> u64 {
    row.default_ctx.min(row.max_ctx.unwrap_or(row.default_ctx)).max(1) as u64
}

fn kv_bytes_per_token(row: &InstRow, card: &toml::Value) -> u64 {
    // KV bytes per token for the card's kv_cache_dtype; auto keeps the installed figure (model dtype).
    let elems = row.kv_elems_per_token.unwrap_or(row.kv_bytes_per_token).max(0) as u64;
    match card.get("kv_cache_dtype").and_then(|v| v.as_str()).map(str::to_ascii_lowercase) {
        Some(d) if d.contains("fp8") => elems,
        Some(d) if d.contains("fp4") => elems.div_ceil(2),
        Some(d) if d.contains("16") => elems * 2,
        _ => row.kv_bytes_per_token.max(0) as u64,
    }
}

fn chooser_input(req: &LoadReq, gpus: &[u32], ctx: u64, max_conc: u64) -> ChooserInput {
    // Budget inputs for a GPU set: Marlin extra only on Blackwell-class GPUs, KV bytes from the card dtype.
    let row = &req.row;
    ChooserInput {
        static_vram_gb: static_weight_gb(
            row.static_vram_gb,
            req.card.get("marlin_extra_gb").and_then(|v| v.as_float()),
            marlin_extra_applies(req.hw.as_ref(), gpus, &req.card),
        ),
        kv_bytes_per_token: kv_bytes_per_token(row, &req.card),
        num_attention_heads: row.num_attention_heads.unwrap_or(1) as u64,
        num_kv_heads: row.num_kv_heads.unwrap_or(1) as u64,
        attention: row.attention.clone().unwrap_or_else(|| "gqa".to_string()),
        max_ctx: row.max_ctx.unwrap_or(row.default_ctx) as u64,
        ctx,
        max_conc,
    }
}

fn marlin_extra_applies(hw: Option<&HardwareState>, gpus: &[u32], card: &toml::Value) -> bool {
    // The Marlin weight copy exists only on compute capability 10+ unless GOETHE_MARLIN=0 (env or card).
    let off = std::env::var("GOETHE_MARLIN").as_deref() == Ok("0")
        || card
            .get("env")
            .and_then(|v| v.get("GOETHE_MARLIN"))
            .and_then(|v| v.as_str())
            == Some("0");
    if off {
        return false;
    }
    gpus.iter()
        .filter_map(|g| {
            hw?.gpus
                .iter()
                .find(|x| x.index == *g)?
                .compute_capability
                .split('.')
                .next()?
                .parse::<u32>()
                .ok()
        })
        .max()
        .unwrap_or(0)
        >= 10
}

fn gpu_name(hw: Option<&HardwareState>, idx: u32) -> String {
    hw.and_then(|h| h.gpus.iter().find(|g| g.index == idx).map(|g| g.name.clone()))
        .unwrap_or_else(|| "gpu".to_string())
}

fn pinned_gpus(raw: Option<&str>, card: &toml::Value, tp: Option<u64>) -> Result<Vec<u32>, String> {
    // --gpus (else the card's preferred_gpus) as sorted indices; it must agree with --tp and list no GPU twice.
    let list: Vec<u32> = match raw {
        Some(s) => s
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|p| p.parse::<u32>().map_err(|_| format!("invalid GPU index '{p}' in --gpus")))
            .collect::<Result<_, _>>()?,
        None => card
            .get("preferred_gpus")
            .and_then(|v| v.as_array())
            .into_iter()
            .flatten()
            .filter_map(|v| v.as_integer().and_then(|i| u32::try_from(i).ok()))
            .collect(),
    };
    if let Some(t) = tp {
        if !list.is_empty() && t as usize != list.len() {
            return Err(format!("--tp {t} disagrees with {} GPUs on --gpus", list.len()));
        }
    }
    let mut sorted = list;
    sorted.sort_unstable();
    if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
        return Err(format!("GPU {} is listed twice in --gpus", w[0]));
    }
    Ok(sorted)
}

fn read_card_value(path: &Path) -> Result<toml::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    toml::from_str(&text).map_err(|e| e.to_string())
}

fn parse_gpu_csv(s: &str) -> Vec<u32> {
    s.split(',').filter_map(|p| p.trim().parse().ok()).collect()
}

fn gpu_csv(gpus: &[u32]) -> String {
    placement::format_gpu_list(gpus)
}

fn co_residency_notice(con: &Connection, gpu: u32, model_id: &str, free_gb: f64) -> Option<String> {
    // Name the other live Goethe instances on this GPU and its free VRAM.
    let mut stmt = con
        .prepare(&format!(
            "SELECT model_id, allocated_vram_gb_per_gpu, gpu_indices FROM loaded_instances
             WHERE status IN {LIVE} AND model_id != ?1 ORDER BY rowid"
        ))
        .ok()?;
    let hosts: Vec<String> = stmt
        .query_map([model_id], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?, r.get::<_, String>(2)?))
        })
        .ok()?
        .filter_map(|r| r.ok())
        .filter(|(_, _, csv)| parse_gpu_csv(csv).contains(&gpu))
        .map(|(m, gb, _)| format!("{m}: {gb:.1} GB"))
        .collect();
    if hosts.is_empty() {
        return None;
    }
    Some(format!(
        "GPU {gpu} currently hosts [{}]. Free VRAM: {free_gb:.1} GB.",
        hosts.join(", ")
    ))
}

fn next_port(con: &Connection, lo: u16, hi: u16) -> Result<u16, String> {
    // Lowest port in the engine range that no instance row holds and nothing listens on.
    for port in lo..=hi {
        let held = con
            .query_row(
                "SELECT 1 FROM loaded_instances WHERE internal_port = ?1",
                [port as i64],
                |_| Ok(()),
            )
            .is_ok();
        if !held && crate::ports::localhost_port_free(port) {
            return Ok(port);
        }
    }
    Err(format!("no free engine port in {lo}-{hi}; set GOETHE_ENGINE_PORTS=<lo>-<hi>"))
}

fn http_ok(port: u16, path: &str, key: &str, timeout_secs: u64) -> bool {
    // GET a loopback engine path with the engine key; only 200 counts.
    let agent = ureq::AgentBuilder::new()
        .timeout_read(Duration::from_secs(timeout_secs))
        .timeout_connect(Duration::from_secs(2))
        .build();
    let mut req = agent.get(&format!("http://127.0.0.1:{port}{path}"));
    if !key.is_empty() {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    req.call().is_ok_and(|r| r.status() == 200)
}

fn cheap_probe_ok(port: u16, key: &str) -> bool {
    ["/v1/models", "/get_model_info"]
        .iter()
        .any(|p| http_ok(port, p, key, 2))
}

fn health_ok(port: u16, key: &str) -> bool {
    http_ok(port, "/health", key, HTTP_PROBE_SECS)
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

pub fn watchdog_tick() {
    // Probe RUNNING engines; restart dead ones within budget, else stop them (verified) and mark FAILED.
    let _guard = WATCHDOG_TICK_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    if shutdown_requested() {
        return;
    }
    let home = goethe_home();
    let Ok(con) = db::open(&home) else {
        return;
    };
    let now = now_epoch_secs();
    for i in insts(&con, "status = 'RUNNING'", &[]) {
        let alive = i.identity.as_ref().is_some_and(proc_ident::leader_alive);
        let mut state = load_probe_state(&con, &i.instance_id, now);
        let cheap = alive && cheap_probe_ok(i.port, &i.engine_key);
        let deep = watchdog_liveness::deep_due(&state, now).then(|| health_ok(i.port, &i.engine_key));
        let action = watchdog_liveness::step(
            &mut state,
            ProbeInput {
                now,
                pid_alive: alive,
                cheap_ok: cheap,
                deep_ok: deep,
            },
        );
        save_probe_state(&con, &i.instance_id, &state);
        let WatchdogAction::MarkDead { reason } = action else {
            continue;
        };
        watchdog_log(&home, &i.instance_id, &i.slot_key, reason, cheap, deep);
        if shutdown_requested() {
            return;
        }
        if !watchdog_may_restart(&con, &i.slot_key) {
            fail_running(&con, &home, &i);
            continue;
        }
        let claimed = con
            .execute(
                "UPDATE loaded_instances SET status = 'RESTARTING' WHERE instance_id = ?1 AND status = 'RUNNING'",
                [&i.instance_id],
            )
            .unwrap_or(0);
        if claimed != 1 {
            continue;
        }
        teardown(&con, &home, std::slice::from_ref(&i));
        if let Some(params) = load_params_for_slot(&con, &i.slot_key, &i.model_id) {
            schedule_restart(params, i.restarts);
        }
    }
}

fn fail_running(con: &Connection, home: &Path, i: &Inst) {
    // Restart budget spent: stop the hung engine (verified) so its VRAM is freed, then keep a FAILED tombstone.
    let claimed = con
        .execute(
            "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1 AND status = 'RUNNING'",
            [&i.instance_id],
        )
        .unwrap_or(0);
    if claimed != 1 {
        return;
    }
    stop_processes(home, std::slice::from_ref(i));
    let _ = con.execute(
        "UPDATE loaded_instances SET status = 'FAILED', pid = 0, pgid = 0 WHERE instance_id = ?1 AND status = 'STOPPING'",
        [&i.instance_id],
    );
    let _ = con.execute("DELETE FROM watchdog_probe_state WHERE instance_id = ?1", [&i.instance_id]);
}

fn schedule_restart(mut params: LoadParams, restarts: i64) {
    // After the backoff and under the slot lock, relaunch only if the slot is still desired and goethed is not stopping.
    let slot = params.slot_key.clone().unwrap_or_default();
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(crate::watchdog_policy::restart_backoff_secs(restarts)));
        let lock = slot_lock(&slot);
        let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        if shutdown_requested() {
            return;
        }
        let desired = db::open(&goethe_home())
            .map(|c| autostart::desired_exists(&c, &slot))
            .unwrap_or(false);
        if !desired {
            return;
        }
        params.restart_count = Some(restarts.max(0) as u64 + 1);
        if let Err((code, msg, _)) = handle_load(params) {
            eprintln!("watchdog restart of {slot} failed ({code}): {msg}");
        }
    });
}

fn watchdog_log(
    home: &Path,
    instance_id: &str,
    slot_key: &str,
    reason: &str,
    cheap_ok: bool,
    deep_ok: Option<bool>,
) {
    append_goethed_log(
        home,
        &format!(
            "watchdog instance={instance_id} slot={slot_key} reason={reason} cheap_ok={cheap_ok} deep_ok={deep_ok:?}\n"
        ),
    );
}

fn load_probe_state(con: &Connection, instance_id: &str, now: i64) -> LivenessState {
    // Persisted liveness state of an instance, or a fresh one due for a deep probe.
    con.query_row(
        "SELECT cheap_fail_since, deep_fail_streak, last_deep_at, COALESCE(cheap_ok_once,0)
         FROM watchdog_probe_state WHERE instance_id = ?1",
        [instance_id],
        |r| {
            Ok(LivenessState {
                cheap_fail_since: r.get(0)?,
                deep_fail_streak: r.get::<_, i64>(1)? as u32,
                last_deep_at: r.get(2)?,
                cheap_ok_once: r.get::<_, i64>(3)? != 0,
            })
        },
    )
    .unwrap_or(LivenessState {
        last_deep_at: now - watchdog_liveness::DEEP_INTERVAL_SECS,
        ..Default::default()
    })
}

fn save_probe_state(con: &Connection, instance_id: &str, st: &LivenessState) {
    let _ = con.execute(
        "INSERT OR REPLACE INTO watchdog_probe_state
        (instance_id, cheap_fail_since, deep_fail_streak, last_deep_at, cheap_ok_once)
        VALUES (?1, ?2, ?3, ?4, ?5)",
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
    // Consume one restart from the slot's window, or refuse when the window is spent.
    let row: Option<(i64, i64)> = con
        .query_row(
            "SELECT window_start, count FROM watchdog_restarts WHERE slot_key = ?1",
            [slot_key],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .ok();
    let (start, count) = row.map(|(s, c)| (Some(s), c)).unwrap_or((None, 0));
    match crate::watchdog_policy::decide_restart(now_epoch_secs(), start, count) {
        crate::watchdog_policy::RestartDecision::DenyFailed => false,
        crate::watchdog_policy::RestartDecision::Allow {
            new_window_start,
            new_count,
        } => {
            con.execute(
                "INSERT OR REPLACE INTO watchdog_restarts (slot_key, window_start, count) VALUES (?1, ?2, ?3)",
                rusqlite::params![slot_key, new_window_start, new_count],
            )
            .ok();
            true
        }
    }
}

fn load_params_for_slot(con: &Connection, slot_key: &str, model_id: &str) -> Option<LoadParams> {
    // Load parameters recorded for exactly this desired slot.
    let (gpus, ctx, max_conc, tp, extra): (Option<String>, Option<i64>, Option<i64>, Option<i64>, String) = con
        .query_row(
            "SELECT gpus, ctx, max_conc, tp, extra_sglang_args FROM desired_loads WHERE slot_key = ?1",
            [slot_key],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .ok()?;
    Some(LoadParams {
        model_id: model_id.to_string(),
        gpus,
        ctx: ctx.map(|v| v as u64),
        max_conc: max_conc.map(|v| v as u64),
        tp: tp.map(|v| v as u64),
        extra_sglang_args: serde_json::from_str(&extra).unwrap_or_default(),
        force: true,
        yes: true,
        restart_count: None,
        slot_key: Some(slot_key.to_string()),
    })
}

fn uuid_simple() -> String {
    let n = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{n:x}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::process::{Child, Command};
    use std::sync::{Mutex, OnceLock};

    fn home_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn with_temp_home<F: FnOnce(&Path, &Connection)>(f: F) {
        let _g = home_lock().lock().unwrap_or_else(|e| e.into_inner());
        let td = tempfile::tempdir().expect("tempdir");
        std::env::set_var("GOETHE_HOME", td.path());
        ensure_layout(td.path()).expect("layout");
        let con = db::open(td.path()).expect("db");
        f(td.path(), &con);
        std::env::remove_var("GOETHE_HOME");
    }

    fn insert_running(con: &Connection, id: &str, pid: i32, ticks: u64, port: u16) {
        con.execute(
            "INSERT INTO loaded_instances
            (instance_id, model_id, pid, pgid, internal_port, gpu_indices, allocated_vram_gb_per_gpu,
             mem_fraction_static, ctx_len, max_conc, status, slot_key, boot_id, start_ticks, engine_key)
            VALUES (?1, 'm', ?2, ?2, ?3, '0', 1.0, 0.5, 4096, 4, 'RUNNING', 'm|0|0', ?4, ?5, 'k')",
            rusqlite::params![id, pid, port as i64, proc_ident::boot_id(), ticks as i64],
        )
        .expect("insert");
    }

    fn sleeper() -> (Child, u64) {
        let child = Command::new("sleep").arg("30").env_remove(proc_ident::INSTANCE_ENV).spawn().expect("sleep");
        let ticks = proc_ident::start_ticks(child.id() as i32).expect("ticks");
        (child, ticks)
    }

    fn ok_server() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        thread::spawn(move || {
            for mut s in listener.incoming().flatten() {
                let mut buf = [0u8; 1024];
                let _ = s.read(&mut buf);
                let _ = s.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok");
            }
        });
        port
    }

    #[test]
    fn unload_never_signals_a_stale_pid_and_keeps_restart_budget() {
        with_temp_home(|_home, con| {
            let (mut child, ticks) = sleeper();
            insert_running(con, "i1", child.id() as i32, ticks + 5, 30001);
            con.execute("INSERT INTO desired_loads (slot_key, model_id) VALUES ('m|0|0', 'm')", []).unwrap();
            handle_unload(Some("m".to_string()), false, false).expect("unload");
            assert!(child.try_wait().unwrap().is_none(), "unload must not signal an unverified pid");
            let left: i64 = con.query_row("SELECT COUNT(*) FROM loaded_instances", [], |r| r.get(0)).unwrap();
            let desired: i64 = con.query_row("SELECT COUNT(*) FROM desired_loads", [], |r| r.get(0)).unwrap();
            let restarts: i64 = con
                .query_row("SELECT COUNT(*) FROM watchdog_restarts", [], |r| r.get(0))
                .unwrap();
            assert_eq!((left, desired, restarts), (0, 0, 0));
            child.kill().unwrap();
            child.wait().unwrap();
        });
    }

    #[test]
    fn unload_clears_desired_even_without_instances() {
        with_temp_home(|_home, con| {
            con.execute("INSERT INTO desired_loads (slot_key, model_id) VALUES ('m|0|0', 'm')", []).unwrap();
            handle_unload(Some("m".to_string()), false, true).expect("unload");
            let desired: i64 = con.query_row("SELECT COUNT(*) FROM desired_loads", [], |r| r.get(0)).unwrap();
            assert_eq!(desired, 0);
        });
    }

    #[test]
    fn reconcile_adopts_only_verified_and_healthy_engines() {
        with_temp_home(|home, con| {
            let port = ok_server();
            let (mut good, good_ticks) = sleeper();
            let (mut stale, stale_ticks) = sleeper();
            insert_running(con, "i-good", good.id() as i32, good_ticks, port);
            insert_running(con, "i-stale", stale.id() as i32, stale_ticks + 9, port + 1);
            con.execute("UPDATE loaded_instances SET slot_key = 'm|0|1' WHERE instance_id = 'i-stale'", []).unwrap();
            let stale_id = Identity {
                instance_id: "i-orphan".to_string(),
                pid: stale.id() as i32,
                boot_id: proc_ident::boot_id().to_string(),
                start_ticks: stale_ticks + 9,
            };
            run_registry::register(home, &stale_id).unwrap();
            reconcile_on_start().expect("reconcile");
            let rows: Vec<(String, String)> = con
                .prepare("SELECT instance_id, status FROM loaded_instances")
                .unwrap()
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect();
            assert_eq!(rows, vec![("i-good".to_string(), "RUNNING".to_string())]);
            assert!(good.try_wait().unwrap().is_none() && stale.try_wait().unwrap().is_none());
            assert!(run_registry::list_entries(home).iter().all(|e| e.instance_id == "i-good"));
            for c in [&mut good, &mut stale] {
                c.kill().unwrap();
                c.wait().unwrap();
            }
        });
    }

    #[test]
    fn pinned_gpus_validates_and_sorts() {
        let card: toml::Value = toml::from_str("preferred_gpus = [3, 2]").unwrap();
        assert_eq!(pinned_gpus(None, &card, None).unwrap(), vec![2, 3]);
        assert_eq!(pinned_gpus(Some("3,2"), &card, Some(2)).unwrap(), vec![2, 3]);
        assert!(pinned_gpus(Some("2,2"), &card, Some(1)).unwrap_err().contains("disagrees"));
        assert!(pinned_gpus(Some("2,2"), &card, None).unwrap_err().contains("twice"));
        assert!(pinned_gpus(Some("x"), &card, None).unwrap_err().contains("invalid"));
    }

    #[test]
    fn default_ctx_uses_configured_ctx_capped_by_max() {
        let row = |default_ctx, max_ctx| InstRow {
            local_path: String::new(),
            static_vram_gb: 1.0,
            kv_bytes_per_token: 100,
            default_ctx,
            default_max_conc: 4,
            default_tp: 1,
            kv_elems_per_token: Some(50),
            max_ctx,
            num_attention_heads: None,
            num_kv_heads: None,
            attention: None,
        };
        assert_eq!(default_ctx(&row(65536, Some(262144))), 65536);
        assert_eq!(default_ctx(&row(65536, Some(32768))), 32768);
        assert_eq!(default_ctx(&row(4096, None)), 4096);
        let fp8: toml::Value = toml::from_str("kv_cache_dtype = \"fp8_e4m3\"").unwrap();
        let auto: toml::Value = toml::from_str("kv_cache_dtype = \"auto\"").unwrap();
        assert_eq!(kv_bytes_per_token(&row(1, None), &fp8), 50);
        assert_eq!(kv_bytes_per_token(&row(1, None), &auto), 100);
    }
}
