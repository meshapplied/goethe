// goethed load, unload, adoption and watchdog supervision of SGLang engines.

use crate::autostart;
use crate::calc::{self, mem_fraction_from_free, shortfall, static_weight_gb, Need};
use crate::chooser::{self, ChooserInput, LoadPlan};
use crate::coordinator;
use crate::db;
use crate::engine_cmd::{self, EngineSpec};
use crate::engine_overhead::{Kind, Learned};
use crate::foreign::{self, Foreign, GpuLive};
use crate::hw::HardwareState;
use crate::listener::{self, PortOwner};
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
use std::sync::{Arc, Mutex, OnceLock, TryLockError};
use std::thread;
use std::time::{Duration, Instant};

static SLOT_LOCKS: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
static WATCHDOG_TICK_LOCK: Mutex<()> = Mutex::new(());
static NEXT_ATTEMPT: OnceLock<Mutex<HashMap<String, i64>>> = OnceLock::new();

const LIVE: &str = "('STARTING', 'RUNNING', 'RESTARTING', 'DRAINING', 'STOPPING')";
const GPU_RELEASE_WAIT: Duration = Duration::from_secs(2);
const HTTP_PROBE_SECS: u64 = 25;
const USER_GATE_WAIT: Duration = Duration::from_secs(590);

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
    #[serde(default)]
    pub replica: bool,
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
    need: Need,
    learned: Learned,
    gpu_name: String,
    tp: u64,
    model_path: PathBuf,
    card: toml::Value,
    gpus: Vec<u32>,
    ctx: u64,
    max_conc: u64,
    cancel: Arc<AtomicBool>,
    hold: coordinator::Hold,
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
    Loaded(Value, String),
    Wait(Vec<u32>),
}

enum Startup {
    Ready,
    Cancelled,
    Failed,
    PortTaken(String),
}

fn insts(con: &Connection, filter: &str, args: &[&dyn ToSql]) -> rusqlite::Result<Vec<Inst>> {
    // Instance rows matching a WHERE clause, with identity, GPUs and engine key parsed; any bad row is an error.
    let sql = format!(
        "SELECT instance_id, model_id, COALESCE(slot_key, model_id), status, pid, boot_id, start_ticks,
                gpu_indices, internal_port, COALESCE(engine_key, ''), COALESCE(restart_count, 0)
         FROM loaded_instances WHERE {filter} ORDER BY rowid"
    );
    let mut stmt = con.prepare(&sql)?;
    let rows = stmt.query_map(args, |r| {
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
    })?;
    rows.collect()
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
    let deadline = Instant::now() + USER_GATE_WAIT;
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
            Step::Loaded(result, message) => return Ok((result, message)),
            Step::Wait(gpus) if req.params.slot_key.is_none() && Instant::now() >= deadline => {
                return Err(err1(format!(
                    "GPU {} is still starting another engine after {}s; load not started, retry when it is RUNNING",
                    gpu_csv(&gpus),
                    USER_GATE_WAIT.as_secs()
                )));
            }
            Step::Wait(_) => coordinator::wait_change(Duration::from_secs(1)),
        }
    }
}

fn plan_and_start(req: &LoadReq, con: &Connection) -> Result<Step, LoadErr> {
    // Under the planning lock: slot and replica checks, plan on live VRAM, then fit, notice and gate.
    let model_id = &req.params.model_id;
    if let Some(slot) = req.params.slot_key.as_deref() {
        let live = format!("SELECT 1 FROM loaded_instances WHERE slot_key = ?1 AND status IN {LIVE} AND pid > 0");
        if con.query_row(&live, [slot], |_| Ok(())).is_ok() {
            return Err(err1(format!("slot {slot} already has a live instance; not starting a second one")));
        }
        if !autostart::desired_exists(con, slot) {
            return Err(err1(format!("slot {slot} is no longer desired; not starting it")));
        }
    }
    let copies = copies_of(con, model_id);
    if let (None, false, Some(first)) = (&req.params.slot_key, req.params.replica, copies.first()) {
        let message = format!(
            "already loaded: {} on GPU {} (port {}); use --replica to start another copy",
            first.0, first.2, first.3
        );
        let result = json!({"instance_id": first.0, "status": first.1, "already_loaded": true});
        return Ok(Step::Loaded(result, message));
    }
    let snap = foreign::snapshot().map_err(err1)?;
    if let Some(g) = req.pinned.iter().find(|g| !snap.iter().any(|s| s.index == **g)) {
        return Err(err1(format!(
            "GPU {g} does not exist; this host has {} GPUs",
            snap.len()
        )));
    }
    let managed = placement::managed_pids(con);
    let live = placement::live_instance_ids(con);
    let netted = placement::placement_inputs(&snap, &managed, &live, &req.denied, coordinator::reserved_gb);
    let raw = placement::placement_inputs(&snap, &managed, &live, &req.denied, |_| 0.0);
    if let Some(tp) = req.pinned_tp {
        placement::ensure_eligible_for_tp(&netted, tp, &req.denied).map_err(err1)?;
    }
    let candidates: Vec<u32> = match req.pinned.is_empty() {
        true => netted.iter().map(|g| g.index).collect(),
        false => req.pinned.clone(),
    };
    let base = chooser_input(req, &snap, &candidates, req.ctx, req.max_conc);
    let pinned = (!req.pinned.is_empty()).then_some(req.pinned.as_slice());
    let tp = req.pinned_tp.or(req.default_tp);
    let first_name = gpu_name(&snap, candidates.first().copied().unwrap_or(0));
    let learned_at = |t: usize| crate::engine_overhead::read_learned(con, model_id, &first_name, t as u64);
    let plan = match chooser::plan_with(&base, &netted, req.hw.as_ref(), tp, pinned, &learned_at) {
        Ok(plan) => plan,
        Err(e) => chooser::plan_with(&base, &raw, req.hw.as_ref(), tp, pinned, &learned_at)
            .map_err(|_| missing_error(e.message, &foreign::foreign_on(&snap, &e.gpus, &managed, &live)))?,
    };
    let busy = coordinator::busy(&plan.gpus);
    let free: HashMap<u32, f64> = plan
        .gpus
        .iter()
        .map(|g| {
            let mb = netted.iter().find(|i| i.index == *g).map(|i| i.free_vram_mb).unwrap_or(0);
            (*g, mb as f64 / 1024.0)
        })
        .collect();
    let gpu_name = gpu_name(&snap, plan.gpus[0]);
    let input = chooser_input(req, &snap, &plan.gpus, plan.ctx, plan.max_conc);
    let learned = crate::engine_overhead::read_learned(con, model_id, &gpu_name, plan.tp as u64);
    let need = chooser::with_learned(chooser::need(&input, plan.tp), learned);
    let found = foreign::foreign_on(&snap, &plan.gpus, &managed, &live);
    let short = plan.gpus.iter().find_map(|g| shortfall(&need, *g, free[g]));
    if let (Some(why), false) = (short, busy) {
        return Err(missing_error(why, &found));
    }
    if !found.is_empty() && !(req.params.force || req.params.yes) {
        return Err((3, foreign::notice(&found, &free, need.total()), json!({ "foreign": found })));
    }
    if busy {
        return Ok(Step::Wait(plan.gpus));
    }
    let auto = pinned.is_none() && tp.is_none();
    let mut notices = format!(
        "{}tp={} on GPUs {}: {}\n",
        if auto { "auto: " } else { "" },
        plan.tp,
        gpu_csv(&plan.gpus),
        calc::describe(&need)
    );
    if !copies.is_empty() {
        let list: Vec<String> = copies.iter().map(|c| format!("{} on GPU {} (port {})", c.0, c.2, c.3)).collect();
        notices.push_str(&format!("another replica of {model_id}; already loaded: {}\n", list.join(", ")));
    }
    for &g in &plan.gpus {
        if let Some(line) = co_residency_notice(con, g, model_id, free[&g]) {
            notices.push_str(&line);
            notices.push('\n');
        }
    }
    let min_free = free.values().copied().fold(f64::INFINITY, f64::min);
    let starting = start_engine(req, con, &plan, need, learned, min_free, gpu_name)?;
    Ok(Step::Started(Box::new(starting), notices))
}

fn copies_of(con: &Connection, model_id: &str) -> Vec<(String, String, String, i64)> {
    // STARTING or RUNNING instances of a model (RUNNING first): id, status, GPUs, port.
    let Ok(mut stmt) = con.prepare(
        "SELECT instance_id, status, gpu_indices, internal_port FROM loaded_instances
         WHERE model_id = ?1 AND status IN ('STARTING', 'RUNNING') ORDER BY status = 'RUNNING' DESC, rowid",
    ) else {
        return Vec::new();
    };
    stmt.query_map([model_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
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
    need: Need,
    learned: Learned,
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
    let budget = need.total();
    let mem_frac = mem_fraction_from_free(&need, min_free).map_err(err1)?;
    let log_path = req.home.join("logs").join(format!("{instance_id}.log"));
    let tp = plan.tp as u64;
    let spec = EngineSpec {
        home: &req.home,
        instance_id: &instance_id,
        served_name: model_id,
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
            "DELETE FROM loaded_instances
             WHERE slot_key = ?1 AND (status IN ('FAILED', 'CANCELLED') OR (status = 'RESTARTING' AND pid = 0))",
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
        stop_one(&req.home, &identity);
        return Err(err1(format!("record instance: {e}; engine stopped")));
    }
    let hold = coordinator::reserve(&instance_id, &plan.gpus, budget);
    if let Err(e) = crate::logs::append_spawn_index(&req.home, &instance_id, model_id, &log_path) {
        eprintln!("append_spawn_index({instance_id}): {e}");
    }
    Ok(Starting {
        home: req.home.clone(),
        cancel: load_slot::begin_slot_load(&slot_key),
        hold,
        params,
        instance_id,
        engine_key,
        identity,
        port,
        log_path,
        log_offset,
        need,
        learned,
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
        let mut starting = starting;
        startup(&mut starting);
        drop(starting);
        load_slot::slot_in_flight_dec(&sk);
    });
    (result, notices)
}

fn startup(s: &mut Starting) {
    // STARTING to RUNNING, CANCELLED or FAILED: two retries after a VRAM shortfall, three on a new port if ours is held.
    let (mut mem_retries, mut port_retries) = (0, 0);
    loop {
        match wait_engine_ready(s) {
            Startup::Ready => {
                if mark_running(s) {
                    learn_overhead(s);
                }
                return;
            }
            Startup::Cancelled => return cancel_engine(s),
            Startup::PortTaken(why) if port_retries < 3 => {
                port_retries += 1;
                let (outcome, retried) = match respawn(s, s.need, true) {
                    Ok(()) => (format!("{why}; retried on port {}", s.port), true),
                    Err(_) if load_slot::is_cancelled(&s.cancel) => return cancel_engine(s),
                    Err(e) => (format!("{why}; {e}"), false),
                };
                append_goethed_log(&s.home, &format!("{}: {outcome}\n", s.instance_id));
                load_slot::push_instance_notice(&s.instance_id, &outcome);
                if retried {
                    continue;
                }
                return fail_engine(s);
            }
            Startup::PortTaken(why) => {
                append_goethed_log(&s.home, &format!("{}: {why}; giving up after 3 ports\n", s.instance_id));
                load_slot::push_instance_notice(&s.instance_id, &why);
                return fail_engine(s);
            }
            Startup::Failed => {}
        }
        if load_slot::is_cancelled(&s.cancel) {
            return cancel_engine(s);
        }
        let log = crate::engine_overhead::read_log_since(&s.log_path, s.log_offset);
        if let Some(extra) = crate::engine_overhead::parse_shortfall_gb(&log) {
            s.learned.static_gb += extra;
            if let Ok(con) = db::open(&s.home) {
                let (model, learned) = (&s.params.model_id, s.learned.static_gb);
                let _ = crate::engine_overhead::store_learned(&con, model, &s.gpu_name, s.tp, Kind::Static, learned);
            }
            if mem_retries < 2 {
                let bigger = Need { weights: s.need.weights + extra, ..s.need };
                match respawn(s, bigger, false) {
                    Ok(()) => {
                        mem_retries += 1;
                        let notice = format!("engine needed {extra:.1} GB more than estimated, retried");
                        load_slot::push_instance_notice(&s.instance_id, &notice);
                        continue;
                    }
                    Err(_) if load_slot::is_cancelled(&s.cancel) => return cancel_engine(s),
                    Err(notice) => load_slot::push_instance_notice(&s.instance_id, &notice),
                }
            }
        }
        return fail_engine(s);
    }
}

fn wait_engine_ready(s: &Starting) -> Startup {
    // Ready needs every listener on the port to be ours, /health to answer and /v1/models to list our model.
    let hard_cap = crate::startup_stall::hard_cap_secs_from_env();
    let mut tracker = crate::startup_stall::StallTracker::new(now_epoch_secs(), log_file_len(&s.log_path));
    loop {
        if load_slot::is_cancelled(&s.cancel) {
            return Startup::Cancelled;
        }
        let alive = proc_ident::leader_alive(&s.identity);
        let ready = match listener::port_owner(&s.identity, s.port) {
            owner @ PortOwner::Foreign { .. } => return Startup::PortTaken(owner.describe(s.port)),
            PortOwner::Ours => alive && health_ok(s.port, &s.engine_key) && serves(s.port, &s.engine_key, &s.params.model_id),
            PortOwner::Nobody => false,
        };
        let step = crate::startup_stall::poll_startup_step(
            &mut tracker,
            now_epoch_secs(),
            alive,
            log_file_len(&s.log_path),
            ready,
            hard_cap,
        );
        match step {
            crate::startup_stall::StartupStep::Ready => return Startup::Ready,
            crate::startup_stall::StartupStep::Failed(_) => return Startup::Failed,
            crate::startup_stall::StartupStep::Continue => thread::sleep(Duration::from_millis(500)),
        }
    }
}

fn learn_overhead(s: &Starting) {
    // Measure what the engine really holds per GPU (NVML, our pids, max over GPUs); beyond the budget, learn it.
    let pids = proc_ident::instance_pids(&s.identity);
    let Ok(snap) = foreign::snapshot() else {
        return;
    };
    let used_mb = s
        .gpus
        .iter()
        .filter_map(|g| snap.iter().find(|x| x.index == *g))
        .map(|g| g.procs.iter().filter(|(p, _)| pids.contains(p)).map(|(_, mb)| *mb).sum::<u64>())
        .max()
        .unwrap_or(0);
    let used = used_mb as f64 / 1024.0;
    let Some(overhead) = calc::learned_overhead(used, &s.need, s.learned.overhead_gb) else {
        return;
    };
    if let Ok(con) = db::open(&s.home) {
        let model = &s.params.model_id;
        let _ = crate::engine_overhead::store_learned(&con, model, &s.gpu_name, s.tp, Kind::Overhead, overhead);
    }
    append_goethed_log(
        &s.home,
        &format!(
            "{}: engine uses {used:.1} GB per GPU, budget was {:.1} GB; overhead learned\n",
            s.instance_id,
            s.need.total()
        ),
    );
}

fn mark_running(s: &Starting) -> bool {
    // STARTING to RUNNING and the desired slot in one transaction, so a concurrent unload can never resurrect it.
    let Ok(mut con) = db::open(&s.home) else {
        return false;
    };
    let Ok(tx) = con.transaction_with_behavior(TransactionBehavior::Immediate) else {
        return false;
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
    tx.commit().is_ok() && changed == 1
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
    settle(&con, s, "CANCELLED");
}

fn fail_engine(s: &Starting) {
    // A start that cannot succeed: stop the engine (verified) and leave a FAILED tombstone for ps.
    if let Ok(con) = db::open(&s.home) {
        settle(&con, s, "FAILED");
    }
}

fn settle(con: &Connection, s: &Starting, status: &str) {
    // Stop a STARTING engine and tombstone its row as status with pid 0, or keep it STOPPING if it survived.
    let _ = if stop_one(&s.home, &s.identity) {
        con.execute(
            "UPDATE loaded_instances SET status = ?2, pid = 0, pgid = 0 WHERE instance_id = ?1 AND status = 'STARTING'",
            [s.instance_id.as_str(), status],
        )
    } else {
        con.execute(
            "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1 AND status = 'STARTING'",
            [&s.instance_id],
        )
    };
    let _ = con.execute("DELETE FROM watchdog_probe_state WHERE instance_id = ?1", [&s.instance_id]);
    load_slot::clear_instance_notices(&s.instance_id);
}

fn stop_one(home: &Path, id: &Identity) -> bool {
    // Stop one engine (verified); its run entry goes only if nothing survived, so a survivor is reaped later.
    let gone = proc_ident::terminate(std::slice::from_ref(id)).survivors.is_empty();
    if gone {
        run_registry::unregister(home, &id.instance_id);
    } else {
        append_goethed_log(home, &format!("{} survived SIGKILL; run entry kept\n", id.instance_id));
    }
    gone
}

fn respawn(s: &mut Starting, need: Need, move_port: bool) -> Result<(), String> {
    // Stop the current engine (verified) and start it again under the same instance id, with a new budget or port.
    let stopped = proc_ident::terminate(std::slice::from_ref(&s.identity));
    foreign::wait_released(&s.gpus, &stopped.pids, GPU_RELEASE_WAIT);
    if !stopped.survivors.is_empty() {
        return Err("the previous engine survived SIGKILL; not respawning over it".to_string());
    }
    let snap = foreign::snapshot()?;
    let free_of = |g: u32| snap.iter().find(|x| x.index == g).map_or(0.0, |x| x.free_mb as f64 / 1024.0);
    let (tight, free) = s
        .gpus
        .iter()
        .map(|g| (*g, free_of(*g)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .ok_or("no GPU")?;
    if let Some(why) = shortfall(&need, tight, free) {
        return Err(why);
    }
    if load_slot::is_cancelled(&s.cancel) {
        return Err("load cancelled during the retry".to_string());
    }
    let con = db::open(&s.home).map_err(|e| e.to_string())?;
    if move_port {
        let _plan = coordinator::plan_lock();
        let (lo, hi) = crate::ports::engine_port_range()?;
        let port = next_port(&con, lo, hi)?;
        let moved = con
            .execute(
                "UPDATE loaded_instances SET internal_port = ?2 WHERE instance_id = ?1 AND status = 'STARTING'",
                rusqlite::params![s.instance_id, port as i64],
            )
            .map_err(|e| e.to_string())?;
        if moved != 1 {
            return Err("instance left STARTING during the retry".to_string());
        }
        s.port = port;
    }
    let mem_frac = mem_fraction_from_free(&need, free)?;
    let spec = EngineSpec {
        home: &s.home,
        instance_id: &s.instance_id,
        served_name: &s.params.model_id,
        engine_key: &s.engine_key,
        model_path: &s.model_path,
        card: &s.card,
        port: s.port,
        ctx: s.ctx,
        max_conc: s.max_conc,
        tp: s.tp,
        mem_frac,
        gpus: &s.gpus,
        load_extra: &s.params.extra_sglang_args,
    };
    let (identity, offset) = spawn_engine(&spec, &s.log_path)?;
    let changed = con
        .execute(
            "UPDATE loaded_instances SET pid = ?1, pgid = ?1, boot_id = ?2, start_ticks = ?3,
             allocated_vram_gb_per_gpu = ?4, mem_fraction_static = ?5
             WHERE instance_id = ?6 AND status = 'STARTING'",
            rusqlite::params![
                identity.pid as i64,
                identity.boot_id,
                identity.start_ticks as i64,
                need.total(),
                mem_frac,
                s.instance_id,
            ],
        )
        .unwrap_or(0);
    if changed != 1 {
        stop_one(&s.home, &identity);
        return Err("instance left STARTING during the retry; new engine stopped".to_string());
    }
    s.identity = identity;
    s.log_offset = offset;
    s.need = need;
    s.hold.resize(need.total());
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

pub fn append_goethed_log(home: &Path, line: &str) {
    let path = home.join("logs/goethed.log");
    let _ = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .and_then(|mut f| std::io::Write::write_all(&mut f, line.as_bytes()));
}

fn stop_processes(home: &Path, list: &[Inst]) -> HashSet<String> {
    // Stop instances (verified, all at once) and wait for their pids to leave NVML; returns instances that survived.
    let ids: Vec<Identity> = list.iter().filter_map(|i| i.identity.clone()).collect();
    let stopped = proc_ident::terminate(&ids);
    let gpus: Vec<u32> = list
        .iter()
        .flat_map(|i| i.gpus.iter().copied())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    if !foreign::wait_released(&gpus, &stopped.pids, GPU_RELEASE_WAIT) {
        append_goethed_log(home, &format!("teardown: pids {:?} still listed on GPUs {gpus:?} after 2s\n", stopped.pids));
    }
    for i in list {
        if stopped.survivors.contains(&i.instance_id) {
            append_goethed_log(home, &format!("teardown: {} survived SIGKILL; kept as STOPPING\n", i.instance_id));
        } else {
            run_registry::unregister(home, &i.instance_id);
        }
    }
    stopped.survivors
}

fn teardown(con: &Connection, home: &Path, list: &[Inst]) {
    // Verified stop of instances; their rows, probe state and notices go, survivors stay STOPPING with identity.
    let survivors = stop_processes(home, list);
    for i in list {
        if survivors.contains(&i.instance_id) {
            let _ = con.execute(
                "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1",
                [&i.instance_id],
            );
            continue;
        }
        let _ = con.execute("DELETE FROM loaded_instances WHERE instance_id = ?1", [&i.instance_id]);
        let _ = con.execute("DELETE FROM watchdog_probe_state WHERE instance_id = ?1", [&i.instance_id]);
        load_slot::clear_instance_notices(&i.instance_id);
    }
}

pub fn graceful_shutdown(home: &Path) -> Result<(), String> {
    // Cancel pending loads and stop every live engine (verified), even ones whose row or state.db is gone.
    let t0 = Instant::now();
    coordinator::cancel_queued(None);
    let con = match db::open(home) {
        Ok(con) => con,
        Err(e) => {
            load_slot::request_cancel_all();
            proc_ident::terminate(&run_registry::tracked());
            return Err(format!("state.db unavailable at shutdown ({e}); stopped tracked engines only"));
        }
    };
    let list = {
        let _plan = coordinator::plan_lock();
        let found = insts(&con, &format!("status IN {LIVE}"), &[]).unwrap_or_default();
        for i in &found {
            let _ = con.execute(
                &format!("UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1 AND status IN {LIVE}"),
                [&i.instance_id],
            );
        }
        found
    };
    load_slot::request_cancel_all();
    let _ = load_slot::wait_all_slots_quiet(Duration::from_millis(500));
    teardown(&con, home, &list);
    let strays = run_registry::tracked();
    let stopped = proc_ident::terminate(&strays);
    for id in strays.iter().filter(|id| !stopped.survivors.contains(&id.instance_id)) {
        run_registry::unregister(home, &id.instance_id);
    }
    append_goethed_log(
        home,
        &format!(
            "shutdown done ms={} instances={} strays={}\n",
            t0.elapsed().as_millis(),
            list.len(),
            strays.len()
        ),
    );
    Ok(())
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
    // One immediate transaction: drop desired slots and tombstones, collect live rows, mark exactly those DRAINING/STOPPING.
    let tx = con.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let only = "(?1 IS NULL OR model_id = ?1)";
    tx.execute(&format!("DELETE FROM desired_loads WHERE {only}"), [model])?;
    tx.execute(
        &format!(
            "DELETE FROM loaded_instances
             WHERE (status IN ('FAILED', 'CANCELLED') OR (status = 'RESTARTING' AND pid = 0)) AND {only}"
        ),
        [model],
    )?;
    let list = insts(&tx, &format!("status IN {LIVE} AND {only}"), &[&model])?;
    for i in &list {
        tx.execute(
            "UPDATE loaded_instances SET status = ?2
             WHERE instance_id = ?1 AND status IN ('STARTING', 'RUNNING', 'RESTARTING')",
            rusqlite::params![i.instance_id, next],
        )?;
    }
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
    // Every instance row (tombstones included) with notices, its own in-flight count, token rate and next attempt.
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
        if let Some(id) = v.get("instance_id").and_then(|x| x.as_str()).map(str::to_string) {
            v["in_flight"] = json!(crate::inflight::instance_count(&id));
        }
        if let Some(mid) = v.get("model_id").and_then(|x| x.as_str()).map(str::to_string) {
            v["tokens_per_s_1m"] = json!(crate::metrics::model_tokens_per_s_1m(&mid));
        }
        if v["status"] == "RESTARTING" {
            let at = v["slot_key"].as_str().and_then(|sk| next_attempts().get(sk).copied());
            v["next_attempt_at"] = json!(at);
        }
        out.push(v);
    }
    Ok(Value::Array(out))
}

pub fn reconcile_on_start() -> Result<(), String> {
    // Adopt verified RUNNING engines that answer; stop other verified engines; clear unverifiable rows unsignaled.
    let home = goethe_home();
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let _ = con.execute(
        "UPDATE loaded_instances SET pid = 0, pgid = 0 WHERE status IN ('FAILED', 'CANCELLED')",
        [],
    );
    let rows = insts(&con, &format!("status IN {LIVE}"), &[]).map_err(|e| e.to_string())?;
    let mut adopted = HashSet::new();
    let mut stale = Vec::new();
    for i in rows {
        let verified = i.identity.clone().filter(proc_ident::leader_alive);
        let _ = con.execute("DELETE FROM watchdog_probe_state WHERE instance_id = ?1", [&i.instance_id]);
        let answers = |id: &Identity| {
            owns_port(Some(id), i.port)
                && (0..3).any(|n| {
                    if n > 0 {
                        thread::sleep(Duration::from_secs(1));
                    }
                    serves(i.port, &i.engine_key, &i.model_id)
                })
        };
        match verified {
            Some(id) if i.status == "RUNNING" && answers(&id) => {
                let _ = run_registry::register(&home, &id);
                append_goethed_log(&home, &format!("adopted instance {} pid {}\n", i.instance_id, id.pid));
                adopted.insert(i.instance_id);
                continue;
            }
            Some(id) => {
                append_goethed_log(
                    &home,
                    &format!("instance {} ({}) not adoptable; stopping its verified engine\n", i.instance_id, i.status),
                );
                stale.push(id);
            }
            None => append_goethed_log(
                &home,
                &format!("cleared instance {} ({}) without signaling: identity does not verify\n", i.instance_id, i.status),
            ),
        }
        let _ = con.execute("DELETE FROM loaded_instances WHERE instance_id = ?1", [&i.instance_id]);
    }
    let stopped = proc_ident::terminate(&stale);
    let mut pids = stopped.pids;
    pids.extend(run_registry::reap_orphans(&home, &adopted));
    let gpus: Vec<u32> = foreign::snapshot().map(|s| s.iter().map(|g| g.index).collect()).unwrap_or_default();
    foreign::wait_released(&gpus, &pids, GPU_RELEASE_WAIT);
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

fn chooser_input(req: &LoadReq, snap: &[GpuLive], gpus: &[u32], ctx: u64, max_conc: u64) -> ChooserInput {
    // Budget inputs for a GPU set: Marlin extra by the live compute capability, KV bytes from the card dtype.
    let row = &req.row;
    ChooserInput {
        static_vram_gb: static_weight_gb(
            row.static_vram_gb,
            req.card.get("marlin_extra_gb").and_then(|v| v.as_float()),
            marlin_extra_applies(snap, gpus, &req.card),
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

fn marlin_extra_applies(snap: &[GpuLive], gpus: &[u32], card: &toml::Value) -> bool {
    // The Marlin weight copy exists on live compute capability 10+ (NVML, not a cached profile) unless GOETHE_MARLIN=0.
    let off = std::env::var("GOETHE_MARLIN").as_deref() == Ok("0")
        || card
            .get("env")
            .and_then(|v| v.get("GOETHE_MARLIN"))
            .and_then(|v| v.as_str())
            == Some("0");
    !off && snap.iter().any(|g| gpus.contains(&g.index) && g.cc_major >= 10)
}

fn gpu_name(snap: &[GpuLive], idx: u32) -> String {
    snap.iter()
        .find(|g| g.index == idx && !g.name.is_empty())
        .map_or_else(|| "gpu".to_string(), |g| g.name.clone())
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
    // Name every live Goethe engine on this GPU (replicas of this model included) and its free VRAM.
    let mut stmt = con
        .prepare(&format!(
            "SELECT model_id, allocated_vram_gb_per_gpu, gpu_indices FROM loaded_instances
             WHERE status IN {LIVE} AND pid > 0 ORDER BY rowid"
        ))
        .ok()?;
    let hosts: Vec<String> = stmt
        .query_map([], |r| {
            Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?, r.get::<_, String>(2)?))
        })
        .ok()?
        .filter_map(|r| r.ok())
        .filter(|(_, _, csv)| parse_gpu_csv(csv).contains(&gpu))
        .map(|(m, gb, _)| format!("{m}: {gb:.1} GB{}", if m == model_id { " (replica)" } else { "" }))
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

fn get(port: u16, path: &str, key: &str, timeout_secs: u64) -> Option<ureq::Response> {
    // GET a loopback engine path with the engine key; only a 200 answer counts.
    let agent = ureq::AgentBuilder::new()
        .timeout_read(Duration::from_secs(timeout_secs))
        .timeout_connect(Duration::from_secs(2))
        .build();
    let mut req = agent.get(&format!("http://127.0.0.1:{port}{path}"));
    if !key.is_empty() {
        req = req.set("Authorization", &format!("Bearer {key}"));
    }
    req.call().ok().filter(|r| r.status() == 200)
}

fn serves(port: u16, key: &str, model_id: &str) -> bool {
    // /v1/models with the engine key must list our served name; a bare 200 from any listener is not enough.
    get(port, "/v1/models", key, 2)
        .and_then(|r| r.into_string().ok())
        .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        .is_some_and(|v| v["data"].as_array().is_some_and(|a| a.iter().any(|m| m["id"].as_str() == Some(model_id))))
}

fn owns_port(identity: Option<&Identity>, port: u16) -> bool {
    identity.is_some_and(|id| listener::port_owner(id, port) == PortOwner::Ours)
}

fn health_ok(port: u16, key: &str) -> bool {
    get(port, "/health", key, HTTP_PROBE_SECS).is_some()
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
    let _guard = match WATCHDOG_TICK_LOCK.try_lock() {
        Ok(g) => g,
        Err(TryLockError::Poisoned(p)) => p.into_inner(),
        Err(TryLockError::WouldBlock) => return,
    };
    if shutdown_requested() {
        return;
    }
    let home = goethe_home();
    let Ok(con) = db::open(&home) else {
        return;
    };
    let now = now_epoch_secs();
    for i in insts(&con, "status = 'RUNNING'", &[]).unwrap_or_default() {
        let alive = i.identity.as_ref().is_some_and(proc_ident::leader_alive);
        let owned = alive && owns_port(i.identity.as_ref(), i.port);
        let mut state = load_probe_state(&con, &i.instance_id, now);
        let cheap = owned && serves(i.port, &i.engine_key, &i.model_id);
        let deep = watchdog_liveness::deep_due(&state, now).then(|| owned && health_ok(i.port, &i.engine_key));
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
                "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1 AND status = 'RUNNING'",
                [&i.instance_id],
            )
            .unwrap_or(0);
        if claimed != 1 {
            continue;
        }
        let params = load_params_for_slot(&con, &i.slot_key, &i.model_id);
        let backoff = crate::watchdog_policy::restart_backoff_secs(i.restarts);
        if params.is_some() {
            next_attempts().insert(i.slot_key.clone(), now_epoch_secs() + backoff as i64);
        }
        let to = if params.is_some() { "RESTARTING" } else { "FAILED" };
        match (retire(&con, &home, &i, "STOPPING", to), params) {
            (true, Some(params)) => schedule_restart(params, i.restarts, backoff),
            _ => {
                next_attempts().remove(&i.slot_key);
            }
        }
    }
}

fn retire(con: &Connection, home: &Path, i: &Inst, from: &str, to: &str) -> bool {
    // Stop a claimed instance (verified) and keep its row as `to` with pid 0 for ps; STOPPING if processes survived.
    if !stop_processes(home, std::slice::from_ref(i)).is_empty() {
        let _ = con.execute(
            "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1 AND status = ?2",
            [&i.instance_id, from],
        );
        return false;
    }
    let _ = con.execute(
        "UPDATE loaded_instances SET status = ?3, pid = 0, pgid = 0 WHERE instance_id = ?1 AND status = ?2",
        [i.instance_id.as_str(), from, to],
    );
    let _ = con.execute("DELETE FROM watchdog_probe_state WHERE instance_id = ?1", [&i.instance_id]);
    load_slot::clear_instance_notices(&i.instance_id);
    true
}

fn fail_running(con: &Connection, home: &Path, i: &Inst) {
    // Restart budget spent: stop the hung engine (verified) so its VRAM is freed, then keep a FAILED tombstone.
    let claimed = con
        .execute(
            "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1 AND status = 'RUNNING'",
            [&i.instance_id],
        )
        .unwrap_or(0);
    if claimed == 1 {
        retire(con, home, i, "STOPPING", "FAILED");
    }
}

fn next_attempts() -> std::sync::MutexGuard<'static, HashMap<String, i64>> {
    NEXT_ATTEMPT
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

fn schedule_restart(mut params: LoadParams, restarts: i64, backoff: u64) {
    // The row is already RESTARTING with its next attempt time; after the backoff re-check under the slot lock, relaunch.
    let slot = params.slot_key.clone().unwrap_or_default();
    thread::spawn(move || {
        thread::sleep(Duration::from_secs(backoff));
        let go = {
            let lock = slot_lock(&slot);
            let _guard = lock.lock().unwrap_or_else(|e| e.into_inner());
            let desired = db::open(&goethe_home())
                .map(|c| autostart::desired_exists(&c, &slot))
                .unwrap_or(false);
            !shutdown_requested() && desired
        };
        params.restart_count = Some(restarts.max(0) as u64 + 1);
        let failed = go.then(|| handle_load(params).err()).flatten();
        next_attempts().remove(&slot);
        if let Some((code, msg, _)) = failed {
            eprintln!("watchdog restart of {slot} failed ({code}): {msg}");
            if let Ok(con) = db::open(&goethe_home()) {
                let _ = con.execute(
                    "UPDATE loaded_instances SET status = 'FAILED' WHERE slot_key = ?1 AND status = 'RESTARTING' AND pid = 0",
                    [&slot],
                );
            }
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
        replica: false,
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

    fn models_server() -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = listener.local_addr().expect("addr").port();
        thread::spawn(move || {
            for mut s in listener.incoming().flatten() {
                let mut buf = [0u8; 1024];
                let _ = s.read(&mut buf);
                let body = r#"{"data":[{"id":"m"}]}"#;
                let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                let _ = s.write_all(format!("{head}{body}").as_bytes());
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
    fn reconcile_adopts_only_engines_that_own_their_port() {
        with_temp_home(|home, con| {
            let port = models_server();
            let squatted = models_server();
            let me = proc_ident::capture("i-good", std::process::id() as i32).unwrap();
            let (mut stale, stale_ticks) = sleeper();
            let (mut squat, squat_ticks) = sleeper();
            insert_running(con, "i-good", me.pid, me.start_ticks, port);
            insert_running(con, "i-stale", stale.id() as i32, stale_ticks + 9, port + 1);
            insert_running(con, "i-squat", squat.id() as i32, squat_ticks, squatted);
            let stale_id = Identity {
                instance_id: "i-orphan".to_string(),
                pid: stale.id() as i32,
                boot_id: proc_ident::boot_id().to_string(),
                start_ticks: stale_ticks + 9,
            };
            run_registry::register(home, &stale_id).unwrap();
            reconcile_on_start().expect("reconcile");
            let rows: Vec<(String, String, i64, String, i64)> = con
                .prepare("SELECT instance_id, status, internal_port, engine_key, pid FROM loaded_instances")
                .unwrap()
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))
                .unwrap()
                .filter_map(|r| r.ok())
                .collect();
            let kept = ("i-good".to_string(), "RUNNING".to_string(), port as i64, "k".to_string(), me.pid as i64);
            assert_eq!(rows, vec![kept], "adoption keeps pid, port and engine key, so the mux routes as before");
            assert!(stale.try_wait().unwrap().is_none(), "an unverified pid is never signaled");
            assert_ne!(proc_ident::start_ticks(squat.id() as i32), Some(squat_ticks), "port squatted: stopped");
            assert!(run_registry::list_entries(home).iter().all(|e| e.instance_id == "i-good"));
            stale.kill().unwrap();
            stale.wait().unwrap();
            let _ = squat.wait();
        });
    }

    #[test]
    fn restore_refuses_a_slot_that_is_already_live() {
        with_temp_home(|home, con| {
            install_m(home, con);
            con.execute("INSERT INTO desired_loads (slot_key, model_id) VALUES ('m|0|0', 'm')", []).unwrap();
            insert_running(con, "i-live", i32::MAX, 1, 30001);
            con.execute("UPDATE loaded_instances SET status = 'STARTING'", []).unwrap();
            let params = LoadParams {
                model_id: "m".to_string(),
                gpus: Some("0".to_string()),
                extra_sglang_args: Vec::new(),
                ctx: None,
                max_conc: None,
                tp: None,
                force: true,
                yes: true,
                restart_count: Some(1),
                slot_key: Some("m|0|0".to_string()),
                replica: false,
            };
            let (code, msg, _) = handle_load(params).unwrap_err();
            assert_eq!(code, 1);
            assert!(msg.contains("already has a live instance"), "{msg}");
        });
    }

    #[test]
    fn watchdog_tombstones_a_dead_engine() {
        with_temp_home(|_home, con| {
            let (mut child, ticks) = sleeper();
            let pid = child.id() as i32;
            child.kill().unwrap();
            child.wait().unwrap();
            insert_running(con, "i-dead", pid, ticks, 30001);
            watchdog_tick();
            let (status, row_pid): (String, i64) = con
                .query_row("SELECT status, pid FROM loaded_instances WHERE instance_id = 'i-dead'", [], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })
                .unwrap();
            assert_eq!((status.as_str(), row_pid), ("FAILED", 0));
        });
    }

    #[test]
    fn reconcile_stops_verified_engines_it_does_not_adopt() {
        with_temp_home(|_home, con| {
            let (mut child, ticks) = sleeper();
            let pid = child.id() as i32;
            insert_running(con, "i-starting", pid, ticks, 30001);
            con.execute("UPDATE loaded_instances SET status = 'STARTING'", []).unwrap();
            reconcile_on_start().expect("reconcile");
            assert_ne!(proc_ident::start_ticks(pid), Some(ticks), "verified STARTING engine must be stopped");
            let left: i64 = con.query_row("SELECT COUNT(*) FROM loaded_instances", [], |r| r.get(0)).unwrap();
            assert_eq!(left, 0);
            let _ = child.wait();
        });
    }

    fn install_m(home: &Path, con: &Connection) {
        let dir = home.join("models/m");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("goethe_card.toml"), "model_id = \"m\"\n").unwrap();
        con.execute(
            "INSERT INTO installed_models (model_id, local_path, architecture, static_vram_gb, kv_bytes_per_token,
             default_ctx, default_max_conc, default_tp) VALUES ('m', 'models/m', 'dense', 1.0, 100, 4096, 4, 1)",
            [],
        )
        .unwrap();
    }

    fn user_load(replica: bool) -> LoadParams {
        LoadParams {
            model_id: "m".to_string(),
            gpus: Some("0".to_string()),
            extra_sglang_args: Vec::new(),
            ctx: None,
            max_conc: None,
            tp: None,
            force: false,
            yes: false,
            restart_count: None,
            slot_key: None,
            replica,
        }
    }

    #[test]
    fn second_load_without_replica_reports_the_running_copy() {
        with_temp_home(|home, con| {
            install_m(home, con);
            insert_running(con, "i-first", i32::MAX, 1, 30007);
            let (result, message) = handle_load(user_load(false)).expect("already loaded is not an error");
            assert_eq!(message, "already loaded: i-first on GPU 0 (port 30007); use --replica to start another copy");
            assert_eq!(result, json!({"instance_id": "i-first", "status": "RUNNING", "already_loaded": true}));
        });
    }

    #[test]
    fn crash_backoff_shows_restarting_then_failed_when_the_attempt_fails() {
        with_temp_home(|_home, con| {
            let (mut child, ticks) = sleeper();
            let pid = child.id() as i32;
            child.kill().unwrap();
            child.wait().unwrap();
            insert_running(con, "i-crash", pid, ticks, 30001);
            con.execute("INSERT INTO desired_loads (slot_key, model_id) VALUES ('m|0|0', 'm')", []).unwrap();
            watchdog_tick();
            let row = |c: &Connection| -> (String, i64) {
                c.query_row("SELECT status, pid FROM loaded_instances WHERE instance_id = 'i-crash'", [], |r| {
                    Ok((r.get(0)?, r.get(1)?))
                })
                .unwrap()
            };
            assert_eq!(row(con), ("RESTARTING".to_string(), 0));
            let ps = handle_ps().unwrap();
            let next = ps[0]["next_attempt_at"].as_i64().expect("next attempt time in ps");
            assert!(next >= now_epoch_secs());
            let failed = (0..60).any(|_| {
                thread::sleep(Duration::from_millis(250));
                row(con).0 == "FAILED"
            });
            assert!(failed, "a failed restart attempt leaves FAILED");
        });
    }

    #[test]
    fn identity_probe_needs_our_model_name() {
        let port = models_server();
        assert!(serves(port, "k", "m"));
        assert!(!serves(port, "k", "other"));
        assert!(!serves(free_port(), "k", "m"));
    }

    fn free_port() -> u16 {
        TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
    }

    #[test]
    fn ps_counts_in_flight_per_instance() {
        with_temp_home(|_home, con| {
            insert_running(con, "i-ps-a", i32::MAX, 1, 30001);
            insert_running(con, "i-ps-b", i32::MAX - 1, 1, 30002);
            let mut guard = crate::inflight::track("m");
            assert_eq!(guard.pick(&["i-ps-a"]), Some(0));
            let rows = handle_ps().unwrap();
            let row = |id: &str| rows.as_array().unwrap().iter().find(|r| r["instance_id"] == id).cloned().unwrap();
            let in_flight = |id: &str| row(id)["in_flight"].clone();
            assert_eq!(in_flight("i-ps-a"), json!(1));
            assert_eq!(in_flight("i-ps-b"), json!(0), "a replica of the same model shows only its own requests");
            drop(guard);
        });
    }

    #[test]
    fn marlin_extra_follows_live_compute_capability() {
        let live = |index, cc_major| GpuLive { index, name: "g".to_string(), cc_major, free_mb: 1, procs: Vec::new() };
        let snap = vec![live(2, 12), live(3, 8)];
        let card: toml::Value = toml::from_str("marlin_extra_gb = 7.8").unwrap();
        assert!(marlin_extra_applies(&snap, &[2], &card));
        assert!(!marlin_extra_applies(&snap, &[3], &card));
        assert!(marlin_extra_applies(&snap, &[2, 3], &card));
        let off: toml::Value = toml::from_str("[env]\nGOETHE_MARLIN = \"0\"").unwrap();
        assert!(!marlin_extra_applies(&snap, &[2], &off));
        assert_eq!(gpu_name(&snap, 3), "g");
        assert_eq!(gpu_name(&snap, 9), "gpu");
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
