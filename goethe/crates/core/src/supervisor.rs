// goethed load, unload, adoption and watchdog supervision of SGLang engines.

use crate::autostart;
use crate::calc::{self, mem_fraction_from_free, shortfall, Need};
use crate::chooser::{self, ChooserInput, LoadPlan};
use crate::coordinator;
use crate::db;
use crate::engines::{self, EngineSpec};
use crate::engine_overhead::Learned;
use crate::foreign::{self, Foreign, GpuLive};
use crate::hw::HardwareState;
use crate::listener::{self, PortOwner};
use crate::load_slot;
use crate::paths::{ensure_layout, goethe_home};
use crate::placement;
use crate::proc_ident::{self, Identity};
#[cfg(test)]
use crate::run_registry;
use crate::shutdown::shutdown_requested;
use crate::watchdog_liveness::{self, LivenessState, ProbeInput, WatchdogAction};
use rusqlite::{Connection, ToSql};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock, TryLockError};
use std::thread;
use std::time::{Duration, Instant};

#[path = "supervisor_load.rs"]
mod supervisor_load;
use supervisor_load::{launch_finish, spawn_engine, stop_one};
#[cfg(test)]
use supervisor_load::tree_cpu_ticks;

#[path = "supervisor_lifecycle.rs"]
mod supervisor_lifecycle;
pub use supervisor_lifecycle::{graceful_shutdown, handle_ps, handle_unload, handle_unload_instance, reconcile_on_start};
use supervisor_lifecycle::stop_processes;

static SLOT_LOCKS: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
static WATCHDOG_TICK_LOCK: Mutex<()> = Mutex::new(());
static NEXT_ATTEMPT: OnceLock<Mutex<HashMap<String, i64>>> = OnceLock::new();
static STUCK_STOPS: Mutex<Vec<(String, String)>> = Mutex::new(Vec::new());

pub(super) const LIVE: &str = "('STARTING', 'RUNNING', 'RESTARTING', 'DRAINING', 'STOPPING')";
pub(super) const GPU_RELEASE_WAIT: Duration = Duration::from_secs(30);
const HTTP_PROBE_SECS: u64 = 25;
const WARM_UP_SECS: u64 = 900;
const USER_GATE_WAIT: Duration = Duration::from_secs(590);

type LoadErr = (i32, String, Value);

pub(super) fn slot_lock(key: &str) -> Arc<Mutex<()>> {
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
    #[serde(default)]
    pub numerics: Option<goethe_proto::Numerics>,
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

pub(super) struct Starting {
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
    cc: String,
    ctx: u64,
    max_conc: u64,
    input: ChooserInput,
    plan_free: f64,
    cancel: Arc<AtomicBool>,
    hold: coordinator::Hold,
}

pub(super) struct Inst {
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

pub(super) enum Startup {
    Ready,
    Cancelled,
    Failed,
    PortTaken(String),
}

pub(super) fn insts(con: &Connection, filter: &str, args: &[&dyn ToSql]) -> rusqlite::Result<Vec<Inst>> {
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
    for extra in [engines::sglang::card_extra_args(&card), params.extra_sglang_args.clone()] {
        engines::sglang::check_extra_args(&extra).map_err(|e| (2, e, Value::Null))?;
    }
    let pinned = pinned_gpus(params.gpus.as_deref(), &card, params.tp).map_err(|e| (2, e, Value::Null))?;
    let denied = crate::gpu_policy::denied_set(&con).map_err(err1)?;
    if let Some(g) = pinned.iter().find(|g| denied.contains(g)) {
        return Err(err1(format!(
            "gpu{g} is denied by policy; run `goethe gpu allow gpu{g}` to use it"
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
            if ticket.cancelled() || shutdown_requested() {
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
    // Under the planning lock: slot and replica checks (a copy still starting is waited for like a fresh load, so "already loaded" names an engine that answers; one in crash backoff counts at once), plan on live VRAM, then fit, notice and gate.
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
    if req.params.slot_key.is_none() && !req.params.replica && copies.is_empty() {
        crate::slots::drop_stale_slots(con, model_id).map_err(|e| err1(e.to_string()))?;
    }
    if let (None, false, Some(first)) = (&req.params.slot_key, req.params.replica, copies.first()) {
        if first.1 == "STARTING" {
            return Ok(Step::Wait(parse_gpu_csv(&first.2)));
        }
        let differs = copy_mismatch(&req.params, &req.pinned, &first.2, first.4);
        let message = if differs.is_empty() {
            format!("already loaded: {} on GPU {} (port {}); use --replica to start another copy", first.0, first.2, first.3)
        } else {
            format!(
                "already loaded: {} on GPU {} (port {}), but {}; nothing changed. Unload it first to load with the new settings, or use --replica for another copy",
                first.0,
                first.2,
                first.3,
                differs.join(" and ")
            )
        };
        let result = json!({"instance_id": first.0, "status": first.1, "already_loaded": true});
        return Ok(Step::Loaded(result, message));
    }
    if let Some(layer) = engines::of_card(&req.card).map_err(err1)?.layer {
        crate::pyenv::require(&req.home, layer, model_id).map_err(err1)?;
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
    let base = chooser_input(req, req.ctx, req.max_conc);
    let pinned = (!req.pinned.is_empty()).then_some(req.pinned.as_slice());
    let tp = req.pinned_tp.or(req.default_tp);
    let runtime = crate::engine_overhead::card_runtime(&req.card);
    let learned_at =
        |t: usize, first: u32| crate::engine_overhead::read_learned(con, model_id, &gpu_name(&snap, first), t as u64, &runtime);
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
    let min_free = free.values().copied().fold(f64::INFINITY, f64::min);
    let mut card = req.card.clone();
    if plan.disable_marlin {
        set_card_env(&mut card, "GOETHE_MARLIN", "0");
    }
    let input = chooser_input_on(req, &card, plan.ctx, plan.max_conc);
    let learned = crate::engine_overhead::read_learned(con, model_id, &gpu_name, plan.tp as u64, &runtime);
    let need = chooser::with_learned(chooser::need(&input, plan.tp, &plan.cc), learned);
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
    let mut reduced = Vec::new();
    if plan.ctx < req.ctx {
        reduced.push(format!("context {} (asked {})", plan.ctx, req.ctx));
    }
    if plan.max_conc < req.max_conc {
        reduced.push(format!("{} parallel requests (asked {})", plan.max_conc, req.max_conc));
    }
    if !reduced.is_empty() {
        notices.push_str(&format!(
            "reduced to fit: {}; pick more GPUs with --gpu, or set --ctx / --max-conc\n",
            reduced.join(", ")
        ));
    }
    if plan.disable_marlin {
        notices.push_str("weight dual-copy off (GOETHE_MARLIN=0): keeps KV room on this card\n");
    }
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
    let starting = start_engine(req, con, &plan, FitSpawn { need, learned, min_free, gpu_name, card })?;
    Ok(Step::Started(Box::new(starting), notices))
}

struct FitSpawn {
    need: Need,
    learned: Learned,
    min_free: f64,
    gpu_name: String,
    card: toml::Value,
}

fn copies_of(con: &Connection, model_id: &str) -> Vec<(String, String, String, i64, i64)> {
    // STARTING, RUNNING or RESTARTING (watchdog backoff) instances of a model, RUNNING first: id, status, GPUs, port, context.
    let Ok(mut stmt) = con.prepare(
        "SELECT instance_id, status, gpu_indices, internal_port, ctx_len FROM loaded_instances
         WHERE model_id = ?1 AND status IN ('STARTING', 'RUNNING', 'RESTARTING') ORDER BY status = 'RUNNING' DESC, rowid",
    ) else {
        return Vec::new();
    };
    stmt.query_map([model_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))
        .map(|rows| rows.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
}

fn copy_mismatch(params: &LoadParams, pinned: &[u32], gpus: &str, ctx_len: i64) -> Vec<String> {
    // What a load asked for explicitly that the running copy does not have (GPUs, context).
    let mut out = Vec::new();
    let mut have = parse_gpu_csv(gpus);
    let mut want = pinned.to_vec();
    have.sort_unstable();
    want.sort_unstable();
    if params.gpus.is_some() && !want.is_empty() && want != have {
        out.push(format!("you asked for GPU {}", gpu_csv(&want)));
    }
    if let Some(ctx) = params.ctx.filter(|c| i64::try_from(*c).ok() != Some(ctx_len)) {
        out.push(format!("you asked for context {ctx} and it runs with {ctx_len}"));
    }
    out
}

fn missing_error(message: String, holders: &[Foreign]) -> LoadErr {
    let mut lines = vec![message];
    lines.extend(foreign::holder_lines(holders));
    (1, lines.join("\n"), Value::Null)
}

fn start_engine(req: &LoadReq, con: &Connection, plan: &LoadPlan, fit: FitSpawn) -> Result<Starting, LoadErr> {
    // Port and slot, spawn, then the STARTING row and ledger entry; any failure after spawn stops the engine.
    let FitSpawn { need, learned, min_free, gpu_name, card } = fit;
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
    let engine_key = engines::new_engine_key().map_err(err1)?;
    let budget = need.total();
    let context = crate::engine_overhead::read_context(con, &gpu_name, plan.tp as u64);
    let mem_frac = mem_fraction_from_free(&need, min_free, context).map_err(err1)?;
    let log_path = req.home.join("logs").join(format!("{instance_id}.log"));
    let tp = plan.tp as u64;
    let spec = EngineSpec {
        home: &req.home,
        instance_id: &instance_id,
        served_name: model_id,
        engine_key: &engine_key,
        model_path: &req.model_path,
        card: &card,
        port,
        ctx: plan.ctx,
        max_conc: plan.max_conc,
        pool_tokens: chooser::kv_tokens(plan.ctx, plan.max_conc, chooser::card_floor(&card)),
        numerics: req.params.numerics,
        tp,
        mem_frac,
        gpus: &plan.gpus,
        cc: &plan.cc,
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
    crate::client::link::state_changed();
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
        input: chooser_input_on(req, &card, plan.ctx, plan.max_conc),
        card,
        gpus: plan.gpus.clone(),
        cc: plan.cc.clone(),
        ctx: plan.ctx,
        max_conc: plan.max_conc,
        plan_free: min_free,
    })
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
    .map_err(|e| match e {
        rusqlite::Error::QueryReturnedNoRows => format!("model {model_id} not installed"),
        e => format!("model {model_id}: the installed models table cannot be read: {e}"),
    })
}

fn default_ctx(row: &InstRow) -> u64 {
    row.default_ctx.min(row.max_ctx.unwrap_or(row.default_ctx)).max(1) as u64
}

fn kv_bytes_per_token(row: &InstRow, card: &toml::Value) -> u64 {
    let elems = row.kv_elems_per_token.unwrap_or(row.kv_bytes_per_token).max(0) as u64;
    crate::install::card_kv_bytes(card, elems, row.kv_bytes_per_token.max(0) as u64)
}

fn chooser_input(req: &LoadReq, ctx: u64, max_conc: u64) -> ChooserInput {
    chooser_input_on(req, &req.card, ctx, max_conc)
}

fn chooser_input_on(req: &LoadReq, card: &toml::Value, ctx: u64, max_conc: u64) -> ChooserInput {
    let row = &req.row;
    let facts = crate::chooser::CardFacts {
        static_vram_gb: row.static_vram_gb,
        kv_bytes_per_token: kv_bytes_per_token(row, card),
        num_attention_heads: row.num_attention_heads.unwrap_or(1) as u64,
        num_kv_heads: row.num_kv_heads.unwrap_or(1) as u64,
        attention: row.attention.clone().unwrap_or_else(|| "gqa".to_string()),
        max_ctx: row.max_ctx.unwrap_or(row.default_ctx) as u64,
    };
    let input = ChooserInput::from_card(
        facts,
        card,
        crate::memory_spec::installed(&req.model_path, card, &req.params.extra_sglang_args),
        ctx,
        max_conc,
    );
    crate::engines::llama_cpp::with_header_kv(input.pin(req.params.ctx, req.params.max_conc), &req.model_path, card)
}

fn set_card_env(card: &mut toml::Value, key: &str, val: &str) {
    let table = card.as_table_mut().expect("card is a table");
    let env = table.entry("env").or_insert_with(|| toml::Value::Table(toml::Table::new()));
    if let Some(env) = env.as_table_mut() {
        env.insert(key.to_string(), toml::Value::String(val.to_string()));
    }
}

fn gpu_name(snap: &[GpuLive], idx: u32) -> String {
    snap.iter()
        .find(|g| g.index == idx && !g.name.is_empty())
        .map_or_else(|| "gpu".to_string(), |g| g.name.clone())
}

fn pinned_gpus(raw: Option<&str>, card: &toml::Value, tp: Option<u64>) -> Result<Vec<u32>, String> {
    // --gpu (else the card's preferred_gpus) as sorted indices; it must agree with --tp and list no GPU twice.
    let list: Vec<u32> = match raw {
        Some(s) => s
            .split(',')
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|p| p.parse::<u32>().map_err(|_| format!("invalid GPU '{p}' in --gpu")))
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
            return Err(format!("--tp {t} disagrees with {} GPUs on --gpu", list.len()));
        }
    }
    let mut sorted = list;
    sorted.sort_unstable();
    if let Some(w) = sorted.windows(2).find(|w| w[0] == w[1]) {
        return Err(format!("gpu{} is listed twice in --gpu", w[0]));
    }
    Ok(sorted)
}

fn read_card_value(path: &Path) -> Result<toml::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    toml::from_str(&text).map_err(|e| e.to_string())
}

pub(super) fn parse_gpu_csv(s: &str) -> Vec<u32> {
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

pub(super) fn next_port(con: &Connection, lo: u16, hi: u16) -> Result<u16, String> {
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

fn warm_up_request(task: Option<&str>, model_id: &str, ctx: u64) -> Option<(&'static str, Value)> {
    // The request that warms this model's kernels: a real-length completion, one input for an embedding model, none for speech (it needs audio).
    let words = (ctx / 3).clamp(256, 3000) as usize;
    let text = "the engine compiles its kernels for this prompt length now ".repeat(words / 10);
    match task {
        Some("transcribe") => None,
        Some("embed") => Some(("/v1/embeddings", serde_json::json!({"model": model_id, "input": text}))),
        _ => Some(("/v1/completions", serde_json::json!({"model": model_id, "prompt": text, "max_tokens": 8, "temperature": 0}))),
    }
}

pub(super) fn warm_up(s: &Starting) {
    // One request before RUNNING, so the first client request does not pay for first-use kernel builds.
    let Some((path, body)) = warm_up_request(s.card.get("task").and_then(|v| v.as_str()), &s.params.model_id, s.ctx) else { return };
    let started = Instant::now();
    let agent = ureq::AgentBuilder::new()
        .timeout_read(Duration::from_secs(WARM_UP_SECS))
        .timeout_connect(Duration::from_secs(2))
        .build();
    let res = agent
        .post(&format!("http://127.0.0.1:{}{path}", s.port))
        .set("Authorization", &format!("Bearer {}", s.engine_key))
        .set("Content-Type", "application/json")
        .send_string(&body.to_string());
    let outcome = match res {
        Ok(_) => format!("warm-up request took {:.0} s", started.elapsed().as_secs_f64()),
        Err(e) => format!("warm-up request skipped ({e})"),
    };
    append_goethed_log(&s.home, &format!("{}: {outcome}\n", s.instance_id));
}

pub(super) fn serves(port: u16, key: &str, model_id: &str) -> bool {
    // /v1/models with the engine key must list our served name; a bare 200 from any listener is not enough.
    get(port, "/v1/models", key, 2)
        .and_then(|r| r.into_string().ok())
        .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        .is_some_and(|v| v["data"].as_array().is_some_and(|a| a.iter().any(|m| m["id"].as_str() == Some(model_id))))
}

pub(super) fn owns_port(identity: Option<&Identity>, port: u16) -> bool {
    identity.is_some_and(|id| listener::port_owner(id, port) == PortOwner::Ours)
}

pub(super) fn health_ok(port: u16, key: &str) -> bool {
    get(port, "/health", key, HTTP_PROBE_SECS).is_some()
}

pub(super) fn log_file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map(|m| m.len()).unwrap_or(0)
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
    retry_stuck_stops(&con, &home);
    let now = crate::sys::unix_secs();
    for i in insts(&con, "status = 'RUNNING'", &[]).unwrap_or_default() {
        let alive = i.identity.as_ref().is_some_and(proc_ident::leader_alive);
        if !alive && i.identity.as_ref().is_some_and(proc_ident::leader_unreadable) {
            append_goethed_log(&home, &format!("watchdog instance={} process table unreadable; liveness unknown, tick skipped\n", i.instance_id));
            continue;
        }
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
            next_attempts().insert(i.slot_key.clone(), crate::sys::unix_secs() + backoff as i64);
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

fn retry_stuck_stops(con: &Connection, home: &Path) {
    // Engines that survived a watchdog stop stay STOPPING with a pid: stop them again each tick until gone, then finish as planned (restart or FAILED).
    let stuck: Vec<(String, String)> = STUCK_STOPS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    for (id, to) in stuck {
        let row = insts(con, "instance_id = ?1 AND status = 'STOPPING' AND pid > 0", &[&id]).ok().and_then(|v| v.into_iter().next());
        let Some(i) = row else {
            STUCK_STOPS.lock().unwrap_or_else(|e| e.into_inner()).retain(|(s, _)| *s != id);
            continue;
        };
        if !retire(con, home, &i, "STOPPING", &to) {
            continue;
        }
        append_goethed_log(home, &format!("watchdog instance={id} stopped on retry; now {to}\n"));
        if to == "RESTARTING" {
            match load_params_for_slot(con, &i.slot_key, &i.model_id) {
                Some(params) => {
                    let backoff = crate::watchdog_policy::restart_backoff_secs(i.restarts);
                    next_attempts().insert(i.slot_key.clone(), crate::sys::unix_secs() + backoff as i64);
                    schedule_restart(params, i.restarts, backoff);
                }
                None => {
                    let _ = con.execute("UPDATE loaded_instances SET status = 'FAILED' WHERE instance_id = ?1 AND status = 'RESTARTING'", [&id]);
                }
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
        let mut stuck = STUCK_STOPS.lock().unwrap_or_else(|e| e.into_inner());
        if !stuck.iter().any(|(id, _)| *id == i.instance_id) {
            append_goethed_log(home, &format!("watchdog instance={} survived the stop; retrying every watchdog tick\n", i.instance_id));
            stuck.push((i.instance_id.clone(), to.to_string()));
        }
        return false;
    }
    STUCK_STOPS.lock().unwrap_or_else(|e| e.into_inner()).retain(|(id, _)| *id != i.instance_id);
    let _ = con.execute(
        "UPDATE loaded_instances SET status = ?3, pid = 0, pgid = 0 WHERE instance_id = ?1 AND status = ?2",
        [i.instance_id.as_str(), from, to],
    );
    let _ = con.execute("DELETE FROM watchdog_probe_state WHERE instance_id = ?1", [&i.instance_id]);
    load_slot::clear_instance_notices(&i.instance_id);
    crate::client::link::state_changed();
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

pub(super) fn next_attempts() -> std::sync::MutexGuard<'static, HashMap<String, i64>> {
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
    match crate::watchdog_policy::decide_restart(crate::sys::unix_secs(), start, count) {
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

pub(crate) fn load_params_for_slot(con: &Connection, slot_key: &str, model_id: &str) -> Option<LoadParams> {
    // Load parameters recorded for exactly this desired slot.
    con.query_row(
        "SELECT gpus, ctx, max_conc, tp, extra_sglang_args, numerics FROM desired_loads WHERE slot_key = ?1",
        [slot_key],
        |r| {
            Ok(LoadParams {
                model_id: model_id.to_string(),
                gpus: r.get(0)?,
                ctx: r.get::<_, Option<i64>>(1)?.map(|v| v as u64),
                max_conc: r.get::<_, Option<i64>>(2)?.map(|v| v as u64),
                tp: r.get::<_, Option<i64>>(3)?.map(|v| v as u64),
                extra_sglang_args: serde_json::from_str(&r.get::<_, String>(4)?).unwrap_or_default(),
                force: true,
                yes: true,
                restart_count: None,
                slot_key: Some(slot_key.to_string()),
                replica: false,
                numerics: r.get::<_, Option<String>>(5)?.as_deref().and_then(goethe_proto::Numerics::parse),
            })
        },
    )
    .ok()
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

    #[test]
    fn a_missing_row_is_not_installed_and_any_other_read_error_keeps_its_cause() {
        let dir = tempfile::tempdir().unwrap();
        let e = |con: &Connection| fetch_installed(con, "org/m").err().unwrap();
        assert_eq!(e(&db::open(dir.path()).unwrap()), "model org/m not installed");
        assert!(e(&Connection::open_in_memory().unwrap()).contains("no such table: installed_models"));
    }

    #[test]
    fn each_task_warms_up_with_a_request_it_answers() {
        let (path, body) = warm_up_request(Some("embed"), "org/e", 2048).unwrap();
        assert_eq!(path, "/v1/embeddings");
        assert!(body["input"].as_str().is_some_and(|t| !t.is_empty()) && body.get("prompt").is_none());
        assert_eq!(warm_up_request(None, "org/m", 8192).unwrap().0, "/v1/completions");
        assert!(warm_up_request(Some("transcribe"), "org/s", 8192).is_none(), "speech needs audio, not text");
    }

    #[test]
    fn ps_names_the_runtime_the_card_serves_with() {
        let home = tempfile::tempdir().unwrap();
        let card = |id: &str, text: &str| {
            let d = crate::paths::model_dir(home.path(), id);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join(crate::install::CARD_FILE), text).unwrap();
        };
        card("org/pack", "runtime = \"goethe-engine\"\n");
        card("org/gguf", "runtime = \"llama.cpp\"\n");
        card("org/plain", "model_id = \"org/plain\"\n");
        assert_eq!(crate::engine_overhead::model_runtime(home.path(), "org/pack"), "goethe-engine");
        assert_eq!(crate::engine_overhead::model_runtime(home.path(), "org/gguf"), "llama.cpp");
        assert_eq!(crate::engine_overhead::model_runtime(home.path(), "org/plain"), "sglang");
        assert_eq!(crate::engine_overhead::model_runtime(home.path(), "org/none"), "sglang");
    }
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
    fn unloading_one_instance_keeps_the_other_and_its_slot() {
        with_temp_home(|_home, con| {
            let (mut child, ticks) = sleeper();
            insert_running(con, "i1", child.id() as i32, ticks + 5, 30001);
            insert_running(con, "i2", child.id() as i32, ticks + 5, 30002);
            con.execute("UPDATE loaded_instances SET slot_key = 'm|1|1' WHERE instance_id = 'i2'", []).unwrap();
            con.execute("INSERT INTO desired_loads (slot_key, model_id) VALUES ('m|0|0', 'm'), ('m|1|1', 'm')", []).unwrap();
            handle_unload_instance("i1", false).expect("unload");
            let left: Vec<(String, String)> = con
                .prepare("SELECT instance_id, status FROM loaded_instances")
                .unwrap()
                .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
                .unwrap()
                .map(Result::unwrap)
                .collect();
            let desired: Vec<String> =
                con.prepare("SELECT slot_key FROM desired_loads").unwrap().query_map([], |r| r.get(0)).unwrap().map(Result::unwrap).collect();
            assert_eq!(left, [("i2".to_string(), "RUNNING".to_string())]);
            assert_eq!(desired, ["m|1|1"]);
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
            insert_running(con, "i-stale", stale.id() as i32, stale_ticks + 9, 1);
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
                numerics: None,
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
            numerics: None,
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
    fn a_load_that_finds_a_copy_still_starting_waits_until_it_runs() {
        // A load returns once the model answers, also when another copy of it is still starting: "already loaded" never points at an engine that is not ready.
        with_temp_home(|home, con| {
            install_m(home, con);
            insert_running(con, "i-boot", i32::MAX, 1, 30009);
            con.execute("UPDATE loaded_instances SET status = 'STARTING' WHERE instance_id = 'i-boot'", []).unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || tx.send(handle_load(user_load(false))).unwrap());
            assert!(rx.recv_timeout(Duration::from_millis(1500)).is_err(), "no answer while the copy is still starting");
            con.execute("UPDATE loaded_instances SET status = 'RUNNING' WHERE instance_id = 'i-boot'", []).unwrap();
            let (result, _) = rx.recv_timeout(Duration::from_secs(10)).expect("answers once it runs").expect("already loaded");
            assert_eq!(result, json!({"instance_id": "i-boot", "status": "RUNNING", "already_loaded": true}));
        });
    }

    #[test]
    fn a_copy_in_restart_backoff_counts_and_different_settings_are_named() {
        with_temp_home(|home, con| {
            install_m(home, con);
            insert_running(con, "i-back", i32::MAX, 1, 30008);
            con.execute("UPDATE loaded_instances SET status = 'RESTARTING', pid = 0 WHERE instance_id = 'i-back'", []).unwrap();
            let (result, message) = handle_load(LoadParams { ctx: Some(8192), ..user_load(false) }).expect("already loaded");
            assert_eq!(result["instance_id"], "i-back", "no second engine during the watchdog backoff");
            assert!(message.contains("you asked for context 8192 and it runs with 4096") && message.contains("nothing changed"), "{message}");
        });
        let p = LoadParams { gpus: Some("1".into()), ..user_load(false) };
        assert_eq!(copy_mismatch(&p, &[1], "0", 4096), vec!["you asked for GPU 1".to_string()]);
        assert!(copy_mismatch(&p, &[0], "0", 4096).is_empty());
        assert!(copy_mismatch(&LoadParams { gpus: None, ..user_load(false) }, &[], "0,1", 4096).is_empty());
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
            assert!(next >= crate::sys::unix_secs());
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
    fn gpu_names_come_from_the_live_snapshot() {
        let live = |index| GpuLive {
            index,
            name: "g".to_string(),
            compute_capability: "12.0".to_string(),
            free_mb: 1,
            procs: Vec::new(),
        };
        let snap = vec![live(2), live(3)];
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

    #[test]
    fn engine_cpu_time_counts_as_startup_progress() {
        let me = std::process::id() as i32;
        let before = tree_cpu_ticks(&[me]);
        let mut x = 0u64;
        for i in 0..30_000_000u64 {
            x = x.wrapping_add(i * i);
        }
        assert!(x > 0);
        assert!(tree_cpu_ticks(&[me]) >= before);
        assert_eq!(tree_cpu_ticks(&[i32::MAX]), 0);
    }
}