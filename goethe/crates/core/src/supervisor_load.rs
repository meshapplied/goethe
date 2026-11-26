// Engine spawn, startup wait, refit and respawn for load.

use super::{
    append_goethed_log, health_ok, log_file_len, next_port, serves, slot_lock, warm_up,
    Starting, Startup, GPU_RELEASE_WAIT,
};
use crate::autostart;
use crate::calc::{self, mem_fraction_from_free, shortfall, Need};
use crate::chooser::{self, ChooserInput};
use crate::coordinator;
use crate::db;
use crate::engines::{self, EngineSpec};
use crate::engine_overhead::Kind;
use crate::foreign;
use crate::listener::{self, PortOwner};
use crate::load_slot;
use crate::proc_ident::{self, Identity};
use crate::run_registry;
use rusqlite::{Connection, TransactionBehavior};
use serde_json::{json, Value};
use std::fs::OpenOptions;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::Stdio;
use std::thread;
use std::time::{Duration, Instant};

pub(super) fn spawn_engine(spec: &EngineSpec, log_path: &Path) -> Result<(Identity, u64), String> {
    // Start the engine in its own session, capture its identity and register its run entry before anything else.
    if let Err(e) = crate::cuda_toolkit::prepare(spec.home) {
        eprintln!("bundled CUDA toolkit layout: {e}");
    }
    let log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .map_err(|e| e.to_string())?;
    let offset = log.metadata().map(|m| m.len()).unwrap_or(0);
    let mut cmd = engines::command(spec)?;
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

pub(super) fn launch_finish(starting: Starting, notices: String) -> (Value, String) {
    // Hand the STARTING engine to a slot-serialized thread that waits for health.
    let result = json!({
        "instance_id": starting.instance_id,
        "port": starting.port,
        "status": "STARTING",
        "slot_key": starting.params.slot_key,
    });
    let sk = starting.params.slot_key.clone().unwrap_or_default();
    let in_flight = load_slot::slot_in_flight(&sk);
    thread::spawn(move || {
        let slot = slot_lock(&sk);
        let _guard = slot.lock().unwrap_or_else(|e| e.into_inner());
        let _in_flight = in_flight;
        let mut starting = starting;
        startup(&mut starting);
        drop(starting);
    });
    (result, notices)
}

pub(super) fn startup(s: &mut Starting) {
    // STARTING to RUNNING, CANCELLED or FAILED: two retries after a VRAM shortfall, three on a new port if ours is held.
    let (mut mem_retries, mut port_retries) = (0, 0);
    loop {
        match wait_engine_ready(s) {
            Startup::Ready => {
                warm_up(s);
                match mark_running(s) {
                    Ok(true) => learn_overhead(s),
                    Ok(false) => {}
                    Err(e) => {
                        append_goethed_log(&s.home, &format!("{}: could not record RUNNING ({e}); stopping the engine\n", s.instance_id));
                        load_slot::push_instance_notice(&s.instance_id, &format!("could not record RUNNING: {e}"));
                        fail_engine(s);
                    }
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
        let weights_gap = crate::engine_overhead::parse_weight_gb(&log)
            .map(|w| w - s.need.weights)
            .filter(|gap| *gap > 0.25);
        if let Some(extra) = weights_gap.or_else(|| crate::engine_overhead::parse_shortfall_gb(&log)) {
            s.learned.static_gb += extra;
            if let Ok(con) = db::open(&s.home) {
                let (model, learned) = (&s.params.model_id, s.learned.static_gb);
                let _ = crate::engine_overhead::store_learned(&con, model, &s.gpu_name, s.tp, Kind::Static, learned, &crate::engine_overhead::card_runtime(&s.card));
            }
            if mem_retries < 2 {
                match refit(s) {
                    Ok(reduced) => {
                        mem_retries += 1;
                        let notice = format!("engine needed {extra:.1} GB more than estimated, retried{reduced}");
                        load_slot::push_instance_notice(&s.instance_id, &notice);
                        continue;
                    }
                    Err(_) if load_slot::is_cancelled(&s.cancel) => return cancel_engine(s),
                    Err(notice) => {
                        append_goethed_log(&s.home, &format!("{}: retry not started: {notice}\n", s.instance_id));
                        load_slot::push_instance_notice(&s.instance_id, &notice);
                    }
                }
            }
        }
        return fail_engine(s);
    }
}

pub(super) fn wait_engine_ready(s: &Starting) -> Startup {
    // Ready needs every listener on the port to be ours, /health to answer and /v1/models to list our model.
    let hard_cap = crate::startup_stall::hard_cap_secs_from_env();
    let mut tracker = crate::startup_stall::StallTracker::new(crate::sys::unix_secs(), log_file_len(&s.log_path));
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
        let progress = log_file_len(&s.log_path) + tree_cpu_ticks(&proc_ident::instance_pids(&s.identity));
        let step = crate::startup_stall::poll_startup_step(&mut tracker, crate::sys::unix_secs(), alive, progress, ready, hard_cap);
        match step {
            crate::startup_stall::StartupStep::Ready => return Startup::Ready,
            crate::startup_stall::StartupStep::Failed(why) => {
                let reason = match why {
                    crate::startup_stall::StartupFail::ProcessGone => "the engine process exited".to_string(),
                    crate::startup_stall::StartupFail::LogStalled => format!(
                        "no log output and no CPU work for {} s; goethed stopped it",
                        crate::startup_stall::LOG_STALL_SECS
                    ),
                    crate::startup_stall::StartupFail::HardCap => {
                        format!("not ready after {hard_cap} s (GOETHE_HEALTH_POLL_TIMEOUT_SEC); goethed stopped it")
                    }
                };
                append_goethed_log(&s.home, &format!("{}: startup failed: {reason}\n", s.instance_id));
                return Startup::Failed;
            }
            crate::startup_stall::StartupStep::Continue => thread::sleep(Duration::from_millis(500)),
        }
    }
}

pub(super) fn tree_cpu_ticks(pids: &[i32]) -> u64 {
    // CPU time of the engine's processes and their reaped children: kernel builds show up here while the log is quiet.
    pids.iter().filter_map(|p| crate::sys::cpu_ticks(*p)).sum()
}

pub(super) fn learn_overhead(s: &Starting) {
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
    learn_weights(s);
    let Some(overhead) = calc::learned_overhead(used, &s.need, s.learned.overhead_gb) else {
        return;
    };
    if let Ok(con) = db::open(&s.home) {
        let model = &s.params.model_id;
        let _ = crate::engine_overhead::store_learned(&con, model, &s.gpu_name, s.tp, Kind::Overhead, overhead, &crate::engine_overhead::card_runtime(&s.card));
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

pub(super) fn learn_weights(s: &Starting) {
    // Weights larger than estimated leave the KV pools short of their tokens: learn the difference for the next load.
    let log = crate::engine_overhead::read_log_since(&s.log_path, s.log_offset);
    if let (Some(pre), Ok(con)) = (crate::engine_overhead::parse_pre_load_gb(&log), db::open(&s.home)) {
        let context = s.plan_free - pre;
        if (0.05..8.0).contains(&context) {
            let _ = crate::engine_overhead::store_context(&con, &s.gpu_name, s.tp, context);
        }
    }
    let Some(loaded) = crate::engine_overhead::parse_weight_gb(&log) else {
        return;
    };
    let extra = loaded - s.need.weights;
    if extra <= 0.25 {
        return;
    }
    if let Ok(con) = db::open(&s.home) {
        let learned = s.learned.static_gb + extra;
        let _ = crate::engine_overhead::store_learned(&con, &s.params.model_id, &s.gpu_name, s.tp, Kind::Static, learned, &crate::engine_overhead::card_runtime(&s.card));
    }
    append_goethed_log(
        &s.home,
        &format!("{}: weights take {loaded:.2} GB per GPU, {extra:.2} GB more than estimated; learned for the next load\n", s.instance_id),
    );
}

pub(super) fn mark_running(s: &Starting) -> Result<bool, String> {
    // STARTING to RUNNING and the desired slot in one transaction, so a concurrent unload can never resurrect it; Ok(false) when the row left STARTING (an unload won).
    let mut con = db::open(&s.home).map_err(|e| e.to_string())?;
    let tx = con.transaction_with_behavior(TransactionBehavior::Immediate).map_err(|e| e.to_string())?;
    let changed = tx
        .execute(
            "UPDATE loaded_instances SET status = 'RUNNING' WHERE instance_id = ?1 AND status = 'STARTING'",
            [&s.instance_id],
        )
        .map_err(|e| e.to_string())?;
    if changed == 1 {
        autostart::mark_desired(&tx, &s.params).map_err(|e| format!("desired load: {e}"))?;
    }
    tx.commit().map_err(|e| e.to_string())?;
    crate::client::link::state_changed();
    Ok(changed == 1)
}

pub(super) fn cancel_engine(s: &Starting) {
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

pub(super) fn fail_engine(s: &Starting) {
    // A start that cannot succeed: stop the engine (verified) and leave a FAILED tombstone for ps.
    if let Ok(con) = db::open(&s.home) {
        settle(&con, s, "FAILED");
    }
}

pub(super) fn settle(con: &Connection, s: &Starting, status: &str) {
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
    crate::client::link::state_changed();
}

pub(super) fn stop_one(home: &Path, id: &Identity) -> bool {
    // Stop one engine (verified); its run entry goes only if nothing survived, so a survivor is reaped later.
    let gone = proc_ident::terminate(std::slice::from_ref(id)).survivors.is_empty();
    if gone {
        run_registry::unregister(home, &id.instance_id);
    } else {
        append_goethed_log(home, &format!("{} survived SIGKILL; run entry kept\n", id.instance_id));
    }
    gone
}

pub(super) fn wait_vram_back(gpus: &[u32], want_gb: f64) {
    // A failed engine's VRAM returns some seconds after its processes exit; wait up to a minute for the pre-load level.
    let deadline = Instant::now() + Duration::from_secs(60);
    while Instant::now() < deadline {
        let Ok(snap) = foreign::snapshot() else { return };
        let free = gpus.iter().map(|g| snap.iter().find(|x| x.index == *g).map_or(0.0, |x| x.free_mb as f64 / 1024.0));
        if free.fold(f64::INFINITY, f64::min) >= want_gb {
            return;
        }
        thread::sleep(Duration::from_millis(500));
    }
}

pub(super) fn refit(s: &mut Starting) -> Result<String, String> {
    // Respawn once at the best settings that fit with what was learned (pins kept), naming what was reduced.
    let stopped = proc_ident::terminate(std::slice::from_ref(&s.identity));
    foreign::wait_released(&s.gpus, &stopped.pids, GPU_RELEASE_WAIT);
    wait_vram_back(&s.gpus, s.plan_free - 1.0);
    let (ctx0, conc0) = (s.ctx, s.max_conc);
    let (tight, free) = tightest_free(&s.gpus)?;
    let ceiling = ChooserInput { ctx: ctx0, max_conc: conc0, ..s.input.clone() };
    let budget = |t: &ChooserInput| chooser::with_learned(chooser::need(t, s.tp as usize, &s.cc), s.learned);
    let Some((ctx, conc, need)) = chooser::settle(&ceiling, |t| {
        let need = budget(t);
        shortfall(&need, tight, free).is_none().then_some((t.ctx, t.max_conc, need))
    }) else {
        return Err(shortfall(&budget(&chooser::bottom(&ceiling)), tight, free).unwrap_or_else(|| "no settings fit".to_string()));
    };
    (s.ctx, s.max_conc) = (ctx, conc);
    if let Err(e) = respawn(s, need, false) {
        (s.ctx, s.max_conc) = (ctx0, conc0);
        return Err(e);
    }
    s.need = need;
    if let Ok(con) = db::open(&s.home) {
        let _ = con.execute(
            "UPDATE loaded_instances SET ctx_len = ?2, max_conc = ?3 WHERE instance_id = ?1",
            rusqlite::params![s.instance_id, s.ctx as i64, s.max_conc as i64],
        );
    }
    let mut reduced = Vec::new();
    if s.ctx < ctx0 {
        reduced.push(format!("context {}", s.ctx));
    }
    if s.max_conc < conc0 {
        reduced.push(format!("{} parallel requests", s.max_conc));
    }
    Ok(if reduced.is_empty() { String::new() } else { format!(" at {}", reduced.join(", ")) })
}

fn tightest_free(gpus: &[u32]) -> Result<(u32, f64), String> {
    // The GPU of the set with the least free VRAM now, and that free figure in GB. O(gpus * visible GPUs).
    let snap = foreign::snapshot()?;
    let free_of = |g: u32| snap.iter().find(|x| x.index == g).map_or(0.0, |x| x.free_mb as f64 / 1024.0);
    gpus.iter().map(|g| (*g, free_of(*g))).min_by(|a, b| a.1.total_cmp(&b.1)).ok_or_else(|| "no GPU".to_string())
}

pub(super) fn respawn(s: &mut Starting, need: Need, move_port: bool) -> Result<(), String> {
    // Stop the current engine (verified) and start it again under the same instance id, with a new budget or port.
    let stopped = proc_ident::terminate(std::slice::from_ref(&s.identity));
    foreign::wait_released(&s.gpus, &stopped.pids, GPU_RELEASE_WAIT);
    if !stopped.survivors.is_empty() {
        return Err("the previous engine survived SIGKILL; not respawning over it".to_string());
    }
    let (tight, free) = tightest_free(&s.gpus)?;
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
    let context = db::open(&s.home).map_or(crate::engine_overhead::DEFAULT_CONTEXT_GB, |c| {
        crate::engine_overhead::read_context(&c, &s.gpu_name, s.tp)
    });
    let mem_frac = mem_fraction_from_free(&need, free, context)?;
    s.plan_free = free;
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
        pool_tokens: chooser::kv_tokens(s.ctx, s.max_conc, s.input.floor),
        numerics: s.params.numerics,
        tp: s.tp,
        mem_frac,
        gpus: &s.gpus,
        cc: &s.cc,
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
