// Unload, graceful shutdown, ps listing and start-of-day reconcile.

use super::{
    append_goethed_log, insts, next_attempts, owns_port, parse_gpu_csv, serves, Inst, LIVE,
    GPU_RELEASE_WAIT,
};
use crate::coordinator;
use crate::db;
use crate::foreign;
use crate::load_slot;
use crate::paths::goethe_home;
use crate::proc_ident::{self, Identity};
use crate::run_registry;
use rusqlite::{Connection, TransactionBehavior};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

pub(super) fn stop_processes(home: &Path, list: &[Inst]) -> HashSet<String> {
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
        append_goethed_log(home, &format!("teardown: pids {:?} still listed on GPUs {gpus:?} after {}s\n", stopped.pids, GPU_RELEASE_WAIT.as_secs()));
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

pub(super) fn teardown(con: &Connection, home: &Path, list: &[Inst]) {
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

pub(super) fn drain_inflight(pending: impl Fn() -> usize, home: &Path) {
    // Wait for the in-flight mux requests `pending` counts to finish, bounded by GOETHE_DRAIN_SECS.
    let max_secs: u64 = std::env::var("GOETHE_DRAIN_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let deadline = Instant::now() + Duration::from_secs(max_secs);
    loop {
        let pending = pending();
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target<'a> {
    All,
    Model(&'a str),
    Instance(&'a str),
}

pub(super) fn claim_for_unload(con: &mut Connection, target: Target, next: &str) -> rusqlite::Result<Vec<Inst>> {
    // One immediate transaction: drop the desired slots and tombstones of the target, collect its live rows, mark exactly those DRAINING/STOPPING.
    let tx = con.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let (model, instance) = match target {
        Target::All => (None, None),
        Target::Model(m) => (Some(m), None),
        Target::Instance(i) => (None, Some(i)),
    };
    let only = "(?1 IS NULL OR model_id = ?1) AND (?2 IS NULL OR instance_id = ?2)";
    tx.execute(
        "DELETE FROM desired_loads WHERE (?1 IS NULL OR model_id = ?1)
           AND (?2 IS NULL OR slot_key = (SELECT COALESCE(slot_key, model_id) FROM loaded_instances WHERE instance_id = ?2))",
        rusqlite::params![model, instance],
    )?;
    tx.execute(
        &format!(
            "DELETE FROM loaded_instances
             WHERE (status IN ('FAILED', 'CANCELLED') OR (status = 'RESTARTING' AND pid = 0)) AND {only}"
        ),
        rusqlite::params![model, instance],
    )?;
    let list = insts(&tx, &format!("status IN {LIVE} AND {only}"), &[&model, &instance])?;
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
    // Every instance of one model, or of all models.
    match (all, model_id.as_deref()) {
        (true, _) => unload(Target::All, force),
        (false, Some(m)) => unload(Target::Model(m), force),
        (false, None) => Err((2, "model_id required".to_string())),
    }
}

pub fn handle_unload_instance(instance_id: &str, force: bool) -> Result<Value, (i32, String)> {
    // One instance; the model's other instances keep serving.
    unload(Target::Instance(instance_id), force)
}

fn unload(target: Target, force: bool) -> Result<Value, (i32, String)> {
    // Clear desired slots first (even with no instances), then drain unless forced and stop every match (verified).
    let home = goethe_home();
    let t0 = Instant::now();
    match target {
        Target::All => coordinator::cancel_queued(None),
        Target::Model(m) => coordinator::cancel_queued(Some(m)),
        Target::Instance(_) => {}
    }
    let mut con = db::open(&home).map_err(|e| (1, e.to_string()))?;
    let list = {
        let _plan = coordinator::plan_lock();
        claim_for_unload(&mut con, target, if force { "STOPPING" } else { "DRAINING" })
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
        if let Target::Instance(id) = target {
            drain_inflight(|| crate::inflight::instance_count(id), &home);
        } else {
            let models: HashSet<&str> = list.iter().map(|i| i.model_id.as_str()).collect();
            drain_inflight(|| models.iter().map(|m| crate::inflight::count(m)).sum(), &home);
        }
        for i in &list {
            let _ = con.execute(
                "UPDATE loaded_instances SET status = 'STOPPING' WHERE instance_id = ?1 AND status = 'DRAINING'",
                [&i.instance_id],
            );
        }
    }
    teardown(&con, &home, &list);
    crate::platform::state_changed();
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
            v["runtime"] = json!(crate::engine_overhead::model_runtime(&home, &mid));
        }
        if let Some(id) = v.get("instance_id").and_then(|x| x.as_str()) {
            let path = home.join("logs").join(format!("{id}.requests.jsonl"));
            if let Some(done) = crate::engines::ninfer::last_request_done(&path) {
                v["prefix_cache_hit_tokens"] = json!(done.prefix_cache_hit_tokens);
                v["prompt_tokens"] = json!(done.prompt_tokens);
                v["computed_prefill_tokens"] = json!(done.computed_prefill_tokens);
                v["tool_call_count"] = json!(done.tool_call_count);
            }
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

