// In-memory gateway view of state.db (keys, routes, aliases, models, trace settings), reloaded when PRAGMA data_version moves.

use crate::db;
use crate::key_usage;
use crate::security::{token_hash, ResolvedKey};
use crate::trace::{self, TraceConfig};
use rusqlite::{Connection, Transaction, TransactionBehavior};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

const RECHECK: Duration = Duration::from_millis(250);
const BUSY_RETRY: Duration = Duration::from_millis(50);
const READ_BUSY_TIMEOUT: Duration = Duration::from_millis(200);

#[derive(Clone, Debug, PartialEq)]
pub struct Route {
    pub instance_id: String,
    pub port: u16,
    pub engine_key: Option<String>,
    pub draining: bool,
}

#[derive(Default)]
pub struct Snapshot {
    keys: HashMap<String, Arc<ResolvedKey>>,
    routes: HashMap<String, Vec<Route>>,
    aliases: HashMap<String, String>,
    models: HashMap<String, i64>,
    trace: Arc<TraceConfig>,
}

impl Snapshot {
    pub fn trace(&self) -> &TraceConfig {
        &self.trace
    }

    pub fn has_key(&self, key: &ResolvedKey) -> bool {
        self.keys.contains_key(&key.token_hash)
    }

    pub fn resolve_model<'a>(&'a self, name: &'a str) -> &'a str {
        // A real model id always wins over an alias spelled the same, so an alias can never hijack a model.
        if self.is_known(name) {
            return name;
        }
        self.aliases.get(name).map(String::as_str).unwrap_or(name)
    }

    pub fn routes(&self, model_id: &str) -> &[Route] {
        self.routes.get(model_id).map(Vec::as_slice).unwrap_or(&[])
    }

    pub fn is_known(&self, model_id: &str) -> bool {
        self.models.contains_key(model_id) || self.routes.contains_key(model_id)
    }

    pub fn created(&self, model_id: &str) -> i64 {
        self.models.get(model_id).copied().unwrap_or(0)
    }

    pub fn serving_models(&self) -> Vec<&str> {
        // Models with at least one non-draining instance, sorted for stable listings.
        let mut ids: Vec<&str> = self
            .routes
            .iter()
            .filter(|(_, rs)| rs.iter().any(|r| !r.draining))
            .map(|(id, _)| id.as_str())
            .collect();
        ids.sort_unstable();
        ids
    }
}

struct DbState {
    con: Connection,
    version: Option<i64>,
}

struct UsageRow {
    key_id: String,
    prompt: i64,
    completion: i64,
}

pub struct GatewayCache {
    home: PathBuf,
    db: Arc<Mutex<Option<DbState>>>,
    snap: RwLock<Arc<Snapshot>>,
    epoch: Instant,
    checked_ms: AtomicU64,
    usage_tx: mpsc::Sender<UsageRow>,
}

impl GatewayCache {
    pub fn open(home: &Path) -> GatewayCache {
        // Cache over home/state.db plus a group-commit writer thread for key usage on the same connection.
        let db = Arc::new(Mutex::new(None));
        let (tx, rx) = mpsc::channel();
        let writer_db = Arc::clone(&db);
        let writer_home = home.to_path_buf();
        let _ = std::thread::Builder::new()
            .name("goethe-usage".into())
            .spawn(move || usage_writer(&writer_home, &writer_db, rx));
        GatewayCache {
            home: home.to_path_buf(),
            db,
            snap: RwLock::new(Arc::new(Snapshot::default())),
            epoch: Instant::now(),
            checked_ms: AtomicU64::new(0),
            usage_tx: tx,
        }
    }

    pub fn current(&self) -> Arc<Snapshot> {
        self.refresh(false);
        self.snapshot()
    }

    pub fn exact(&self) -> Arc<Snapshot> {
        self.refresh(true);
        self.snapshot()
    }

    pub fn resolve_token(&self, token: &str) -> Option<Arc<ResolvedKey>> {
        // A miss rechecks the database so a key created a moment ago works on its first request.
        let hash = token_hash(token);
        if let Some(k) = self.current().keys.get(&hash) {
            return Some(Arc::clone(k));
        }
        self.exact().keys.get(&hash).cloned()
    }

    pub fn record_usage(&self, key_id: &str, prompt: i64, completion: i64) {
        let row = UsageRow {
            key_id: key_id.to_string(),
            prompt,
            completion,
        };
        let _ = self.usage_tx.send(row);
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        Arc::clone(&self.snap.read().unwrap_or_else(|e| e.into_inner()))
    }

    fn refresh(&self, force: bool) {
        // Reload when another connection committed; unforced checks run at most every RECHECK.
        let now = self.epoch.elapsed().as_millis() as u64 + 1;
        let last = self.checked_ms.load(Ordering::Relaxed);
        if !force && last != 0 && now.saturating_sub(last) < RECHECK.as_millis() as u64 {
            return;
        }
        self.checked_ms.store(now, Ordering::Relaxed);
        let mut guard = self.db.lock().unwrap_or_else(|e| e.into_inner());
        open_if_needed(&mut guard, &self.home);
        let Some(state) = guard.as_mut() else {
            return;
        };
        let version = match state.con.query_row("PRAGMA data_version", [], |r| r.get::<_, i64>(0)) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("goethe mux: state.db version check failed: {e}");
                guard.take();
                return;
            }
        };
        if state.version == Some(version) {
            return;
        }
        match load_snapshot(&state.con) {
            Ok(snap) => {
                let mut current = self.snap.write().unwrap_or_else(|e| e.into_inner());
                let _ = std::mem::replace(&mut *current, Arc::new(snap));
                state.version = Some(version);
            }
            Err(e) => eprintln!("goethe mux: state.db reload failed: {e}"),
        }
    }
}

fn open_if_needed(slot: &mut Option<DbState>, home: &Path) {
    // Open the shared connection lazily; a short busy timeout keeps request threads from stalling on writers.
    if slot.is_some() {
        return;
    }
    match db::open(home) {
        Ok(con) => {
            let _ = con.busy_timeout(READ_BUSY_TIMEOUT);
            slot.replace(DbState { con, version: None });
        }
        Err(e) => eprintln!("goethe mux: open state.db failed: {e}"),
    }
}

fn load_snapshot(con: &Connection) -> rusqlite::Result<Snapshot> {
    // Read keys, RUNNING/DRAINING instances, aliases and installed models in one read transaction.
    let tx = con.unchecked_transaction()?;
    let mut snap = Snapshot::default();
    let mut stmt = tx.prepare(
        "SELECT token_hash, key_id, rpm_limit, expires_at, model_scope, is_admin FROM api_keys",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            ResolvedKey {
                token_hash: r.get(0)?,
                key_id: r.get(1)?,
                rpm_limit: r.get::<_, Option<i64>>(2)?.unwrap_or(120),
                expires_at: r.get(3)?,
                model_scope: r.get(4)?,
                is_admin: r.get::<_, i64>(5)? != 0,
            },
        ))
    })?;
    for row in rows {
        let (hash, key) = row?;
        snap.keys.insert(hash, Arc::new(key));
    }
    let mut stmt = tx.prepare(
        "SELECT instance_id, model_id, internal_port, status, engine_key FROM loaded_instances
         WHERE status IN ('RUNNING', 'DRAINING') ORDER BY instance_id",
    )?;
    let rows = stmt.query_map([], |r| {
        Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
            r.get::<_, i64>(2)?,
            r.get::<_, String>(3)?,
            r.get::<_, Option<String>>(4)?,
        ))
    })?;
    for row in rows {
        let (instance_id, model_id, port, status, engine_key) = row?;
        let Ok(port) = u16::try_from(port) else {
            continue;
        };
        snap.routes.entry(model_id).or_default().push(Route {
            instance_id,
            port,
            engine_key: engine_key.filter(|k| !k.is_empty()),
            draining: status == "DRAINING",
        });
    }
    let mut stmt = tx.prepare("SELECT alias, model_id FROM model_aliases")?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
    for row in rows {
        let (alias, model_id) = row?;
        snap.aliases.insert(alias, model_id);
    }
    let mut stmt = tx.prepare(
        "SELECT model_id, COALESCE(CAST(strftime('%s', installed_at) AS INTEGER), 0) FROM installed_models",
    )?;
    let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))?;
    for row in rows {
        let (model_id, created) = row?;
        snap.models.insert(model_id, created);
    }
    snap.trace = Arc::new(trace::load_config(&tx));
    Ok(snap)
}

fn usage_writer(home: &Path, db: &Mutex<Option<DbState>>, rx: mpsc::Receiver<UsageRow>) {
    // Group commit: drain everything queued, write it in one transaction, repeat; busy databases are retried.
    let mut pending: HashMap<String, (i64, i64, i64)> = HashMap::new();
    while let Ok(first) = rx.recv() {
        for row in std::iter::once(first).chain(rx.try_iter()) {
            let e = pending.entry(row.key_id).or_insert((0, 0, 0));
            e.0 += 1;
            e.1 += row.prompt;
            e.2 += row.completion;
        }
        loop {
            match write_usage(home, db, &pending) {
                Ok(()) => {
                    pending.clear();
                    break;
                }
                Err(e) if e.contains("locked") || e.contains("busy") => std::thread::sleep(BUSY_RETRY),
                Err(e) => {
                    eprintln!("goethe mux: key usage write failed, dropping {} key(s): {e}", pending.len());
                    pending.clear();
                    break;
                }
            }
        }
    }
}

fn write_usage(home: &Path, db: &Mutex<Option<DbState>>, pending: &HashMap<String, (i64, i64, i64)>) -> Result<(), String> {
    // Take the write lock without waiting so request threads never queue behind another process's writer.
    let mut guard = db.lock().unwrap_or_else(|e| e.into_inner());
    open_if_needed(&mut guard, home);
    let state = guard.as_mut().ok_or("state.db unavailable")?;
    let _ = state.con.busy_timeout(Duration::ZERO);
    let begun = Transaction::new_unchecked(&state.con, TransactionBehavior::Immediate);
    let _ = state.con.busy_timeout(READ_BUSY_TIMEOUT);
    let tx = begun.map_err(|e| e.to_string())?;
    for (key_id, (requests, prompt, completion)) in pending {
        key_usage::add_usage(&tx, key_id, *requests, *prompt, *completion)?;
    }
    tx.commit().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn insert_key(con: &Connection, key_id: &str, token: &str) {
        con.execute(
            "INSERT INTO api_keys (key_id, name, token_hash, raw_prefix, rpm_limit) VALUES (?1, 'n', ?2, 'p', 5)",
            rusqlite::params![key_id, token_hash(token)],
        )
        .unwrap();
    }

    fn insert_instance(con: &Connection, id: &str, model: &str, port: i64, status: &str, key: Option<&str>) {
        con.execute(
            "INSERT INTO loaded_instances (instance_id, model_id, pid, pgid, internal_port, gpu_indices,
             allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status, engine_key)
             VALUES (?1, ?2, 1, 1, ?3, '0', 1.0, 0.5, 512, 2, ?4, ?5)",
            rusqlite::params![id, model, port, status, key],
        )
        .unwrap();
    }

    #[test]
    fn revocation_invalidates_the_cache() {
        let dir = tempfile::tempdir().unwrap();
        let other = db::open(dir.path()).unwrap();
        insert_key(&other, "k1", "goethe_sk_one");
        let cache = GatewayCache::open(dir.path());
        assert_eq!(cache.resolve_token("goethe_sk_one").unwrap().key_id, "k1");
        assert!(cache.resolve_token("goethe_sk_two").is_none());
        insert_key(&other, "k2", "goethe_sk_two");
        assert_eq!(cache.resolve_token("goethe_sk_two").unwrap().rpm_limit, 5, "a miss rechecks at once");
        other.execute("DELETE FROM api_keys WHERE key_id = 'k1'", []).unwrap();
        std::thread::sleep(RECHECK + Duration::from_millis(20));
        assert!(cache.resolve_token("goethe_sk_one").is_none(), "revoked key must be refused after one recheck");
        let two = cache.resolve_token("goethe_sk_two").unwrap();
        assert!(cache.exact().has_key(&two));
        other.execute("DELETE FROM api_keys WHERE key_id = 'k2'", []).unwrap();
        assert!(!cache.exact().has_key(&two), "an exact snapshot sees a revocation at once");
    }

    #[test]
    fn routes_track_status_and_engine_keys() {
        let dir = tempfile::tempdir().unwrap();
        let other = db::open(dir.path()).unwrap();
        insert_instance(&other, "inst-a", "org/m", 30001, "RUNNING", Some("ek1"));
        insert_instance(&other, "inst-b", "org/m", 30002, "RUNNING", None);
        insert_instance(&other, "inst-c", "other", 30003, "STARTING", None);
        other
            .execute("INSERT INTO model_aliases (alias, model_id) VALUES ('chat', 'org/m')", [])
            .unwrap();
        let cache = GatewayCache::open(dir.path());
        let snap = cache.exact();
        assert_eq!(snap.resolve_model("chat"), "org/m");
        assert_eq!(snap.routes("org/m").len(), 2);
        assert_eq!(snap.routes("org/m")[0].engine_key.as_deref(), Some("ek1"));
        assert_eq!(snap.routes("org/m")[1].engine_key, None);
        assert!(snap.routes("other").is_empty(), "STARTING instances are not routable");
        assert_eq!(snap.serving_models(), vec!["org/m"]);
        other
            .execute("UPDATE loaded_instances SET status = 'DRAINING' WHERE model_id = 'org/m'", [])
            .unwrap();
        let snap = cache.exact();
        assert!(snap.routes("org/m").iter().all(|r| r.draining));
        assert!(snap.serving_models().is_empty());
    }

    #[test]
    fn model_ids_win_over_aliases_with_the_same_name() {
        let dir = tempfile::tempdir().unwrap();
        let other = db::open(dir.path()).unwrap();
        insert_instance(&other, "inst-a", "org/big", 30001, "RUNNING", None);
        insert_instance(&other, "inst-b", "org/small", 30002, "RUNNING", None);
        other
            .execute("INSERT INTO model_aliases (alias, model_id) VALUES ('org/small', 'org/big')", [])
            .unwrap();
        let snap = GatewayCache::open(dir.path()).exact();
        assert_eq!(snap.resolve_model("org/small"), "org/small");
        assert_eq!(snap.resolve_model("org/big"), "org/big");
    }

    #[test]
    fn usage_writes_are_batched_and_do_not_invalidate() {
        let dir = tempfile::tempdir().unwrap();
        let cache = GatewayCache::open(dir.path());
        let _ = cache.exact();
        for _ in 0..5 {
            cache.record_usage("k9", 2, 3);
        }
        let other = db::open(dir.path()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let (req, pt, ct, ..) = key_usage::totals_for_key(&other, "k9").unwrap();
            if req == 5 {
                assert_eq!((pt, ct), (10, 15));
                break;
            }
            assert!(Instant::now() < deadline, "usage never flushed");
            std::thread::sleep(Duration::from_millis(10));
        }
        let guard = cache.db.lock().unwrap();
        let state = guard.as_ref().unwrap();
        let v: i64 = state.con.query_row("PRAGMA data_version", [], |r| r.get(0)).unwrap();
        assert_eq!(state.version, Some(v), "own usage writes must not force a reload");
    }
}
