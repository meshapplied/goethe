// What the node reports on GET /state: hello, then the live load of its instances, which goethe-server routes on (CONTRACTS 4.3). Telemetry goes to the console on the heartbeat, not here.

use super::serve::line;
use goethe_proto::{Instance, StateEvent, STATE_SCHEMA};
use hyper::body::Bytes;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::mpsc;
use tokio::time::MissedTickBehavior;

const LIVE_TICK: Duration = Duration::from_millis(250);
const GAUGE_EVERY: Duration = Duration::from_secs(5);
const GAUGE_TIMEOUT: Duration = Duration::from_secs(1);
const LIVE_STATUSES: &str = "('STARTING', 'RUNNING', 'RESTARTING', 'DRAINING', 'STOPPING')";

pub struct Node {
    pub home: PathBuf,
    hello: Bytes,
}

type Gauges = HashMap<String, (Option<u64>, Option<f64>)>;

#[derive(Default)]
struct Live {
    con: Option<rusqlite::Connection>,
    last: Option<Vec<Instance>>,
    gauges: Gauges,
    polled: Option<Instant>,
}

impl Node {
    pub fn new(home: PathBuf, node_id: &str, install_hash: &str) -> Node {
        let hello = StateEvent::Hello {
            schema: STATE_SCHEMA.into(),
            node_id: node_id.into(),
            install_hash: install_hash.into(),
            goethe_version: env!("CARGO_PKG_VERSION").into(),
            proto: 1,
        };
        Node { home, hello: line(&hello) }
    }
}

pub(super) fn open(node: Arc<Node>) -> mpsc::Receiver<Bytes> {
    let (tx, rx) = mpsc::channel(8);
    tokio::spawn(feed(node, tx));
    rx
}

async fn feed(node: Arc<Node>, tx: mpsc::Sender<Bytes>) {
    // hello, then the instances whenever they change, at most 4 times a second, until the stream closes.
    if tx.send(node.hello.clone()).await.is_err() {
        return;
    }
    let mut live = Some(Live::default());
    let mut tick = tokio::time::interval(LIVE_TICK);
    tick.set_missed_tick_behavior(MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            _ = tick.tick() => {}
            _ = tx.closed() => return,
        }
        let (home, l) = (node.home.clone(), live.take().unwrap_or_default());
        let Ok((l, out)) = tokio::task::spawn_blocking(move || step(&home, l)).await else {
            continue;
        };
        live = Some(l);
        if let Some(out) = out {
            if tx.send(out).await.is_err() {
                return;
            }
        }
    }
}

fn step(home: &Path, mut live: Live) -> (Live, Option<Bytes>) {
    // Blocking: read the instances, refresh the SGLang gauges every 5 s, and a live event only when something changed.
    if live.con.is_none() {
        live.con = crate::db::open(home).ok();
    }
    let Some(con) = live.con.as_ref() else {
        return (live, None);
    };
    let rows = instances(con);
    if live.polled.is_none_or(|t| t.elapsed() >= GAUGE_EVERY) {
        live.gauges = poll_gauges(home, &rows);
        live.polled = Some(Instant::now());
    }
    let now: Vec<Instance> = rows
        .into_iter()
        .map(|r| {
            let (queued, kv_cache_pct) = live.gauges.get(&r.instance.instance_id).copied().unwrap_or_default();
            Instance { queued, kv_cache_pct, ..r.instance }
        })
        .collect();
    if live.last.as_ref() == Some(&now) {
        return (live, None);
    }
    let t_ms = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let out = line(&StateEvent::Live { t_ms, instances: now.clone() });
    live.last = Some(now);
    (live, Some(out))
}

struct Row {
    instance: Instance,
    port: u16,
    engine_key: Option<String>,
    card: Option<String>,
}

fn instances(con: &rusqlite::Connection) -> Vec<Row> {
    // Live instances with their aliases, in-flight count and max concurrency; port, engine key and card stay local for the gauge poll.
    let mut aliases: HashMap<String, Vec<String>> = HashMap::new();
    let pairs: Vec<(String, String)> = con
        .prepare("SELECT alias, model_id FROM model_aliases ORDER BY alias")
        .and_then(|mut s| s.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?.collect())
        .unwrap_or_default();
    for (alias, model) in pairs {
        aliases.entry(model).or_default().push(alias);
    }
    let sql = format!(
        "SELECT l.instance_id, l.model_id, l.status, l.max_conc, l.internal_port, l.engine_key, i.local_path FROM loaded_instances l
         LEFT JOIN installed_models i ON i.model_id = l.model_id WHERE l.status IN {LIVE_STATUSES} ORDER BY l.rowid"
    );
    let row = |r: &rusqlite::Row| -> rusqlite::Result<Row> {
        let (id, model): (String, String) = (r.get(0)?, r.get(1)?);
        Ok(Row {
            instance: Instance {
                in_flight: crate::inflight::instance_count(&id) as u64,
                aliases: aliases.get(&model).cloned().unwrap_or_default(),
                instance_id: id,
                model_id: model,
                status: r.get(2)?,
                max_conc: r.get::<_, i64>(3)?.max(0) as u64,
                queued: None,
                kv_cache_pct: None,
            },
            port: u16::try_from(r.get::<_, i64>(4)?).unwrap_or(0),
            engine_key: r.get::<_, Option<String>>(5)?.filter(|k| !k.is_empty()),
            card: r.get(6)?,
        })
    };
    con.prepare(&sql).and_then(|mut s| s.query_map([], row)?.collect()).unwrap_or_default()
}

fn poll_gauges(home: &Path, rows: &[Row]) -> Gauges {
    // Queue length and KV cache use from each running SGLang instance's /metrics, 1 s at most each; any failure leaves that instance without gauges.
    let agent = ureq::AgentBuilder::new().timeout(GAUGE_TIMEOUT).build();
    let sglang = |local: &str| {
        let text = std::fs::read_to_string(home.join(local).join("goethe_card.toml")).ok()?;
        let card = toml::from_str::<toml::Value>(&text).ok()?;
        (crate::engines::of_card(&card).ok()?.id == crate::engines::sglang::SGLANG.id).then_some(())
    };
    rows.iter()
        .filter(|r| r.instance.status == "RUNNING" && r.port > 0 && r.card.as_deref().and_then(sglang).is_some())
        .filter_map(|r| {
            let mut req = agent.get(&format!("http://127.0.0.1:{}/metrics", r.port));
            if let Some(k) = &r.engine_key {
                req = req.set("Authorization", &format!("Bearer {k}"));
            }
            let text = req.call().ok()?.into_string().ok()?;
            Some((r.instance.instance_id.clone(), gauges(&text)))
        })
        .collect()
}

fn gauges(text: &str) -> (Option<u64>, Option<f64>) {
    // Prometheus text: the queued requests summed over label sets, the highest token usage (a 0 to 1 fraction) as a percent.
    let (mut queued, mut usage) = (None::<f64>, None::<f64>);
    for l in text.lines().filter(|l| !l.starts_with('#')) {
        let (name, rest) = match l.rfind('}') {
            Some(i) => (l[..i].split('{').next().unwrap_or(""), &l[i + 1..]),
            None => l.split_once(' ').unwrap_or((l, "")),
        };
        let Some(v) = rest.split_whitespace().next().and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite()) else {
            continue;
        };
        match name.trim() {
            "sglang:num_queue_reqs" => queued = Some(queued.unwrap_or(0.0) + v),
            "sglang:token_usage" => usage = Some(usage.unwrap_or(0.0).max(v)),
            _ => {}
        }
    }
    (queued.map(|q| q.max(0.0) as u64), usage.map(|u| (u * 1000.0).round() / 10.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauges_sum_the_queue_and_take_the_fullest_cache() {
        let text = "# TYPE sglang:num_queue_reqs gauge\nsglang:num_queue_reqs{dp=\"0\"} 2\nsglang:num_queue_reqs{dp=\"1\"} 1.0\nsglang:token_usage{dp=\"0\"} 0.5\nsglang:token_usage{dp=\"1\"} 0.25 1700000000\nsglang:num_running_reqs 9\n";
        assert_eq!(gauges(text), (Some(3), Some(50.0)));
        assert_eq!(gauges("other 1\nsglang:token_usage NaN\n"), (None, None));
    }
}
