// goethed heartbeat task: runs each step on the blocking pool, waits on a timer or a change signal, never on the request path.

use goethe_core::heartbeat::{Outcome, Reason, Worker, DEBOUNCE, MIN_GAP};
use goethe_core::platform::{self, CHANGE_LINK};
use goethe_core::snapshot::{self, Collector};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Notify;
use tokio::time::{sleep_until, Instant};

pub fn spawn(home: PathBuf) {
    // Install the change hook (a Notify permit, never blocking) and start the loop.
    let notify = Arc::new(Notify::new());
    let hook = notify.clone();
    platform::set_change_hook(Box::new(move || hook.notify_one()));
    let started_at = snapshot::now_rfc3339();
    tokio::spawn(run(home, notify, started_at));
}

fn fresh(home: &std::path::Path, started_at: &str) -> Worker {
    Worker::new(home.to_path_buf(), Collector::new(started_at.to_string()))
}

async fn run(home: PathBuf, notify: Arc<Notify>, started_at: String) {
    // One step at a time; every wake waits out the 2 s gap or the server's Retry-After, and a state change the 1 s debounce too.
    let mut worker = Some(fresh(&home, &started_at));
    let mut reason = Reason::Start;
    loop {
        let w = worker.take().unwrap_or_else(|| fresh(&home, &started_at));
        let (next, not_before) = match tokio::task::spawn_blocking(move || {
            let mut w = w;
            let (outcome, delay) = w.step(reason);
            (w, outcome, delay)
        })
        .await
        {
            Ok((w, outcome, delay)) => {
                worker = Some(w);
                let gap = match outcome {
                    Outcome::RateLimited(wait) => wait.max(MIN_GAP),
                    _ => MIN_GAP,
                };
                (Instant::now() + delay, Instant::now() + gap)
            }
            Err(_) => (Instant::now() + Duration::from_secs(60), Instant::now() + Duration::from_secs(60)),
        };
        reason = loop {
            tokio::select! {
                _ = sleep_until(next) => break Reason::Interval,
                _ = notify.notified() => {
                    let kinds = platform::take_pending();
                    if kinds == 0 {
                        continue;
                    }
                    let wait = if kinds & CHANGE_LINK != 0 { not_before } else { (Instant::now() + DEBOUNCE).max(not_before) };
                    sleep_until(wait).await;
                    let kinds = kinds | platform::take_pending();
                    break if kinds & CHANGE_LINK != 0 { Reason::Start } else { Reason::Change };
                }
            }
        };
    }
}
