// Daemon-wide load coordination: one planning lock, a VRAM ledger of STARTING engines and a per-GPU startup gate.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

#[derive(Default)]
struct State {
    starting: HashMap<String, (Vec<u32>, f64)>,
    queued: HashMap<u64, (String, Arc<AtomicBool>)>,
    next_ticket: u64,
}

static PLAN: Mutex<()> = Mutex::new(());
static CHANGED: Condvar = Condvar::new();

fn state() -> MutexGuard<'static, State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE
        .get_or_init(|| Mutex::new(State::default()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

pub fn plan_lock() -> MutexGuard<'static, ()> {
    PLAN.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn reserved_gb(gpu: u32) -> f64 {
    state()
        .starting
        .values()
        .filter(|(gpus, _)| gpus.contains(&gpu))
        .map(|(_, gb)| *gb)
        .sum()
}

pub fn busy(gpus: &[u32]) -> bool {
    state()
        .starting
        .values()
        .any(|(held, _)| held.iter().any(|g| gpus.contains(g)))
}

pub fn reserve(instance_id: &str, gpus: &[u32], gb_per_gpu: f64) {
    state()
        .starting
        .insert(instance_id.to_string(), (gpus.to_vec(), gb_per_gpu));
}

pub fn release(instance_id: &str) {
    state().starting.remove(instance_id);
    CHANGED.notify_all();
}

pub fn wait_change(timeout: Duration) {
    let guard = state();
    let _ = CHANGED.wait_timeout(guard, timeout);
}

pub struct Ticket {
    id: u64,
    cancel: Arc<AtomicBool>,
}

impl Ticket {
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        state().queued.remove(&self.id);
    }
}

pub fn enqueue(model_id: &str) -> Ticket {
    // Register a load that may wait for the gate so unload and shutdown can cancel it.
    let mut st = state();
    st.next_ticket += 1;
    let id = st.next_ticket;
    let cancel = Arc::new(AtomicBool::new(false));
    st.queued.insert(id, (model_id.to_string(), cancel.clone()));
    Ticket { id, cancel }
}

pub fn cancel_queued(model_id: Option<&str>) {
    // Cancel queued loads of one model (or all) and wake them so they return.
    for (mid, flag) in state().queued.values() {
        if model_id.is_none_or(|m| m == mid) {
            flag.store(true, Ordering::SeqCst);
        }
    }
    CHANGED.notify_all();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_counts_starting_reservations_per_gpu() {
        reserve("inst-ledger-a", &[90, 91], 10.0);
        reserve("inst-ledger-b", &[91], 2.5);
        assert!((reserved_gb(90) - 10.0).abs() < 1e-9);
        assert!((reserved_gb(91) - 12.5).abs() < 1e-9);
        assert!(busy(&[91]) && busy(&[89, 90]) && !busy(&[92]));
        reserve("inst-ledger-a", &[90, 91], 14.0);
        assert!((reserved_gb(90) - 14.0).abs() < 1e-9);
        release("inst-ledger-a");
        assert_eq!(reserved_gb(90), 0.0);
        assert!(!busy(&[90]));
        assert!((reserved_gb(91) - 2.5).abs() < 1e-9);
        release("inst-ledger-b");
        assert!(!busy(&[91]));
    }

    #[test]
    fn cancel_queued_hits_only_matching_model_and_drop_dequeues() {
        let a = enqueue("model-q-a");
        let b = enqueue("model-q-b");
        cancel_queued(Some("model-q-a"));
        assert!(a.cancelled() && !b.cancelled());
        let id = b.id;
        drop(b);
        assert!(!state().queued.contains_key(&id));
    }
}
