// Per-slot load/restart cancellation and instance notices for the CLI.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone)]
struct SlotState {
    in_flight: Arc<AtomicU32>,
    done: Arc<(Mutex<bool>, Condvar)>,
}

static SLOTS: OnceLock<Mutex<HashMap<String, SlotState>>> = OnceLock::new();
static CANCEL: OnceLock<Mutex<HashMap<String, Arc<AtomicBool>>>> = OnceLock::new();
static NOTICES: OnceLock<Mutex<HashMap<String, Vec<String>>>> = OnceLock::new();

fn slots() -> &'static Mutex<HashMap<String, SlotState>> {
    SLOTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cancels() -> &'static Mutex<HashMap<String, Arc<AtomicBool>>> {
    CANCEL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn notices_map() -> &'static Mutex<HashMap<String, Vec<String>>> {
    NOTICES.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cancel_flag(slot_key: &str) -> Arc<AtomicBool> {
    let mut g = cancels().lock().unwrap_or_else(|e| e.into_inner());
    g.entry(slot_key.to_string())
        .or_insert_with(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

pub fn begin_slot_load(slot_key: &str) -> Arc<AtomicBool> {
    let flag = cancel_flag(slot_key);
    flag.store(false, Ordering::SeqCst);
    let mut g = slots().lock().unwrap_or_else(|e| e.into_inner());
    g.entry(slot_key.to_string())
        .or_insert_with(|| SlotState {
            in_flight: Arc::new(AtomicU32::new(0)),
            done: Arc::new((Mutex::new(false), Condvar::new())),
        });
    flag
}

pub fn slot_in_flight_inc(slot_key: &str) {
    let g = slots().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = g.get(slot_key) {
        s.in_flight.fetch_add(1, Ordering::SeqCst);
    }
}

pub fn slot_in_flight_dec(slot_key: &str) {
    let g = slots().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(s) = g.get(slot_key) {
        let prev = s.in_flight.fetch_sub(1, Ordering::SeqCst);
        if prev == 1 {
            let (lock, cv) = &*s.done;
            if let Ok(mut guard) = lock.lock() {
                guard.clone_from(&true);
                cv.notify_all();
            }
        }
    }
}

pub fn request_cancel_slot(slot_key: &str) {
    cancel_flag(slot_key).store(true, Ordering::SeqCst);
}

pub fn request_cancel_all() {
    let g = cancels().lock().unwrap_or_else(|e| e.into_inner());
    for flag in g.values() {
        flag.store(true, Ordering::SeqCst);
    }
}

pub fn is_cancelled(cancel: &AtomicBool) -> bool {
    cancel.load(Ordering::SeqCst)
}

pub fn wait_slot_quiet(slot_key: &str, timeout: Duration) -> bool {
    let state = {
        let g = slots().lock().unwrap_or_else(|e| e.into_inner());
        g.get(slot_key).cloned()
    };
    let Some(state) = state else {
        return true;
    };
    let deadline = Instant::now() + timeout;
    loop {
        if state.in_flight.load(Ordering::SeqCst) == 0 {
            return true;
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return false;
        }
        let (lock, cv) = &*state.done;
        let guard = lock.lock().unwrap_or_else(|e| e.into_inner());
        let _ = cv.wait_timeout(guard, remaining).ok();
    }
}

pub fn wait_all_slots_quiet(timeout: Duration) -> bool {
    let keys: Vec<String> = {
        let g = slots().lock().unwrap_or_else(|e| e.into_inner());
        g.keys().cloned().collect()
    };
    for k in keys {
        if !wait_slot_quiet(&k, timeout) {
            return false;
        }
    }
    true
}

pub fn push_instance_notice(instance_id: &str, msg: &str) {
    let mut g = notices_map().lock().unwrap_or_else(|e| e.into_inner());
    g.entry(instance_id.to_string())
        .or_default()
        .push(msg.to_string());
}

pub fn instance_notices(instance_id: &str) -> Vec<String> {
    let g = notices_map().lock().unwrap_or_else(|e| e.into_inner());
    g.get(instance_id).cloned().unwrap_or_default()
}

pub fn clear_instance_notices(instance_id: &str) {
    let mut g = notices_map().lock().unwrap_or_else(|e| e.into_inner());
    g.remove(instance_id);
}
