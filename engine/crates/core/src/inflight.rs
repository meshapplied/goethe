// Per-model in-flight request counters shared by the gateway and unload drain.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

static COUNTS: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();

fn counts() -> &'static Mutex<HashMap<String, usize>> {
    COUNTS.get_or_init(|| Mutex::new(HashMap::new()))
}

pub struct InflightGuard {
    model_id: String,
}

pub fn track(model_id: &str) -> InflightGuard {
    if let Ok(mut map) = counts().lock() {
        map.entry(model_id.to_string())
            .and_modify(|n| *n += 1)
            .or_insert(1);
    }
    InflightGuard {
        model_id: model_id.to_string(),
    }
}

pub fn count(model_id: &str) -> usize {
    counts()
        .lock()
        .ok()
        .and_then(|map| map.get(model_id).copied())
        .unwrap_or(0)
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        if let Ok(mut map) = counts().lock() {
            let left = map.get(&self.model_id).copied().unwrap_or(0).saturating_sub(1);
            if left == 0 {
                map.remove(&self.model_id);
            } else {
                map.insert(self.model_id.clone(), left);
            }
        }
    }
}
