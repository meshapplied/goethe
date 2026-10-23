// Per-model and per-instance in-flight request counters shared by the gateway and unload drain.

use std::collections::HashMap;
use std::sync::{Mutex, MutexGuard, OnceLock};

#[derive(Default)]
struct Counts {
    models: HashMap<String, usize>,
    instances: HashMap<String, usize>,
    turn: usize,
}

fn counts() -> MutexGuard<'static, Counts> {
    static COUNTS: OnceLock<Mutex<Counts>> = OnceLock::new();
    COUNTS
        .get_or_init(|| Mutex::new(Counts::default()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

pub struct InflightGuard {
    model_id: String,
    instance_id: Option<String>,
}

pub fn track(model_id: &str) -> InflightGuard {
    counts().models.entry(model_id.to_string()).and_modify(|n| *n += 1).or_insert(1);
    InflightGuard {
        model_id: model_id.to_string(),
        instance_id: None,
    }
}

pub fn count(model_id: &str) -> usize {
    counts().models.get(model_id).copied().unwrap_or(0)
}

pub fn instance_count(instance_id: &str) -> usize {
    counts().instances.get(instance_id).copied().unwrap_or(0)
}

pub fn total() -> usize {
    counts().models.values().sum()
}

impl InflightGuard {
    pub fn pick(&mut self, instances: &[&str]) -> Option<usize> {
        // Count this request against the least busy instance; ties rotate so idle replicas share traffic.
        if instances.is_empty() || self.instance_id.is_some() {
            return None;
        }
        let mut c = counts();
        let start = c.turn % instances.len();
        c.turn = c.turn.wrapping_add(1);
        let best = (0..instances.len())
            .map(|k| (start + k) % instances.len())
            .min_by_key(|&i| c.instances.get(instances[i]).copied().unwrap_or(0))?;
        c.instances.entry(instances[best].to_string()).and_modify(|n| *n += 1).or_insert(1);
        self.instance_id = Some(instances[best].to_string());
        Some(best)
    }
}

fn decrement(map: &mut HashMap<String, usize>, key: &str) {
    match map.get(key).copied() {
        Some(n) if n > 1 => map.insert(key.to_string(), n - 1),
        _ => map.remove(key),
    };
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let mut c = counts();
        decrement(&mut c.models, &self.model_id);
        if let Some(inst) = &self.instance_id {
            decrement(&mut c.instances, inst);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_in_flight_routing_spreads_and_tracks() {
        let ids = ["lif-a", "lif-b"];
        let mut g1 = track("lif-model");
        let first = g1.pick(&ids).unwrap();
        let mut g2 = track("lif-model");
        let second = g2.pick(&ids).unwrap();
        assert_ne!(first, second, "an idle replica must win over a busy one");
        assert_eq!(count("lif-model"), 2);
        assert_eq!(instance_count("lif-a"), 1);
        assert_eq!(instance_count("lif-b"), 1);
        drop(g1);
        let mut g3 = track("lif-model");
        assert_eq!(g3.pick(&ids), Some(first), "the freed replica is the least loaded");
        assert_eq!(g3.pick(&ids), None, "a guard counts against one instance only");
        drop(g2);
        drop(g3);
        assert_eq!(count("lif-model"), 0);
        assert_eq!(instance_count("lif-a") + instance_count("lif-b"), 0);
    }

    #[test]
    fn ties_rotate_between_idle_replicas() {
        let ids = ["rot-a", "rot-b", "rot-c"];
        let mut seen = std::collections::HashSet::new();
        for _ in 0..ids.len() {
            let mut g = track("rot-model");
            seen.insert(g.pick(&ids).unwrap());
        }
        assert_eq!(seen.len(), ids.len());
        assert_eq!(track("none").pick(&[]), None);
    }
}
