// What a node reports on its goethe-server connection (hello, live instance load) and the remote block its console snapshot carries.

use crate::command::{Policy, TrustEntry};
use serde::{Deserialize, Serialize};

pub const STATE_SCHEMA: &str = "goethe.state.v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum StateEvent {
    Hello {
        schema: String,
        node_id: String,
        install_hash: String,
        goethe_version: String,
        proto: u32,
    },
    Live {
        t_ms: u64,
        instances: Vec<Instance>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    pub instance_id: String,
    pub model_id: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub status: String,
    pub in_flight: u64,
    pub max_conc: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub queued: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_cache_pct: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Remote {
    pub policy: Policy,
    pub trust: Vec<TrustEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_parse_and_round_trip() {
        for line in include_str!("../fixtures/state.ndjson").lines().filter(|l| !l.trim().is_empty()) {
            let event: StateEvent = serde_json::from_str(line).unwrap();
            let again: StateEvent = serde_json::from_value(serde_json::to_value(&event).unwrap()).unwrap();
            assert_eq!(event, again);
            if let StateEvent::Live { instances, .. } = &event {
                assert_eq!((instances[0].queued, instances[0].kv_cache_pct), (Some(1), Some(42.5)));
                assert!(instances[1].queued.is_none() && !serde_json::to_string(&instances[1]).unwrap().contains("kv_cache_pct"), "optional gauges stay off the wire");
            }
        }
        let remote: Remote = serde_json::from_str(include_str!("../fixtures/remote.json")).unwrap();
        assert!(remote.policy.change && !remote.policy.destructive);
        assert!(serde_json::from_str::<StateEvent>(r#"{"type":"snapshot","snapshot":{}}"#).is_err(), "telemetry never rides this stream");
    }
}
