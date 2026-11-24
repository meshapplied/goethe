// What a node reports on its connection (hello, snapshot, live instance load) and the ack the platform sends back.

use crate::command::{Policy, TrustEntry};
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
    Snapshot {
        snapshot: Value,
    },
    Live {
        t_ms: u64,
        instances: Vec<Instance>,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Instance {
    pub instance_id: String,
    pub model_id: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    pub status: String,
    pub in_flight: u64,
    pub max_conc: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Remote {
    pub policy: Policy,
    pub trust: Vec<TrustEntry>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ack {
    pub usage_to: i64,
    pub errors_to: i64,
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
        }
        let ack: Ack = serde_json::from_str(include_str!("../fixtures/ack.json")).unwrap();
        assert_eq!(ack, Ack { usage_to: 1_795_000_020, errors_to: 1_795_000_000 });
        let remote: Remote = serde_json::from_str(include_str!("../fixtures/remote.json")).unwrap();
        assert!(remote.policy.change && !remote.policy.destructive);
        assert!(serde_json::from_str::<Ack>(include_str!("../fixtures/bad/ack_extra_field.json")).is_err());
    }
}
