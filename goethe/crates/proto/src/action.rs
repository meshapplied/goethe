// The closed catalog of remote actions: names, argument shapes, kinds of change, and the limits a node enforces whoever signed.

use crate::unb64;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const MINUTE_MS: u64 = 60_000;
const SHARE_MIN_S: u64 = 60;
const SHARE_MAX_S: u64 = 7 * 86_400;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Change,
    Network,
    Keys,
    Update,
    Destructive,
    Trust,
}

impl Kind {
    pub const ALL: [Kind; 6] = [Kind::Change, Kind::Network, Kind::Keys, Kind::Update, Kind::Destructive, Kind::Trust];

    pub fn name(self) -> &'static str {
        match self {
            Kind::Change => "change",
            Kind::Network => "network",
            Kind::Keys => "keys",
            Kind::Update => "update",
            Kind::Destructive => "destructive",
            Kind::Trust => "trust",
        }
    }

    pub fn parse(name: &str) -> Option<Kind> {
        Kind::ALL.into_iter().find(|k| k.name() == name)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", content = "args")]
pub enum Action {
    #[serde(rename = "model.install")]
    ModelInstall(ModelInstall),
    #[serde(rename = "model.load")]
    ModelLoad(ModelLoad),
    #[serde(rename = "model.unload")]
    ModelUnload(ModelUnload),
    #[serde(rename = "model.remove")]
    ModelRemove(ModelRef),
    #[serde(rename = "gpu.allow")]
    GpuAllow(Gpus),
    #[serde(rename = "gpu.deny")]
    GpuDeny(Gpus),
    #[serde(rename = "api.key.create")]
    KeyCreate(KeyCreate),
    #[serde(rename = "api.key.revoke")]
    KeyRevoke(KeyRef),
    #[serde(rename = "api.key.rotate")]
    KeyRotate(KeyRotate),
    #[serde(rename = "api.expose")]
    Expose(Expose),
    #[serde(rename = "api.allow")]
    Allow(Allow),
    #[serde(rename = "api.share")]
    Share(Share),
    #[serde(rename = "goethe.update")]
    Update(Update),
    #[serde(rename = "goethe.update.rollback")]
    UpdateRollback(Empty),
    #[serde(rename = "trust.add")]
    TrustAdd(TrustAdd),
    #[serde(rename = "trust.remove")]
    TrustRemove(KeyRef),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelInstall {
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Numerics {
    Invariant,
    Tolerance,
}

impl Numerics {
    pub const ALL: [Numerics; 2] = [Numerics::Invariant, Numerics::Tolerance];

    pub fn name(self) -> &'static str {
        match self {
            Numerics::Invariant => "invariant",
            Numerics::Tolerance => "tolerance",
        }
    }

    pub fn parse(s: &str) -> Option<Numerics> {
        Numerics::ALL.into_iter().find(|n| n.name() == s)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelLoad {
    pub model_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpus: Option<Vec<u32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ctx: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_conc: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tp: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub numerics: Option<Numerics>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelUnload {
    pub instance_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelRef {
    pub model_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gpus {
    pub gpus: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyCreate {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires: Option<String>,
    pub browser_pub: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyRef {
    pub key_id: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyRotate {
    pub key_id: String,
    pub browser_pub: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExposeMode {
    Local,
    Vpn,
    Public,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Expose {
    pub mode: ExposeMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Allow {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub add: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remove: Option<Vec<String>>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Share {
    pub keys: Vec<String>,
    pub allow: Vec<String>,
    pub ttl_s: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Update {
    pub version: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Empty {}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrustAdd {
    pub key_id: String,
    pub public_key: String,
    pub user_id: String,
    pub cap: Kind,
}

impl Action {
    pub fn kind_of(name: &str) -> Option<Kind> {
        // The kind of change an action name belongs to; None for a name outside the catalog.
        Some(match name {
            "model.install" | "model.load" | "model.unload" | "gpu.allow" | "gpu.deny" => Kind::Change,
            "api.expose" | "api.allow" | "api.share" => Kind::Network,
            "api.key.create" | "api.key.rotate" | "api.key.revoke" => Kind::Keys,
            "goethe.update" | "goethe.update.rollback" => Kind::Update,
            "model.remove" => Kind::Destructive,
            "trust.add" | "trust.remove" => Kind::Trust,
            _ => return None,
        })
    }

    pub fn max_window_ms(name: &str) -> u64 {
        // Longest validity a signed command may carry: updates get two hours so a rollout can finish, everything else five minutes.
        match name {
            "goethe.update" | "goethe.update.rollback" => 120 * MINUTE_MS,
            _ => 5 * MINUTE_MS,
        }
    }

    pub fn parse(name: &str, args: &Value) -> Option<Action> {
        // The typed action, or None when the name is unknown, a field is missing or unknown, or a node-side limit is broken.
        let action: Action = serde_json::from_value(json!({ "action": name, "args": args })).ok()?;
        action.valid().then_some(action)
    }

    pub fn name(&self) -> &'static str {
        match self {
            Action::ModelInstall(_) => "model.install",
            Action::ModelLoad(_) => "model.load",
            Action::ModelUnload(_) => "model.unload",
            Action::ModelRemove(_) => "model.remove",
            Action::GpuAllow(_) => "gpu.allow",
            Action::GpuDeny(_) => "gpu.deny",
            Action::KeyCreate(_) => "api.key.create",
            Action::KeyRevoke(_) => "api.key.revoke",
            Action::KeyRotate(_) => "api.key.rotate",
            Action::Expose(_) => "api.expose",
            Action::Allow(_) => "api.allow",
            Action::Share(_) => "api.share",
            Action::Update(_) => "goethe.update",
            Action::UpdateRollback(_) => "goethe.update.rollback",
            Action::TrustAdd(_) => "trust.add",
            Action::TrustRemove(_) => "trust.remove",
        }
    }

    pub fn kind(&self) -> Kind {
        Self::kind_of(self.name()).expect("every catalog action has a kind")
    }

    fn valid(&self) -> bool {
        // Limits that hold whoever signed: shares never open to everyone and stay within the CLI's own time limits.
        match self {
            Action::ModelInstall(a) => !a.model_id.is_empty(),
            Action::ModelLoad(a) => !a.model_id.is_empty() && a.gpus.as_ref().is_none_or(|g| !g.is_empty()),
            Action::ModelUnload(a) => !a.instance_id.is_empty(),
            Action::ModelRemove(a) => !a.model_id.is_empty(),
            Action::GpuAllow(a) | Action::GpuDeny(a) => !a.gpus.is_empty(),
            Action::KeyCreate(a) => !a.name.is_empty() && p256_point(&a.browser_pub),
            Action::KeyRevoke(a) | Action::TrustRemove(a) => !a.key_id.is_empty(),
            Action::KeyRotate(a) => !a.key_id.is_empty() && p256_point(&a.browser_pub),
            Action::Expose(_) | Action::UpdateRollback(_) => true,
            Action::Allow(a) => a.add.as_deref().unwrap_or_default().iter().all(|c| !everyone(c)),
            Action::Share(a) => {
                !a.keys.is_empty()
                    && !a.allow.is_empty()
                    && !a.allow.iter().any(|c| everyone(c))
                    && (SHARE_MIN_S..=SHARE_MAX_S).contains(&a.ttl_s)
            }
            Action::Update(a) => !a.version.is_empty(),
            Action::TrustAdd(a) => !a.key_id.is_empty() && !a.user_id.is_empty() && p256_point(&a.public_key),
        }
    }
}

fn everyone(cidr: &str) -> bool {
    cidr.trim().ends_with("/0")
}

fn p256_point(text: &str) -> bool {
    unb64(text).is_some_and(|b| b.len() == 65 && b[0] == 4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entries(text: &str) -> Vec<(String, Value)> {
        let list: Vec<Value> = serde_json::from_str(text).unwrap();
        list.into_iter().map(|e| (e["action"].as_str().unwrap().to_string(), e["args"].clone())).collect()
    }

    #[test]
    fn every_catalog_example_parses_with_its_kind_and_round_trips() {
        let good = entries(include_str!("../fixtures/actions.json"));
        assert_eq!(good.len(), 17);
        for (name, args) in good {
            let action = Action::parse(&name, &args).unwrap_or_else(|| panic!("{name} should parse"));
            assert_eq!(action.name(), name);
            assert_eq!(Some(action.kind()), Action::kind_of(&name));
            assert_eq!(serde_json::to_value(&action).unwrap(), json!({ "action": name, "args": args }));
        }
    }

    #[test]
    fn unknown_actions_unknown_fields_and_broken_limits_are_refused() {
        for (name, args) in entries(include_str!("../fixtures/bad/actions.json")) {
            assert!(Action::parse(&name, &args).is_none(), "{name} {args} should be refused");
        }
    }

    #[test]
    fn kinds_are_ordered_from_least_to_most_powerful() {
        assert!(Kind::Change < Kind::Network && Kind::Network < Kind::Keys && Kind::Keys < Kind::Update);
        assert!(Kind::Update < Kind::Destructive && Kind::Destructive < Kind::Trust);
        assert_eq!(Action::max_window_ms("goethe.update"), 7_200_000);
        assert_eq!(Action::max_window_ms("model.load"), 300_000);
        for k in Kind::ALL {
            assert_eq!(serde_json::to_value(k).unwrap(), json!(k.name()));
            assert_eq!(Kind::parse(k.name()), Some(k));
        }
        assert_eq!(Kind::parse("shell"), None);
        for n in Numerics::ALL {
            assert_eq!((serde_json::to_value(n).unwrap(), Numerics::parse(n.name())), (json!(n.name()), Some(n)));
        }
        assert_eq!(Numerics::parse("fast"), None);
    }
}
