// Signed console commands: the command body, the batch a passkey signs once, and every stateless check a node runs before acting.

use crate::action::{Action, Kind};
use crate::webauthn::{self, Assertion};
use crate::{b64, hex, sha256, unb64};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const COMMAND_SCHEMA: &str = "goethe.command.v1";
pub const BATCH_SCHEMA: &str = "goethe.batch.v1";
const CLOCK_SKEW_MS: u64 = 60_000;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Command {
    pub schema: String,
    pub cmd_id: String,
    pub batch_id: String,
    pub org_id: String,
    pub node_id: String,
    pub action: String,
    pub args: Value,
    pub issued_by: String,
    pub key_id: String,
    pub issued_ms: u64,
    pub expires_ms: u64,
    pub rollout_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Batch {
    pub schema: String,
    pub batch_id: String,
    pub org_id: String,
    pub issued_by: String,
    pub key_id: String,
    pub issued_ms: u64,
    pub expires_ms: u64,
    pub commands: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignedCommand {
    pub manifest_b64: String,
    pub body_b64: String,
    pub assertion: Assertion,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustEntry {
    pub key_id: String,
    pub public_key: String,
    pub user_id: String,
    pub cap: Kind,
    pub added_by: String,
    pub added_ms: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Policy {
    pub change: bool,
    pub network: bool,
    pub keys: bool,
    pub update: bool,
    pub destructive: bool,
    pub trust: bool,
}

impl Policy {
    pub fn allows(&self, kind: Kind) -> bool {
        match kind {
            Kind::Change => self.change,
            Kind::Network => self.network,
            Kind::Keys => self.keys,
            Kind::Update => self.update,
            Kind::Destructive => self.destructive,
            Kind::Trust => self.trust,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Refusal {
    #[serde(rename = "REMOTE_ACTION_UNKNOWN")]
    ActionUnknown,
    #[serde(rename = "REMOTE_ARGS")]
    Args,
    #[serde(rename = "REMOTE_WRONG_NODE")]
    WrongNode,
    #[serde(rename = "REMOTE_SIG")]
    Sig,
    #[serde(rename = "REMOTE_KEY_UNKNOWN")]
    KeyUnknown,
    #[serde(rename = "REMOTE_KEY_CAP")]
    KeyCap,
    #[serde(rename = "REMOTE_POLICY")]
    Policy,
    #[serde(rename = "REMOTE_EXPIRED")]
    Expired,
    #[serde(rename = "REMOTE_REPLAY")]
    Replay,
    #[serde(rename = "REMOTE_BUSY")]
    Busy,
}

impl Refusal {
    pub fn code(self) -> &'static str {
        match self {
            Refusal::ActionUnknown => "REMOTE_ACTION_UNKNOWN",
            Refusal::Args => "REMOTE_ARGS",
            Refusal::WrongNode => "REMOTE_WRONG_NODE",
            Refusal::Sig => "REMOTE_SIG",
            Refusal::KeyUnknown => "REMOTE_KEY_UNKNOWN",
            Refusal::KeyCap => "REMOTE_KEY_CAP",
            Refusal::Policy => "REMOTE_POLICY",
            Refusal::Expired => "REMOTE_EXPIRED",
            Refusal::Replay => "REMOTE_REPLAY",
            Refusal::Busy => "REMOTE_BUSY",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum CommandEvent {
    Accepted {
        cmd_id: String,
    },
    Refused {
        cmd_id: String,
        code: Refusal,
        message: String,
    },
    Progress {
        cmd_id: String,
        phase: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        pct: Option<u8>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
    Done {
        cmd_id: String,
        ok: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        result: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        code: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        message: Option<String>,
    },
}

pub struct Expect<'a> {
    pub org_id: &'a str,
    pub node_id: &'a str,
    pub rp_id: &'a str,
    pub origin: &'a str,
    pub now_ms: u64,
    pub policy: Policy,
    pub trust: &'a [TrustEntry],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Verified {
    pub command: Command,
    pub action: Action,
    pub signer: TrustEntry,
    pub body_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prepared {
    pub manifest_b64: String,
    pub challenge: [u8; 32],
    pub bodies_b64: Vec<String>,
}

pub fn prepare(commands: &[Command]) -> Option<Prepared> {
    // The batch one passkey tap signs: every command must share batch id, org, issuer, key and validity, or there is no batch.
    let first = commands.first()?;
    let shared = |c: &Command| {
        c.batch_id == first.batch_id
            && c.org_id == first.org_id
            && c.issued_by == first.issued_by
            && c.key_id == first.key_id
            && c.issued_ms == first.issued_ms
            && c.expires_ms == first.expires_ms
    };
    if !commands.iter().all(shared) {
        return None;
    }
    let bodies: Vec<Vec<u8>> = commands.iter().map(serde_json::to_vec).collect::<Result<_, _>>().ok()?;
    let batch = Batch {
        schema: BATCH_SCHEMA.into(),
        batch_id: first.batch_id.clone(),
        org_id: first.org_id.clone(),
        issued_by: first.issued_by.clone(),
        key_id: first.key_id.clone(),
        issued_ms: first.issued_ms,
        expires_ms: first.expires_ms,
        commands: bodies.iter().map(|b| hex(&sha256(b))).collect(),
    };
    let manifest = serde_json::to_vec(&batch).ok()?;
    Some(Prepared { manifest_b64: b64(&manifest), challenge: sha256(&manifest), bodies_b64: bodies.iter().map(|b| b64(b)).collect() })
}

pub fn verify(signed: &SignedCommand, x: &Expect) -> Result<Verified, Refusal> {
    // Every check a node runs before acting, in order, except replay, which needs the node's memory of the commands it has seen.
    let body = unb64(&signed.body_b64).ok_or(Refusal::Args)?;
    let cmd: Command = serde_json::from_slice(&body).map_err(|_| Refusal::Args)?;
    if cmd.schema != COMMAND_SCHEMA {
        return Err(Refusal::Args);
    }
    let kind = Action::kind_of(&cmd.action).ok_or(Refusal::ActionUnknown)?;
    if cmd.node_id != x.node_id || cmd.org_id != x.org_id {
        return Err(Refusal::WrongNode);
    }
    let manifest = unb64(&signed.manifest_b64).ok_or(Refusal::Sig)?;
    let batch: Batch = serde_json::from_slice(&manifest).map_err(|_| Refusal::Sig)?;
    let body_sha256 = hex(&sha256(&body));
    let listed = batch.schema == BATCH_SCHEMA
        && batch.batch_id == cmd.batch_id
        && batch.org_id == cmd.org_id
        && batch.issued_by == cmd.issued_by
        && batch.key_id == cmd.key_id
        && batch.issued_ms == cmd.issued_ms
        && batch.expires_ms == cmd.expires_ms
        && batch.commands.contains(&body_sha256)
        && signed.assertion.key_id == cmd.key_id;
    if !listed {
        return Err(Refusal::Sig);
    }
    let signer = x.trust.iter().find(|t| t.key_id == cmd.key_id).ok_or(Refusal::KeyUnknown)?;
    if !webauthn::verify(&signed.assertion, &signer.public_key, &sha256(&manifest), x.rp_id, x.origin) {
        return Err(Refusal::Sig);
    }
    let window = cmd.expires_ms.saturating_sub(cmd.issued_ms);
    let early = cmd.issued_ms > x.now_ms.saturating_add(CLOCK_SKEW_MS);
    if early || x.now_ms >= cmd.expires_ms || window == 0 || window > Action::max_window_ms(&cmd.action) {
        return Err(Refusal::Expired);
    }
    if !x.policy.allows(kind) {
        return Err(Refusal::Policy);
    }
    if kind > signer.cap {
        return Err(Refusal::KeyCap);
    }
    let action = Action::parse(&cmd.action, &cmd.args).ok_or(Refusal::Args)?;
    Ok(Verified { command: cmd, action, signer: signer.clone(), body_sha256 })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::Passkey;
    use crate::{CONSOLE_ORIGIN, RP_ID};
    use serde_json::json;

    const NOW: u64 = 1_800_000_000_000;
    const ALL: Policy = Policy { change: true, network: true, keys: true, update: true, destructive: true, trust: true };

    fn cmd(node: &str, action: &str, args: Value) -> Command {
        Command {
            schema: COMMAND_SCHEMA.into(),
            cmd_id: format!("c-{node}"),
            batch_id: "b1".into(),
            org_id: "org1".into(),
            node_id: node.into(),
            action: action.into(),
            args,
            issued_by: "user1".into(),
            key_id: "k1".into(),
            issued_ms: NOW,
            expires_ms: NOW + 60_000,
            rollout_id: None,
        }
    }

    fn check(signed: &SignedCommand, trust: &[TrustEntry], policy: Policy, now_ms: u64) -> Result<Verified, Refusal> {
        let x = Expect { org_id: "org1", node_id: "n1", rp_id: RP_ID, origin: CONSOLE_ORIGIN, now_ms, policy, trust };
        verify(signed, &x)
    }

    fn load() -> Value {
        json!({ "model_id": "qwen3-8b" })
    }

    #[test]
    fn one_tap_signs_a_batch_and_each_node_accepts_only_its_own() {
        let pk = Passkey::new("k1");
        let trust = [pk.trust("user1", Kind::Trust)];
        let signed = pk.sign(&[cmd("n1", "model.load", load()), cmd("n2", "model.load", load())]);
        let ok = check(&signed[0], &trust, ALL, NOW).unwrap();
        assert_eq!(ok.command.cmd_id, "c-n1");
        assert_eq!(ok.action, Action::ModelLoad(crate::ModelLoad { model_id: "qwen3-8b".into(), gpus: None, ctx: None, max_conc: None, tp: None }));
        assert_eq!(check(&signed[1], &trust, ALL, NOW), Err(Refusal::WrongNode));
    }

    #[test]
    fn every_broken_rule_is_refused_with_its_code() {
        let pk = Passkey::new("k1");
        let trust = [pk.trust("user1", Kind::Change)];
        let good = pk.sign(&[cmd("n1", "model.load", load())]).remove(0);
        assert!(check(&good, &trust, ALL, NOW).is_ok());

        let mut tampered = good.clone();
        let mut body: Value = serde_json::from_slice(&unb64(&tampered.body_b64).unwrap()).unwrap();
        body["args"]["model_id"] = json!("other");
        tampered.body_b64 = b64(&serde_json::to_vec(&body).unwrap());

        let mut extra_body = serde_json::to_value(cmd("n1", "model.load", load())).unwrap();
        extra_body["sneaky"] = json!(true);
        let mut unknown_field = good.clone();
        unknown_field.body_b64 = b64(&serde_json::to_vec(&extra_body).unwrap());

        let stranger = Passkey::new("k2");
        let mut long = cmd("n1", "model.load", load());
        long.expires_ms = NOW + 10 * 60_000;
        let shell = cmd("n1", "shell.run", json!({ "cmd": "id" }));
        let mut later = cmd("n1", "model.load", load());
        later.issued_ms = NOW + 365 * 86_400_000;
        later.expires_ms = later.issued_ms + 60_000;
        let world = cmd("n1", "api.share", json!({ "keys": ["k"], "allow": ["0.0.0.0/0"], "ttl_s": 600 }));
        let bad_arg = cmd("n1", "model.load", json!({ "model_id": "m", "extra": 1 }));

        let cases: Vec<(SignedCommand, &[TrustEntry], Policy, u64, Refusal)> = vec![
            (tampered, &trust, ALL, NOW, Refusal::Sig),
            (unknown_field, &trust, ALL, NOW, Refusal::Args),
            (good.clone(), &[], ALL, NOW, Refusal::KeyUnknown),
            (good.clone(), &trust, ALL, NOW + 60_000, Refusal::Expired),
            (good.clone(), &trust, Policy::default(), NOW, Refusal::Policy),
            (pk.sign(&[long]).remove(0), &trust, ALL, NOW, Refusal::Expired),
            (pk.sign(&[later]).remove(0), &trust, ALL, NOW, Refusal::Expired),
            (pk.sign(&[shell]).remove(0), &trust, ALL, NOW, Refusal::ActionUnknown),
            (pk.sign(&[cmd("n1", "model.remove", load())]).remove(0), &trust, ALL, NOW, Refusal::KeyCap),
            (pk.sign(&[bad_arg]).remove(0), &trust, ALL, NOW, Refusal::Args),
            (stranger.sign(&[cmd("n1", "model.load", load())]).remove(0), &trust, ALL, NOW, Refusal::Sig),
            (pk.sign_as(&[cmd("n1", "model.load", load())], RP_ID, "https://evil.example", true).remove(0), &trust, ALL, NOW, Refusal::Sig),
            (pk.sign_as(&[cmd("n1", "model.load", load())], "evil.example", CONSOLE_ORIGIN, true).remove(0), &trust, ALL, NOW, Refusal::Sig),
            (pk.sign_as(&[cmd("n1", "model.load", load())], RP_ID, CONSOLE_ORIGIN, false).remove(0), &trust, ALL, NOW, Refusal::Sig),
        ];
        for (i, (signed, trust, policy, now, want)) in cases.into_iter().enumerate() {
            assert_eq!(check(&signed, trust, policy, now).map(|_| ()), Err(want), "case {i}");
        }
        let world_trust = [pk.trust("user1", Kind::Trust)];
        assert_eq!(check(&pk.sign(&[world]).remove(0), &world_trust, ALL, NOW).map(|_| ()), Err(Refusal::Args));
    }

    #[test]
    fn a_batch_must_share_issuer_key_and_validity() {
        let mut other = cmd("n2", "model.load", load());
        other.expires_ms += 1;
        assert!(prepare(&[cmd("n1", "model.load", load()), other]).is_none());
        assert!(prepare(&[]).is_none());
    }

    #[test]
    fn command_and_event_fixtures_parse_and_round_trip() {
        let c: Command = serde_json::from_str(include_str!("../fixtures/command.json")).unwrap();
        assert!(Action::parse(&c.action, &c.args).is_some());
        assert_eq!(serde_json::from_value::<Command>(serde_json::to_value(&c).unwrap()).unwrap(), c);
        for line in include_str!("../fixtures/command_events.ndjson").lines().filter(|l| !l.trim().is_empty()) {
            let e: CommandEvent = serde_json::from_str(line).unwrap();
            assert_eq!(serde_json::to_value(&e).unwrap(), serde_json::from_str::<Value>(line).unwrap());
        }
    }

    #[test]
    fn refusal_codes_serialize_as_their_code() {
        for r in [Refusal::ActionUnknown, Refusal::Args, Refusal::WrongNode, Refusal::Sig, Refusal::KeyUnknown, Refusal::KeyCap, Refusal::Policy, Refusal::Expired, Refusal::Replay, Refusal::Busy] {
            assert_eq!(serde_json::to_value(r).unwrap(), json!(r.code()));
        }
    }
}
