// Signed console commands on the node: verify against remote.json and the link, refuse replays, run the action through the code the CLI and goethed use, and keep each outcome.

use super::remote;
use crate::errcode::{self, Code};
use crate::supervisor::LoadParams;
use goethe_proto::{verify, Action, CommandEvent, Expect, Kind, Refusal, SignedCommand, TrustEntry, Verified, CONSOLE_ORIGIN, RP_ID};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const SEEN_FILE: &str = "remote_cmds.json";
const KEEP_OUTCOMES: usize = 256;

static SEEN_LOCK: Mutex<()> = Mutex::new(());
static RUNNING: Mutex<Vec<(PathBuf, Kind)>> = Mutex::new(Vec::new());

pub struct Engines {
    pub load: fn(LoadParams) -> Result<Value, String>,
    pub unload: fn(&str) -> Result<(), String>,
}

pub const SUPERVISOR: Engines = Engines { load: supervisor_load, unload: supervisor_unload };

fn supervisor_load(params: LoadParams) -> Result<Value, String> {
    // What goethed's load RPC runs, after the limits check the CLI makes first; nothing waits here, the console sees the instance come up in the state.
    crate::config::validate_limits(&crate::paths::goethe_home(), &params.model_id, params.ctx, params.max_conc)?;
    crate::supervisor::handle_load(params).map(|(v, _)| v).map_err(|(_, msg, _)| msg)
}

fn supervisor_unload(instance_id: &str) -> Result<(), String> {
    // Only this instance; the model's other instances keep serving.
    crate::supervisor::handle_unload_instance(instance_id, false).map(|_| ()).map_err(|(_, msg)| msg)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct Seen {
    cmd_id: String,
    body_sha256: String,
    expires_ms: u64,
    seen_ms: u64,
    events: Vec<CommandEvent>,
}

struct Failure {
    code: Code,
    message: String,
}

fn fail(code: Code, message: impl Into<String>) -> Failure {
    Failure { code, message: message.into() }
}

struct Claim {
    home: PathBuf,
    kind: Kind,
}

impl Drop for Claim {
    fn drop(&mut self) {
        RUNNING.lock().unwrap_or_else(|e| e.into_inner()).retain(|(h, k)| !(*h == self.home && *k == self.kind));
    }
}

fn claim(home: &Path, kind: Kind) -> Option<Claim> {
    // One command of a kind at a time per home; None while another runs.
    let mut running = RUNNING.lock().unwrap_or_else(|e| e.into_inner());
    if running.iter().any(|(h, k)| h == home && *k == kind) {
        return None;
    }
    running.push((home.to_path_buf(), kind));
    Some(Claim { home: home.to_path_buf(), kind })
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

pub fn handle(home: &Path, signed: &SignedCommand, emit: &mut dyn FnMut(CommandEvent)) {
    // Blocking: verify, check replay, run, and emit accepted or refused, progress, then done.
    handle_with(home, signed, &SUPERVISOR, now_ms(), emit)
}

pub fn handle_with(home: &Path, signed: &SignedCommand, engines: &Engines, now_ms: u64, emit: &mut dyn FnMut(CommandEvent)) {
    // The replay check, the claim and the stored accepted event happen under one lock, so two copies of a command never both run.
    let cmd_id = signed.cmd_id().unwrap_or_default();
    let v = match checked(home, signed, now_ms) {
        Ok(v) => v,
        Err((r, msg)) => return refuse(home, &cmd_id, r, &msg, emit),
    };
    let kind = v.action.kind();
    let accepted = CommandEvent::Accepted { cmd_id: cmd_id.clone() };
    let claimed = {
        let _guard = SEEN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let mut seen = match load_seen(home) {
            Ok(s) => s,
            Err(e) => return refuse(home, &cmd_id, Refusal::Busy, &e, emit),
        };
        match seen.iter_mut().find(|s| s.cmd_id == cmd_id) {
            Some(s) if s.body_sha256 != v.body_sha256 => Err(Refusal::Replay),
            Some(s) if finished(&s.events) => Ok(Err(s.events.clone())),
            Some(_) if RUNNING.lock().unwrap_or_else(|e| e.into_inner()).iter().any(|(h, k)| h == home && *k == kind) => Err(Refusal::Busy),
            Some(s) => {
                s.events.push(CommandEvent::Done { cmd_id: cmd_id.clone(), ok: false, result: None, code: Some(Code::CmdFailed.label().into()), message: Some("goethed stopped before the command finished".into()) });
                let events = s.events.clone();
                let _ = save_seen(home, &seen);
                Ok(Err(events))
            }
            None => match claim(home, kind) {
                None => Err(Refusal::Busy),
                Some(c) => {
                    keep(&mut seen, Seen { cmd_id: cmd_id.clone(), body_sha256: v.body_sha256.clone(), expires_ms: v.command.expires_ms, seen_ms: now_ms, events: vec![accepted.clone()] }, now_ms);
                    match save_seen(home, &seen) {
                        Ok(()) => Ok(Ok(c)),
                        Err(e) => return refuse(home, &cmd_id, Refusal::Busy, &e, emit),
                    }
                }
            },
        }
    };
    let _claim = match claimed {
        Err(r) => return refuse(home, &cmd_id, r, "", emit),
        Ok(Err(events)) => return events.into_iter().for_each(&mut *emit),
        Ok(Ok(c)) => c,
    };
    emit(accepted.clone());
    let progress = CommandEvent::Progress { cmd_id: cmd_id.clone(), phase: v.action.name().into(), pct: None, message: None };
    emit(progress.clone());
    let done = match run(home, &v, engines, now_ms) {
        Ok(result) => {
            crate::client::link::state_changed();
            CommandEvent::Done { cmd_id: cmd_id.clone(), ok: true, result, code: None, message: None }
        }
        Err(f) => {
            errcode::record_at(home, f.code, &f.message);
            CommandEvent::Done { cmd_id: cmd_id.clone(), ok: false, result: None, code: Some(f.code.label().into()), message: Some(f.message) }
        }
    };
    let events = vec![accepted, progress, done.clone()];
    if let Err(e) = remember(home, Seen { cmd_id, body_sha256: v.body_sha256, expires_ms: v.command.expires_ms, seen_ms: now_ms, events }, now_ms) {
        errcode::record_at(home, Code::CmdFailed, &e);
    }
    emit(done);
}

fn finished(events: &[CommandEvent]) -> bool {
    events.iter().any(|e| matches!(e, CommandEvent::Done { .. }))
}

fn checked(home: &Path, signed: &SignedCommand, now_ms: u64) -> Result<Verified, (Refusal, String)> {
    // Every stateless check against this node's link and remote.json; a node that is not linked has no workspace, so every command is for another node.
    let creds = crate::client::link::load_credentials(home).ok().flatten().ok_or((Refusal::WrongNode, "this node is not linked".to_string()))?;
    let r = remote::status(home);
    let x = Expect { org_id: &creds.workspace.id, node_id: &creds.node_id, rp_id: RP_ID, origin: CONSOLE_ORIGIN, now_ms, policy: r.policy, trust: &r.trust };
    let v = verify(signed, &x).map_err(|r| (r, String::new()))?;
    if !supported(&v.action) {
        return Err((Refusal::ActionUnknown, format!("{} is not run remotely by this goethe yet", v.action.name())));
    }
    Ok(v)
}

fn supported(a: &Action) -> bool {
    matches!(
        a,
        Action::ModelLoad(_)
            | Action::ModelUnload(_)
            | Action::GpuAllow(_)
            | Action::GpuDeny(_)
            | Action::KeyCreate(_)
            | Action::KeyRotate(_)
            | Action::KeyRevoke(_)
            | Action::TrustAdd(_)
            | Action::TrustRemove(_)
    )
}

fn not_admin(home: &Path, key_id: &str, why: &str) -> Result<(), Failure> {
    // Admin keys are managed only at the machine: the console never rotates or revokes one.
    let admin: Option<bool> = db(home)?.query_row("SELECT is_admin FROM api_keys WHERE key_id = ?1", [key_id], |r| r.get(0)).ok();
    match admin {
        None => Err(fail(Code::RemoteArgs, format!("key {key_id} not found"))),
        Some(true) => Err(fail(Code::RemoteArgs, format!("an admin key {why}"))),
        Some(false) => Ok(()),
    }
}

fn refuse(home: &Path, cmd_id: &str, r: Refusal, detail: &str, emit: &mut dyn FnMut(CommandEvent)) {
    // Every refusal is logged under its code too, so it reaches the console like any other error.
    let message = if detail.is_empty() { refusal_text(r).to_string() } else { detail.to_string() };
    errcode::record_at(home, refusal_code(r), &message);
    emit(CommandEvent::Refused { cmd_id: cmd_id.to_string(), code: r, message });
}

fn refusal_code(r: Refusal) -> Code {
    match r {
        Refusal::ActionUnknown => Code::RemoteActionUnknown,
        Refusal::Args => Code::RemoteArgs,
        Refusal::WrongNode => Code::RemoteWrongNode,
        Refusal::Sig => Code::RemoteSig,
        Refusal::KeyUnknown => Code::RemoteKeyUnknown,
        Refusal::KeyCap => Code::RemoteKeyCap,
        Refusal::Policy => Code::RemotePolicy,
        Refusal::Expired => Code::RemoteExpired,
        Refusal::Replay => Code::RemoteReplay,
        Refusal::Busy => Code::RemoteBusy,
    }
}

fn refusal_text(r: Refusal) -> &'static str {
    match r {
        Refusal::ActionUnknown => "action not in this node's catalog",
        Refusal::Args => "malformed command or bad arguments",
        Refusal::WrongNode => "command is for another node or workspace",
        Refusal::Sig => "signature or batch does not verify",
        Refusal::KeyUnknown => "passkey not pinned on this node",
        Refusal::KeyCap => "passkey not allowed this kind of change",
        Refusal::Policy => "this node has not enabled this kind of change (goethe remote allow)",
        Refusal::Expired => "command expired or validity window too long",
        Refusal::Replay => "command id reused for a different command",
        Refusal::Busy => "a command of the same kind is already running",
    }
}

fn seen_path(home: &Path) -> PathBuf {
    home.join(SEEN_FILE)
}

fn load_seen(home: &Path) -> Result<Vec<Seen>, String> {
    // The commands this node has seen; a missing file is none, an unreadable or damaged one refuses every command, since forgetting would let a replay through.
    let p = seen_path(home);
    match std::fs::read_to_string(&p) {
        Ok(t) => serde_json::from_str(&t).map_err(|_| format!("{} is damaged; remove it once every command sent in the last two hours has expired", p.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("cannot read {}: {e}", p.display())),
    }
}

fn save_seen(home: &Path, seen: &[Seen]) -> Result<(), String> {
    let data = serde_json::to_vec(seen).map_err(|e| e.to_string())?;
    crate::paths::write_private(&seen_path(home), &data).map_err(|e| format!("cannot write {}: {e}", seen_path(home).display()))
}

fn keep(seen: &mut Vec<Seen>, entry: Seen, now_ms: u64) {
    // Store or replace one outcome; an id is kept until its command expires, and the newest 256 outcomes are kept whatever their age.
    seen.retain(|s| s.cmd_id != entry.cmd_id);
    seen.push(entry);
    seen.sort_by_key(|s| std::cmp::Reverse(s.seen_ms));
    let mut i = 0;
    seen.retain(|s| {
        i += 1;
        i <= KEEP_OUTCOMES || s.expires_ms > now_ms
    });
}

fn remember(home: &Path, entry: Seen, now_ms: u64) -> Result<(), String> {
    let _guard = SEEN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut seen = load_seen(home)?;
    keep(&mut seen, entry, now_ms);
    save_seen(home, &seen)
}

pub fn outcome(home: &Path, cmd_id: &str) -> Option<Vec<CommandEvent>> {
    // The stored events of a command this node has seen, for GET /cmd/{id} after a reconnect.
    let _guard = SEEN_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    load_seen(home).ok()?.into_iter().find(|s| s.cmd_id == cmd_id).map(|s| s.events)
}

pub fn load_params(a: &goethe_proto::ModelLoad) -> Result<LoadParams, String> {
    // The load goethe load sends goethed: GPUs as a list, tensor parallel from the GPUs when they are given; never past busy GPUs or foreign processes (no --yes).
    if let (Some(g), Some(tp)) = (&a.gpus, a.tp) {
        if g.len() != tp as usize {
            return Err(format!("tp {tp} does not match the {} GPUs given", g.len()));
        }
    }
    let gpus = a.gpus.as_ref().map(|g| g.iter().map(u32::to_string).collect::<Vec<_>>().join(","));
    let tp = a.gpus.as_ref().map_or(a.tp.map(u64::from), |g| Some(g.len() as u64));
    serde_json::from_value(json!({ "model_id": a.model_id, "gpus": gpus, "ctx": a.ctx, "max_conc": a.max_conc, "tp": tp })).map_err(|e| e.to_string())
}

fn db(home: &Path) -> Result<rusqlite::Connection, Failure> {
    crate::db::open(home).map_err(|e| fail(Code::CmdFailed, e.to_string()))
}

fn run(home: &Path, v: &Verified, engines: &Engines, now_ms: u64) -> Result<Option<Value>, Failure> {
    // One action, through the functions the CLI and goethed's local RPC call; a sealed secret is the only secret that leaves.
    let cmd_id = &v.command.cmd_id;
    let failed = |e: String| fail(Code::CmdFailed, e);
    match &v.action {
        Action::ModelLoad(a) => {
            let params = load_params(a).map_err(|e| fail(Code::RemoteArgs, e))?;
            let r = (engines.load)(params).map_err(failed)?;
            Ok(Some(match r.get("instance_id") {
                Some(id) => json!({ "instance_id": id }),
                None => r,
            }))
        }
        Action::ModelUnload(a) => {
            let known: Option<i64> = db(home)?.query_row("SELECT 1 FROM loaded_instances WHERE instance_id = ?1", [&a.instance_id], |r| r.get(0)).ok();
            known.ok_or_else(|| fail(Code::RemoteArgs, format!("no instance {} on this node", a.instance_id)))?;
            (engines.unload)(&a.instance_id).map_err(failed).map(|()| None)
        }
        Action::GpuAllow(a) | Action::GpuDeny(a) => {
            let deny = matches!(v.action, Action::GpuDeny(_));
            for g in &a.gpus {
                crate::gpu_policy::validate_gpu_index(home, *g).map_err(|e| fail(Code::RemoteArgs, e))?;
            }
            let con = db(home)?;
            for g in &a.gpus {
                crate::gpu_policy::set_denied(&con, *g, deny).map_err(failed)?;
            }
            Ok(None)
        }
        Action::KeyCreate(a) => {
            sealable(&a.browser_pub, cmd_id)?;
            let models = a.models.as_ref().map(|m| m.join(","));
            let (key_id, token) = crate::security::create_key(home, &a.name, None, a.expires.as_deref(), models.as_deref(), false).map_err(|e| fail(Code::RemoteArgs, e))?;
            sealed(&a.browser_pub, cmd_id, &key_id, &token)
        }
        Action::KeyRotate(a) => {
            sealable(&a.browser_pub, cmd_id)?;
            not_admin(home, &a.key_id, "gets a new secret only at the machine (goethe api keys rotate)")?;
            let token = crate::security::rotate_secret(home, &a.key_id).map_err(failed)?;
            sealed(&a.browser_pub, cmd_id, &a.key_id, &token)
        }
        Action::KeyRevoke(a) => {
            not_admin(home, &a.key_id, "is revoked only at the machine (goethe api keys)")?;
            crate::security::revoke_key(home, &a.key_id).map_err(|e| fail(Code::RemoteArgs, e)).map(|()| None)
        }
        Action::TrustAdd(a) => {
            let entry = TrustEntry { key_id: a.key_id.clone(), public_key: a.public_key.clone(), user_id: a.user_id.clone(), cap: a.cap, added_by: v.signer.user_id.clone(), added_ms: now_ms };
            remote::pin(home, entry, &v.signer.key_id).map_err(failed).map(|()| None)
        }
        Action::TrustRemove(a) => remote::unpin(home, &a.key_id, &v.signer.key_id, now_ms).map_err(|e| fail(Code::RemoteArgs, e)).map(|_| None),
        other => Err(fail(Code::RemoteActionUnknown, format!("{} is not run remotely by this goethe yet", other.name()))),
    }
}

fn sealable(browser_pub: &str, cmd_id: &str) -> Result<(), Failure> {
    // Checked before a secret exists, so a key the browser could never open is not made or rotated.
    goethe_proto::seal(browser_pub, cmd_id, "", b"").map(|_| ()).ok_or_else(|| fail(Code::RemoteArgs, "the one-time browser key is not a P-256 point"))
}

fn sealed(browser_pub: &str, cmd_id: &str, key_id: &str, token: &str) -> Result<Option<Value>, Failure> {
    let s = goethe_proto::seal(browser_pub, cmd_id, key_id, token.as_bytes()).ok_or_else(|| fail(Code::CmdFailed, "sealing the secret failed"))?;
    Ok(Some(json!({ "key_id": key_id, "prefix": &token[..14], "sealed": s })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::link::{Credentials, Secret, Workspace};
    use goethe_proto::testkit::{Browser, Passkey};
    use goethe_proto::{Command, Sealed, COMMAND_SCHEMA};

    const NOW: u64 = 1_800_000_000_000;

    fn fake_load(p: LoadParams) -> Result<Value, String> {
        if p.model_id == "missing" {
            return Err("missing is not installed".into());
        }
        Ok(json!({ "instance_id": format!("inst-{}", p.model_id), "port": 1 }))
    }

    fn fake_unload(instance_id: &str) -> Result<(), String> {
        if instance_id == "inst-1" { Ok(()) } else { Err("unexpected instance".into()) }
    }

    const FAKE: Engines = Engines { load: fake_load, unload: fake_unload };

    fn linked() -> tempfile::TempDir {
        let h = tempfile::tempdir().unwrap();
        let c = Credentials {
            platform_url: "https://platform.example.com".into(),
            node_id: "n1".into(),
            workspace: Workspace { id: "org1".into(), name: "Lab".into() },
            token: Secret::new("mesh_node_0123456789abcdef0123456789abcdef"),
            linked_at: "2026-11-11T10:00:00Z".into(),
        };
        crate::client::link::save_credentials(h.path(), &c).unwrap();
        for k in Kind::ALL {
            remote::set_allowed(h.path(), k, true).unwrap();
        }
        h
    }

    fn cmd(id: &str, action: &str, args: Value) -> Command {
        Command {
            schema: COMMAND_SCHEMA.into(),
            cmd_id: id.into(),
            batch_id: "b1".into(),
            org_id: "org1".into(),
            node_id: "n1".into(),
            action: action.into(),
            args,
            issued_by: "user1".into(),
            key_id: "k1".into(),
            issued_ms: NOW,
            expires_ms: NOW + 60_000,
            rollout_id: None,
        }
    }

    fn pinned(home: &Path, cap: Kind) -> Passkey {
        let pk = Passkey::new("k1");
        remote::pin(home, pk.trust("user1", cap), remote::LOCAL).unwrap();
        pk
    }

    fn send(home: &Path, signed: &SignedCommand, now: u64) -> Vec<CommandEvent> {
        let mut out = Vec::new();
        handle_with(home, signed, &FAKE, now, &mut |e| out.push(e));
        out
    }

    fn refusal(events: &[CommandEvent]) -> Option<Refusal> {
        match events {
            [CommandEvent::Refused { code, .. }] => Some(*code),
            _ => None,
        }
    }

    fn done(events: &[CommandEvent]) -> (bool, Option<Value>, Option<String>) {
        match events.last() {
            Some(CommandEvent::Done { ok, result, code, .. }) => (*ok, result.clone(), code.clone()),
            other => panic!("no done: {other:?}"),
        }
    }

    fn load(model: &str) -> Value {
        json!({ "model_id": model })
    }

    #[test]
    fn a_signed_load_runs_once_and_a_retry_gets_the_stored_outcome() {
        let h = linked();
        let pk = pinned(h.path(), Kind::Change);
        let signed = pk.sign(&[cmd("c1", "model.load", load("org/m"))]).remove(0);
        let first = send(h.path(), &signed, NOW);
        assert!(matches!(first[0], CommandEvent::Accepted { .. }) && matches!(first[1], CommandEvent::Progress { .. }));
        assert_eq!(done(&first), (true, Some(json!({ "instance_id": "inst-org/m" })), None));
        assert_eq!(send(h.path(), &signed, NOW + 1), first, "the same bytes again replay the stored events");
        assert_eq!(outcome(h.path(), "c1"), Some(first));
        let other = pk.sign(&[cmd("c1", "model.load", load("org/other"))]).remove(0);
        assert_eq!(refusal(&send(h.path(), &other, NOW)), Some(Refusal::Replay));
        let failed = send(h.path(), &pk.sign(&[cmd("c2", "model.load", load("missing"))]).remove(0), NOW);
        assert_eq!(done(&failed), (false, None, Some("CMD_FAILED".into())));
    }

    #[test]
    fn each_broken_rule_is_refused_with_its_code_and_logged() {
        let h = linked();
        let pk = pinned(h.path(), Kind::Change);
        let good = pk.sign(&[cmd("c1", "model.load", load("org/m"))]).remove(0);
        let mut tampered = pk.sign(&[cmd("c2", "model.load", load("org/m"))]).remove(0);
        tampered.body_b64 = good.body_b64.clone();
        let mut unsigned = good.clone();
        unsigned.assertion.signature_b64 = String::new();
        let mut wrong = cmd("c3", "model.load", load("org/m"));
        wrong.node_id = "n2".into();
        let mut stranger_cmd = cmd("c4", "model.load", load("org/m"));
        stranger_cmd.key_id = "k9".into();
        let cases = [
            (unsigned, NOW, Refusal::Sig),
            (tampered, NOW, Refusal::Sig),
            (good.clone(), NOW + 60_000, Refusal::Expired),
            (pk.sign(&[wrong]).remove(0), NOW, Refusal::WrongNode),
            (Passkey::new("k9").sign(&[stranger_cmd]).remove(0), NOW, Refusal::KeyUnknown),
            (pk.sign(&[cmd("c5", "model.remove", load("org/m"))]).remove(0), NOW, Refusal::KeyCap),
            (pk.sign(&[cmd("c6", "model.load", json!({ "model_id": "m", "gpus": [] }))]).remove(0), NOW, Refusal::Args),
        ];
        for (i, (signed, now, want)) in cases.into_iter().enumerate() {
            assert_eq!(refusal(&send(h.path(), &signed, now)), Some(want), "case {i}");
        }
        let log = std::fs::read_to_string(h.path().join("errors.log")).unwrap();
        for code in ["REMOTE_SIG", "REMOTE_EXPIRED", "REMOTE_WRONG_NODE", "REMOTE_KEY_UNKNOWN", "REMOTE_KEY_CAP", "REMOTE_ARGS"] {
            assert!(log.contains(code), "{code} not logged");
        }
        remote::set_allowed(h.path(), Kind::Change, false).unwrap();
        assert_eq!(refusal(&send(h.path(), &good, NOW)), Some(Refusal::Policy));
        assert_eq!(outcome(h.path(), "c1"), None, "a refused command leaves no outcome");
    }

    #[test]
    fn an_unlinked_node_refuses_everything_and_a_busy_kind_waits() {
        let h = tempfile::tempdir().unwrap();
        let pk = pinned(h.path(), Kind::Trust);
        remote::set_allowed(h.path(), Kind::Change, true).unwrap();
        let signed = pk.sign(&[cmd("c1", "model.load", load("org/m"))]).remove(0);
        assert_eq!(refusal(&send(h.path(), &signed, NOW)), Some(Refusal::WrongNode));
        let h = linked();
        let pk = pinned(h.path(), Kind::Trust);
        let signed = pk.sign(&[cmd("c1", "model.load", load("org/m"))]).remove(0);
        let held = claim(h.path(), Kind::Change).unwrap();
        assert_eq!(refusal(&send(h.path(), &signed, NOW)), Some(Refusal::Busy));
        assert!(done(&send(h.path(), &pk.sign(&[cmd("c2", "trust.remove", json!({ "key_id": "kx" }))]).remove(0), NOW)).2.is_some(), "another kind still runs");
        drop(held);
        assert!(done(&send(h.path(), &signed, NOW)).0);
        let install = pk.sign(&[cmd("c3", "model.install", load("org/m"))]).remove(0);
        assert_eq!(refusal(&send(h.path(), &install, NOW)), Some(Refusal::ActionUnknown));
    }

    #[test]
    fn a_trusted_key_adds_the_next_one_and_a_lower_cap_cannot() {
        let h = linked();
        let owner = pinned(h.path(), Kind::Trust);
        let member = Passkey::new("k2");
        let add = json!({ "key_id": "k2", "public_key": member.public_key_b64(), "user_id": "user2", "cap": "change" });
        assert!(done(&send(h.path(), &owner.sign(&[cmd("c1", "trust.add", add.clone())]).remove(0), NOW)).0);
        let entry = remote::status(h.path()).trust.into_iter().find(|t| t.key_id == "k2").unwrap();
        assert_eq!((entry.cap, entry.added_by.as_str()), (Kind::Change, "user1"));
        assert_eq!(remote::load(h.path()).unwrap().log.last().unwrap().by, "k1");
        let mut by_member = cmd("c2", "model.load", load("org/m"));
        by_member.key_id = "k2".into();
        assert!(done(&send(h.path(), &member.sign(&[by_member]).remove(0), NOW)).0, "the new key signs within its cap");
        let mut escalate = cmd("c3", "trust.add", json!({ "key_id": "k3", "public_key": member.public_key_b64(), "user_id": "user2", "cap": "trust" }));
        escalate.key_id = "k2".into();
        assert_eq!(refusal(&send(h.path(), &member.sign(&[escalate]).remove(0), NOW)), Some(Refusal::KeyCap));
        assert!(done(&send(h.path(), &owner.sign(&[cmd("c4", "trust.remove", json!({ "key_id": "k2" }))]).remove(0), NOW)).0);
        let mut after = cmd("c5", "model.load", load("org/m"));
        after.key_id = "k2".into();
        assert_eq!(refusal(&send(h.path(), &member.sign(&[after]).remove(0), NOW)), Some(Refusal::KeyUnknown));
    }

    fn opened(browser: Browser, events: &[CommandEvent], cmd_id: &str) -> (String, String) {
        let r = done(events).1.unwrap();
        let sealed: Sealed = serde_json::from_value(r["sealed"].clone()).unwrap();
        let key_id = r["key_id"].as_str().unwrap().to_string();
        let secret = String::from_utf8(browser.open(&sealed, cmd_id, &key_id).unwrap()).unwrap();
        assert!(secret.starts_with(r["prefix"].as_str().unwrap()));
        (key_id, secret)
    }

    #[test]
    fn a_new_or_rotated_secret_reaches_only_the_browser_that_asked() {
        let h = linked();
        let pk = pinned(h.path(), Kind::Keys);
        let browser = Browser::new();
        let create = json!({ "name": "phone", "models": ["org/m"], "browser_pub": browser.public_b64() });
        let events = send(h.path(), &pk.sign(&[cmd("c1", "api.key.create", create)]).remove(0), NOW);
        let (key_id, secret) = opened(browser, &events, "c1");
        let stored = std::fs::read_to_string(h.path().join(SEEN_FILE)).unwrap();
        assert!(!stored.contains(&secret), "the outcome keeps only the sealed secret");
        let con = crate::db::open(h.path()).unwrap();
        let (hash, admin, scope): (String, bool, String) = con.query_row("SELECT token_hash, is_admin, model_scope FROM api_keys WHERE key_id = ?1", [&key_id], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).unwrap();
        assert_eq!((hash, admin, scope.as_str()), (crate::security::token_hash(&secret), false, "org/m"));
        let browser = Browser::new();
        let rotate = json!({ "key_id": key_id, "browser_pub": browser.public_b64() });
        let (_, fresh) = opened(browser, &send(h.path(), &pk.sign(&[cmd("c2", "api.key.rotate", rotate)]).remove(0), NOW), "c2");
        assert_ne!(fresh, secret);
        let (admin_id, _) = crate::security::create_key(h.path(), "ops", None, None, None, true).unwrap();
        let rotate = json!({ "key_id": admin_id, "browser_pub": Browser::new().public_b64() });
        assert_eq!(done(&send(h.path(), &pk.sign(&[cmd("c3", "api.key.rotate", rotate)]).remove(0), NOW)).2.as_deref(), Some("REMOTE_ARGS"));
        let revoke = json!({ "key_id": admin_id });
        assert_eq!(done(&send(h.path(), &pk.sign(&[cmd("c4", "api.key.revoke", revoke)]).remove(0), NOW)).2.as_deref(), Some("REMOTE_ARGS"));
        assert!(crate::db::open(h.path()).unwrap().query_row("SELECT 1 FROM api_keys WHERE key_id = ?1", [&admin_id], |_| Ok(())).is_ok(), "the admin key stays");
    }

    #[test]
    fn unload_stops_only_the_named_instance() {
        let h = linked();
        let pk = pinned(h.path(), Kind::Change);
        let con = crate::db::open(h.path()).unwrap();
        con.execute(
            "INSERT INTO loaded_instances (instance_id, model_id, pid, pgid, internal_port, gpu_indices, allocated_vram_gb_per_gpu, mem_fraction_static, ctx_len, max_conc, status) VALUES ('inst-1', 'org/m', 1, 1, 9000, '0', 1.0, 0.8, 4096, 4, 'RUNNING')",
            [],
        )
        .unwrap();
        assert!(done(&send(h.path(), &pk.sign(&[cmd("c1", "model.unload", json!({ "instance_id": "inst-1" }))]).remove(0), NOW)).0);
        let gone = send(h.path(), &pk.sign(&[cmd("c2", "model.unload", json!({ "instance_id": "inst-2" }))]).remove(0), NOW);
        assert_eq!(done(&gone).2.as_deref(), Some("REMOTE_ARGS"));
    }

    #[test]
    fn load_arguments_map_onto_the_load_goethe_load_sends() {
        let a = |gpus: Option<Vec<u32>>, tp: Option<u32>| goethe_proto::ModelLoad { model_id: "org/m".into(), gpus, ctx: Some(8192), max_conc: Some(4), tp };
        let p = load_params(&a(Some(vec![2, 3]), None)).unwrap();
        assert_eq!((p.gpus.as_deref(), p.tp, p.ctx, p.max_conc, p.yes, p.force, p.replica), (Some("2,3"), Some(2), Some(8192), Some(4), false, false, false));
        assert_eq!(load_params(&a(None, Some(4))).unwrap().tp, Some(4));
        assert!(load_params(&a(Some(vec![2]), Some(2))).is_err());
    }

    #[test]
    fn a_damaged_replay_record_refuses_instead_of_forgetting() {
        let h = linked();
        let pk = pinned(h.path(), Kind::Change);
        let signed = pk.sign(&[cmd("c1", "model.load", load("org/m"))]).remove(0);
        assert!(done(&send(h.path(), &signed, NOW)).0);
        std::fs::write(h.path().join(SEEN_FILE), "{not json").unwrap();
        assert_eq!(refusal(&send(h.path(), &signed, NOW + 1)), Some(Refusal::Busy), "the same command again is not run a second time");
        assert_eq!(refusal(&send(h.path(), &pk.sign(&[cmd("c2", "model.load", load("org/m"))]).remove(0), NOW)), Some(Refusal::Busy));
        std::fs::remove_file(h.path().join(SEEN_FILE)).unwrap();
        std::fs::create_dir(h.path().join(SEEN_FILE)).unwrap();
        assert_eq!(refusal(&send(h.path(), &pk.sign(&[cmd("c3", "model.load", load("org/m"))]).remove(0), NOW)), Some(Refusal::Busy), "an outcome that cannot be kept is not run");
    }

    #[test]
    fn old_outcomes_go_once_expired_but_the_newest_256_stay() {
        let h = tempfile::tempdir().unwrap();
        for i in 0..300u64 {
            remember(h.path(), Seen { cmd_id: format!("c{i}"), body_sha256: String::new(), expires_ms: NOW + i, seen_ms: i, events: Vec::new() }, NOW + 1000).unwrap();
        }
        assert_eq!(load_seen(h.path()).unwrap().len(), KEEP_OUTCOMES);
        assert!(outcome(h.path(), "c299").is_some() && outcome(h.path(), "c10").is_none());
        remember(h.path(), Seen { cmd_id: "live".into(), body_sha256: String::new(), expires_ms: NOW + 5000, seen_ms: 0, events: Vec::new() }, NOW + 1000).unwrap();
        assert!(outcome(h.path(), "live").is_some(), "an unexpired id stays even past 256");
    }
}
