// Values for a command from the words of a request (typed spans, written --flags) and from what is live on this machine.

use super::parse::{
    is_auth_key, is_url, find_cidrs, find_domain, find_duration, find_gpus, find_hf_or_path, find_named_token, find_path_token,
    find_quoted, is_duration, split_words,
};
use super::schema::{IntentCard, Kind};
use crate::complete;
use crate::spec::Fact;
use std::collections::HashMap;

fn parse_explicit_flags(card: &IntentCard, tokens: &[String], out: &mut HashMap<String, Vec<String>>) {
    // Parse explicit --flag, --flag=val, and -s tokens written directly by the user.
    let mut i = 0;
    while i < tokens.len() {
        let tok = &tokens[i];
        if let Some(rest) = tok.strip_prefix("--") {
            let (name, inline_val) = rest.split_once('=').map(|(a, b)| (a, Some(b))).unwrap_or((rest, None));
            if let Some(f) = card.flags.iter().find(|f| f.long == name) {
                if !f.takes_value {
                    out.entry(f.long.clone()).or_default();
                } else if let Some(v) = inline_val {
                    out.entry(f.long.clone()).or_default().push(v.to_string());
                } else if i + 1 < tokens.len() && !tokens[i + 1].starts_with('-') {
                    i += 1;
                    out.entry(f.long.clone()).or_default().push(tokens[i].clone());
                }
            }
        } else if let Some(rest) = tok.strip_prefix('-') {
            let mut chars = rest.chars();
            if let (Some(ch), None) = (chars.next(), chars.next()) {
                if let Some(f) = card.flags.iter().find(|f| f.short == Some(ch)) {
                    if !f.takes_value {
                        out.entry(f.long.clone()).or_default();
                    } else if i + 1 < tokens.len() && !tokens[i + 1].starts_with('-') {
                        i += 1;
                        out.entry(f.long.clone()).or_default().push(tokens[i].clone());
                    }
                }
            }
        }
        i += 1;
    }
}

#[derive(Clone, Debug, Default)]
pub struct World {
    pub daemon_up: bool,
    pub installed_known: bool,
    pub installed: Vec<String>,
    pub loaded: Vec<String>,
    pub aliases: Vec<(String, String)>,
    pub gpus: Vec<(String, String)>,
    pub interfaces: Vec<(String, String)>,
    pub keys: Vec<(String, String)>,
    pub log_targets: Vec<(String, String)>,
}

impl World {
    pub fn from_host() -> Self {
        // One snapshot ask plans from. Unreadable installs are unknown, never "nothing installed".
        let home = goethe_core::paths::goethe_home();
        let daemon_up = goethe_core::service::goethed_running();
        let (installed_known, mut installed, mut loaded) = read_models(&home);
        installed.sort();
        loaded.sort();
        let aliases = goethe_core::alias::list().unwrap_or_default();
        let gpus = cand_pairs(complete::gpus());
        let interfaces = cand_pairs(complete::interfaces());
        let keys = cand_pairs(complete::key_ids());
        let log_targets = cand_pairs(complete::log_targets());
        World { daemon_up, installed_known, installed, loaded, aliases, gpus, interfaces, keys, log_targets }
    }
}

impl World {
    pub fn key_id(&self, name_or_id: &str) -> String {
        // A key named by its name stands for its id; anything else is kept as written.
        self.keys.iter().find(|(_, name)| name == name_or_id).map_or_else(|| name_or_id.to_string(), |(id, _)| id.clone())
    }

    pub fn has(&self, fact: Fact, model: Option<&str>) -> bool {
        // Whether a fact holds now; a fact about a model holds when there is no model to act on.
        match (fact, model) {
            (Fact::Daemon, _) => self.daemon_up,
            (Fact::Key, _) => !self.keys.is_empty(),
            (Fact::Installed, Some(m)) => !super::planner::needs_install(m, self),
            (Fact::Loaded, Some(m)) => self.loaded.iter().any(|id| id == m),
            (Fact::Installed | Fact::Loaded, None) => true,
        }
    }
}

fn read_models(home: &std::path::Path) -> (bool, Vec<String>, Vec<String>) {
    let Ok(con) = goethe_core::db::open(home) else {
        return (false, Vec::new(), Vec::new());
    };
    let installed = match con.prepare("SELECT model_id FROM installed_models") {
        Ok(mut stmt) => match stmt.query_map([], |r| r.get::<_, String>(0)) {
            Ok(rows) => rows.flatten().collect(),
            Err(_) => return (false, Vec::new(), Vec::new()),
        },
        Err(_) => return (false, Vec::new(), Vec::new()),
    };
    let loaded = match con.prepare(
        "SELECT DISTINCT model_id FROM loaded_instances WHERE status IN ('STARTING','RUNNING','RESTARTING','DRAINING')",
    ) {
        Ok(mut stmt) => stmt.query_map([], |r| r.get::<_, String>(0)).map(|rows| rows.flatten().collect()).unwrap_or_default(),
        Err(_) => Vec::new(),
    };
    (true, installed, loaded)
}

fn cand_pairs(cands: Vec<clap_complete::engine::CompletionCandidate>) -> Vec<(String, String)> {
    cands
        .into_iter()
        .map(|c| {
            let val = c.get_value().to_string_lossy().into_owned();
            let help = c.get_help().map(|h| h.to_string()).unwrap_or_default();
            (val, help)
        })
        .collect()
}

#[derive(Clone, Debug, Default)]
pub struct SlotExtraction {
    pub positionals: HashMap<String, Vec<String>>,
    pub flags: Vec<(String, Vec<String>)>,
    pub passthrough: Vec<String>,
    pub ambiguous: Option<(String, Vec<String>)>,
}

pub fn extract_slots(card: &IntentCard, clause: &str, world: &World) -> SlotExtraction {
    // Bind explicit --flags, then typed spans: positionals by their kind, and value flags whose kind is a typed token no positional took.
    let mut out = SlotExtraction::default();
    let (main_text, passthrough) = split_passthrough(clause, card.accepts_passthrough);
    out.passthrough = passthrough;
    let tokens = split_words(main_text);
    let mut used_flags: HashMap<String, Vec<String>> = HashMap::new();
    parse_explicit_flags(card, &tokens, &mut used_flags);
    for f in &card.flags {
        if let Some(vals) = used_flags.remove(&f.long) {
            out.flags.push((f.long.clone(), vals));
        }
    }
    for pos in &card.positionals {
        match find_kind(pos.kind, &pos.possible_values, main_text, &tokens, world) {
            PosMatch::Values(vs) if !vs.is_empty() && vs.iter().all(|v| takes_model_value(card, pos.kind, v, world)) => {
                out.positionals.insert(pos.id.clone(), vs);
            }
            PosMatch::Ambiguous(cands) if out.ambiguous.is_none() => {
                out.ambiguous = Some((pos.id.clone(), cands));
            }
            _ => {}
        }
    }
    let taken: Vec<String> = out.positionals.values().flatten().cloned().collect();
    for f in card.flags.iter().filter(|f| f.takes_value) {
        let Some(kind) = f.kind.filter(|k| typed_token(*k)) else {
            continue;
        };
        if out.flags.iter().any(|(k, _)| k == &f.long) {
            continue;
        }
        if card.flags.iter().filter(|g| g.kind == Some(kind)).count() > 1 {
            continue;
        }
        if let PosMatch::Values(vs) = find_kind(kind, &f.possible_values, main_text, &tokens, world) {
            if !vs.is_empty() && !vs.iter().any(|v| taken.contains(v)) {
                out.flags.push((f.long.clone(), vs));
            }
        }
    }
    out
}

fn typed_token(kind: Kind) -> bool {
    // Kinds whose values are recognisable tokens on their own, so a flag of that kind can be filled from plain words.
    matches!(kind, Kind::Cidr | Kind::Gpu | Kind::Key | Kind::KeyId | Kind::Interface | Kind::Domain | Kind::Url | Kind::AuthKey)
}

pub enum PosMatch {
    None,
    Values(Vec<String>),
    Ambiguous(Vec<String>),
}

pub fn find_kind(kind: Kind, possible: &[String], text: &str, tokens: &[String], world: &World) -> PosMatch {
    // The value of one kind in a request, from typed spans and the live state.
    let one = |v: Option<String>| v.map(|x| PosMatch::Values(vec![x])).unwrap_or(PosMatch::None);
    let many = |vs: Vec<String>| if vs.is_empty() { PosMatch::None } else { PosMatch::Values(vs) };
    match kind {
        Kind::Model | Kind::InstalledModel | Kind::ModelOrAll => match find_hf_or_path(tokens) {
            Some(m) => PosMatch::Values(vec![m]),
            None => match_installed_model(tokens, world),
        },
        Kind::ModelInput => extract_run_args(text, tokens, world),
        Kind::Gpu => many(find_gpus(text)),
        Kind::Cidr => many(find_cidrs(tokens)),
        Kind::Duration | Kind::When => one(find_duration(tokens)),
        Kind::Key => match_api_key(tokens, world),
        Kind::KeyId => match match_api_key(tokens, world) {
            PosMatch::Values(vs) => PosMatch::Values(vs.iter().map(|v| world.key_id(v)).collect()),
            other => other,
        },
        Kind::KeyName => one(find_named_token(text, tokens, &["named", "called", "name", "create", "key"])),
        Kind::Alias => match world.aliases.iter().find(|(a, _)| tokens.iter().any(|t| t.eq_ignore_ascii_case(a))) {
            Some((a, _)) => PosMatch::Values(vec![a.clone()]),
            None => one(find_named_token(text, tokens, &["alias", "as", "named", "called", "remove"])),
        },
        Kind::NewAlias => one(find_named_token(text, tokens, &["alias", "as", "named", "called", "set"])),
        Kind::Domain => one(find_domain(tokens)),
        Kind::LogTarget => match world.log_targets.iter().find(|(t, _)| tokens.iter().any(|w| w.eq_ignore_ascii_case(t))) {
            Some((t, _)) => PosMatch::Values(vec![t.clone()]),
            None => match_installed_model(tokens, world),
        },
        Kind::ExposeTarget => extract_expose_target(tokens, world),
        Kind::Interface => one(world.interfaces.iter().find(|(i, _)| tokens.iter().any(|t| t.eq_ignore_ascii_case(i))).map(|(i, _)| i.clone())),
        Kind::Folder => one(find_path_token(tokens)),
        Kind::Url => one(tokens.iter().find(|t| is_url(t)).cloned()),
        Kind::AuthKey => one(tokens.iter().find(|t| is_auth_key(t)).cloned()),
        Kind::Choice => one(possible.iter().find(|v| tokens.iter().any(|t| t.eq_ignore_ascii_case(v))).cloned()),
        Kind::Number => one(tokens.iter().find(|t| !t.is_empty() && t.chars().all(|c| c.is_ascii_digit())).cloned()),
        Kind::Text => one(find_quoted(text).filter(|q| !q.starts_with('-'))),
    }
}

fn extract_run_args(text: &str, tokens: &[String], world: &World) -> PosMatch {
    // Extract optional [MODEL] and optional quoted/prompt [INPUT] for goethe run.
    let mut out = Vec::new();
    if let Some(m) = find_hf_or_path(tokens) {
        out.push(m);
    } else if let PosMatch::Values(vs) = match_installed_model(tokens, world) {
        out.extend(vs.into_iter().take(1));
    }
    if let Some(q) = find_quoted(text) {
        out.push(q);
    }
    if out.is_empty() {
        PosMatch::None
    } else {
        PosMatch::Values(out)
    }
}

fn extract_expose_target(tokens: &[String], world: &World) -> PosMatch {
    // Match an interface name from live state or a raw IPv4 address for goethe api expose.
    for (iface, _) in &world.interfaces {
        if tokens.iter().any(|t| t.eq_ignore_ascii_case(iface)) {
            return PosMatch::Values(vec![iface.clone()]);
        }
    }
    for t in tokens {
        if t.parse::<std::net::IpAddr>().is_ok() {
            return PosMatch::Values(vec![t.clone()]);
        }
    }
    PosMatch::None
}

pub fn named_models(text: &str, world: &World) -> Vec<String> {
    // The installed models or aliases a request's words name (words of three or more characters with a letter, so "gpu 1" names none).
    let words: Vec<String> = split_words(text).into_iter().filter(|t| t.len() >= 3 && t.chars().any(|c| c.is_ascii_alphabetic())).collect();
    match match_installed_model(&words, world) {
        PosMatch::Values(v) | PosMatch::Ambiguous(v) => v,
        PosMatch::None => Vec::new(),
    }
}

pub fn is_model_value(value: &str, world: &World) -> bool {
    world.installed.iter().any(|m| m == value) || world.aliases.iter().any(|(a, _)| a == value) || find_hf_or_path(&[value.to_string()]).is_some()
}

pub fn takes_model_value(card: &IntentCard, kind: Kind, value: &str, world: &World) -> bool {
    // A command on an installed model that nothing installs first (remove, not load) takes only one that is installed: "delete the file /tmp/x" is no model to remove.
    !matches!(kind, Kind::InstalledModel | Kind::ModelOrAll)
        || card.does.needs.contains(&Fact::Installed)
        || !world.installed_known
        || world.installed.iter().any(|m| m == value)
        || world.aliases.iter().any(|(a, _)| a == value)
}

fn match_installed_model(tokens: &[String], world: &World) -> PosMatch {
    // An alias or id written out wins; else the installed models whose name holds the most of the request's words (a tie is a pick).
    for (alias, _) in &world.aliases {
        if tokens.iter().any(|t| t.eq_ignore_ascii_case(alias)) {
            return PosMatch::Values(vec![alias.clone()]);
        }
    }
    for m in &world.installed {
        if tokens.iter().any(|t| t.eq_ignore_ascii_case(m)) {
            return PosMatch::Values(vec![m.clone()]);
        }
    }
    let words: Vec<String> = tokens.iter().filter(|t| !t.starts_with('-')).flat_map(|t| name_parts(t)).collect();
    let score = |id: &str| {
        let parts = name_parts(id);
        words.iter().filter(|w| parts.iter().any(|p| p == *w || (w.len() >= 3 && p.starts_with(w.as_str())))).count()
    };
    let best = world.installed.iter().map(|m| score(m)).max().unwrap_or(0);
    if best == 0 {
        return PosMatch::None;
    }
    let hits: Vec<String> = world.installed.iter().filter(|m| score(m) == best).cloned().collect();
    if hits.len() == 1 { PosMatch::Values(hits) } else { PosMatch::Ambiguous(hits) }
}

fn name_parts(s: &str) -> Vec<String> {
    s.to_ascii_lowercase().split(|c: char| !c.is_ascii_alphanumeric()).filter(|p| !p.is_empty()).map(str::to_string).collect()
}

pub fn match_api_key(tokens: &[String], world: &World) -> PosMatch {
    // A live key written by its name or id, as a whole word; several are a pick, and a long hex word with no live match is taken as an id.
    let said = |w: &str| !w.is_empty() && tokens.iter().any(|t| t.eq_ignore_ascii_case(w));
    let mut hits: Vec<String> = world.keys.iter().filter_map(|(id, name)| if said(name) { Some(name.clone()) } else { said(id).then(|| id.clone()) }).collect();
    hits.sort();
    hits.dedup();
    if hits.len() == 1 {
        PosMatch::Values(hits)
    } else if hits.len() > 1 {
        PosMatch::Ambiguous(hits)
    } else {
        tokens.iter().find(|t| t.len() >= 10 && t.chars().all(|c| c.is_ascii_hexdigit())).map_or(PosMatch::None, |t| PosMatch::Values(vec![t.clone()]))
    }
}

pub fn typed_values(text: &str, world: &World) -> Vec<String> {
    // Every value in a request that has a recognisable shape: a model id or path, addresses, a duration, GPUs, a domain, a live interface or key, an auth key, a URL, quoted text, a written --flag.
    let tokens = split_words(text.split(" -- ").next().unwrap_or(text));
    let mut out: Vec<String> = Vec::new();
    out.extend(find_hf_or_path(&tokens));
    out.extend(find_cidrs(&tokens));
    out.extend(find_duration(&tokens));
    out.extend(find_gpus(text));
    out.extend(find_domain(&tokens).filter(|d| !d.contains('/')));
    out.extend(world.interfaces.iter().filter(|(i, _)| tokens.iter().any(|t| t.eq_ignore_ascii_case(i))).map(|(i, _)| i.clone()));
    if let PosMatch::Values(vs) = match_api_key(&tokens, world) {
        out.extend(vs);
    }
    out.extend(tokens.iter().filter(|t| is_auth_key(t) || is_url(t)).cloned());
    out.extend(find_quoted(text));
    out.extend(tokens.iter().filter(|t| t.len() > 2 && t.starts_with("--")).map(|t| t.split('=').next().unwrap_or(t).to_string()));
    let mut seen = std::collections::HashSet::new();
    out.retain(|v| seen.insert(v.clone()));
    out
}

pub fn validate_manual(kind: Kind, possible: &[String], values: &[String], world: &World) -> Result<Vec<String>, String> {
    // A value typed by hand, checked for its kind; junk is refused with what it should look like.
    if values.is_empty() || values.iter().all(|v| v.trim().is_empty()) {
        return Err("empty value".into());
    }
    if let Some(v) = values.iter().find(|v| v.starts_with('-')) {
        return Err(format!("a value cannot start with '-': {v}"));
    }
    match kind {
        Kind::Cidr => {
            let ok: Vec<String> = values.iter().flat_map(|v| find_cidrs(std::slice::from_ref(v))).collect();
            if ok.is_empty() {
                return Err("need an IP or CIDR (e.g. 203.0.113.7/32)".into());
            }
            Ok(ok)
        }
        Kind::Duration => {
            let d = values.iter().find(|v| is_duration(v)).cloned().or_else(|| find_duration(values));
            d.map(|x| vec![x]).ok_or_else(|| "need a duration (e.g. 1m, 2h)".into())
        }
        Kind::Domain => find_domain(values).map(|d| vec![d]).ok_or_else(|| "need a domain name".into()),
        Kind::Gpu => {
            let gpus = find_gpus(&values.join(" "));
            if gpus.is_empty() { Err("need a GPU as goethe gpu lists it (e.g. gpu0)".into()) } else { Ok(gpus) }
        }
        Kind::Model | Kind::InstalledModel | Kind::ModelOrAll => {
            if let Some(m) = find_hf_or_path(values) {
                return Ok(vec![m]);
            }
            match match_installed_model(values, world) {
                PosMatch::Values(vs) => Ok(vs),
                PosMatch::Ambiguous(cands) => Err(format!("ambiguous - pick one: {}", cands.join(", "))),
                PosMatch::None => {
                    if values.len() == 1 && values[0].contains('/') {
                        Ok(values.to_vec())
                    } else {
                        Err("use HF org/name, a local path, or an installed model name".into())
                    }
                }
            }
        }
        Kind::KeyName | Kind::NewAlias if values.len() > 1 => Err("one word, such as my-app".into()),
        Kind::KeyId => Ok(values.iter().map(|v| world.key_id(v)).collect()),
        Kind::Number if values.len() > 1 || values[0].parse::<f64>().is_err() => Err("need a number".into()),
        Kind::Choice if !possible.iter().any(|p| p == &values[0]) => Err(format!("one of: {}", possible.join(", "))),
        _ => Ok(values.to_vec()),
    }
}

pub fn slot_candidates(kind: Kind, possible: &[String], world: &World) -> Vec<(String, String)> {
    // The live values to pick from for a kind (value, help); kinds without a list are typed.
    match kind {
        Kind::InstalledModel | Kind::ModelOrAll | Kind::ModelInput => {
            let mut out: Vec<(String, String)> = world.installed.iter().map(|m| (m.clone(), "installed model".into())).collect();
            for (a, m) in &world.aliases {
                out.push((a.clone(), format!("alias for {m}")));
            }
            out
        }
        Kind::Gpu => world.gpus.clone(),
        Kind::Key => world.keys.iter().map(|(id, name)| (if name.is_empty() { id.clone() } else { name.clone() }, format!("id {id}"))).collect(),
        Kind::KeyId => world.keys.iter().map(|(id, name)| (id.clone(), name.clone())).collect(),
        Kind::Alias => world.aliases.clone(),
        Kind::LogTarget => world.log_targets.clone(),
        Kind::ExposeTarget | Kind::Interface => world.interfaces.clone(),
        Kind::Choice => possible.iter().map(|v| (v.clone(), String::new())).collect(),
        _ => Vec::new(),
    }
}

fn split_passthrough(clause: &str, accepts: bool) -> (&str, Vec<String>) {
    if !accepts {
        return (clause, Vec::new());
    }
    if let Some((left, right)) = clause.split_once(" -- ") {
        return (left, split_words(right));
    }
    (clause, Vec::new())
}
