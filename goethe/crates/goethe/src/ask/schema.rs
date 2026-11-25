// Ask's view of the CLI: one card per command of the spec tree, with its arguments, risk, and what it needs and gives.
use crate::spec::{self, ArgDoc, ArgKind, Asked, CmdMeta, Fact, Node};
pub use crate::spec::{Risk, Value as Kind};
use clap::Command;
use serde_json::{json, Value};

#[derive(Clone, Debug)]
pub struct PositionalSpec {
    pub id: String,
    pub value_name: String,
    pub required: bool,
    pub multiple: bool,
    pub kind: Kind,
    pub help: String,
    pub possible_values: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct FlagSpec {
    pub long: String,
    pub short: Option<char>,
    pub takes_value: bool,
    pub value_name: String,
    pub kind: Option<Kind>,
    pub asked: Asked,
    pub widens: bool,
    pub default_value: Option<String>,
    pub possible_values: Vec<String>,
    pub help: String,
}

#[derive(Clone, Debug)]
pub struct IntentCard {
    pub id: String,
    pub argv_prefix: Vec<String>,
    pub about: String,
    pub aliases: Vec<String>,
    pub positionals: Vec<PositionalSpec>,
    pub flags: Vec<FlagSpec>,
    pub accepts_passthrough: bool,
    pub risk: Risk,
    pub does: CmdMeta,
    pub search_text: String,
    pub embed_text: String,
}

impl IntentCard {
    pub fn summary(&self) -> String {
        // The about's first phrase, short enough for one row beside the command, without a leading "The".
        let first = self.about.split([':', ';', ',', '.', '(']).next().unwrap_or("").trim();
        first.strip_prefix("The ").map_or_else(|| first.to_string(), capitalize)
    }

    pub fn model_arg(&self) -> Option<&PositionalSpec> {
        self.positionals.iter().find(|p| matches!(p.kind, Kind::Model | Kind::InstalledModel | Kind::ModelInput))
    }

    pub fn takes_model(&self) -> bool {
        self.model_arg().is_some() || self.positionals.iter().any(|p| p.kind == Kind::LogTarget)
    }
}

pub fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    c.next().map_or_else(String::new, |f| f.to_uppercase().chain(c).collect())
}

#[derive(Clone, Debug)]
pub struct CliCatalog {
    pub intents: Vec<IntentCard>,
    pub schema_hash: String,
}

impl CliCatalog {
    pub fn get(&self, id: &str) -> Option<&IntentCard> {
        self.intents.iter().find(|i| i.id == id)
    }

    pub fn giver(&self, fact: Fact) -> Option<&IntentCard> {
        // The one command that makes a fact true (a test keeps it to exactly one).
        self.intents.iter().find(|c| c.does.gives.contains(&fact))
    }

    pub fn to_json(&self) -> Value {
        let intents: Vec<Value> = self
            .intents
            .iter()
            .map(|c| {
                json!({
                    "id": c.id,
                    "argv_prefix": c.argv_prefix,
                    "about": c.about,
                    "risk": c.risk.label(),
                    "accepts_passthrough": c.accepts_passthrough,
                    "positionals": c.positionals.iter().map(|p| json!({
                        "id": p.id, "value_name": p.value_name, "required": p.required, "multiple": p.multiple,
                    })).collect::<Vec<_>>(),
                    "flags": c.flags.iter().map(|f| json!({
                        "long": format!("--{}", f.long),
                        "short": f.short.map(|s| format!("-{s}")),
                        "takes_value": f.takes_value,
                        "default": f.default_value,
                        "possible_values": f.possible_values,
                    })).collect::<Vec<_>>(),
                })
            })
            .collect();
        json!({ "schema_hash": self.schema_hash, "count": self.intents.len(), "commands": intents })
    }
}

pub fn build_catalog(root: &Command) -> CliCatalog {
    // Every command of the spec tree is a card; the hash covers the whole tree, help included, so a cached catalog never outlives a CLI change.
    let doc = spec::node(root, "goethe");
    let schema_hash = fnv(&serde_json::to_string(&doc).unwrap_or_default());
    let mut all = Vec::new();
    doc.walk(&mut all);
    let about_of = |path: &str| all.iter().find(|n| n.path == path).map_or_else(String::new, |n| n.about.clone());
    let intents = all
        .iter()
        .filter(|n| n.path != "goethe")
        .map(|n| card(n, n.path.rsplit_once(' ').filter(|(p, _)| p.contains(' ')).map_or_else(String::new, |(p, _)| about_of(p))))
        .collect();
    CliCatalog { intents, schema_hash }
}

fn card(n: &Node, parent_about: String) -> IntentCard {
    // Ranking reads the command's own text; the embedding also reads its parent's description, the context a subcommand's meaning comes from.
    let argv_prefix = n.argv();
    let id = argv_prefix[1..].join(".");
    let positionals = n.args.iter().filter(|a| a.kind == ArgKind::Positional).map(positional).collect();
    let flags = n.args.iter().filter(|a| a.kind == ArgKind::Flag && a.long.is_some()).map(flag).collect();
    let does = n.does.unwrap_or(CmdMeta::change());
    let mut c = IntentCard {
        id,
        argv_prefix,
        about: n.about.clone(),
        aliases: n.aliases.clone(),
        positionals,
        flags,
        accepts_passthrough: n.args.iter().any(|a| a.kind == ArgKind::Trailing),
        risk: does.risk,
        does,
        search_text: String::new(),
        embed_text: String::new(),
    };
    c.search_text = search_text(&c);
    c.embed_text = format!("{} {parent_about}", c.search_text).trim_end().to_string();
    c
}

fn positional(a: &ArgDoc) -> PositionalSpec {
    PositionalSpec {
        id: a.id.clone(),
        value_name: a.value_name.clone().unwrap_or_else(|| a.id.to_uppercase()),
        required: a.required,
        multiple: a.multiple,
        kind: a.value.unwrap_or(Kind::Text),
        help: a.help.clone(),
        possible_values: a.possible_values.clone(),
    }
}

fn flag(a: &ArgDoc) -> FlagSpec {
    FlagSpec {
        long: a.long.clone().unwrap_or_default(),
        short: a.short,
        takes_value: a.value_name.is_some(),
        value_name: a.value_name.clone().unwrap_or_default(),
        kind: a.value,
        asked: a.asked,
        widens: a.widens,
        default_value: a.default.clone(),
        possible_values: a.possible_values.clone(),
        help: a.help.clone(),
    }
}

fn search_text(card: &IntentCard) -> String {
    // What ranking reads for a card: its words, its own description, aliases, and each argument's name and help.
    let argv_tail = card.argv_prefix[1..].join(" ");
    let mut parts = vec![card.id.replace('.', " "), argv_tail, card.about.clone()];
    parts.extend(card.aliases.iter().cloned());
    parts.extend(card.does.says.iter().map(ToString::to_string));
    parts.extend(card.positionals.iter().filter(|p| !p.help.is_empty()).flat_map(|p| [p.value_name.to_lowercase(), p.help.clone()]));
    parts.extend(card.flags.iter().map(|f| format!("{} {}", f.long.replace('-', " "), f.help).trim().to_string()));
    parts.join(" ")
}

fn fnv(text: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}
