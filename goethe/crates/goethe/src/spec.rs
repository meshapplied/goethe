// What each command risks, needs and gives, and what each argument takes: read by Ask, the CLI reference and tab completion.

use clap::builder::{ArgExt, CommandExt};
use clap::{Arg, Command};
use clap_complete::engine::{ArgValueCandidates, CompletionCandidate};
use serde::{Serialize, Serializer};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Risk {
    ReadOnly,
    Mutating,
    Download,
    Network,
    Destructive,
}

impl Serialize for Risk {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.label())
    }
}

impl Risk {
    pub fn label(self) -> &'static str {
        match self {
            Risk::ReadOnly => "read",
            Risk::Mutating => "change",
            Risk::Download => "download",
            Risk::Network => "network",
            Risk::Destructive => "destructive",
        }
    }

    pub fn is_destructive(self) -> bool {
        matches!(self, Risk::Destructive)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Fact {
    Daemon,
    Installed,
    Loaded,
    Key,
}

impl Fact {
    pub fn why(self) -> &'static str {
        // Why the command that makes this true runs before the one that needs it.
        match self {
            Fact::Daemon => "Goethe is stopped",
            Fact::Installed => "not installed yet",
            Fact::Loaded => "the API serves loaded models",
            Fact::Key => "no API key yet",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct CmdMeta {
    pub risk: Risk,
    pub needs: &'static [Fact],
    pub gives: &'static [Fact],
    pub confirms: bool,
    pub terminal: bool,
    #[serde(skip)]
    pub says: &'static [&'static str],
}

impl CommandExt for CmdMeta {}

impl CmdMeta {
    const fn of(risk: Risk) -> Self {
        CmdMeta { risk, needs: &[], gives: &[], confirms: false, terminal: false, says: &[] }
    }

    pub const fn read() -> Self {
        Self::of(Risk::ReadOnly)
    }

    pub const fn change() -> Self {
        Self::of(Risk::Mutating)
    }

    pub const fn download() -> Self {
        Self::of(Risk::Download)
    }

    pub const fn network() -> Self {
        Self::of(Risk::Network)
    }

    pub const fn destructive() -> Self {
        Self::of(Risk::Destructive)
    }

    pub const fn needs(self, needs: &'static [Fact]) -> Self {
        CmdMeta { needs, ..self }
    }

    pub const fn gives(self, gives: &'static [Fact]) -> Self {
        CmdMeta { gives, ..self }
    }

    pub const fn says(self, says: &'static [&'static str]) -> Self {
        // Other words a person uses for this command; Ask needs one of them (or the name) before it plans a command that changes something.
        CmdMeta { says, ..self }
    }

    pub const fn confirms(self) -> Self {
        // The command asks before it deletes or replaces; Ask asks on its own screen and passes --yes.
        CmdMeta { confirms: true, ..self }
    }

    pub const fn terminal(self) -> Self {
        // The command takes over the terminal (a chat screen), so it never runs in the background.
        CmdMeta { terminal: true, ..self }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Value {
    Model,
    InstalledModel,
    ModelOrAll,
    ModelInput,
    LogTarget,
    Gpu,
    Cidr,
    Duration,
    When,
    KeyName,
    KeyId,
    Key,
    Alias,
    NewAlias,
    Domain,
    Folder,
    Interface,
    ExposeTarget,
    Url,
    AuthKey,
    Number,
    Choice,
    Text,
}

impl Value {
    fn completer(self) -> Option<fn() -> Vec<CompletionCandidate>> {
        // The live values the shell offers for this kind; kinds without a list complete nothing.
        use crate::complete as c;
        let f: fn() -> Vec<CompletionCandidate> = match self {
            Value::InstalledModel | Value::ModelInput => c::models,
            Value::ModelOrAll => c::models_or_all,
            Value::LogTarget => c::log_targets,
            Value::Gpu => c::gpus,
            Value::KeyId => c::key_ids,
            Value::Key => c::key_names,
            Value::Alias => c::aliases,
            Value::Interface => c::interfaces,
            Value::ExposeTarget => c::expose_targets,
            Value::Url => c::platform_urls,
            Value::AuthKey => c::authkey_stdin,
            _ => return None,
        };
        Some(f)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Asked {
    No,
    Always,
    With(&'static str),
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ArgMeta {
    pub value: Value,
    pub asked: Asked,
}

impl ArgExt for ArgMeta {}

impl ArgMeta {
    pub const fn of(value: Value) -> Self {
        ArgMeta { value, asked: Asked::No }
    }

    pub const fn asked(self) -> Self {
        // Optional for the CLI (it asks on a terminal itself), but Ask asks for it before the command runs in the background.
        ArgMeta { asked: Asked::Always, ..self }
    }

    pub const fn asked_with(self, sibling: &'static str) -> Self {
        // Optional for the CLI (a remembered value may stand in), but Ask asks for it once `sibling` is given.
        ArgMeta { asked: Asked::With(sibling), ..self }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Widens;

impl ArgExt for Widens {}

pub fn cmd_meta(cmd: &Command) -> Option<CmdMeta> {
    cmd.get::<CmdMeta>().copied()
}

pub fn arg_meta(arg: &Arg) -> Option<ArgMeta> {
    arg.get::<ArgMeta>().copied()
}

pub fn value_of(arg: &Arg) -> Option<Value> {
    // What an argument takes: its declared kind, else a choice from its possible values or a number from its parser; None for a plain flag.
    if !arg.get_action().takes_values() {
        return None;
    }
    if let Some(m) = arg_meta(arg) {
        return Some(m.value);
    }
    if !arg.get_possible_values().is_empty() {
        return Some(Value::Choice);
    }
    use std::any::TypeId;
    let t = arg.get_value_parser().type_id();
    let numeric = [TypeId::of::<u16>(), TypeId::of::<u32>(), TypeId::of::<u64>(), TypeId::of::<usize>(), TypeId::of::<i64>(), TypeId::of::<f64>()];
    numeric.iter().any(|n| t == *n).then_some(Value::Number)
}

pub fn with_completers(cmd: Command) -> Command {
    // Every argument whose kind has live values gets the shell completer for them.
    cmd.mut_args(|a| match arg_meta(&a).and_then(|m| m.value.completer()) {
        Some(f) => a.add(ArgValueCandidates::new(f)),
        None => a,
    })
    .mut_subcommands(with_completers)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ArgKind {
    Positional,
    Flag,
    Trailing,
}

#[derive(Clone, Debug, Serialize)]
pub struct ArgDoc {
    pub id: String,
    pub kind: ArgKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short: Option<char>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
    pub required: bool,
    pub multiple: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub possible_values: Vec<String>,
    #[serde(skip_serializing_if = "Asked::is_no")]
    pub asked: Asked,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub widens: bool,
    pub help: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Node {
    pub path: String,
    pub about: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub does: Option<CmdMeta>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub examples: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<ArgDoc>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub commands: Vec<Node>,
}

impl Asked {
    fn is_no(&self) -> bool {
        *self == Asked::No
    }
}

impl Node {
    pub fn argv(&self) -> Vec<String> {
        self.path.split(' ').map(str::to_string).collect()
    }

    pub fn walk<'n>(&'n self, out: &mut Vec<&'n Node>) {
        // This command, then every command under it, depth first.
        out.push(self);
        for c in &self.commands {
            c.walk(out);
        }
    }

    #[cfg(test)]
    fn find_mut(&mut self, path: &[String]) -> Option<&mut Node> {
        match path.split_first() {
            None => Some(self),
            Some((head, rest)) => self.commands.iter_mut().find(|c| c.path.rsplit(' ').next() == Some(head.as_str()))?.find_mut(rest),
        }
    }
}

fn arg_doc(a: &Arg) -> ArgDoc {
    let kind = if a.is_last_set() {
        ArgKind::Trailing
    } else if a.is_positional() {
        ArgKind::Positional
    } else {
        ArgKind::Flag
    };
    let takes = a.get_action().takes_values();
    ArgDoc {
        id: a.get_id().to_string(),
        kind,
        long: a.get_long().map(str::to_string),
        short: a.get_short(),
        value_name: takes.then(|| a.get_value_names().and_then(|v| v.first()).map_or_else(|| a.get_id().to_string().to_uppercase(), |n| n.to_string())),
        value: value_of(a),
        required: a.is_required_set(),
        multiple: takes && (matches!(a.get_action(), clap::ArgAction::Append) || a.get_num_args().is_some_and(|r| r.max_values() > 1)),
        default: a.get_default_values().first().map(|v| v.to_string_lossy().into_owned()),
        possible_values: if takes { a.get_possible_values().iter().filter(|v| !v.is_hide_set()).map(|v| v.get_name().to_string()).collect() } else { Vec::new() },
        asked: arg_meta(a).map_or(Asked::No, |m| m.asked),
        widens: a.get::<Widens>().is_some(),
        help: a.get_help().map(ToString::to_string).unwrap_or_default(),
    }
}

pub fn node(cmd: &Command, path: &str) -> Node {
    // One command and everything under it, as the CLI shows them: hidden commands and arguments, and clap's own help, are left out.
    node_under(cmd, path, &[])
}

fn node_under(cmd: &Command, path: &str, inherited: &[ArgDoc]) -> Node {
    // A command's own arguments, then the global ones of the commands above it (the program-wide ones on goethe itself stay there).
    let visible = |a: &&Arg| !a.is_hide_set() && !matches!(a.get_id().as_str(), "help" | "version");
    let own: Vec<ArgDoc> = cmd.get_arguments().filter(visible).map(arg_doc).collect();
    let mut down = inherited.to_vec();
    if path.contains(' ') {
        down.extend(cmd.get_arguments().filter(visible).filter(|a| a.is_global_set()).map(arg_doc));
    }
    Node {
        path: path.to_string(),
        about: cmd.get_about().map(ToString::to_string).unwrap_or_default(),
        aliases: cmd.get_visible_aliases().map(str::to_string).collect(),
        does: cmd_meta(cmd),
        examples: Vec::new(),
        args: own.into_iter().chain(inherited.iter().cloned()).collect(),
        commands: cmd
            .get_subcommands()
            .filter(|c| !c.is_hide_set() && c.get_name() != "help")
            .map(|c| node_under(c, &format!("{path} {}", c.get_name()), &down))
            .collect(),
    }
}

#[cfg(test)]
fn examples(cmd: &Command, out: &mut Vec<String>) {
    // The copyable lines (marked with an arrow) under each command's help.
    if let Some(after) = cmd.get_after_help() {
        let text = after.to_string();
        out.extend(text.lines().filter_map(|l| l.split_once("\u{2192} goethe ")).map(|(_, rest)| {
            let rest = rest.split("  ").next().unwrap_or(rest).trim().trim_end_matches(')');
            format!("goethe {rest}")
        }));
    }
    for sub in cmd.get_subcommands() {
        examples(sub, out);
    }
}

#[cfg(test)]
fn filled(example: &str) -> Vec<&str> {
    // An example with a sample value for each placeholder, as goethe would be run.
    example
        .split_whitespace()
        .map(|w| match w {
            "MODEL" => "org/m",
            "FOLDER" => "f",
            "KEY_ID" | "<key-id>" => "a1",
            "DOMAIN" => "a.example.com",
            "<their-ip>/32" => "203.0.113.7/32",
            w => w,
        })
        .collect()
}

#[cfg(test)]
fn example_path(root: &Command, example: &str) -> Result<Vec<String>, String> {
    // The command an example runs, found by parsing it the way goethe would.
    let m = root.clone().try_get_matches_from(filled(example)).map_err(|e| format!("{example}: {}", e.kind()))?;
    let mut path = Vec::new();
    let mut at = &m;
    while let Some((name, sub)) = at.subcommand() {
        path.push(name.to_string());
        at = sub;
    }
    Ok(path)
}

#[cfg(test)]
pub fn reference() -> Result<Node, String> {
    // The whole CLI with its examples: what docs/cli.json holds.
    use clap::CommandFactory;
    let root = crate::Cli::command();
    let mut doc = node(&root, "goethe");
    let mut lines = Vec::new();
    examples(&root, &mut lines);
    lines.extend(crate::HELP_SECTIONS.iter().flat_map(|(_, rows)| rows.iter()).map(|(c, _)| c.to_string()).filter(|c| !matches!(c.split_whitespace().nth(1), Some("ask" | "?") | None)));
    for ex in lines {
        let path = example_path(&root, &ex)?;
        let at = doc.find_mut(&path).ok_or_else(|| format!("{ex}: no command {path:?} in the reference"))?;
        if !at.examples.iter().any(|e| filled(e) == filled(&ex)) {
            at.examples.push(ex);
        }
    }
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    fn visible(cmd: &Command, path: &mut Vec<String>, out: &mut Vec<(String, Command)>) {
        for sub in cmd.get_subcommands().filter(|s| !s.is_hide_set() && s.get_name() != "help") {
            path.push(sub.get_name().to_string());
            out.push((path.join(" "), sub.clone()));
            visible(sub, path, out);
            path.pop();
        }
    }

    fn commands() -> Vec<(String, Command)> {
        let mut out = Vec::new();
        visible(&crate::Cli::command(), &mut vec!["goethe".to_string()], &mut out);
        out
    }

    #[test]
    fn every_visible_command_says_what_it_risks() {
        // A new command without a CmdMeta fails here, listed by name.
        let gaps: Vec<String> = commands().into_iter().filter(|(_, c)| cmd_meta(c).is_none()).map(|(p, _)| p).collect();
        assert!(gaps.is_empty(), "commands without CmdMeta: {gaps:#?}");
    }

    #[test]
    fn every_visible_argument_that_takes_a_value_says_what_it_takes() {
        // A value-taking argument needs a declared kind unless clap's possible values or number parser already say it.
        let mut gaps = Vec::new();
        for (path, cmd) in commands() {
            for a in cmd.get_arguments().filter(|a| !a.is_hide_set() && !a.is_last_set() && a.get_action().takes_values()) {
                if value_of(a).is_none() {
                    gaps.push(format!("{path} {}", a.get_id()));
                }
            }
        }
        assert!(gaps.is_empty(), "arguments without a value kind: {gaps:#?}");
    }

    #[test]
    fn asked_with_names_a_sibling_argument() {
        // A typo in asked_with would never ask; each one must name an argument of the same command.
        for (path, cmd) in commands() {
            for a in cmd.get_arguments() {
                if let Some(Asked::With(s)) = arg_meta(a).map(|m| m.asked) {
                    assert!(cmd.get_arguments().any(|b| b.get_id() == s), "{path} --{}: asked_with {s} is not an argument here", a.get_id());
                }
            }
        }
    }

    #[test]
    fn each_fact_has_exactly_one_command_that_gives_it() {
        // Ask puts the giver of a missing fact first, so there must be one and only one.
        for fact in [Fact::Daemon, Fact::Installed, Fact::Loaded, Fact::Key] {
            let givers: Vec<String> = commands().into_iter().filter(|(_, c)| cmd_meta(c).is_some_and(|m| m.gives.contains(&fact))).map(|(p, _)| p).collect();
            assert_eq!(givers.len(), 1, "{fact:?} is given by {givers:?}");
        }
    }

    #[test]
    fn every_visible_command_and_argument_has_help() {
        // Gaps in the descriptions are listed here, so they can be written.
        let doc = reference().expect("every example parses");
        let mut all = Vec::new();
        doc.walk(&mut all);
        let mut gaps: Vec<String> = all.iter().filter(|n| n.path != "goethe" && n.about.trim().is_empty()).map(|n| format!("{} (about)", n.path)).collect();
        for n in &all {
            gaps.extend(n.args.iter().filter(|a| a.help.trim().is_empty()).map(|a| format!("{} {}", n.path, a.id)));
        }
        assert!(gaps.is_empty(), "missing help: {gaps:#?}");
    }

    #[test]
    fn cli_json_is_current() {
        // docs/cli.json is generated from the CLI; a CLI change without a refreshed file fails here.
        let want = serde_json::to_string_pretty(&reference().expect("every example parses")).expect("serializable") + "\n";
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../docs/cli.json");
        if std::env::var_os("GOETHE_WRITE_CLI_JSON").is_some() {
            std::fs::write(&path, &want).expect("write docs/cli.json");
        }
        let have = std::fs::read_to_string(&path).unwrap_or_default();
        assert!(have == want, "docs/cli.json is out of date; refresh it with: GOETHE_WRITE_CLI_JSON=1 cargo test -p goethe cli_json");
    }

    #[test]
    fn completers_follow_the_value_kind() {
        // The arguments that completed before still complete, from the same lists.
        let cli = with_completers(crate::Cli::command());
        let arg = |path: &[&str], id: &str| {
            let mut c = &cli;
            for p in path {
                c = c.find_subcommand(p).unwrap_or_else(|| panic!("no {p}"));
            }
            c.get_arguments().find(|a| a.get_id() == id).unwrap_or_else(|| panic!("no {id}")).get::<ArgValueCandidates>().is_some()
        };
        for (path, id) in [
            (&["install"][..], "gpu"),
            (&["remove"][..], "model"),
            (&["load"][..], "model"),
            (&["load"][..], "gpu"),
            (&["unload"][..], "model"),
            (&["run"][..], "args"),
            (&["logs"][..], "target"),
            (&["gpu", "allow"][..], "gpus"),
            (&["login"][..], "platform_url"),
            (&["login"][..], "authkey"),
            (&["api", "expose"][..], "target"),
            (&["api", "expose", "vpn"][..], "interface"),
            (&["api", "keys", "revoke"][..], "ids"),
            (&["models", "alias", "remove"][..], "aliases"),
            (&["models", "alias", "set"][..], "model"),
            (&["gpu", "deny"][..], "gpus"),
            (&["api", "share", "stop"][..], "key"),
            (&["api", "share"][..], "key"),
            (&["trace", "on"][..], "key"),
            (&["trace", "off"][..], "key"),
            (&["learn", "on"][..], "model"),
        ] {
            assert!(arg(path, id), "{path:?} {id} lost its completer");
        }
    }
}
