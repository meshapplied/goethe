// Ask's checks: held-out request sets with their bars, scripted dialogues, the catalog against clap, and the GASK format.

use super::embed::{HybridRetriever, RetrievalMode, ShortlistBackend, StaticModel};
use super::planner::{PendingDecision, PlanAction, PlanLine, PlanState, PlanStep};
use super::schema::{build_catalog, IntentCard, Kind, Risk};
use super::slots::World;
use crate::Cli;
use clap::{Arg, ArgAction, Command, CommandFactory, Parser};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::time::Instant;

fn fixture_world() -> World {
    // A machine with three installed models (two Qwen sizes, to test a tie), an alias, GPUs, interfaces and keys; Goethe running.
    World {
        daemon_up: true,
        installed_known: true,
        installed: vec![
            "Qwen/Qwen2.5-7B-Instruct".into(),
            "Qwen/Qwen2.5-14B-Instruct".into(),
            "mistralai/Mistral-7B-Instruct".into(),
        ],
        aliases: vec![("fast".into(), "mistralai/Mistral-7B-Instruct".into())],
        gpus: vec![("gpu0".into(), "RTX 4090".into()), ("gpu1".into(), "RTX 4090".into()), ("gpu2".into(), "RTX 4090".into())],
        interfaces: vec![("wg0".into(), "10.0.0.1".into()), ("tailscale0".into(), "100.64.1.1".into())],
        keys: vec![("95f818903035".into(), "friend-demo".into()), ("4a12c09e1120".into(), "my-laptop".into())],
        log_targets: vec![("goethed".into(), "the daemon log".into()), ("gateway".into(), "the gateway log".into())],
        ..World::default()
    }
}

fn validate_cli(argv: &[String]) -> Result<(), clap::Error> {
    Cli::try_parse_from(argv).map(|_| ())
}

fn bench_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ask").join("bench")
}

fn load_rows(name: &str) -> Vec<Value> {
    let raw = std::fs::read_to_string(bench_dir().join(name)).expect("read bench file");
    serde_json::from_str::<Vec<Value>>(&raw).expect("parse benchmark json")
}

fn jstr(v: &Value, k: &str) -> String {
    v.get(k).and_then(Value::as_str).unwrap_or("").to_string()
}

fn jstrs(v: &Value, k: &str) -> Vec<String> {
    v.get(k)
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect())
        .unwrap_or_default()
}

fn collect_clap_paths(cmd: &Command, prefix: &mut Vec<String>, out: &mut HashSet<String>) {
    // Visible clap subcommand dotted ids (skip ask/help/hidden).
    for sub in cmd.get_subcommands() {
        if sub.is_hide_set() {
            continue;
        }
        let name = sub.get_name();
        if matches!(name, "help" | "ask") {
            continue;
        }
        prefix.push(name.to_string());
        out.insert(prefix.join("."));
        collect_clap_paths(sub, prefix, out);
        prefix.pop();
    }
}

fn minimal_argv(card: &IntentCard) -> Vec<String> {
    // argv_prefix plus a placeholder of the right kind for each positional clap says is missing, so clap can validate the leaf.
    let mut argv = card.argv_prefix.clone();
    let missing = validate_cli(&argv).err().map(|e| super::planner::missing_names(&e)).unwrap_or_default();
    let needed = |p: &&super::schema::PositionalSpec| missing.iter().any(|m| m.trim_start_matches('<').starts_with(&format!("{}>", p.value_name)));
    for pos in card.positionals.iter().filter(needed) {
        argv.push(match pos.kind {
            Kind::Model | Kind::InstalledModel | Kind::ModelOrAll | Kind::ModelInput => "mistralai/Mistral-7B-Instruct".into(),
            Kind::Duration | Kind::When => "1h".into(),
            Kind::Domain => "example.com".into(),
            Kind::Cidr => "203.0.113.7/32".into(),
            Kind::Gpu => "gpu0".into(),
            Kind::Key | Kind::KeyId | Kind::KeyName => "demo".into(),
            Kind::Alias | Kind::NewAlias => "fast".into(),
            Kind::Folder => "/tmp".into(),
            Kind::LogTarget => "goethed".into(),
            Kind::ExposeTarget | Kind::Interface => "wg0".into(),
            Kind::Url => "https://goethe.sh".into(),
            Kind::AuthKey => "msh_ak_x".into(),
            Kind::Number => "1".into(),
            Kind::Choice => pos.possible_values.first().cloned().unwrap_or_default(),
            Kind::Text => "x".into(),
        });
    }
    argv
}

#[test]
fn clap_catalog_covers_visible_commands() {
    // Every visible clap command is a command Ask can choose.
    let root = Cli::command();
    let cat = build_catalog(&root);
    let cat_ids: HashSet<&str> = cat.intents.iter().map(|i| i.id.as_str()).collect();
    let mut clap_paths = HashSet::new();
    collect_clap_paths(&root, &mut Vec::new(), &mut clap_paths);
    for path in &clap_paths {
        let covered = cat_ids.contains(path.as_str())
            || cat_ids.iter().any(|id| id.starts_with(&format!("{path}.")));
        assert!(covered, "clap command {path} missing from ask catalog");
    }
    assert!(cat.intents.len() >= 45, "expected full CLI coverage, got {}", cat.intents.len());
}

#[test]
fn clap_catalog_argv_prefixes_validate() {
    // Every command, with sample values for what clap requires, parses.
    let cat = build_catalog(&Cli::command());
    let mut failures = Vec::new();
    for card in &cat.intents {
        let argv = minimal_argv(card);
        if let Err(e) = validate_cli(&argv) {
            failures.push(format!("{} {:?}: {e}", card.id, argv));
        }
    }
    assert!(failures.is_empty(), "clap invalid catalog argv:\n{}", failures.join("\n"));
}

fn load_step(model: &str) -> PlanStep {
    let mut positionals = HashMap::new();
    positionals.insert("model".into(), vec![model.into()]);
    PlanStep {
        clause: model.into(),
        intent_id: "load".into(),
        argv: vec!["goethe".into(), "load".into(), model.into()],
        risk: Risk::Mutating,
        positionals,
        flags: Vec::new(),
        passthrough: Vec::new(),
        missing: Vec::new(),
        ambiguous: None,
        alternatives: Vec::new(),
        needs_intent_pick: false,
        verified: true,
        verify_err: None,
        reason: None,
        part: 0,
    }
}

fn cmds(lines: &[PlanLine]) -> Vec<String> {
    lines.iter().map(|l| l.argv.join(" ")).collect()
}

#[test]
fn world_orders_up_and_install_before_load() {
    // A missing model is installed first and a stopped Goethe started first; a path or an installed model needs neither.
    let cat = build_catalog(&Cli::command());
    let mut world = fixture_world();
    world.daemon_up = true;
    let order = |m: &str, world: &World| PlanState::with_goals(vec![load_step(m)], world, &cat, &validate_cli).lines;
    let lines = order("orcarouter/OrcaCyber-Nano-1.0", &world);
    assert_eq!(cmds(&lines), ["goethe install orcarouter/OrcaCyber-Nano-1.0", "goethe load orcarouter/OrcaCyber-Nano-1.0"]);
    assert!(lines[0].why.contains("not installed"));
    world.daemon_up = false;
    assert_eq!(cmds(&order("mistralai/Mistral-7B-Instruct", &world)), ["goethe up", "goethe load mistralai/Mistral-7B-Instruct"]);
    assert_eq!(cmds(&order("/tmp/org/tiny", &world)), ["goethe up", "goethe load /tmp/org/tiny"]);
    let both = order("orcarouter/OrcaCyber-Nano-1.0", &world);
    assert_eq!(cmds(&both), ["goethe up", "goethe install orcarouter/OrcaCyber-Nano-1.0", "goethe load orcarouter/OrcaCyber-Nano-1.0"], "install needs Goethe running too, and up is listed once");
}

#[test]
fn a_misread_request_is_never_ready_to_change_anything() {
    // Found on a real box: a model no command takes, a negation, a path that is no installed model, a stray word beside a command's name, the whole internet allowed in.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let plan = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli);
    let shown = |p: &PlanState| p.lines.iter().map(|l| l.argv.join(" ")).collect::<Vec<_>>();
    for q in ["uninstall mistral", "uninstall the qwen 7b model", "do not unload mistral", "never remove mistral", "don't stop goethe", "delete the file /tmp/x", "start the car"] {
        let p = plan(q);
        assert!(!p.is_ready(), "{q} is ready: {:?}", shown(&p));
    }
    let open = plan("allow 0.0.0.0/0");
    assert!(!open.is_ready() || open.has_destructive(), "the whole internet on one Enter: {:?}", shown(&open));
    assert!(super::embed::says("loads", "load") && super::embed::says("loading", "load") && !super::embed::says("loaded", "load"), "a past form asks about a state");
    for (q, want) in [("remove mistral", "goethe remove mistralai/Mistral-7B-Instruct"), ("load fast", "goethe load fast"), ("uninstall goethe", "goethe uninstall"), ("stop goethe", "goethe down")] {
        let p = plan(q);
        assert!(p.is_ready() && shown(&p).last().is_some_and(|l| l == want), "{q}: {:?}", shown(&p));
    }
}

#[test]
fn nothing_is_deleted_or_stopped_beside_words_about_something_else() {
    // Found by judging thousands of paraphrased requests: an alias stood for its model in "remove fast" and deleted the model; a changing command's own word beside a word no command knows ("kill all processes", "open youtube") ran the command.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let plan = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli);
    let shown = |p: &PlanState| p.lines.iter().map(|l| l.argv.join(" ")).collect::<Vec<_>>();
    for q in ["remove fast", "please remove fast", "kill all processes", "open youtube"] {
        let p = plan(q);
        assert!(!p.is_ready(), "{q} is ready: {:?}", shown(&p));
    }
    for (q, want) in [("remove mistral", "goethe remove mistralai/Mistral-7B-Instruct"), ("kill goethe", "goethe down"), ("stop sharing", "goethe api share stop"), ("cancel the share", "goethe api share stop"), ("shut everything down", "goethe down")] {
        let p = plan(q);
        assert!(p.is_ready() && shown(&p).last().is_some_and(|l| l == want), "{q}: {:?}", shown(&p));
    }
}

#[test]
fn a_request_of_many_steps_binds_each_value_to_its_own_step() {
    // The owner's request: two models on two GPUs, a third GPU blocked, a key scoped to each model; a step without a command word continues the one before, and "each" repeats over the models bound earlier.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let want = [
        ("load", "goethe load mistralai/Mistral-7B-Instruct --gpu gpu0"),
        ("load", "goethe load Qwen/Qwen2.5-7B-Instruct --gpu gpu1"),
        ("gpu.deny", "goethe gpu deny gpu2"),
        ("api.keys.create", "--models mistralai/Mistral-7B-Instruct"),
        ("api.keys.create", "--models Qwen/Qwen2.5-7B-Instruct"),
    ];
    for q in [
        "load mistral on gpu0 and qwen 7b on gpu1, deny gpu2, then create an api key for mistral and another for qwen 7b",
        "load mistral on gpu0 and qwen 7b on gpu1, deny gpu2, then create an api key for each",
    ] {
        let p = PlanState::build(q, &cat, &ret, &world, &validate_cli);
        let got: Vec<(String, String)> = p.goals.iter().filter(|g| g.reason.is_none()).map(|g| (g.intent_id.clone(), g.argv.join(" "))).collect();
        assert_eq!(got.len(), want.len(), "{q}: {got:?}");
        assert!(StaticModel::load_default().is_none() || p.goals.iter().all(|g| !g.needs_intent_pick), "{q}: with the embedding model every step's command is clear: {got:?}");
        for ((id, argv), (want_id, want_argv)) in got.iter().zip(want) {
            assert!(id == want_id && argv.contains(want_argv), "{q}: {id} `{argv}` should be {want_id} with `{want_argv}`; all: {got:?}");
        }
    }
}

#[test]
fn a_number_takes_the_flag_its_words_name_and_a_count_must_match() {
    // A number goes to the number flag whose own name or help has the word beside it; a count ("two keys") must match the steps planned; any other number blocks the plan.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let plan = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli);
    let shown = |p: &PlanState| p.goals.iter().map(|g| g.argv.join(" ")).collect::<Vec<_>>();
    for (q, want) in [
        ("load mistral with 32k context", "goethe load mistralai/Mistral-7B-Instruct --ctx 32768"),
        ("load mistral with a context of 8192", "goethe load mistralai/Mistral-7B-Instruct --ctx 8192"),
        ("load mistral split across 2 gpus", "goethe load mistralai/Mistral-7B-Instruct --tp 2"),
    ] {
        let p = plan(q);
        assert!(p.is_ready() && shown(&p).last().is_some_and(|l| l == want), "{q}: {:?}", shown(&p));
    }
    for q in ["load mistral with 32k", "load mistral 7", "load mistral and qwen 7b, then create three api keys for each"] {
        let p = plan(q);
        assert!(!p.is_ready(), "{q}: a number no flag or count takes is not dropped: {:?}", shown(&p));
    }
    let two = plan("load mistral and qwen 7b, then create two api keys, one for each");
    assert_eq!(two.goals.iter().filter(|g| g.intent_id == "api.keys.create").count(), 2, "{:?}", shown(&two));
    assert!(two.goals.iter().all(|g| !g.argv.iter().any(|a| a == "two")), "a count is no key name: {:?}", shown(&two));
    assert!(two.unused(two.parts.len() - 1, &world).is_empty(), "the count matches: {:?}", two.unused(two.parts.len() - 1, &world));
}

#[test]
fn each_step_is_checked_against_the_state_the_steps_before_it_leave() {
    // A GPU must exist and not be denied, on the machine or by an earlier step; what a step undoes is gone for the steps after it; a stopped Goethe serves nothing.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let plan = |q: &str, w: &World| PlanState::build(q, &cat, &ret, w, &validate_cli);
    let mut world = fixture_world();
    world.denied = vec!["gpu1".into()];
    for q in ["deny gpu0, then load mistral on gpu0", "load mistral on gpu1", "load mistral on gpu7"] {
        let p = plan(q, &world);
        assert!(!p.is_ready(), "{q} is ready: {:?}", cmds(&p.lines));
        assert!(p.goals.iter().any(|g| g.verify_err.as_deref().is_some_and(|e| e.contains("gpu"))), "{q}: the reason names the GPU: {:?}", p.goals.iter().map(|g| &g.verify_err).collect::<Vec<_>>());
    }
    let allowed = plan("allow gpu1, then load mistral on gpu1", &world);
    assert!(allowed.is_ready(), "{:?}", allowed.goals.iter().map(|g| &g.verify_err).collect::<Vec<_>>());
    assert_eq!(cmds(&plan("stop goethe and load mistral", &world).lines), ["goethe down", "goethe up", "goethe load mistralai/Mistral-7B-Instruct"]);
    let mut stale = fixture_world();
    stale.daemon_up = false;
    stale.loaded = vec!["mistralai/Mistral-7B-Instruct".into()];
    assert!(!stale.has(crate::spec::Fact::Loaded, Some("mistralai/Mistral-7B-Instruct")), "a stopped Goethe serves nothing, whatever its rows say");
    stale.daemon_up = true;
    assert!(stale.has(crate::spec::Fact::Loaded, Some("mistralai/Mistral-7B-Instruct")));
}

#[test]
fn it_means_the_model_before_and_a_negated_part_is_left_out() {
    // "it" is the model the step before bound; a part that says not to do something is left out, never planned, and the rest still plans.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let plan = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli);
    let it = plan("load qwen 7b, then chat with it");
    assert!(it.goals.iter().any(|g| g.intent_id == "run" && g.argv.iter().any(|a| a == "Qwen/Qwen2.5-7B-Instruct")), "{:?}", it.goals.iter().map(|g| g.argv.join(" ")).collect::<Vec<_>>());
    let p = plan("load mistral but don't share it");
    assert_eq!(p.goals.iter().map(|g| g.argv.join(" ")).collect::<Vec<_>>(), ["goethe load mistralai/Mistral-7B-Instruct"]);
    assert!(p.is_ready() && p.parts.iter().any(|x| x.left_out && x.text.contains("don't")), "{:?}", p.parts);
}

#[test]
fn a_step_repeated_for_each_model_stays_repeated_when_picked_and_answered() {
    // Picking the command of a part that says "each" keeps one command per earlier model, and each one asks for its own key name.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let mut p = PlanState::build("load mistral and qwen 7b, then create an api key for each", &cat, &ret, &world, &validate_cli);
    let keys = p.parts.len() - 1;
    p.apply_to(keys, PlanAction::SelectIntent { intent_id: "api.keys.create".into() }, &cat, &world, &validate_cli);
    assert_eq!(p.goals.iter().filter(|g| g.intent_id == "api.keys.create").count(), 2, "{:?}", cmds(&p.lines));
    for name in ["first", "second"] {
        match p.next_decision(&cat, &world) {
            PendingDecision::PromptSlot { missing } => p.apply(PlanAction::SetSlot { is_flag: false, name: missing.name, values: vec![name.into()] }, &cat, &world, &validate_cli),
            other => panic!("{name}: {other:?}"),
        }
    }
    assert!(p.is_ready(), "{:?}", cmds(&p.lines));
    let lines = cmds(&p.lines);
    assert!(lines.contains(&"goethe api keys create first --models mistralai/Mistral-7B-Instruct".to_string()) && lines.contains(&"goethe api keys create second --models Qwen/Qwen2.5-7B-Instruct".to_string()), "{lines:?}");
    let mut menu = super::MenuAsk::new(Cli::command(), Box::new(validate_cli));
    menu.world = world;
    menu.plan = Some(p);
    menu.sync(&cat);
    assert_eq!(menu.frame.parts[keys].label, "api keys create x2", "the screen says the part runs twice");
}

#[test]
fn the_owners_request_reads_as_meant() {
    // The owner's own words: a wish in the past form names its command, a word only one command has names it, and a step repeated over models pairs up with as many values of another kind, in order.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let plan = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli);
    let argvs = |p: &PlanState| p.goals.iter().filter(|g| g.reason.is_none()).map(|g| (g.intent_id.clone(), g.argv.join(" "))).collect::<Vec<_>>();
    let wish = plan("i want qwen 7b loaded");
    assert!(wish.is_ready() && argvs(&wish).iter().any(|(id, _)| id == "load"), "{:?}", argvs(&wish));
    assert!(!plan("is qwen 7b loaded").goals.iter().any(|g| g.intent_id == "load" && !g.needs_intent_pick), "a question plans no load");
    let block = plan("block gpu2");
    assert!(block.is_ready() && argvs(&block) == [("gpu.deny".to_string(), "goethe gpu deny gpu2".to_string())], "{:?}", argvs(&block));
    let p = plan("install mistral and qwen 7b, load them on gpu0 and gpu1, block gpu2, then create two api keys, one for each");
    let got = argvs(&p);
    for want in ["goethe load mistralai/Mistral-7B-Instruct --gpu gpu0", "goethe load Qwen/Qwen2.5-7B-Instruct --gpu gpu1", "goethe gpu deny gpu2", "goethe api keys create --models mistralai/Mistral-7B-Instruct", "goethe api keys create --models Qwen/Qwen2.5-7B-Instruct"] {
        assert!(got.iter().any(|(_, a)| a == want), "missing `{want}`: {got:?}");
    }
    assert!(p.goals.iter().all(|g| !g.needs_intent_pick), "{got:?}");
    let literal = plan("i want like have 2 models installed on gpu0 and gpu1, but block gpu2, then give me two scoped to (one model only) API keys for both");
    let got = argvs(&literal);
    assert!(got.first().is_some_and(|(id, _)| id == "install") && got.iter().any(|(_, a)| a == "goethe gpu deny gpu2"), "{got:?}");
    assert!(!literal.is_ready() && !literal.has_destructive(), "it asks, never guesses: {got:?}");
}

#[test]
fn filler_in_front_and_the_order_of_values_never_change_the_plan() {
    // Metamorphic: a polite wish before a request, or its values in another order, plans the very same commands.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let plan = |q: &str| cmds(&PlanState::build(q, &cat, &ret, &world, &validate_cli).lines);
    for q in ["load mistral", "remove mistral", "stop goethe", "deny gpu2", "load mistral with 32k context on gpu0", "load mistral on gpu0 and qwen 7b on gpu1", "create an api key for mistral"] {
        for wish in ["please", "could you", "i want to", "can you please"] {
            assert_eq!(plan(&format!("{wish} {q}")), plan(q), "'{wish} {q}'");
        }
    }
    assert_eq!(plan("load mistral on gpu0 with 32k context"), plan("load mistral with 32k context on gpu0"));
    assert_eq!(plan("share the api for 2 hours with key friend-demo for 203.0.113.7"), plan("share the api with key friend-demo for 203.0.113.7 for 2 hours"));
}

#[test]
fn a_part_page_reads_short_descriptions_and_value_labels() {
    // Rows drop a leading "The"; a missing value reads "Key name:", not "Which key name?".
    let cat = build_catalog(&Cli::command());
    let about = |argv: &str| cat.intents.iter().find(|c| c.argv_prefix.join(" ") == argv).map(IntentCard::summary).unwrap();
    assert_eq!(about("goethe api"), "API address and how to connect");
    assert_eq!(about("goethe install"), "Download a model from Hugging Face or local path");
    assert!(cat.intents.iter().all(|c| !c.summary().starts_with("The ")));
    let missing = |label: &str| super::planner::MissingSlot { is_flag: false, name: String::new(), label: label.into(), kind: Kind::KeyName, possible: Vec::new(), shown: String::new() };
    assert_eq!(super::question(&missing("KEY_NAME")), "Key name:");
    assert_eq!(super::question(&missing("--allow")), "Allow:");
}

#[test]
fn a_request_runs_what_it_says_or_asks() {
    // The rows of the failure table: a value is used or reported, never dropped; a destructive guess is never ready.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let plan = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli);
    let argv = |p: &PlanState| primary(p).map(|g| g.argv.join(" ")).unwrap_or_default();
    for (q, value) in [("expose the api on wg0", "wg0"), ("expose the api on 10.0.0.5", "10.0.0.5"), ("login with auth key msh_ak_123", "msh_ak_123")] {
        let p = plan(q);
        assert!(!p.is_ready() || argv(&p).contains(value), "{q}: ready without {value}: {}", argv(&p));
    }
    assert_eq!(argv(&plan("expose the api on wg0")), "goethe api expose wg0");
    assert_eq!(argv(&plan("login with auth key msh_ak_123")), "goethe login --authkey msh_ak_123");

    let mut p = plan("show status for 203.0.113.7");
    assert!(!p.is_ready(), "an address status does not take is reported");
    let PendingDecision::Unused { part, values } = p.next_decision(&cat, &world) else { panic!("{:?}", p.next_decision(&cat, &world)) };
    assert_eq!(values, ["203.0.113.7"]);
    p.apply_to(part, PlanAction::Ignore { value: values[0].clone() }, &cat, &world, &validate_cli);
    assert!(p.is_ready(), "left out, it runs: {:?}", p.next_decision(&cat, &world));

    for q in ["remove alias fast", "remove the alias fast", "remove 203.0.113.7 from the allowlist"] {
        let p = plan(q);
        assert!(!(p.is_ready() && primary(&p).is_some_and(|g| g.intent_id == "remove")), "{q}: deleting a model is never the ready guess");
    }

    let share = plan("share the api for 2 hours with key friend-demo for 203.0.113.7");
    let got = argv(&share);
    assert!(got.starts_with("goethe api share 2h --key friend-demo --allow 203.0.113.7"), "{got}");

    let key = plan("create an api key");
    assert!(key.lines.iter().any(|l| l.placeholders == ["<KEY_NAME>"]), "the missing name shows in the command: {:?}", key.lines);
}

#[test]
fn a_written_model_id_means_a_command_that_takes_a_model() {
    // No word lists: the id narrows the field to commands that take a model, the ranker picks among them, and a vague request asks.
    let mut world = fixture_world();
    world.daemon_up = true;
    let cat = build_catalog(&Cli::command());
    let model = StaticModel::load_default();
    let dense = model.is_some();
    let ret = HybridRetriever::build(&cat, model);
    let plan = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli);
    for (q, want) in [
        ("load orcarouter/OrcaCyber-Nano-1.0", vec!["goethe install orcarouter/OrcaCyber-Nano-1.0", "goethe load orcarouter/OrcaCyber-Nano-1.0"]),
        ("download orcarouter/OrcaCyber-Nano-1.0 please", vec!["goethe install orcarouter/OrcaCyber-Nano-1.0"]),
        ("stop serving mistralai/Mistral-7B-Instruct", vec!["goethe unload mistralai/Mistral-7B-Instruct"]),
        ("delete mistralai/Mistral-7B-Instruct from disk", vec!["goethe remove mistralai/Mistral-7B-Instruct"]),
        ("serve mistralai/Mistral-7B-Instruct on the gpus", vec!["goethe load mistralai/Mistral-7B-Instruct"]),
        ("show logs for mistralai/Mistral-7B-Instruct", vec!["goethe logs mistralai/Mistral-7B-Instruct"]),
        ("chat with mistralai/Mistral-7B-Instruct", vec!["goethe run mistralai/Mistral-7B-Instruct"]),
    ] {
        let p = plan(q);
        let top = p.goals.iter().find(|g| g.reason.is_none()).map(|s| s.argv.join(" "));
        assert_eq!(top.as_deref(), want.last().copied(), "{q}: the right command is the plan or the first choice");
        if dense || p.is_ready() {
            assert!(p.is_ready(), "{q}: {:?}", p.goals.first().map(|s| &s.alternatives));
            assert_eq!(cmds(&p.lines), want, "{q}");
        }
        assert!(p.lines.iter().all(|l| !l.why.is_empty()), "{q}: every command says what it does");
    }
    for (q, offered) in [
        ("i want mistralai/Mistral-7B-Instruct running", "load"),
        ("uninstall mistralai/Mistral-7B-Instruct", "remove"),
    ] {
        let p = plan(q);
        let step = p.goals.first().expect(q);
        assert!(step.alternatives.iter().all(|a| cat.get(&a.id).is_some_and(|c| c.positionals.iter().any(|x| matches!(x.kind, Kind::Model | Kind::InstalledModel | Kind::ModelInput | Kind::LogTarget)))), "{q}: only model commands are offered");
        assert!(step.needs_intent_pick || step.intent_id == offered, "{q}: vague, so it asks: {:?}", step.alternatives);
        assert!(step.alternatives.iter().any(|a| a.id == offered), "{q}: {offered} is among the choices");
    }
}

#[test]
fn a_model_named_by_part_of_its_name_is_found_or_asked() {
    // Words of the request against the installed names: one clear match is used, a tie is a choice, nothing is a question.
    let world = fixture_world();
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let step = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli).goals.into_iter().next().unwrap();
    assert_eq!(step("load qwen 14b").argv, ["goethe", "load", "Qwen/Qwen2.5-14B-Instruct"]);
    assert_eq!(step("load qwen").ambiguous.map(|a| a.1.len()), Some(2));
    assert!(step("load my model").missing.iter().any(|m| m.name == "model"));
}

#[test]
fn a_request_in_parts_plans_each_part() {
    // "X and Y": each part is a command, the model is shared, repeats merge, and an API for a model gets it served and a key.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let mut world = fixture_world();
    world.daemon_up = false;
    world.keys.clear();
    let live = world.clone();
    let mut p = PlanState::build("install and give me an api to orcarouter/OrcaCyber-Nano-1.0", &cat, &ret, &world, &validate_cli);
    if p.parts.len() < 2 {
        assert!(StaticModel::load_default().is_none(), "with the embedding model this is two parts: {:?}", p.parts);
    } else {
        let mut asked = Vec::new();
        for _ in 0..4 {
            match p.next_decision(&cat, &live) {
                PendingDecision::PickIntent { clause, options } => {
                    assert!(options.iter().any(|o| o.id == "api"), "{clause}: {options:?}");
                    assert!(clause.contains("api"), "the unclear part is asked about by its own words: {clause}");
                    asked.push("which command");
                    p.apply(PlanAction::SelectIntent { intent_id: "api".into() }, &cat, &live, &validate_cli);
                }
                PendingDecision::PromptSlot { missing } => {
                    asked.push("key name");
                    p.apply(PlanAction::SetSlot { is_flag: false, name: missing.name, values: vec!["my-app".into()] }, &cat, &live, &validate_cli);
                }
                PendingDecision::ReadyToConfirm => break,
                other => panic!("{other:?}"),
            }
        }
        assert_eq!(asked.last(), Some(&"key name"), "the key name is asked last, once goethe api is settled: {asked:?}");
        assert!(p.is_ready(), "{:?}", p.goals);
        assert_eq!(
            cmds(&p.lines),
            [
                "goethe up",
                "goethe install orcarouter/OrcaCyber-Nano-1.0",
                "goethe api keys create my-app",
                "goethe load orcarouter/OrcaCyber-Nano-1.0",
                "goethe api",
            ]
        );
    }
    world.daemon_up = true;
    let plan = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli);
    assert_eq!(cmds(&plan("install org/foo and load it").lines), ["goethe install org/foo", "goethe load org/foo"], "shared model, one install");
    assert_eq!(cmds(&plan("unload mistral and stop goethe").lines), ["goethe unload mistralai/Mistral-7B-Instruct", "goethe down"]);
    assert_eq!(cmds(&plan("turn learn on and then turn trace on").lines), ["goethe learn on", "goethe trace on"]);
}

#[test]
fn requests_split_into_any_number_of_parts() {
    // Two, three or four parts; the same command twice stays two parts; a bare value after a comma is a detail of its part; steps land under the part that needs them.
    let cat = build_catalog(&Cli::command());
    let model = StaticModel::load_default();
    let dense = model.is_some();
    let ret = HybridRetriever::build(&cat, model);
    let world = fixture_world();
    let plan = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli);
    let three = plan("show status, show gpus and list models");
    if dense {
        assert_eq!(three.parts.len(), 3, "{:?}", three.parts);
        let offers = |i: usize, id: &str| three.parts[i].options.iter().any(|o| o.id == id);
        assert!(offers(0, "status") && offers(1, "gpu") && offers(2, "models"), "{:?}", three.parts);
    }
    let two_loads = plan("load qwen 7b and load mistral");
    assert_eq!(two_loads.parts.len(), 2);
    assert_eq!(cmds(&two_loads.lines), ["goethe load Qwen/Qwen2.5-7B-Instruct", "goethe load mistralai/Mistral-7B-Instruct"], "each part keeps its own model");
    let four = plan("show status, install org/foo and load it, then show logs");
    assert_eq!(four.parts.iter().map(|p| p.text.as_str()).collect::<Vec<_>>(), ["show status", "install org/foo", "load it", "show logs"]);
    let owner = |p: &PlanState, cmd: &str| p.lines.iter().find(|l| l.argv.join(" ") == cmd).map(|l| l.part);
    assert_eq!(owner(&four, "goethe install org/foo"), Some(1));
    assert_eq!(owner(&four, "goethe load org/foo"), Some(2), "the shared model, under its own part");
    assert!(four.lines.iter().any(|l| l.part == 3), "the fourth part has its own command: {:?}", four.lines);
    let detail = plan("api allow add 203.0.113.7/32, 198.51.100.4/32");
    assert_eq!(detail.parts.len(), 1, "a bare address is a detail, not a request: {:?}", detail.parts);
    assert_eq!(cmds(&detail.lines), ["goethe api allow add 203.0.113.7/32 198.51.100.4/32"]);
}

#[test]
fn one_request_with_and_is_never_lost() {
    // "and" inside one request ("check the machine and the setup"): its command stays in the plan, or Ask asks about the extra part.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let mut lost = Vec::new();
    for row in load_rows("paraphrase.json") {
        let q = jstr(&row, "q");
        if !q.contains(" and ") {
            continue;
        }
        let p = PlanState::build(&q, &cat, &ret, &world, &validate_cli);
        let asks = !p.is_ready();
        let has_it = p.goals.iter().any(|g| g.intent_id == jstr(&row, "intent"));
        if !asks && p.goals.len() > 1 && !has_it {
            lost.push(q);
        }
    }
    assert!(lost.is_empty(), "a ready plan without the sentence's own command: {lost:?}");
}

#[test]
fn plan_build_never_panics_on_unicode_junk() {
    // Pasted junk never panics planning or ranking.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let live = fixture_world();
    let long = "\u{5B57}".repeat(4000);
    let cases = [
        "\u{2026}".to_string(),
        "\u{2014} \u{2013} \u{2212}".to_string(),
        "\u{1F525} load mistral".to_string(),
        "expose api locally \u{2026}".to_string(),
        "share for 2\u{1F600} hours".to_string(),
        "\u{1F600}m \u{2460}h caf\u{00E9}".to_string(),
        "status | tee /tmp/x".into(),
        "up && down".into(),
        "\"unterminated".into(),
        "'also open".into(),
        "\u{FEFF}show status".into(),
        "load ./\u{6A21}\u{578B}/foo".into(),
        "api allow add 203.0.113.7/32 \u{2026} please".to_string(),
        long,
        "".into(),
        "   ".into(),
        "\n\t".into(),
        "a && b || c; d | e > f < g `h` $(i)".into(),
    ];
    for q in &cases {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = PlanState::build(q, &cat, &ret, &live, &validate_cli);
            let _ = ret.rank(q, 5);
        }));
        assert!(result.is_ok(), "panic on ask input {q:?}");
    }
}

#[test]
fn schema_drift_synthetic_zero_shot() {
    // A command Ask has never seen is still found from its help text, flag included.
    let synthetic = Command::new("goethe")
        .subcommand(Command::new("status").about("Show daemon status."))
        .subcommand(
            Command::new("telemetry")
                .about("Inspect local prometheus telemetry counters.")
                .arg(Arg::new("export_csv").long("export-csv").action(ArgAction::SetTrue).help("Export counters as csv.")),
        );
    let cat = build_catalog(&synthetic);
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let live = World { daemon_up: true, installed_known: true, ..World::default() };
    let val = |argv: &[String]| -> Result<(), clap::Error> { synthetic.clone().try_get_matches_from(argv).map(|_| ()) };
    let plan = PlanState::build("inspect telemetry counters --export-csv", &cat, &ret, &live, &val);
    assert!(plan.is_ready(), "synthetic command should route zero-shot: {plan:?}");
    assert_eq!(plan.goals.first().map(|s| s.argv.clone()), Some(vec!["goethe".into(), "telemetry".into(), "--export-csv".into()]));
}

#[test]
fn wordpiece_and_gask_v1_roundtrip() {
    // A minimal GASK v1 table in memory: tokenizing and mean-pooling work; an adapter matrix or trailing bytes are refused.
    let vocab = ["[UNK]", "show", "gpu", "##s", "status"];
    let table = |adapter: u32| {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GASK");
        for v in [1u32, vocab.len() as u32, 4, 0, adapter] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        for t in vocab {
            bytes.extend_from_slice(&(t.len() as u16).to_le_bytes());
            bytes.extend_from_slice(t.as_bytes());
        }
        for _ in 0..vocab.len() {
            bytes.extend_from_slice(&(1.0f32 / 127.0).to_le_bytes());
        }
        let rows: [[i8; 4]; 5] = [[0, 0, 0, 0], [127, 0, 0, 0], [0, 127, 0, 0], [0, 0, 127, 0], [0, 0, 0, 127]];
        bytes.extend(rows.iter().flatten().map(|q| *q as u8));
        bytes
    };
    let model = StaticModel::from_bytes(&table(0)).expect("parse GASK v1");
    assert_eq!(model.dim(), 4);
    assert_eq!(model.tokenize("show gpus"), vec![1, 2, 3]);
    let v = model.embed("show gpus");
    assert!((v[0] - v[1]).abs() < 1e-4 && (v[1] - v[2]).abs() < 1e-4 && v[3].abs() < 1e-4);
    assert!(StaticModel::from_bytes(&table(1)).is_err(), "an adapter is refused");
    let mut long = table(0);
    long.push(0);
    assert!(StaticModel::from_bytes(&long).is_err(), "trailing bytes are refused");
}

#[test]
fn multi_turn_dialogue_simulator() {
    // Scripted answers through the plan: ambiguous commands, ambiguous models and missing flags.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let live = fixture_world();
    type TurnScenario<'a> = (&'a str, Vec<(&'a str, &'a str)>, Vec<&'a str>);
    let scenarios: &[TurnScenario<'_>] = &[
        ("remove qwen", vec![("slot", "Qwen/Qwen2.5-7B-Instruct")], vec!["goethe", "remove", "Qwen/Qwen2.5-7B-Instruct"]),
        ("load qwen", vec![("slot", "Qwen/Qwen2.5-14B-Instruct")], vec!["goethe", "load", "Qwen/Qwen2.5-14B-Instruct"]),
        (
            "share api 2h",
            vec![("slot", "friend-demo"), ("slot", "203.0.113.7/32")],
            vec!["goethe", "api", "share", "2h", "--key", "friend-demo", "--allow", "203.0.113.7/32"],
        ),
        (
            "expose api public",
            vec![("slot", "203.0.113.7/32")],
            vec!["goethe", "api", "expose", "public", "--allow", "203.0.113.7/32"],
        ),
        (
            "open the api on a public address for 2 hours",
            vec![("intent", "api.share"), ("slot", "friend-demo"), ("slot", "203.0.113.7/32")],
            vec!["goethe", "api", "share", "2h", "--key", "friend-demo", "--allow", "203.0.113.7/32"],
        ),
        (
            "create an api key",
            vec![("slot", "prod-app")],
            vec!["goethe", "api", "keys", "create", "prod-app"],
        ),
        ("revoke an api key", vec![("slot", "4a12c09e1120")], vec!["goethe", "api", "keys", "revoke", "4a12c09e1120"]),
        (
            "allow client ip addresses",
            vec![("intent", "api.allow.add"), ("slot", "198.51.100.5/32")],
            vec!["goethe", "api", "allow", "add", "198.51.100.5/32"],
        ),
        ("deny a gpu", vec![("slot", "gpu2")], vec!["goethe", "gpu", "deny", "gpu2"]),
        ("show logs", vec![("intent", "logs"), ("slot", "goethed")], vec!["goethe", "logs", "goethed"]),
        ("check domain", vec![("slot", "llm.example.com")], vec!["goethe", "api", "domain", "check", "llm.example.com"]),
        ("set model alias", vec![("slot", "m7"), ("slot", "mistralai/Mistral-7B-Instruct")], vec!["goethe", "models", "alias", "set", "m7", "mistralai/Mistral-7B-Instruct"]),
    ];
    for (q, answers, expected) in scenarios {
        let mut plan = PlanState::build(q, &cat, &ret, &live, &validate_cli);
        for (kind, val) in answers {
            match (plan.next_decision(&cat, &live), *kind) {
                (PendingDecision::PickIntent { .. }, "intent") => {
                    plan.apply(PlanAction::SelectIntent { intent_id: (*val).to_string() }, &cat, &live, &validate_cli);
                }
                (PendingDecision::PickSlot { .. } | PendingDecision::PromptSlot { .. }, "intent") => {}
                (PendingDecision::PickSlot { missing, .. } | PendingDecision::PromptSlot { missing }, "slot") => {
                    plan.apply(
                        PlanAction::SetSlot { is_flag: missing.is_flag, name: missing.name, values: vec![(*val).to_string()] },
                        &cat,
                        &live,
                        &validate_cli,
                    );
                }
                (other, _) => panic!("unexpected decision for '{q}': {other:?}"),
            }
        }
        assert!(plan.is_ready(), "dialogue not ready for '{q}': {plan:?}");
        let got: Vec<&str> = primary(&plan).map(|s| s.argv.iter().map(String::as_str).collect()).unwrap_or_default();
        assert_eq!(got, *expected, "dialogue for '{q}'");
    }
}

struct EvalSummary {
    name: &'static str,
    core_exact: f64,
    para_top1: f64,
    para_top3: f64,
    ambig_clarify: f64,
    multi_exact: f64,
    flags_exact: f64,
    meta_reject: f64,
    ood_abstain: f64,
    clap_valid: f64,
    p95_ms: f64,
}

fn primary(p: &PlanState) -> Option<&PlanStep> {
    // The first command the request itself asked for, after any step Goethe's state put before it.
    p.goals.iter().find(|g| g.reason.is_none())
}

fn eval_system(name: &'static str, report_misses: bool, ret: &dyn ShortlistBackend, cat: &super::schema::CliCatalog, live: &World) -> EvalSummary {
    // One ranking setup over every held-out set: how often it is right, and how often clap accepts what it plans.
    let core = load_rows("core.json");
    let para = load_rows("paraphrase.json");
    let ambig = load_rows("ambiguous.json");
    let multi = load_rows("multistep.json");
    let flags = load_rows("flags_and_passthrough.json");
    let ood = load_rows("ood_and_adversarial.json");
    let mut latencies_ms = Vec::new();
    let mut emitted_total = 0usize;
    let mut emitted_valid = 0usize;
    let mut core_ok = 0usize;
    for c in &core {
        let q = jstr(c, "q");
        let intent = jstr(c, "intent");
        let argv = jstrs(c, "argv");
        let t0 = Instant::now();
        let p = PlanState::build(&q, cat, ret, live, &validate_cli);
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        if p.is_ready() {
            emitted_total += 1;
            emitted_valid += usize::from(primary(&p).is_some_and(|s| s.verified));
        }
        let got_ok = primary(&p).is_some_and(|s| s.intent_id == intent && s.argv == argv && s.verified);
        if got_ok {
            core_ok += 1;
        } else if report_misses {
            eprintln!("  [core miss] '{}' -> {:?} (ready={})", q, primary(&p).map(|s| (&s.intent_id, &s.argv, s.needs_intent_pick)), p.is_ready());
        }
    }
    let mut para_1 = 0usize;
    let mut para_3 = 0usize;
    for c in &para {
        let q = jstr(c, "q");
        let intent = jstr(c, "intent");
        let t0 = Instant::now();
        let ranked = ret.rank(&q, 3);
        let p = PlanState::build(&q, cat, ret, live, &validate_cli);
        latencies_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        if p.is_ready() {
            emitted_total += 1;
            emitted_valid += usize::from(primary(&p).is_some_and(|s| s.verified));
        }
        if ranked.first().is_some_and(|r| r.id == intent) {
            para_1 += 1;
        }
        if ranked.iter().take(3).any(|r| r.id == intent) {
            para_3 += 1;
        } else if report_misses {
            eprintln!("  [para miss] '{}' want={} got={:?}", q, intent, ranked.iter().map(|r| (&r.id, r.score)).collect::<Vec<_>>());
        }
    }
    let mut ambig_ok = 0usize;
    for c in &ambig {
        let q = jstr(c, "q");
        let expected_in_top3 = c.get("expected_in_top3").map(|_| jstrs(c, "expected_in_top3"));
        let expected_slot = c.get("expected_slot").and_then(Value::as_str);
        let p = PlanState::build(&q, cat, ret, live, &validate_cli);
        let ok = match p.next_decision(cat, live) {
            PendingDecision::PickIntent { options, .. } => expected_in_top3
                .as_ref()
                .map(|want| options.iter().take(3).any(|o| want.contains(&o.id)))
                .unwrap_or(true),
            PendingDecision::PickSlot { missing, .. } | PendingDecision::PromptSlot { missing, .. } => {
                expected_slot.map(|w| w == missing.name).unwrap_or(true)
            }
            _ => false,
        };
        if ok {
            ambig_ok += 1;
        } else if report_misses {
            eprintln!("  [ambig miss] '{}' -> {:?}", q, p.next_decision(cat, live));
        }
    }
    let mut multi_ok = 0usize;
    for c in &multi {
        let q = jstr(c, "q");
        let p = PlanState::build(&q, cat, ret, live, &validate_cli);
        if p.is_ready() {
            emitted_total += 1;
            emitted_valid += usize::from(primary(&p).is_some_and(|s| s.verified));
        }
        let ok = if c.get("argv").is_some() {
            let argv = jstrs(c, "argv");
            primary(&p).is_some_and(|s| s.argv == argv && s.verified)
        } else if c.get("single").and_then(Value::as_bool).unwrap_or(false) {
            p.abstain.is_some()
                || primary(&p).is_some_and(|s| s.needs_intent_pick || s.verified || !s.missing.is_empty() || s.ambiguous.is_some())
        } else {
            false
        };
        if ok {
            multi_ok += 1;
        } else if report_misses {
            eprintln!(
                "  [multi miss] '{}' -> step={:?} abstain={:?} pick={}",
                q,
                primary(&p).map(|s| (&s.intent_id, &s.argv, s.verified)),
                p.abstain,
                primary(&p).map(|s| s.needs_intent_pick).unwrap_or(false)
            );
        }
    }
    let mut flags_ok = 0usize;
    for c in &flags {
        let q = jstr(c, "q");
        let argv = jstrs(c, "argv");
        let p = PlanState::build(&q, cat, ret, live, &validate_cli);
        if p.is_ready() {
            emitted_total += 1;
            emitted_valid += usize::from(primary(&p).is_some_and(|s| s.verified));
        }
        if primary(&p).is_some_and(|s| s.argv == argv && s.verified) {
            flags_ok += 1;
        } else if report_misses {
            eprintln!("  [flags miss] '{}' -> got {:?} want {:?}", q, primary(&p).map(|s| &s.argv), argv);
        }
    }
    let mut meta_total = 0usize;
    let mut meta_ok = 0usize;
    let mut ood_total = 0usize;
    let mut ood_ok = 0usize;
    for c in &ood {
        let q = jstr(c, "q");
        let kind = jstr(c, "kind");
        let p = PlanState::build(&q, cat, ret, live, &validate_cli);
        let abstained = p.abstain.is_some();
        if kind == "metachar" {
            meta_total += 1;
            if abstained {
                meta_ok += 1;
            }
        } else {
            ood_total += 1;
            if abstained {
                ood_ok += 1;
            } else if report_misses {
                eprintln!("  [ood miss] '{}' ({}) -> {:?}", q, kind, primary(&p).map(|s| (&s.intent_id, s.alternatives.first().map(|a| a.score))));
            }
        }
    }
    latencies_ms.sort_by(|a, b| a.total_cmp(b));
    let p95_ms = latencies_ms[(latencies_ms.len() * 95 / 100).min(latencies_ms.len().saturating_sub(1))];
    EvalSummary {
        name,
        core_exact: core_ok as f64 / core.len() as f64,
        para_top1: para_1 as f64 / para.len() as f64,
        para_top3: para_3 as f64 / para.len() as f64,
        ambig_clarify: ambig_ok as f64 / ambig.len() as f64,
        multi_exact: multi_ok as f64 / multi.len() as f64,
        flags_exact: flags_ok as f64 / flags.len() as f64,
        meta_reject: meta_ok as f64 / meta_total.max(1) as f64,
        ood_abstain: ood_ok as f64 / ood_total.max(1) as f64,
        clap_valid: if emitted_total == 0 { 1.0 } else { emitted_valid as f64 / emitted_total as f64 },
        p95_ms,
    }
}

#[test]
fn benchmark_suite_and_ci_gates() {
    // Words only, the embedding model only, and both; the last one available must meet its bars.
    let cat = build_catalog(&Cli::command());
    let live = fixture_world();
    let t_load = Instant::now();
    let model = StaticModel::load_default();
    let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
    let has_model = model.is_some();
    let bm25 = HybridRetriever::build_with_mode(&cat, None, RetrievalMode::LexicalOnly);
    let r_bm25 = eval_system("words only", !has_model, &bm25, &cat, &live);
    let mut rows = vec![r_bm25];
    if let Some(m) = model {
        let model_bytes = std::fs::read(StaticModel::default_path()).expect("read model");
        let m2 = StaticModel::from_bytes(&model_bytes).expect("parse model");
        let dense = HybridRetriever::build_with_mode(&cat, Some(m), RetrievalMode::DenseOnly);
        let hybrid = HybridRetriever::build_with_mode(&cat, Some(m2), RetrievalMode::Hybrid);
        rows.push(eval_system("embedding only", false, &dense, &cat, &live));
        rows.push(eval_system("words and embedding", true, &hybrid, &cat, &live));
    }
    eprintln!("\n=== goethe ask bench (model load {load_ms:.2} ms, embedding model: {has_model}) ===");
    eprintln!("{:<30} | core:{:5.1}%", "a random guess", 100.0 / cat.intents.len() as f64);
    for r in &rows {
        eprintln!(
            "{:<30} | core:{:5.1}% | para@1:{:5.1}% @3:{:5.1}% | ambig:{:5.1}% | multi:{:5.1}% | flags:{:5.1}% | meta:{:5.1}% | ood:{:5.1}% | clap:{:5.1}% | p95:{:5.2}ms",
            r.name,
            r.core_exact * 100.0,
            r.para_top1 * 100.0,
            r.para_top3 * 100.0,
            r.ambig_clarify * 100.0,
            r.multi_exact * 100.0,
            r.flags_exact * 100.0,
            r.meta_reject * 100.0,
            r.ood_abstain * 100.0,
            r.clap_valid * 100.0,
            r.p95_ms
        );
    }
    let primary = rows.last().unwrap();
    assert_eq!(primary.clap_valid, 1.0, "clap validity must be 100%");
    assert_eq!(primary.meta_reject, 1.0, "metacharacter rejection must be 100%");
    assert!(primary.ood_abstain >= 0.98, "non-Goethe requests refused {:.3} < 0.98", primary.ood_abstain);
    if has_model {
        assert!(primary.core_exact >= 0.95, "core {:.3} < 0.95", primary.core_exact);
        assert!(primary.para_top3 >= 0.90, "paraphrase top-3 {:.3} < 0.90", primary.para_top3);
        assert!(primary.ambig_clarify >= 0.90, "ambiguous asks {:.3} < 0.90", primary.ambig_clarify);
        assert!(primary.multi_exact >= 0.90, "multi-step {:.3} < 0.90", primary.multi_exact);
        assert!(primary.flags_exact >= 0.90, "flags {:.3} < 0.90", primary.flags_exact);
    } else {
        assert!(primary.core_exact >= 0.90, "words only: core {:.3} < 0.90", primary.core_exact);
        assert!(primary.para_top3 >= 0.88, "words only: paraphrase top-3 {:.3} < 0.88", primary.para_top3);
        assert!(primary.ambig_clarify >= 0.88, "words only: ambiguous asks {:.3} < 0.88", primary.ambig_clarify);
        assert!(primary.multi_exact >= 0.85, "words only: multi-step {:.3} < 0.85", primary.multi_exact);
        assert!(primary.flags_exact >= 0.90, "words only: flags {:.3} < 0.90", primary.flags_exact);
    }
}

#[test]
fn a_wish_before_the_verb_does_not_change_the_command_and_a_request_never_loads_and_unloads_one_model() {
    // "I want you to load X" is "load X"; a request that gives something and undoes it on the same model is refused, not offered.
    let mut world = fixture_world();
    world.daemon_up = true;
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let plan = |q: &str| PlanState::build(q, &cat, &ret, &world, &validate_cli);
    for q in ["I want you to load mistralai/Mistral-7B-Instruct", "can you please load mistralai/Mistral-7B-Instruct"] {
        assert_eq!(plan(q).goals.first().map(|g| g.intent_id.as_str()), plan("load mistralai/Mistral-7B-Instruct").goals.first().map(|g| g.intent_id.as_str()), "{q}");
    }
    assert_eq!(super::parse::without_wish("I want you to load x"), "load x");
    assert_eq!(super::parse::without_wish("tell me about the weather"), "tell me about the weather", "only a leading wish goes");
    assert_eq!(super::parse::without_wish("Can you, please load x"), "load x");
    for q in ["load mistralai/Mistral-7B-Instruct and unload mistralai/Mistral-7B-Instruct", "install mistralai/Mistral-7B-Instruct and remove mistralai/Mistral-7B-Instruct"] {
        let p = plan(q);
        assert!(!p.is_ready(), "{q}: {:?}", cmds(&p.lines));
        assert!(matches!(p.next_decision(&cat, &world), PendingDecision::Invalid { .. }), "{q}: {:?}", p.goals.iter().map(|g| (&g.intent_id, &g.verify_err)).collect::<Vec<_>>());
    }
}


#[test]
fn a_gated_command_is_never_ranked_without_its_own_word_and_a_filler_never_changes_the_best_command() {
    // Generated from the catalog: every command's words, wrapped in wishes and joined with junk, in both rankers. Undoing and destructive commands need their word; a wish in front changes nothing.
    let cat = build_catalog(&Cli::command());
    let rankers: Vec<(&str, HybridRetriever)> = vec![
        ("hybrid", HybridRetriever::build(&cat, StaticModel::load_default())),
        ("lexical", HybridRetriever::build_with_mode(&cat, None, RetrievalMode::LexicalOnly)),
    ];
    let wishes = ["", "i want you to ", "please ", "can you ", "i would like to ", "could you please "];
    let junk = ["a model", "the thing", "now", "for me", "qwen", "my gpu", "it", "everything", "load", "some api"];
    let gated: Vec<&IntentCard> = cat.intents.iter().filter(|c| super::embed::gated(c)).collect();
    assert!(gated.len() >= 10, "{} gated", gated.len());
    let mut broken = Vec::new();
    for (mode, ret) in &rankers {
        for c in &cat.intents {
            let words = c.id.replace('.', " ");
            let bare = ret.rank(&words, 1).first().map(|r| r.id.clone());
            for w in wishes {
                let q = format!("{w}{words}");
                if ret.rank(&super::parse::without_wish(&q), 1).first().map(|r| r.id.clone()) != bare {
                    broken.push(format!("{mode}: {q:?} is not {bare:?}"));
                }
            }
        }
        for g in &gated {
            for j in junk {
                for w in wishes {
                    let q = format!("{w}{j}");
                    let said = super::embed::names(g).iter().any(|n| q.split_whitespace().any(|t| super::embed::says(t, n)));
                    let got = ret.rank(&q, 80).into_iter().find(|r| r.id == g.id).map_or(0.0, |r| r.score);
                    if !said && got > 0.0 {
                        broken.push(format!("{mode}: {q:?} ranks {} at {got}", g.id));
                    }
                }
            }
        }
    }
    println!("{} broken\n{}", broken.len(), broken.join("\n"));
    assert!(broken.is_empty());
}

#[test]
fn two_unrelated_changing_words_in_one_part_are_asked_never_run() {
    // "load/unload X", for every pair of commands that change something: the louder word never wins.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let leaf = |c: &IntentCard| c.id.rsplit('.').next().unwrap_or("").to_string();
    let changing: Vec<&IntentCard> = cat.intents.iter().filter(|c| super::embed::gated(c) && c.takes_model() && ret.rank(&leaf(c), 1).first().is_some_and(|t| t.id == c.id)).collect();
    assert!(changing.len() >= 4, "{} words that name their command alone", changing.len());
    let mut ready = Vec::new();
    for a in &changing {
        for b in &changing {
            let (la, lb) = (leaf(a), leaf(b));
            if la == lb || a.id.starts_with(&format!("{}.", b.id)) || b.id.starts_with(&format!("{}.", a.id)) {
                continue;
            }
            let q = format!("{la}/{lb} mistralai/Mistral-7B-Instruct");
            let p = PlanState::build(&q, &cat, &ret, &world, &validate_cli);
            if p.is_ready() {
                ready.push(format!("{q} -> {:?}", p.lines.iter().map(|l| l.argv.join(" ")).collect::<Vec<_>>()));
            }
        }
    }
    println!("{} ready\n{}", ready.len(), ready.join("\n"));
    assert!(ready.is_empty());
}

#[test]
fn a_ready_plan_only_ever_holds_commands_the_request_names() {
    // Random sentences from the catalog's own words, wishes and junk: no goal of a plan that is ready to confirm changes something its words do not say.
    let cat = build_catalog(&Cli::command());
    let ret = HybridRetriever::build(&cat, StaticModel::load_default());
    let world = fixture_world();
    let mut vocab: Vec<String> = cat.intents.iter().flat_map(|c| super::embed::names(c).into_iter().chain(c.id.split('.').map(str::to_string))).collect();
    vocab.extend(["please", "i", "want", "you", "to", "qwen", "mistralai/Mistral-7B-Instruct", "the", "my", "model", "gpu", "api", "everything", "now", "it", "and", "then", "203.0.113.7", "--yes"].map(str::to_string));
    let mut seed = 0x2545F4914F6CDD1Du64;
    let mut next = |n: usize| {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % n as u64) as usize
    };
    let (mut bad, mut ready) = (Vec::new(), 0);
    for _ in 0..1000 {
        let q = (0..1 + next(7)).map(|_| vocab[next(vocab.len())].clone()).collect::<Vec<_>>().join(" ");
        let p = PlanState::build(&q, &cat, &ret, &world, &validate_cli);
        if !p.is_ready() {
            continue;
        }
        ready += 1;
        let toks = super::embed::lex_tokens(&q.to_ascii_lowercase());
        for g in &p.goals {
            let Some(c) = cat.get(&g.intent_id).filter(|c| super::embed::gated(c)) else { continue };
            let said = super::embed::names(c).iter().any(|n| if n.contains(' ') { q.to_ascii_lowercase().contains(n.as_str()) } else { toks.iter().any(|t| super::embed::says(t, n)) });
            if !said {
                bad.push(format!("{q:?} plans {}", g.intent_id));
            }
        }
    }
    println!("{} bad of {ready} ready\n{}", bad.len(), bad.join("\n"));
    assert!(ready > 50 && bad.is_empty());
}
