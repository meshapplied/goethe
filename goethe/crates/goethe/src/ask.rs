// NL -> ordered Goethe plan. Jev-style: typed steps from a fixed catalog; clap validates; never shell.

use clap::Command;
use std::ffi::OsString;

const SEED: &str = include_str!("../ask/fixtures/seed.json");
const TAU: f32 = 0.42;
const FORBIDDEN: &[&str] = &["|", ";", "`", "$(", "${", ">", "<", "\n", "\r", "&&", "||"];
const OOD: &[&str] = &[
    "sudo", "apt ", "yum ", "dnf ", "docker", "kubectl", "pip ", "npm ", "systemctl", "chmod ", "ssh ",
    "curl ", "wget ", "rm -", "git push", "git clone", "what is the weather", "write me a", "tell me a joke",
    "book a flight", "buy bitcoin", "translate this", "explain quantum", "open youtube", "play some music",
    "send email", "hack the", "drop database", "compile the linux", "password for", "who won", "make me a",
    "ignore previous", "launch the", "brew ", "conda ", "poetry ",
];
const SPLIT: &[&str] = &[" and then ", " then ", " after that ", " also ", " and also "];

struct Intent {
    id: String,
    template: Vec<String>,
    model_slot: bool,
    labels: Vec<String>,
    blurb: String,
}

struct Step {
    argv: Vec<String>,
    blurb: String,
    id: String,
}

pub fn run(args: &[String], root: Command, validate: &dyn Fn(&[String]) -> Result<(), String>) -> Result<(), String> {
    let (yes, dry, dump, verbose, utterance) = parse_flags(args)?;
    let cat = catalog(&root);
    if dump {
        println!("{}", dump_catalog(&cat));
        return Ok(());
    }
    if utterance.trim().is_empty() {
        return Err("usage: goethe ask [--yes] [--dry-run] [--dump-catalog] [-v] <utterance...>".into());
    }
    let (plan, why) = plan(&utterance, &cat);
    if verbose {
        eprintln!("ask: {why}");
    }
    let Some(plan) = plan else {
        eprintln!("ask: abstain ({why})");
        hint_model(&utterance);
        std::process::exit(2);
    };
    for step in &plan {
        validate(&step.argv).map_err(|e| format!("ask: clap rejected {}: {e}", join(&step.argv)))?;
    }
    print_plan(&plan);
    if dry {
        return Ok(());
    }
    goethe_core::host::confirm(&format!("run these {} step(s)?", plan.len()), yes)?;
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    for (i, step) in plan.iter().enumerate() {
        eprintln!("ask: [{}/{}] {}", i + 1, plan.len(), join(&step.argv));
        let st = std::process::Command::new(&exe)
            .args(step.argv.iter().skip(1).map(OsString::from))
            .status()
            .map_err(|e| e.to_string())?;
        if !st.success() {
            std::process::exit(st.code().unwrap_or(1));
        }
    }
    Ok(())
}

fn plan(utt: &str, cat: &[Intent]) -> (Option<Vec<Step>>, String) {
    if FORBIDDEN.iter().any(|f| utt.contains(f)) {
        return (None, "shell metacharacters".into());
    }
    let lower = utt.to_ascii_lowercase();
    if OOD.iter().any(|m| lower.contains(m)) {
        return (None, "out of domain".into());
    }
    if let Some(p) = compose(utt, cat) {
        return p;
    }
    let clauses = split_clauses(utt);
    if clauses.len() > 1 {
        let mut steps = Vec::new();
        let mut notes = Vec::new();
        for c in &clauses {
            match resolve_one(c, cat) {
                Ok(s) => {
                    notes.push(s.id.clone());
                    steps.push(s);
                }
                Err(e) => return (None, format!("clause `{c}`: {e}")),
            }
        }
        return (Some(dedupe_steps(steps)), notes.join(" -> "));
    }
    match resolve_one(utt, cat) {
        Ok(s) => {
            let id = s.id.clone();
            (Some(vec![s]), id)
        }
        Err(e) => (None, e),
    }
}

fn compose(utt: &str, cat: &[Intent]) -> Option<(Option<Vec<Step>>, String)> {
    // General multi-step plan from verbs + one model ref (Hub org/name or filesystem path). No brand recipes.
    let lower = utt.to_ascii_lowercase();
    let model = parse_model_ref(utt).ok();
    let want_up = any(&lower, &["set up", "setup", "bring up", "bootstrap", "start goethe", "start fresh", "get goethe"]);
    let want_install = any(&lower, &["install", "deploy", "download", "add model", "get this model"]);
    let want_load = any(&lower, &[" --load", "load ", " load", "serve", "run it", " and run", "deploy", "get running"]);
    let want_api = any(&lower, &["api", "expose", "connect"]);
    let want_doctor = any(&lower, &["doctor", "health check", "diagnose"]);
    let want_status = any(&lower, &["status", "are you up"]) && !want_up;
    let multi = want_up || (want_install && want_load) || (want_install && want_api) || (want_up && (want_install || want_doctor));
    if !multi && !(want_install && model.is_some() && want_load) {
        return None;
    }
    let mut steps = Vec::new();
    if want_up {
        match must_step(cat, "up", utt) {
            Ok(s) => steps.push(s),
            Err(e) => return Some((None, e)),
        }
    }
    let install_and_serve = want_load || (want_up && model.is_some());
    if want_install || (want_up && model.is_some()) {
        match &model {
            Some(m) if install_and_serve => steps.push(step_install_load(m)),
            Some(m) => steps.push(step_install(m)),
            None => {
                return Some((
                    None,
                    "needs a Hugging Face id org/name (or org/name:QUANT) or a local path".into(),
                ));
            }
        }
    } else if want_load {
        match &model {
            Some(m) => steps.push(step_load(m)),
            None => {
                return Some((
                    None,
                    "needs a Hugging Face id org/name (or org/name:QUANT) or a local path".into(),
                ));
            }
        }
    }
    if want_api {
        match must_step(cat, "api", utt) {
            Ok(s) => steps.push(s),
            Err(e) => return Some((None, e)),
        }
    } else if want_doctor && model.is_none() {
        match must_step(cat, "doctor", utt) {
            Ok(s) => steps.push(s),
            Err(e) => return Some((None, e)),
        }
    }
    if want_status || (want_up && model.is_some() && !want_api) {
        match must_step(cat, "status", utt) {
            Ok(s) => steps.push(s),
            Err(e) => return Some((None, e)),
        }
    } else if want_up && model.is_none() && !want_doctor {
        match must_step(cat, "doctor", utt) {
            Ok(s) => steps.push(s),
            Err(e) => return Some((None, e)),
        }
        match must_step(cat, "status", utt) {
            Ok(s) => steps.push(s),
            Err(e) => return Some((None, e)),
        }
    }
    if steps.is_empty() {
        return None;
    }
    Some((Some(dedupe_steps(steps)), "compose".into()))
}

fn any(hay: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| hay.contains(n))
}

fn resolve_one(utt: &str, cat: &[Intent]) -> Result<Step, String> {
    let lower = utt.to_ascii_lowercase();
    let mut ranked: Vec<(&Intent, f32)> = cat.iter().map(|i| (i, score(&lower, i))).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if ranked.is_empty() || ranked[0].1 < TAU {
        return Err(format!("low confidence ({:.2})", ranked.first().map(|r| r.1).unwrap_or(0.0)));
    }
    let top = ranked[0].1;
    for (intent, s) in ranked.iter().take(5) {
        if *s < TAU || top - *s > 0.12 {
            break;
        }
        match fill(&intent.template, intent.model_slot, utt) {
            Ok(argv) => {
                return Ok(Step {
                    argv,
                    blurb: intent.blurb.clone(),
                    id: intent.id.clone(),
                });
            }
            Err(_) if intent.model_slot => {
                return Err("needs Hugging Face org/name (or org/name:QUANT) or a local path".into());
            }
            Err(_) => continue,
        }
    }
    Err("no fillable intent".into())
}

fn must_step(cat: &[Intent], id: &str, utt: &str) -> Result<Step, String> {
    let intent = cat.iter().find(|i| i.id == id).ok_or_else(|| format!("missing intent {id}"))?;
    let argv = fill(&intent.template, intent.model_slot, utt)?;
    Ok(Step {
        argv,
        blurb: intent.blurb.clone(),
        id: intent.id.clone(),
    })
}

fn step_install_load(model: &str) -> Step {
    Step {
        argv: vec!["goethe".into(), "install".into(), model.into(), "--load".into()],
        blurb: "install and serve".into(),
        id: "install.load".into(),
    }
}

fn step_install(model: &str) -> Step {
    Step {
        argv: vec!["goethe".into(), "install".into(), model.into()],
        blurb: "install model".into(),
        id: "install".into(),
    }
}

fn step_load(model: &str) -> Step {
    Step {
        argv: vec!["goethe".into(), "load".into(), model.into()],
        blurb: "serve model".into(),
        id: "load".into(),
    }
}

fn split_clauses(utt: &str) -> Vec<String> {
    let mut parts = vec![utt.to_string()];
    for sep in SPLIT {
        parts = parts.into_iter().flat_map(|p| p.split(sep).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect::<Vec<_>>()).collect();
    }
    parts
}

fn dedupe_steps(steps: Vec<Step>) -> Vec<Step> {
    let mut out = Vec::new();
    for s in steps {
        if out.last().map(|p: &Step| p.argv == s.argv).unwrap_or(false) {
            continue;
        }
        out.push(s);
    }
    out
}

fn print_plan(plan: &[Step]) {
    println!("plan ({}):", plan.len());
    for (i, s) in plan.iter().enumerate() {
        println!("  {}. {:<28} {}", i + 1, s.blurb, join(&s.argv));
    }
}

fn hint_model(utt: &str) {
    if parse_model_ref(utt).is_err() {
        eprintln!("hint: pass org/name, org/name:QUANT, or a path, e.g. goethe ask \"setup org/model --load\"");
    }
}

fn catalog(root: &Command) -> Vec<Intent> {
    let mut out = Vec::new();
    walk(root, &mut vec!["goethe".into()], &mut out);
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(SEED) {
        for row in v.as_array().into_iter().flatten() {
            let id = row["id"].as_str().unwrap_or("").to_string();
            if id.is_empty() {
                continue;
            }
            let template: Vec<String> = row["template"].as_array().into_iter().flatten().filter_map(|x| x.as_str().map(str::to_string)).collect();
            if template.first().map(String::as_str) != Some("goethe") || FORBIDDEN.iter().any(|f| template.iter().any(|t| t.contains(f))) {
                continue;
            }
            let model_slot = template.iter().any(|t| t == "{model}");
            let blurb = row["blurb"].as_str().unwrap_or(&id).to_string();
            let mut labels = vec![id.replace('.', " ")];
            for p in row["paraphrases"].as_array().into_iter().flatten().filter_map(|x| x.as_str()) {
                labels.push(p.to_string());
            }
            if let Some(existing) = out.iter_mut().find(|i| i.id == id) {
                existing.template = template;
                existing.model_slot = model_slot;
                existing.blurb = blurb;
                for l in labels {
                    if !existing.labels.iter().any(|e| e == &l) {
                        existing.labels.push(l);
                    }
                }
            } else {
                out.push(Intent { id, template, model_slot, labels, blurb });
            }
        }
    }
    for i in &mut out {
        if i.blurb.is_empty() {
            i.blurb = i.id.replace('.', " ");
        }
    }
    out
}

fn walk(cmd: &Command, prefix: &mut Vec<String>, out: &mut Vec<Intent>) {
    for sub in cmd.get_subcommands() {
        if sub.is_hide_set() {
            continue;
        }
        let name = sub.get_name().to_string();
        prefix.push(name.clone());
        let id = prefix[1..].join(".");
        let about = sub.get_about().map(|s| s.to_string()).unwrap_or_default();
        let mut labels = vec![id.replace('.', " "), name.clone()];
        if !about.is_empty() {
            labels.push(about.clone());
        }
        let mut template = prefix.clone();
        let mut model_slot = false;
        for arg in sub.get_arguments() {
            if arg.is_hide_set() || arg.get_long().is_some() || arg.get_short().is_some() || !arg.is_positional() {
                continue;
            }
            let aid = arg.get_id().as_str();
            if aid == "model" || aid == "args" {
                template.push("{model}".into());
                model_slot = true;
                break;
            }
        }
        out.push(Intent {
            id,
            template,
            model_slot,
            labels,
            blurb: if about.is_empty() { name } else { about },
        });
        if sub.get_subcommands().next().is_some() {
            walk(sub, prefix, out);
        }
        prefix.pop();
    }
}

fn score(q: &str, intent: &Intent) -> f32 {
    let q_toks: Vec<&str> = q.split_whitespace().filter(|t| t.len() > 1).collect();
    if q_toks.is_empty() {
        return 0.0;
    }
    let mut best = 0.0f32;
    for label in &intent.labels {
        let l = label.to_ascii_lowercase();
        if l == *q {
            return 1.0;
        }
        if q.contains(&l) && l.len() > 5 {
            best = best.max(0.95);
        }
        let l_toks: Vec<&str> = l.split_whitespace().filter(|t| t.len() > 1).collect();
        if l_toks.is_empty() {
            continue;
        }
        let hit = q_toks.iter().filter(|t| l_toks.iter().any(|l| l == *t || l.contains(**t) || t.contains(l))).count();
        best = best.max(hit as f32 / q_toks.len().max(l_toks.len()) as f32);
    }
    if q_toks.iter().any(|t| intent.id.split('.').any(|p| p == *t)) {
        best = best.max(0.55);
    }
    best
}

fn fill(template: &[String], model_slot: bool, utt: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::with_capacity(template.len());
    for part in template {
        if part == "{model}" {
            out.push(parse_model_ref(utt)?);
        } else if part.starts_with('{') && part.ends_with('}') {
            return Err(format!("unsupported slot {part}"));
        } else {
            out.push(part.clone());
        }
    }
    if model_slot && out.iter().all(|a| !is_model_ref(a)) {
        return Err("needs Hugging Face org/name or a local path".into());
    }
    if out.first().map(String::as_str) != Some("goethe") {
        return Err("argv[0] must be goethe".into());
    }
    if out.iter().any(|a| FORBIDDEN.iter().any(|f| a.contains(f))) {
        return Err("forbidden token in argv".into());
    }
    Ok(out)
}

fn parse_model_ref(utt: &str) -> Result<String, String> {
    // Hugging Face org/name or org/name:QUANT, or a filesystem path (/ ./ ../ ~/).
    let tokens: Vec<&str> = utt
        .split_whitespace()
        .map(|t| t.trim_matches(|c: char| matches!(c, ',' | '?' | '!' | '"' | '\'' | ')' | '(')))
        .filter(|t| !t.is_empty() && !t.starts_with('-'))
        .collect();
    for t in &tokens {
        if is_path_ref(t) {
            return Ok((*t).to_string());
        }
    }
    for t in &tokens {
        if is_hf_id(t) {
            return Ok((*t).to_string());
        }
    }
    Err("no Hugging Face org/name or local path".into())
}

fn is_model_ref(s: &str) -> bool {
    is_path_ref(s) || is_hf_id(s)
}

fn is_path_ref(s: &str) -> bool {
    s.starts_with('/') || s.starts_with("./") || s.starts_with("../") || s.starts_with("~/") || (s.len() > 1 && s.chars().nth(1) == Some(':') && s.chars().next().is_some_and(|c| c.is_ascii_alphabetic()))
}

fn is_hf_id(s: &str) -> bool {
    let base = s.split_once(':').map(|(b, _)| b).unwrap_or(s);
    let parts: Vec<&str> = base.split('/').collect();
    if parts.len() != 2 {
        return false;
    }
    parts.iter().all(|p| {
        !p.is_empty()
            && *p != "."
            && *p != ".."
            && p.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    })
}

fn parse_flags(args: &[String]) -> Result<(bool, bool, bool, bool, String), String> {
    let mut yes = false;
    let mut dry = false;
    let mut dump = false;
    let mut verbose = false;
    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--yes" | "-y" => yes = true,
            "--dry-run" => dry = true,
            "--dump-catalog" => dump = true,
            "--verbose" | "-v" => verbose = true,
            "-h" | "--help" => return Err("usage: goethe ask [--yes] [--dry-run] [--dump-catalog] [-v] <utterance...>".into()),
            s if s.starts_with('-') => return Err(format!("unknown ask flag: {s}")),
            _ => return Ok((yes, dry, dump, verbose, args[i..].join(" "))),
        }
        i += 1;
    }
    Ok((yes, dry, dump, verbose, String::new()))
}

fn join(argv: &[String]) -> String {
    argv.iter()
        .map(|a| {
            if a.chars().any(|c| c.is_whitespace()) {
                format!("\"{}\"", a.replace('"', "\\\""))
            } else {
                a.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn dump_catalog(cat: &[Intent]) -> String {
    let rows: Vec<serde_json::Value> = cat
        .iter()
        .map(|i| serde_json::json!({"id": i.id, "template": i.template, "blurb": i.blurb, "labels": i.labels}))
        .collect();
    serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Command;

    fn root() -> Command {
        Command::new("goethe")
            .subcommand(Command::new("status").about("show status"))
            .subcommand(Command::new("doctor").about("check the machine"))
            .subcommand(Command::new("up").about("start Goethe"))
            .subcommand(Command::new("down").about("stop Goethe"))
            .subcommand(Command::new("api").about("show API"))
            .subcommand(
                Command::new("install")
                    .about("download a model")
                    .arg(clap::Arg::new("model").required(true).num_args(1..)),
            )
    }

    fn load_cases(name: &str) -> Vec<serde_json::Value> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("ask/fixtures").join(name);
        serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
    }

    fn first_argv(utt: &str, cat: &[Intent]) -> Option<Vec<String>> {
        plan(utt, cat).0.map(|p| p[0].argv.clone())
    }

    #[test]
    fn gate_core_para_ood_slots() {
        let cat = catalog(&root());
        let core = load_cases("test_core.json");
        let core_ok = core
            .iter()
            .filter(|c| {
                first_argv(c["utterance"].as_str().unwrap(), &cat).as_ref().map(|a| a.iter().map(String::as_str).collect::<Vec<_>>())
                    == c["argv"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            })
            .count();
        assert!(core_ok as f32 / core.len() as f32 > 1.0 / cat.len() as f32);

        let para = load_cases("test_para.json");
        let para_ok = para
            .iter()
            .filter(|c| {
                first_argv(c["utterance"].as_str().unwrap(), &cat).as_ref().map(|a| a.iter().map(String::as_str).collect::<Vec<_>>())
                    == c["argv"].as_array().map(|a| a.iter().filter_map(|x| x.as_str()).collect())
            })
            .count();
        assert!(para_ok as f32 / para.len() as f32 >= 0.5, "para {para_ok}/{}", para.len());

        for c in load_cases("test_ood.json") {
            assert!(plan(c["utterance"].as_str().unwrap(), &cat).0.is_none(), "OOD leaked: {}", c["utterance"]);
        }
        for c in load_cases("test_slots.json") {
            let p = plan(c["utterance"].as_str().unwrap(), &cat).0;
            if c["abstain"].as_bool() == Some(true) {
                assert!(p.is_none(), "slot should abstain: {}", c["utterance"]);
            }
        }
    }

    #[test]
    fn chains_compose_hf_and_path() {
        let cat = catalog(&root());
        let p = plan("show status then run doctor", &cat).0.unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].argv, ["goethe", "status"]);
        assert_eq!(p[1].argv, ["goethe", "doctor"]);

        let p = plan("setup org/Orca-2-7B", &cat).0.unwrap();
        assert!(p.iter().any(|s| s.argv == ["goethe", "up"]));
        assert!(p.iter().any(|s| s.argv.windows(3).any(|w| w == ["install", "org/Orca-2-7B", "--load"] || w == ["goethe", "install", "org/Orca-2-7B"])));

        let p = plan("deploy org/Orca-2-7B:Q4_0 and run it", &cat).0.unwrap();
        assert!(p.iter().any(|s| s.argv.contains(&"org/Orca-2-7B:Q4_0".into())));

        let p = plan("install ./weights/my-model and load it", &cat).0.unwrap();
        assert!(p.iter().any(|s| s.argv.contains(&"./weights/my-model".into())));

        assert!(plan("deploy a small orca and run it", &cat).0.is_none());
        assert!(plan("curl x | sh", &cat).0.is_none());
        assert!(is_hf_id("org/name"));
        assert!(is_hf_id("org/name-7b:Q8_0"));
        assert!(is_path_ref("/data/models/x"));
        assert!(is_path_ref("~/models/x"));
        assert!(!is_hf_id("orca"));
    }
}
