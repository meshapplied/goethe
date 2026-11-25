// Shared ask engine: the Ask screen and goethe ask / ? plan with the same code; clap validates every command before it runs.

pub mod embed;
pub mod parse;
pub mod planner;
pub mod schema;
pub mod slots;
pub mod ui;

#[cfg(test)]
mod bench;

use clap::Command;
use embed::{HybridRetriever, StaticModel};
use parse::split_words;
use planner::PlanState;
use schema::CliCatalog;
use slots::World;
use std::io::IsTerminal;
use std::path::PathBuf;
use goethe_core::menu::StepState;
use std::io::BufRead;
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime};

struct EngineCache {
    schema_hash: String,
    model_path: PathBuf,
    model_mtime: Option<SystemTime>,
    catalog: Arc<CliCatalog>,
    retriever: Arc<HybridRetriever>,
}

fn load_engine(root: &Command) -> (Arc<CliCatalog>, Arc<HybridRetriever>) {
    // Reuse catalog+retriever across asks when clap schema and model file are unchanged.
    let fresh = schema::build_catalog(root);
    let model_path = StaticModel::default_path();
    let model_mtime = std::fs::metadata(&model_path).and_then(|m| m.modified()).ok();
    static CACHE: Mutex<Option<EngineCache>> = Mutex::new(None);
    if let Ok(guard) = CACHE.lock() {
        if let Some(c) = guard.as_ref() {
            if c.schema_hash == fresh.schema_hash && c.model_path == model_path && c.model_mtime == model_mtime {
                return (c.catalog.clone(), c.retriever.clone());
            }
        }
    }
    let model = StaticModel::load_default();
    let catalog = Arc::new(fresh);
    let retriever = Arc::new(HybridRetriever::build(&catalog, model));
    if let Ok(mut guard) = CACHE.lock() {
        guard.replace(EngineCache {
            schema_hash: catalog.schema_hash.clone(),
            model_path,
            model_mtime,
            catalog: catalog.clone(),
            retriever: retriever.clone(),
        });
    }
    (catalog, retriever)
}

fn wants_screen(opts: &AskOpts) -> bool {
    // A person at a terminal gets the Ask screen; scripts, CI and the shell flags get plain output.
    let ci = std::env::var("CI").is_ok_and(|v| !v.is_empty() && v != "0" && !v.eq_ignore_ascii_case("false"));
    !opts.json && !opts.dry_run && !opts.no_input && !opts.yes && !ci && std::io::stdin().is_terminal() && std::io::stdout().is_terminal()
}

fn open_screen(root: Command, validate: fn(&[String]) -> Result<(), clap::Error>, utterance: &str) -> i32 {
    let mut driver = menu_driver(root, validate);
    if !utterance.trim().is_empty() {
        driver.submit(utterance);
    }
    match goethe_core::menu::ask_session(driver.as_mut()) {
        Ok(_) => 0,
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

#[derive(Default)]
pub struct AskOpts {
    pub yes: bool,
    pub dry_run: bool,
    pub no_input: bool,
    pub json: bool,
    pub verbose: bool,
}

struct AskFlags {
    opts: AskOpts,
    dump_catalog: bool,
    help: bool,
    utterance: String,
}

pub fn run(args: &[String], root: Command, validate: fn(&[String]) -> Result<(), clap::Error>) -> i32 {
    // goethe ask: its options, then the Ask screen on a terminal or plain output otherwise.
    let parsed = match parse_args(args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            return 2;
        }
    };
    if parsed.help {
        print_help();
        return 0;
    }
    if parsed.dump_catalog {
        let catalog = schema::build_catalog(&root);
        println!("{}", serde_json::to_string_pretty(&catalog.to_json()).expect("a JSON value serializes"));
        return 0;
    }
    if wants_screen(&parsed.opts) {
        return open_screen(root, validate, &parsed.utterance);
    }
    if parsed.utterance.trim().is_empty() {
        eprintln!("usage: goethe ask [--yes] [--dry-run] [--no-input] [--json] <request...>");
        return 2;
    }
    run_once(&parsed.utterance, &root, &validate, &parsed.opts)
}

fn run_once(query: &str, root: &Command, validate: &planner::Validate, opts: &AskOpts) -> i32 {
    // No screen: plan once from the live state, then print it or run it. Never prompts.
    let (catalog, retriever) = load_engine(root);
    if opts.verbose {
        let path = StaticModel::default_path();
        let mode = match (retriever.has_dense(), StaticModel::from_file(&path)) {
            (true, _) => format!("words and the embedding model ({} dimensions)", retriever.dense_dim()),
            (false, Err(e)) if path.is_file() => format!("words only; the embedding model could not be read: {e}"),
            (false, _) => format!("words only; no embedding model at {} (goethe update fetches it)", path.display()),
        };
        eprintln!("ask: ranking by {mode}; {} commands; CLI {}", catalog.intents.len(), catalog.schema_hash);
    }
    let world = World::from_host();
    let plan = PlanState::build(query, &catalog, retriever.as_ref(), &world, validate);
    if opts.json {
        println!("{}", serde_json::to_string_pretty(&plan.to_json(&catalog, &world)).expect("a JSON value serializes"));
        return if plan.abstain.is_some() { 2 } else if plan.is_ready() { 0 } else { 3 };
    }
    if let Some(reason) = &plan.abstain {
        ui::print_abstain(reason);
        return 2;
    }
    if !plan.is_ready() {
        eprintln!("ask: this needs a detail first; run goethe ask on a terminal, or pass --json to see what");
        return 3;
    }
    if opts.dry_run {
        for line in &plan.lines {
            println!("{}", ui::format_argv(&line.argv));
        }
        return 0;
    }
    ui::print_plan(&plan);
    if !opts.yes && !plan.all_readonly() {
        eprintln!("ask: these commands change things; pass --yes to run them, or --dry-run to only print them");
        return 3;
    }
    run_list(&plan_argvs(&plan), &Watch::Terminal)
}

pub fn menu_driver(
    root: Command,
    validate: impl Fn(&[String]) -> Result<(), clap::Error> + 'static,
) -> Box<dyn goethe_core::menu::AskDriver> {
    // Ask on the menu and `goethe ask` on a terminal share this driver.
    Box::new(MenuAsk::new(root, Box::new(validate)))
}

type Validate = Box<planner::Validate<'static>>;

struct MenuAsk {
    root: Command,
    validate: Validate,
    frame: goethe_core::menu::AskFrame,
    plan: Option<PlanState>,
    world: World,
    catalog: Option<Arc<CliCatalog>>,
    retriever: Option<Arc<HybridRetriever>>,
    run: Option<Run>,
}

#[derive(Clone, Default)]
struct StepLog {
    state: StepState,
    started: Option<Instant>,
    ended: Option<Instant>,
    out: Vec<String>,
}

#[derive(Default)]
struct RunLog {
    steps: Vec<StepLog>,
    finished: Option<i32>,
    group: Option<u32>,
}

struct Run {
    argvs: Vec<Vec<String>>,
    whys: Vec<String>,
    parts: Vec<usize>,
    log: Arc<Mutex<RunLog>>,
    stop: Arc<AtomicBool>,
    seen_end: bool,
}

const KEPT_LINES: usize = 200;

impl MenuAsk {
    fn new(root: Command, validate: Validate) -> Self {
        MenuAsk {
            root,
            validate,
            frame: goethe_core::menu::AskFrame::default(),
            plan: None,
            world: World::from_host(),
            catalog: None,
            retriever: None,
            run: None,
        }
    }

    fn engine(&mut self) -> (Arc<CliCatalog>, Arc<HybridRetriever>) {
        if let (Some(c), Some(r)) = (&self.catalog, &self.retriever) {
            return (c.clone(), r.clone());
        }
        let (c, r) = load_engine(&self.root);
        self.catalog = Some(c.clone());
        self.retriever = Some(r.clone());
        (c, r)
    }

    fn sync(&mut self, catalog: &CliCatalog) {
        // The screen's view of the plan: one page per part (its commands, its open question), then the commands in run order.
        use goethe_core::menu::{AskFrame, AskPart, AskStep};
        use planner::PendingDecision;
        let Some(plan) = &self.plan else {
            self.frame = AskFrame::default();
            return;
        };
        let mut frame = AskFrame { said: plan.raw_query.clone(), ..AskFrame::default() };
        if let Some(reason) = &plan.abstain {
            frame.note = Some(reason.clone());
            self.frame = frame;
            return;
        }
        for (i, part) in plan.parts.iter().enumerate() {
            let goal = plan.goals.iter().find(|g| g.part == i && g.reason.is_none());
            let settled = goal.filter(|g| !g.needs_intent_pick);
            let mut view = AskPart {
                text: part.text.clone(),
                options: part.options.iter().map(|o| catalog.get(&o.id).map_or((o.id.clone(), String::new()), |c| (c.argv_prefix.join(" "), c.summary()))).collect(),
                chosen: settled.and_then(|g| part.options.iter().position(|o| o.id == g.intent_id)),
                unsure: !part.left_out && settled.is_none(),
                left_out: part.left_out,
                label: settled.and_then(|g| catalog.get(&g.intent_id)).map_or_else(|| short(&part.text), |c| times(&c.argv_prefix[1..].join(" "), plan.goals.iter().filter(|g| g.part == i && g.reason.is_none()).count())),
                ..AskPart::default()
            };
            view.unused = plan.unused(i, &self.world);
            match plan.part_decision(i, catalog, &self.world) {
                PendingDecision::PickSlot { missing, options } if !part.left_out => {
                    view.question = Some(question(&missing));
                    view.choices = options;
                }
                PendingDecision::PromptSlot { missing } if !part.left_out => view.question = Some(question(&missing)),
                PendingDecision::Invalid { message } if !part.left_out => {
                    view.unsure = true;
                    frame.note = Some(message);
                }
                _ => {}
            }
            frame.parts.push(view);
        }
        frame.steps = plan
            .lines
            .iter()
            .filter(|l| l.argv.len() >= 2)
            .map(|l| AskStep { cmd: std::iter::once(ui::format_argv(&l.argv)).chain(l.placeholders.iter().cloned()).collect::<Vec<_>>().join(" "), why: l.why.clone(), part: l.part, ..AskStep::default() })
            .collect();
        frame.confirm = confirm_text(plan, catalog);
        frame.attached = opens_a_screen(plan, catalog);
        frame.can_run = plan.is_ready();
        if plan.parts.iter().all(|p| p.left_out) {
            frame.note = Some("Every part is left out; nothing would run.".into());
        }
        self.frame = frame;
    }

    fn act(&mut self, part: usize, action: planner::PlanAction) -> goethe_core::menu::AskFrame {
        let (catalog, _) = self.engine();
        if let Some(plan) = self.plan.as_mut() {
            plan.apply_to(part, action, &catalog, &self.world, self.validate.as_ref());
        }
        self.sync(&catalog);
        self.frame.clone()
    }
}

fn times(label: &str, n: usize) -> String {
    // A part that runs its command once per model ("each") says how many times.
    if n > 1 { format!("{label} x{n}") } else { label.to_string() }
}

fn short(text: &str) -> String {
    // A tab label for a part with no command yet: its first words.
    text.split_whitespace().take(3).collect::<Vec<_>>().join(" ")
}

fn confirms(catalog: &CliCatalog, id: &str) -> bool {
    catalog.get(id).is_some_and(|c| c.does.confirms)
}

fn confirm_text(plan: &PlanState, catalog: &CliCatalog) -> Option<String> {
    // Commands that ask before deleting or replacing things get the question here, on the screen.
    let asks: Vec<String> = plan.lines.iter().filter(|l| confirms(catalog, &l.intent_id)).map(|l| ui::format_argv(&l.argv)).collect();
    if !asks.is_empty() {
        return Some(format!("{} would ask before going ahead. Press Enter again to run it.", asks.join(", ")));
    }
    plan.has_destructive().then(|| "This opens or removes access. Press Enter again to run it.".to_string())
}

fn opens_a_screen(plan: &PlanState, catalog: &CliCatalog) -> bool {
    // A command that takes over the terminal (a chat screen) runs there, not in the background.
    plan.goals.iter().any(|s| catalog.get(&s.intent_id).is_some_and(|c| c.does.terminal))
}

fn run_argv(line: &planner::PlanLine, catalog: &CliCatalog) -> Vec<String> {
    // The screen already asked; the command's own question cannot be answered without a terminal, so it gets --yes.
    let mut argv = line.argv.clone();
    let has_yes = catalog.get(&line.intent_id).is_some_and(|c| c.flags.iter().any(|f| f.long == "yes"));
    if confirms(catalog, &line.intent_id) && has_yes && !argv.iter().any(|a| a == "--yes") {
        argv.push("--yes".into());
    }
    argv
}

fn read_lines(pipe: Box<dyn std::io::Read + Send>, log: Arc<Mutex<RunLog>>, step: usize) -> std::thread::JoinHandle<()> {
    // A progress line redrawn with carriage returns counts as its last state.
    std::thread::spawn(move || {
        for line in std::io::BufReader::new(pipe).split(b'\n').map_while(Result::ok) {
            let text = String::from_utf8_lossy(&line);
            let last = text.rsplit('\r').find(|t| !t.trim().is_empty()).unwrap_or("");
            push_line(&log, step, last);
        }
    })
}

fn push_line(log: &Mutex<RunLog>, step: usize, line: &str) {
    // Output as shown on the screen: no control characters, no trailing space, one blank line at most.
    let clean: String = line.chars().filter(|c| !c.is_control()).collect::<String>().trim_end().to_string();
    let Ok(mut l) = log.lock() else {
        return;
    };
    let Some(s) = l.steps.get_mut(step) else {
        return;
    };
    if clean.is_empty() && s.out.last().is_none_or(String::is_empty) {
        return;
    }
    s.out.push(clean);
    let extra = s.out.len().saturating_sub(KEPT_LINES);
    s.out.drain(..extra);
}

fn set_state(log: &Mutex<RunLog>, step: usize, state: StepState) {
    if let Some(s) = log.lock().ok().as_mut().and_then(|l| l.steps.get_mut(step)) {
        match state {
            StepState::Running => s.started = Some(Instant::now()),
            StepState::Done | StepState::Failed | StepState::Stopped => s.ended = Some(Instant::now()),
            StepState::Planned => {}
        }
        s.state = state;
    }
}

fn progress_line(argvs: &[Vec<String>], log: &Mutex<RunLog>) -> String {
    // The Ask row on the main menu while a run goes on in the background, and once it ended.
    let Ok(l) = log.lock() else {
        return String::new();
    };
    let total = argvs.len();
    let done = l.steps.iter().filter(|s| s.state == StepState::Done).count();
    let failed = l.steps.iter().position(|s| matches!(s.state, StepState::Failed | StepState::Stopped));
    let running = l.steps.iter().position(|s| s.state == StepState::Running);
    match (l.finished, failed, running) {
        (Some(_), Some(i), _) => format!("failed: {}", ui::format_argv(&argvs[i])),
        (Some(_), None, _) => format!("done: {done} of {total} commands ran"),
        (None, _, Some(i)) => format!("running {} of {total}: {}", i + 1, ui::format_argv(&argvs[i])),
        (None, _, None) => format!("running {done} of {total}"),
    }
}

fn question(missing: &planner::MissingSlot) -> String {
    // "KEY_NAME" or "--allow" reads "Key name:" or "Allow:".
    format!("{}:", schema::capitalize(&missing.label.trim_start_matches('-').replace(['_', '-'], " ").to_ascii_lowercase()))
}

impl goethe_core::menu::AskDriver for MenuAsk {
    fn frame(&mut self) -> goethe_core::menu::AskFrame {
        // While a run is on: each step's state, time and newest line. Once it ends: each one's output, the live state read again, done or the failure.
        use goethe_core::menu::AskStep;
        let Some(run) = self.run.as_mut() else {
            return self.frame.clone();
        };
        let (steps, finished) = match run.log.lock() {
            Ok(l) => (l.steps.clone(), l.finished),
            Err(_) => (Vec::new(), Some(1)),
        };
        let now = Instant::now();
        let mut frame = self.frame.clone();
        frame.steps = run
            .argvs
            .iter()
            .enumerate()
            .map(|(i, argv)| {
                let s = steps.get(i).cloned().unwrap_or_default();
                let secs = s.started.map(|t| s.ended.unwrap_or(now).duration_since(t).as_secs());
                AskStep { cmd: ui::format_argv(argv), why: run.whys[i].clone(), part: run.parts[i], state: s.state, secs, latest: s.out.iter().rev().find(|l| !l.is_empty()).cloned(), output: s.out }
            })
            .collect();
        frame.can_run = false;
        frame.confirm = None;
        frame.running = finished.is_none();
        frame.ran = finished.is_some();
        let stopped = run.stop.load(Ordering::Relaxed);
        if stopped {
            frame.steps.iter_mut().filter(|s| s.state == StepState::Failed).for_each(|s| s.state = StepState::Stopped);
        }
        if let Some(code) = finished {
            if !run.seen_end {
                run.seen_end = true;
                self.world = World::from_host();
            }
            frame.done = code == 0;
            frame.note = frame.steps.iter().find(|s| matches!(s.state, StepState::Failed | StepState::Stopped)).map(|s| match s.state {
                StepState::Stopped => format!("Stopped: you pressed Ctrl-C during {}.", s.cmd),
                _ => format!("Stopped: {} failed (exit {code}).", s.cmd),
            });
        }
        frame
    }

    fn submit(&mut self, request: &str) -> goethe_core::menu::AskFrame {
        let (catalog, retriever) = self.engine();
        self.run = None;
        self.world = World::from_host();
        self.plan = Some(PlanState::build(request, &catalog, retriever.as_ref(), &self.world, self.validate.as_ref()));
        self.sync(&catalog);
        self.frame.clone()
    }

    fn choose(&mut self, part: usize, option: usize) -> goethe_core::menu::AskFrame {
        let id = self.plan.as_ref().and_then(|p| p.parts.get(part)).and_then(|p| p.options.get(option)).map(|o| o.id.clone());
        match id {
            Some(intent_id) => self.act(part, planner::PlanAction::SelectIntent { intent_id }),
            None => self.frame.clone(),
        }
    }

    fn write(&mut self, part: usize, text: &str) -> goethe_core::menu::AskFrame {
        let (catalog, retriever) = self.engine();
        let refused = self.plan.as_mut().and_then(|plan| plan.rewrite(part, text, &catalog, retriever.as_ref(), self.validate.as_ref()).err());
        self.sync(&catalog);
        if refused.is_some() {
            self.frame.note = refused;
        }
        self.frame.clone()
    }

    fn answer(&mut self, part: usize, text: &str) -> goethe_core::menu::AskFrame {
        // A typed value for the part's open question, checked like a hand-typed value before it is used.
        let (catalog, _) = self.engine();
        let missing = match self.plan.as_ref().map(|p| p.part_decision(part, &catalog, &self.world)) {
            Some(planner::PendingDecision::PromptSlot { missing } | planner::PendingDecision::PickSlot { missing, .. }) => missing,
            _ => return self.frame.clone(),
        };
        match slots::validate_manual(missing.kind, &missing.possible, &split_words(text), &self.world) {
            Ok(values) => self.act(part, planner::PlanAction::SetSlot { is_flag: missing.is_flag, name: missing.name, values }),
            Err(e) => {
                self.sync(&catalog);
                self.frame.note = Some(e);
                self.frame.clone()
            }
        }
    }

    fn pick_value(&mut self, part: usize, index: usize) -> goethe_core::menu::AskFrame {
        let (catalog, _) = self.engine();
        match self.plan.as_ref().map(|p| p.part_decision(part, &catalog, &self.world)) {
            Some(planner::PendingDecision::PickSlot { missing, options }) if index < options.len() => {
                let values = vec![options[index].0.clone()];
                self.act(part, planner::PlanAction::SetSlot { is_flag: missing.is_flag, name: missing.name, values })
            }
            _ => self.frame.clone(),
        }
    }

    fn drop_value(&mut self, part: usize, index: usize) -> goethe_core::menu::AskFrame {
        // "leave wg0 out": the value stays in the words but is no longer required to be used.
        match self.frame.parts.get(part).and_then(|p| p.unused.get(index)).cloned() {
            Some(value) => self.act(part, planner::PlanAction::Ignore { value }),
            None => self.frame.clone(),
        }
    }

    fn leave_out(&mut self, part: usize) -> goethe_core::menu::AskFrame {
        self.act(part, planner::PlanAction::LeaveOut)
    }

    fn start(&mut self) -> goethe_core::menu::AskFrame {
        // Run the whole list in the background; the screen (or the menu's Ask row) follows it.
        let (catalog, _) = self.engine();
        let Some(plan) = self.plan.as_ref().filter(|p| p.is_ready() && self.run.is_none()) else {
            return self.frame.clone();
        };
        let lines: Vec<&planner::PlanLine> = plan.lines.iter().filter(|l| l.argv.len() >= 2).collect();
        let argvs: Vec<Vec<String>> = lines.iter().map(|l| run_argv(l, &catalog)).collect();
        let log = Arc::new(Mutex::new(RunLog { steps: vec![StepLog::default(); argvs.len()], finished: None, group: None }));
        let stop = Arc::new(AtomicBool::new(false));
        let watch = Watch::Screen { log: log.clone(), stop: stop.clone() };
        let list = argvs.clone();
        std::thread::spawn(move || {
            let code = run_list(&list, &watch);
            if let Watch::Screen { log, .. } = &watch {
                if let Ok(mut l) = log.lock() {
                    l.finished = Some(code);
                }
            }
        });
        let whys = lines.iter().map(|l| l.why.clone()).collect();
        let parts = lines.iter().map(|l| l.part).collect();
        self.run = Some(Run { argvs, whys, parts, log, stop, seen_end: false });
        self.frame.done = false;
        self.frame.note = None;
        self.frame()
    }

    fn run_attached(&mut self) {
        let Some(plan) = self.plan.as_ref().filter(|p| p.is_ready()) else {
            return;
        };
        let code = run_list(&plan_argvs(plan), &Watch::Terminal);
        self.world = World::from_host();
        self.frame.can_run = false;
        self.frame.done = code == 0;
        self.frame.note = (code != 0).then(|| format!("Stopped: it ended with exit {code}."));
    }

    fn stop(&mut self) {
        if let Some(run) = &self.run {
            run.stop.store(true, Ordering::Relaxed);
        }
    }

    fn progress(&self) -> Option<Box<dyn Fn() -> (String, bool)>> {
        let run = self.run.as_ref()?;
        let (argvs, log) = (run.argvs.clone(), run.log.clone());
        Some(Box::new(move || (progress_line(&argvs, &log), log.lock().is_ok_and(|l| l.finished.is_none()))))
    }
}

enum Watch {
    Terminal,
    Screen { log: Arc<Mutex<RunLog>>, stop: Arc<AtomicBool> },
}

impl Watch {
    fn running(&self, group: Option<u32>) {
        if let Watch::Screen { log, .. } = self {
            if let Ok(mut l) = log.lock() {
                l.group = group;
            }
        }
    }

    fn mark(&self, i: usize, state: StepState) {
        if let Watch::Screen { log, .. } = self {
            set_state(log, i, state);
        }
    }

    fn say(&self, step: usize, text: &str) {
        match self {
            Watch::Terminal => eprintln!("error: {text}"),
            Watch::Screen { log, .. } => push_line(log, step, text),
        }
    }
}

fn run_list(argvs: &[Vec<String>], watch: &Watch) -> i32 {
    // One command after another via current_exe (never a shell); the first failure ends the list. On a terminal they print as usual; on the Ask screen their output is read line by line and Ctrl-C stops them.
    let exe = match std::env::current_exe() {
        Ok(e) => e,
        Err(e) => {
            watch.say(0, &format!("cannot locate goethe binary: {e}"));
            return 1;
        }
    };
    for (i, argv) in argvs.iter().enumerate() {
        let shown = ui::format_argv(argv);
        watch.mark(i, StepState::Running);
        if let Watch::Terminal = watch {
            eprintln!("==> {shown}");
        }
        let mut cmd = ProcessCommand::new(&exe);
        cmd.args(&argv[1..]);
        if let Watch::Screen { .. } = watch {
            cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
            std::os::unix::process::CommandExt::process_group(&mut cmd, 0);
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                watch.mark(i, StepState::Failed);
                watch.say(i, &format!("could not start {shown}: {e}"));
                return 1;
            }
        };
        watch.running(Some(child.id()));
        let readers: Vec<_> = match watch {
            Watch::Screen { log, .. } => {
                let out = child.stdout.take().map(|o| read_lines(Box::new(o), log.clone(), i));
                let err = child.stderr.take().map(|e| read_lines(Box::new(e), log.clone(), i));
                [out, err].into_iter().flatten().collect()
            }
            Watch::Terminal => Vec::new(),
        };
        let code = loop {
            if let Watch::Screen { stop, .. } = watch {
                if stop.load(Ordering::Relaxed) {
                    stop_group(child.id());
                }
            }
            match child.try_wait() {
                Ok(Some(st)) => break st.code().unwrap_or(130),
                Ok(None) => std::thread::sleep(std::time::Duration::from_millis(100)),
                Err(_) => break 1,
            }
        };
        watch.running(None);
        for r in readers {
            let _ = r.join();
        }
        if code != 0 {
            watch.mark(i, StepState::Failed);
            if let Watch::Terminal = watch {
                eprintln!("ask: {shown} failed (exit {code})");
            }
            return 1;
        }
        watch.mark(i, StepState::Done);
    }
    0
}

fn stop_group(leader: u32) {
    // SIGTERM to a step's whole process group (it was started as the leader), so the commands it started stop and let go of its output too.
    if let Ok(pgid) = i32::try_from(leader) {
        unsafe { libc::kill(-pgid, libc::SIGTERM) };
    }
}

impl Drop for MenuAsk {
    fn drop(&mut self) {
        // Leaving goethe while a run goes on (Ctrl-C) stops its current command instead of leaving it behind without a screen.
        if let Some(group) = self.run.as_ref().and_then(|r| r.log.lock().ok().and_then(|l| l.group)) {
            stop_group(group);
        }
    }
}

fn plan_argvs(plan: &PlanState) -> Vec<Vec<String>> {
    plan.lines.iter().map(|l| l.argv.clone()).filter(|a| a.len() >= 2).collect()
}

fn parse_args(args: &[String]) -> Result<AskFlags, String> {
    // Ask's own options first (an unknown one is refused, so a typo never becomes part of the request), then the request words.
    let mut flags = AskFlags {
        opts: AskOpts::default(),
        dump_catalog: false,
        help: false,
        utterance: String::new(),
    };
    let mut rest = Vec::new();
    let mut in_utterance = false;
    for tok in args.iter().skip(2) {
        if in_utterance {
            rest.push(tok.clone());
        } else {
            match tok.as_str() {
                "-h" | "--help" => flags.help = true,
                "-y" | "--yes" => flags.opts.yes = true,
                "--dry-run" => flags.opts.dry_run = true,
                "--no-input" => flags.opts.no_input = true,
                "--json" => flags.opts.json = true,
                "--dump-catalog" => flags.dump_catalog = true,
                "-v" | "--verbose" => flags.opts.verbose = true,
                "--" => in_utterance = true,
                t if t.starts_with('-') => return Err(format!("unknown option {t}; goethe ask --help lists the options")),
                _ => {
                    in_utterance = true;
                    rest.push(tok.clone());
                }
            }
        }
    }
    flags.utterance = rest.join(" ");
    Ok(flags)
}

fn print_help() {
    println!(
        "Turn a request in plain English into Goethe commands, check them with the CLI, and run them (never a shell).\n\n\
         Usage: goethe ask [OPTIONS] [REQUEST...]\n\
         Also: goethe ? REQUEST, or Ask on the main menu (goethe with no arguments).\n\n\
         Options:\n\
           -y, --yes           Run plans that change things; without it, only plans that just read run\n\
               --dry-run       Print the commands instead of running them\n\
               --no-input      Never ask; exit 3 if a detail or a confirmation is missing\n\
               --json          Print the plan, or what it still needs, as JSON\n\
               --dump-catalog  Print the commands Ask chooses from, as JSON\n\
           -v, --verbose       Say how commands are ranked\n\
           -h, --help          Show this help\n\n\
         Exit codes:\n\
           0  Ran, printed a ready plan, or stopped on request\n\
           1  A command failed\n\
           2  Not a Goethe request, shell characters, or wrong usage\n\
           3  A detail or a confirmation is missing"
    );
}
