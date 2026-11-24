// One request (one or more parts) to an ordered list of clap commands: rank each part, read the live state, ask for what is missing, validate.

use super::embed::{ScoredIntent, ShortlistBackend};
use super::parse::{find_hf_or_path, find_shell_metachar, is_obvious_ood, negation, split_parts, split_words, without_wish};
use super::schema::{CliCatalog, IntentCard, Kind, Risk};
use super::slots::{extract_slots, slot_candidates, World};
use crate::spec::{Asked, Fact};
use serde_json::{json, Value};
use std::collections::HashMap;

pub const TAU_OOD: f32 = 0.30;
pub const TAU_HI: f32 = 0.56;
pub const DELTA_MARGIN: f32 = 0.08;

pub type Validate<'a> = dyn Fn(&[String]) -> Result<(), clap::Error> + 'a;

#[derive(Clone, Debug)]
pub struct MissingSlot {
    pub is_flag: bool,
    pub name: String,
    pub label: String,
    pub kind: Kind,
    pub possible: Vec<String>,
    pub shown: String,
}

#[derive(Clone, Debug)]
pub struct PlanStep {
    pub clause: String,
    pub intent_id: String,
    pub argv: Vec<String>,
    pub risk: Risk,
    pub positionals: HashMap<String, Vec<String>>,
    pub flags: Vec<(String, Vec<String>)>,
    pub passthrough: Vec<String>,
    pub missing: Vec<MissingSlot>,
    pub ambiguous: Option<(String, Vec<String>)>,
    pub alternatives: Vec<ScoredIntent>,
    pub needs_intent_pick: bool,
    pub verified: bool,
    pub verify_err: Option<String>,
    pub reason: Option<String>,
    pub part: usize,
}

#[derive(Clone, Debug)]
pub enum PlanAction {
    SelectIntent { intent_id: String },
    SetSlot { is_flag: bool, name: String, values: Vec<String> },
    Ignore { value: String },
    LeaveOut,
}

#[derive(Clone, Debug)]
pub enum PendingDecision {
    Abstain { reason: String },
    Invalid { message: String },
    PickIntent { clause: String, options: Vec<ScoredIntent> },
    PickSlot { missing: MissingSlot, options: Vec<(String, String)> },
    PromptSlot { missing: MissingSlot },
    Unused { part: usize, values: Vec<String> },
    ReadyToConfirm,
}

#[derive(Clone, Debug)]
pub struct PlanLine {
    pub argv: Vec<String>,
    pub why: String,
    pub intent_id: String,
    pub risk: Risk,
    pub verified: bool,
    pub part: usize,
    pub placeholders: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PlanPart {
    pub text: String,
    pub options: Vec<ScoredIntent>,
    pub left_out: bool,
    pub ignored: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct PlanState {
    pub raw_query: String,
    pub parts: Vec<PlanPart>,
    pub goals: Vec<PlanStep>,
    pub lines: Vec<PlanLine>,
    pub abstain: Option<String>,
    model_ref: Option<String>,
    world: World,
}

impl PlanState {
    pub fn build(
        query: &str,
        catalog: &CliCatalog,
        retriever: &dyn ShortlistBackend,
        world: &World,
        validate: &Validate,
    ) -> Self {
        // A request is one or more parts; each part is one command ranked on what to do (a written model id is a value, shared by the parts), then what the live state needs first.
        let trimmed = query.trim();
        if trimmed.is_empty() {
            return Self::abstained(trimmed, "Type what you want Goethe to do, such as: load qwen and give me an API.", world);
        }
        if let Some(meta) = find_shell_metachar(trimmed) {
            return Self::abstained(trimmed, &format!("Shell characters such as '{meta}' are not allowed; say it in words."), world);
        }
        if let Some(word) = negation(trimmed) {
            return Self::abstained(trimmed, &format!("Ask plans what to do, not what not to do ('{word}'); say what you want done."), world);
        }
        if is_obvious_ood(trimmed) {
            return Self::abstained(trimmed, "That is not something Goethe does; goethe help lists what it does.", world);
        }
        let model_ref = written_model(trimmed, catalog);
        let rank = |text: &str| retriever.rank(&without(text, model_ref.as_deref()), catalog.intents.len());
        let mut texts = request_parts(trimmed, &rank);
        if texts.len() < 2 {
            texts = vec![(trimmed.to_string(), rank(trimmed))];
        }
        let several = texts.len() > 1;
        let model_used = texts.iter().any(|(_, open)| open.first().is_some_and(|t| takes_model(catalog, &t.id)));
        let mut parts = Vec::new();
        let mut goals = Vec::new();
        for (i, (text, open)) in texts.into_iter().enumerate() {
            let (options, about_model) = part_options(&text, open, model_ref.as_deref(), model_used, catalog);
            let Some(top) = options.first().cloned() else {
                continue;
            };
            if !about_model && !several && top.score < TAU_OOD {
                continue;
            }
            let needs_intent_pick = !clear(&options, if about_model { TAU_OOD } else { TAU_HI }) || changes_beside_a_stray_word(&top.id, &text, catalog, retriever, world);
            let input = StepBuildInput { clause: &text, intent_id: &top.id, alternatives: options.clone(), needs_intent_pick };
            let mut goal = build_step(input, catalog, world, validate);
            goal.part = i;
            share_model(&mut goal, model_ref.as_deref(), catalog, world, validate);
            goals.push(goal);
            parts.push(PlanPart { text, options, left_out: false, ignored: Vec::new() });
        }
        if goals.is_empty() {
            return Self::abstained(trimmed, "No Goethe command matches that request closely enough; goethe help lists the commands.", world);
        }
        let mut plan = PlanState { raw_query: trimmed.to_string(), parts, goals, lines: Vec::new(), abstain: None, model_ref, world: world.clone() };
        plan.refresh(catalog, validate);
        plan
    }

    fn abstained(query: &str, reason: &str, world: &World) -> Self {
        PlanState { raw_query: query.to_string(), parts: Vec::new(), goals: Vec::new(), lines: Vec::new(), abstain: Some(reason.to_string()), model_ref: None, world: world.clone() }
    }

    fn focus(&self) -> usize {
        // Which command a part is comes first, then the values it is missing, in order.
        let pick = self.goals.iter().position(|g| g.needs_intent_pick);
        pick.or_else(|| self.goals.iter().position(|g| g.ambiguous.is_some() || !g.missing.is_empty())).or_else(|| self.goals.iter().position(|g| g.verify_err.is_some())).unwrap_or(0)
    }

    fn part_focus(&self, part: usize) -> Option<usize> {
        // Within one part: which command first, then its missing values, then its own command.
        let mine = |g: &&PlanStep| g.part == part;
        let at = |f: &dyn Fn(&PlanStep) -> bool| self.goals.iter().position(|g| mine(&g) && f(g));
        at(&|g| g.needs_intent_pick).or_else(|| at(&|g| g.ambiguous.is_some() || !g.missing.is_empty())).or_else(|| at(&|g| g.verify_err.is_some())).or_else(|| at(&|g| g.reason.is_none()))
    }

    pub fn next_decision(&self, catalog: &CliCatalog, world: &World) -> PendingDecision {
        // The single next question for this request, from its first unresolved part; then any value no command uses.
        match self.decide(Some(self.focus()), catalog, world) {
            PendingDecision::ReadyToConfirm => (0..self.parts.len()).find_map(|p| self.unused_decision(p, world)).unwrap_or(PendingDecision::ReadyToConfirm),
            other => other,
        }
    }

    pub fn part_decision(&self, part: usize, catalog: &CliCatalog, world: &World) -> PendingDecision {
        // The question one part still has, if any; then the values in its words that no command uses.
        match self.decide(self.part_focus(part), catalog, world) {
            PendingDecision::ReadyToConfirm => self.unused_decision(part, world).unwrap_or(PendingDecision::ReadyToConfirm),
            other => other,
        }
    }

    fn unused_decision(&self, part: usize, world: &World) -> Option<PendingDecision> {
        let values = self.unused(part, world);
        (!values.is_empty()).then_some(PendingDecision::Unused { part, values })
    }

    pub fn unused(&self, part: usize, world: &World) -> Vec<String> {
        // Values written in a part that no command of the plan takes and that were not left out, and a model it names when none of its settled commands acts on one ("uninstall llama" never uninstalls Goethe); Ask never runs something other than what was said.
        let Some(p) = self.parts.get(part).filter(|p| !p.left_out) else {
            return Vec::new();
        };
        let used = |v: &str| self.goals.iter().any(|g| g.argv.iter().any(|a| a == v || *a == world.key_id(v)));
        let mut out: Vec<String> = super::slots::typed_values(&p.text, world).into_iter().filter(|v| !used(v) && !p.ignored.contains(v)).collect();
        let mine: Vec<&PlanStep> = self.goals.iter().filter(|g| g.part == part).collect();
        let settled = !mine.is_empty() && mine.iter().all(|g| !g.needs_intent_pick && g.ambiguous.is_none() && g.missing.is_empty());
        if settled && !mine.iter().any(|g| g.argv.iter().any(|a| super::slots::is_model_value(a, world))) {
            let named: Vec<String> = super::slots::named_models(&p.text, world).into_iter().filter(|m| !p.ignored.contains(m) && !out.contains(m)).collect();
            out.extend(named);
        }
        out
    }

    fn decide(&self, at: Option<usize>, catalog: &CliCatalog, world: &World) -> PendingDecision {
        if let Some(reason) = &self.abstain {
            return PendingDecision::Abstain { reason: reason.clone() };
        }
        let Some(step) = at.and_then(|i| self.goals.get(i)) else {
            return PendingDecision::Abstain { reason: "Every part is left out; nothing would run.".into() };
        };
        if step.needs_intent_pick {
            return PendingDecision::PickIntent { clause: step.clause.clone(), options: step.alternatives.clone() };
        }
        if let Some((slot_name, cands)) = &step.ambiguous {
            let opts: Vec<(String, String)> = cands
                .iter()
                .map(|c| (c.clone(), world.aliases.iter().find(|(a, _)| a == c).map_or_else(|| "installed model".to_string(), |(_, m)| format!("alias for {m}"))))
                .collect();
            let pos = catalog.get(&step.intent_id).and_then(|c| c.positionals.iter().find(|p| &p.id == slot_name));
            let missing = MissingSlot {
                is_flag: false,
                name: slot_name.clone(),
                label: pos.map_or_else(|| slot_name.clone(), |p| p.value_name.clone()),
                kind: pos.map_or(Kind::InstalledModel, |p| p.kind),
                possible: Vec::new(),
                shown: format!("<{}>", pos.map_or_else(|| slot_name.to_uppercase(), |p| p.value_name.clone())),
            };
            return PendingDecision::PickSlot { missing, options: opts };
        }
        if let Some(message) = step.verify_err.clone().filter(|_| step.missing.is_empty()) {
            return PendingDecision::Invalid { message };
        }
        if let Some(miss) = step.missing.first() {
            let opts = slot_candidates(miss.kind, &miss.possible, world);
            if !opts.is_empty() {
                return PendingDecision::PickSlot { missing: miss.clone(), options: opts };
            }
            return PendingDecision::PromptSlot { missing: miss.clone() };
        }
        PendingDecision::ReadyToConfirm
    }

    #[cfg(test)]
    pub fn with_goals(goals: Vec<PlanStep>, world: &World, catalog: &CliCatalog, validate: &Validate) -> Self {
        // A plan of given commands, ordered after what the live state needs first.
        let mut plan = PlanState { raw_query: String::new(), parts: Vec::new(), goals, lines: Vec::new(), abstain: None, model_ref: None, world: world.clone() };
        plan.refresh(catalog, validate);
        plan
    }

    #[cfg(test)]
    pub fn apply(
        &mut self,
        action: PlanAction,
        catalog: &CliCatalog,
        world: &World,
        validate: &Validate,
    ) {
        // One answer for the part that asked first.
        let part = self.goals.get(self.focus()).map_or(0, |g| g.part);
        self.apply_to(part, action, catalog, world, validate);
    }

    pub fn apply_to(
        &mut self,
        part: usize,
        action: PlanAction,
        catalog: &CliCatalog,
        world: &World,
        validate: &Validate,
    ) {
        // One answer for one part: its command, a value it is missing, or leaving it out.
        let model_ref = self.model_ref.clone();
        match action {
            PlanAction::Ignore { value } => {
                if let Some(p) = self.parts.get_mut(part) {
                    p.ignored.push(value);
                }
            }
            PlanAction::LeaveOut => {
                self.goals.retain(|g| g.part != part);
                if let Some(p) = self.parts.get_mut(part) {
                    p.left_out = true;
                }
            }
            PlanAction::SelectIntent { intent_id } => {
                let Some(p) = self.parts.get_mut(part) else {
                    return;
                };
                p.left_out = false;
                let input = StepBuildInput { clause: &p.text, intent_id: &intent_id, alternatives: p.options.clone(), needs_intent_pick: false };
                let mut goal = build_step(input, catalog, world, validate);
                goal.part = part;
                share_model(&mut goal, model_ref.as_deref(), catalog, world, validate);
                self.goals.retain(|g| g.part != part || g.reason.is_some());
                let at = self.goals.iter().position(|g| g.part > part).unwrap_or(self.goals.len());
                self.goals.insert(at, goal);
            }
            PlanAction::SetSlot { is_flag, name, values } => {
                let Some(at) = self.part_focus(part) else {
                    return;
                };
                let goal = &mut self.goals[at];
                if is_flag {
                    match goal.flags.iter_mut().find(|(k, _)| k == &name) {
                        Some(entry) => entry.1 = values,
                        None => goal.flags.push((name, values)),
                    }
                } else {
                    goal.positionals.insert(name, values);
                    goal.ambiguous = None;
                }
                recompile_step(goal, catalog, validate);
            }
        }
        self.refresh(catalog, validate);
    }

    pub fn rewrite(
        &mut self,
        part: usize,
        text: &str,
        catalog: &CliCatalog,
        retriever: &dyn ShortlistBackend,
        validate: &Validate,
    ) -> Result<(), String> {
        // New words for one part ("type something else"): rank them like a part of the request and take the best command.
        let text = text.trim();
        if text.is_empty() || part >= self.parts.len() {
            return Ok(());
        }
        if let Some(meta) = find_shell_metachar(text) {
            return Err(format!("Shell characters such as '{meta}' are not allowed; say it in words."));
        }
        if let Some(word) = negation(text) {
            return Err(format!("Ask plans what to do, not what not to do ('{word}'); say what you want done."));
        }
        let model_ref = self.model_ref.clone().or_else(|| written_model(text, catalog));
        let open = retriever.rank(&without(text, model_ref.as_deref()), catalog.intents.len());
        let model_used = self.goals.iter().any(|g| g.part != part && takes_model(catalog, &g.intent_id));
        let (options, about_model) = part_options(text, open, model_ref.as_deref(), model_used, catalog);
        let Some(top) = options.first().cloned() else {
            return Err("No Goethe command matches those words.".into());
        };
        self.model_ref = model_ref;
        self.parts[part] = PlanPart { text: text.to_string(), options: options.clone(), left_out: false, ignored: Vec::new() };
        let needs_intent_pick = !clear(&options, if about_model { TAU_OOD } else { TAU_HI }) || changes_beside_a_stray_word(&top.id, text, catalog, retriever, &self.world);
        let input = StepBuildInput { clause: text, intent_id: &top.id, alternatives: options.clone(), needs_intent_pick };
        let mut goal = build_step(input, catalog, &self.world, validate);
        goal.part = part;
        share_model(&mut goal, self.model_ref.as_deref(), catalog, &self.world, validate);
        self.goals.retain(|g| g.part != part);
        let at = self.goals.iter().position(|g| g.part > part).unwrap_or(self.goals.len());
        self.goals.insert(at, goal);
        self.refresh(catalog, validate);
        Ok(())
    }

    pub fn is_ready(&self) -> bool {
        self.abstain.is_none()
            && !self.goals.is_empty()
            && self.goals.iter().all(|s| !s.needs_intent_pick && s.ambiguous.is_none() && s.missing.is_empty() && s.verified)
            && self.lines.iter().all(|l| l.verified)
            && (0..self.parts.len()).all(|p| self.unused(p, &self.world).is_empty())
    }

    pub fn has_destructive(&self) -> bool {
        self.lines.iter().any(|l| l.risk.is_destructive())
    }

    pub fn all_readonly(&self) -> bool {
        !self.lines.is_empty() && self.lines.iter().all(|l| l.risk == Risk::ReadOnly)
    }

    pub fn to_json(&self, catalog: &CliCatalog, world: &World) -> Value {
        // The plan, or what it still needs, for --json.
        match self.next_decision(catalog, world) {
            PendingDecision::Abstain { reason } => json!({"status": "refused", "reason": reason, "steps": []}),
            PendingDecision::Invalid { message } => json!({"status": "clarify", "kind": "invalid", "message": message}),
            PendingDecision::PickIntent { clause, options } => json!({
                "status": "clarify", "kind": "command", "part": clause,
                "candidates": options.iter().map(|o| json!({
                    "id": o.id, "argv": catalog.get(&o.id).map(|c| c.argv_prefix.join(" ")).unwrap_or_default(), "score": o.score,
                })).collect::<Vec<_>>(),
            }),
            PendingDecision::PickSlot { missing, options } => json!({
                "status": "clarify", "kind": "value", "missing": missing.name, "is_flag": missing.is_flag,
                "options": options.iter().map(|(v, _)| v).collect::<Vec<_>>(),
            }),
            PendingDecision::Unused { part, values } => json!({"status": "clarify", "kind": "unused", "part": part, "values": values}),
            PendingDecision::PromptSlot { missing } => {
                json!({"status": "clarify", "kind": "value", "missing": missing.name, "is_flag": missing.is_flag, "options": []})
            }
            PendingDecision::ReadyToConfirm => {
                let steps = self.lines.iter().map(|l| {
                    json!({"command": l.intent_id, "risk": l.risk.label(), "argv": l.argv, "why": l.why, "verified": l.verified})
                }).collect::<Vec<_>>();
                json!({"status": "ready", "destructive": self.has_destructive(), "steps": steps})
            }
        }
    }

    fn refresh(&mut self, catalog: &CliCatalog, validate: &Validate) {
        // Steps Goethe's state needs come first: each part's command, after the giver of every fact it needs that is not true yet (recursively). An auto step keeps the values already given to it.
        let (old_auto, user): (Vec<PlanStep>, Vec<PlanStep>) = std::mem::take(&mut self.goals).into_iter().partition(|g| g.reason.is_some());
        let mut have: Vec<(Fact, Option<String>)> = Vec::new();
        let mut goals = Vec::new();
        for goal in user {
            if let Some(card) = catalog.get(&goal.intent_id).filter(|_| !goal.needs_intent_pick) {
                let model = step_model(&goal, card).or_else(|| self.model_ref.clone().filter(|_| card.model_arg().is_none()));
                for need in card.does.needs {
                    self.satisfy(*need, model.as_deref(), goal.part, catalog, validate, &old_auto, &mut have, &mut goals);
                }
                have.extend(card.does.gives.iter().map(|f| (*f, model.clone())));
            }
            goals.push(goal);
        }
        refuse_undoing_what_it_gives(&mut goals, catalog);
        self.goals = goals;
        self.lines = order_lines(&self.goals, &self.world, catalog);
    }

    #[expect(clippy::too_many_arguments, reason = "one recursive step of the regression; the state it threads is the plan being built")]
    fn satisfy(
        &self,
        fact: Fact,
        model: Option<&str>,
        part: usize,
        catalog: &CliCatalog,
        validate: &Validate,
        old_auto: &[PlanStep],
        have: &mut Vec<(Fact, Option<String>)>,
        goals: &mut Vec<PlanStep>,
    ) {
        let model = model.filter(|_| matches!(fact, Fact::Installed | Fact::Loaded));
        if self.world.has(fact, model) || have.iter().any(|(f, m)| *f == fact && m.as_deref() == model) {
            return;
        }
        let Some(giver) = catalog.giver(fact) else {
            return;
        };
        for need in giver.does.needs {
            self.satisfy(*need, model, part, catalog, validate, old_auto, have, goals);
        }
        let kept = old_auto.iter().find(|s| s.intent_id == giver.id && s.part == part && step_model(s, giver).as_deref() == model);
        let step = kept.cloned().unwrap_or_else(|| {
            let input = StepBuildInput { clause: "", intent_id: &giver.id, alternatives: Vec::new(), needs_intent_pick: false };
            let mut step = build_step(input, catalog, &self.world, validate);
            if let (Some(m), Some(pos)) = (model, giver.model_arg()) {
                step.positionals.insert(pos.id.clone(), vec![m.to_string()]);
                recompile_step(&mut step, catalog, validate);
            }
            step.reason = Some(fact.why().to_string());
            step.part = part;
            step
        });
        have.extend(giver.does.gives.iter().map(|f| (*f, model.map(str::to_string))));
        goals.push(step);
    }
}

fn step_model(step: &PlanStep, card: &IntentCard) -> Option<String> {
    card.model_arg().and_then(|p| step.positionals.get(&p.id)).and_then(|v| v.first()).cloned()
}

fn request_parts(request: &str, rank: &dyn Fn(&str) -> Vec<ScoredIntent>) -> Vec<(String, Vec<ScoredIntent>)> {
    // Each piece that is a request on its own is a part; a piece that is not (a detail such as "at every boot") joins the part before it, or the next one.
    let mut parts: Vec<(String, Vec<ScoredIntent>)> = Vec::new();
    let mut lead = String::new();
    for piece in split_parts(request) {
        let open = rank(&piece);
        let request_like = open.first().is_some_and(|t| t.score >= TAU_OOD);
        match parts.last_mut() {
            _ if request_like && lead.is_empty() => parts.push((piece, open)),
            _ if request_like => {
                let text = format!("{} {piece}", std::mem::take(&mut lead));
                let open = rank(&text);
                parts.push((text, open));
            }
            Some(last) => {
                last.0 = format!("{} {piece}", last.0);
                last.1 = rank(&last.0);
            }
            None => lead = format!("{lead} {piece}").trim().to_string(),
        }
    }
    parts
}

fn part_options(text: &str, mut open: Vec<ScoredIntent>, model_ref: Option<&str>, model_used: bool, catalog: &CliCatalog) -> (Vec<ScoredIntent>, bool) {
    // The commands one part can be: only model commands when the part is about the written model (it names it and no other part uses it, or its best guess takes a model), and only commands that take every --flag written out.
    let top_takes_model = open.first().is_some_and(|t| takes_model(catalog, &t.id));
    let names_it = model_ref.is_some_and(|m| text.contains(m));
    let about_model = model_ref.is_some() && (top_takes_model || (names_it && !model_used));
    if about_model {
        open.retain(|r| takes_model(catalog, &r.id));
    }
    let typed: Vec<&str> = text.split(" -- ").next().unwrap_or(text).split_whitespace().filter_map(|t| t.strip_prefix("--")).filter_map(|t| t.split('=').next()).filter(|t| !t.is_empty()).collect();
    let takes_all = |id: &str| catalog.get(id).is_some_and(|c| typed.iter().all(|f| c.flags.iter().any(|g| g.long == *f)));
    if !typed.is_empty() && open.iter().any(|r| takes_all(&r.id)) {
        open.retain(|r| takes_all(&r.id));
    }
    prefer_named_child(&mut open, text);
    open.truncate(5);
    (open, about_model)
}

fn prefer_named_child(open: &mut Vec<ScoredIntent>, text: &str) {
    // Naming a command's words points at it: a subcommand named by its own word takes its parent's place and score, and a command named by two or more of its words, more than the best guess, stands level with it, so the plan asks instead of guessing.
    let words = split_words(&text.to_ascii_lowercase());
    let named = |id: &str| id.split('.').filter(|w| words.iter().any(|t| t == w)).count();
    if let Some(top) = open.first().map(|t| (named(&t.id), t.score)) {
        if let Some(k) = open.iter().position(|c| named(&c.id) >= 2 && named(&c.id) > top.0).filter(|k| *k > 0) {
            let mut c = open.remove(k);
            c.score = c.score.max(top.1);
            open.insert(1, c);
        }
    }
    let mut i = 0;
    while i < open.len() {
        let parent = format!("{}.", open[i].id);
        let child = open.iter().skip(i + 1).position(|c| c.id.strip_prefix(&parent).is_some_and(|w| !w.contains('.') && words.iter().any(|t| t == w)));
        if let Some(k) = child {
            let mut child = open.remove(i + 1 + k);
            child.score = child.score.max(open[i].score);
            open.insert(i, child);
        }
        i += 1;
    }
}

fn changes_beside_a_stray_word(id: &str, text: &str, catalog: &CliCatalog, retriever: &dyn ShortlistBackend, world: &World) -> bool {
    // A command that changes something is asked, never taken, while a word means nothing to any command and is no value or model: "start the car" is not "goethe start".
    if !catalog.get(id).is_some_and(super::embed::gated) {
        return false;
    }
    let mut rest = text.to_string();
    for v in super::slots::typed_values(text, world) {
        rest = rest.replace(&v, " ");
    }
    super::embed::lex_tokens(&rest).iter().any(|t| !t.chars().all(|c| c.is_ascii_digit()) && !retriever.knows(t) && super::slots::named_models(t, world).is_empty())
}

fn clear(ranked: &[ScoredIntent], bar: f32) -> bool {
    // One command stands out: over the bar and ahead of the next by the margin (a parent command and its own subcommand do not compete).
    let Some(top) = ranked.first() else {
        return false;
    };
    let lead = ranked.get(1).is_none_or(|s| top.score - s.score >= DELTA_MARGIN || top.id.starts_with(&format!("{}.", s.id)));
    top.score >= bar && lead
}

fn share_model(goal: &mut PlanStep, model_ref: Option<&str>, catalog: &CliCatalog, world: &World, validate: &Validate) {
    // A part whose command needs a model and names none uses the request's model ("install and load it").
    let (Some(model), Some(card)) = (model_ref, catalog.get(&goal.intent_id)) else {
        return;
    };
    let Some(pos) = card.model_arg().filter(|p| super::slots::takes_model_value(card, p.kind, model, world)) else {
        return;
    };
    if goal.positionals.contains_key(&pos.id) || goal.ambiguous.as_ref().is_some_and(|(n, _)| n != &pos.id) {
        return;
    }
    goal.positionals.insert(pos.id.clone(), vec![model.to_string()]);
    goal.ambiguous = None;
    recompile_step(goal, catalog, validate);
}

fn refuse_undoing_what_it_gives(goals: &mut [PlanStep], catalog: &CliCatalog) {
    // A request that gives a fact and undoes it again on the same model contradicts itself (a stray "Unload"): that step is invalid until the request is fixed.
    let mut given: Vec<(Fact, Option<String>, String)> = Vec::new();
    for g in goals.iter_mut().filter(|g| !g.needs_intent_pick) {
        let Some(card) = catalog.get(&g.intent_id) else {
            continue;
        };
        let model = step_model(g, card);
        if let Some((_, _, by)) = given.iter().find(|(f, m, _)| *m == model && card.does.undoes.contains(f)) {
            g.verified = false;
            g.verify_err = Some(format!("this request asks for both {by} and {}{}; say which one you want", g.intent_id, model.as_deref().map(|m| format!(" {m}")).unwrap_or_default()));
        }
        given.extend(card.does.gives.iter().map(|f| (*f, model.clone(), g.intent_id.clone())));
    }
}

fn written_model(text: &str, catalog: &CliCatalog) -> Option<String> {
    // A model id or path written in the request; "Load/Unload" is two command words, not an id.
    let leaves: Vec<&str> = catalog.intents.iter().filter_map(|c| c.id.rsplit('.').next()).collect();
    find_hf_or_path(&split_words(text)).filter(|m| !m.split('/').all(|h| leaves.contains(&h.to_ascii_lowercase().as_str())))
}

fn without(text: &str, model_ref: Option<&str>) -> String {
    // The words that say what to do; the model id is a value, not part of what to do.
    let text = &without_wish(text);
    let Some(m) = model_ref else {
        return text.to_string();
    };
    let rest = split_words(text).into_iter().filter(|t| !t.contains(m)).collect::<Vec<_>>().join(" ");
    if rest.trim().is_empty() { text.to_string() } else { rest }
}

pub(crate) fn order_lines(goals: &[PlanStep], world: &World, catalog: &CliCatalog) -> Vec<PlanLine> {
    // The steps in order as commands to run; a command already listed is not repeated.
    let mut lines: Vec<PlanLine> = Vec::new();
    for goal in goals {
        let line = PlanLine {
            argv: goal.argv.clone(),
            why: goal.reason.clone().unwrap_or_else(|| primary_why(goal, world, catalog)),
            intent_id: goal.intent_id.clone(),
            risk: goal.risk,
            verified: goal.verified,
            part: goal.part,
            placeholders: goal.missing.iter().map(|m| m.shown.clone()).collect(),
        };
        if goal.needs_intent_pick || goal.argv.is_empty() || !lines.iter().any(|l| l.argv == line.argv) {
            lines.push(line);
        }
    }
    lines
}

fn primary_why(step: &PlanStep, world: &World, catalog: &CliCatalog) -> String {
    // What the command does, from its own help, so it never drifts from the CLI; a model it would load that already serves says so.
    let Some(card) = catalog.get(&step.intent_id) else {
        return String::new();
    };
    match step_model(step, card) {
        Some(m) if card.does.gives.contains(&Fact::Loaded) && world.has(Fact::Loaded, Some(&m)) => format!("{m} is already loaded"),
        _ => card.summary(),
    }
}

pub(crate) fn needs_install(model: &str, world: &World) -> bool {
    if super::parse::is_local_path(model) {
        return false;
    }
    if !world.installed_known {
        return true;
    }
    let target = world.aliases.iter().find(|(a, _)| a == model).map(|(_, id)| id.as_str()).unwrap_or(model);
    !world.installed.iter().any(|id| id == target || id == model)
}

fn takes_model(catalog: &CliCatalog, id: &str) -> bool {
    // Commands that take a model: the only ones a part about a written model id can be.
    catalog.get(id).is_some_and(IntentCard::takes_model)
}

struct StepBuildInput<'a> {
    clause: &'a str,
    intent_id: &'a str,
    alternatives: Vec<ScoredIntent>,
    needs_intent_pick: bool,
}

fn build_step(
    input: StepBuildInput<'_>,
    catalog: &CliCatalog,
    world: &World,
    validate: &Validate,
) -> PlanStep {
    // One command for a part: values from its words, then the argv clap checks.
    let Some(card) = catalog.get(input.intent_id) else {
        return PlanStep {
            clause: input.clause.to_string(),
            intent_id: input.intent_id.to_string(),
            argv: Vec::new(),
            risk: Risk::ReadOnly,
            positionals: Default::default(),
            flags: Vec::new(),
            passthrough: Vec::new(),
            missing: Vec::new(),
            ambiguous: None,
            alternatives: input.alternatives,
            needs_intent_pick: true,
            verified: false,
            verify_err: Some(format!("unknown command '{}'", input.intent_id)),
            reason: None,
            part: 0,
        };
    };
    let ext = extract_slots(card, input.clause, world);
    let mut step = PlanStep {
        clause: input.clause.to_string(),
        intent_id: input.intent_id.to_string(),
        argv: Vec::new(),
        risk: card.risk,
        positionals: ext.positionals,
        flags: ext.flags,
        passthrough: ext.passthrough,
        missing: Vec::new(),
        ambiguous: ext.ambiguous,
        alternatives: input.alternatives,
        needs_intent_pick: input.needs_intent_pick,
        verified: false,
        verify_err: None,
        reason: None,
        part: 0,
    };
    recompile_step(&mut step, catalog, validate);
    step
}

pub(crate) fn recompile_step(step: &mut PlanStep, catalog: &CliCatalog, validate: &Validate) {
    // The argv from the values bound so far; clap says which required values are still missing, and an asked flag joins them once its trigger is given.
    let Some(card) = catalog.get(&step.intent_id) else {
        return;
    };
    let mut argv = card.argv_prefix.clone();
    for pos in &card.positionals {
        if let Some(vals) = step.positionals.get(&pos.id) {
            argv.extend(vals.iter().cloned());
        }
    }
    for (flag, vals) in &step.flags {
        argv.push(format!("--{flag}"));
        argv.extend(vals.iter().cloned());
    }
    if !step.passthrough.is_empty() {
        argv.push("--".into());
        argv.extend(step.passthrough.iter().cloned());
    }
    let ambiguous = step.ambiguous.as_ref().map(|(n, _)| n.clone());
    let mut missing = Vec::new();
    step.verified = false;
    step.verify_err = None;
    let flag_widens = step.flags.iter().any(|(k, _)| card.flags.iter().any(|f| &f.long == k && f.widens));
    let lets_everyone_in = step.positionals.values().chain(step.flags.iter().map(|(_, v)| v)).flatten().any(|v| v.ends_with("/0"));
    step.risk = if flag_widens || lets_everyone_in { Risk::Destructive } else { card.risk };
    let dash = step.positionals.values().flatten().find(|v| v.starts_with('-'));
    match dash.map_or_else(|| validate(&argv), |v| Err(clap::Error::raw(clap::error::ErrorKind::InvalidValue, format!("a value cannot start with '-': {v}")))) {
        Ok(()) => step.verified = step.ambiguous.is_none(),
        Err(e) if e.kind() == clap::error::ErrorKind::MissingRequiredArgument => {
            missing.extend(missing_names(&e).iter().filter_map(|shown| missing_of(card, shown)).filter(|m| Some(&m.name) != ambiguous.as_ref()));
        }
        Err(e) if step.ambiguous.is_none() => step.verify_err = Some(first_line(&e.to_string())),
        Err(_) => {}
    }
    let given = |id: &str| step.flags.iter().any(|(k, _)| k == id) || step.positionals.contains_key(id);
    for f in &card.flags {
        let ask = match f.asked {
            Asked::Always => true,
            Asked::With(sibling) => given(sibling),
            Asked::No => false,
        };
        if ask && !given(&f.long) && !missing.iter().any(|m: &MissingSlot| m.name == f.long) {
            missing.push(flag_slot(f));
        }
    }
    if !missing.is_empty() {
        step.verified = false;
    }
    step.missing = missing;
    step.argv = argv;
}

pub(crate) fn missing_names(e: &clap::Error) -> Vec<String> {
    // The arguments clap reports missing, as it shows them: "<NAME>", "<GPU>..." or "--key <NAME_OR_ID>".
    use clap::error::{ContextKind, ContextValue};
    e.context()
        .filter(|(k, _)| *k == ContextKind::InvalidArg)
        .flat_map(|(_, v)| match v {
            ContextValue::Strings(s) => s.clone(),
            ContextValue::String(s) => vec![s.clone()],
            _ => Vec::new(),
        })
        .collect()
}

fn missing_of(card: &IntentCard, shown: &str) -> Option<MissingSlot> {
    // One reported argument, found on the card by its long name or its value name.
    if let Some(long) = shown.strip_prefix("--") {
        let long = long.split([' ', '=']).next().unwrap_or(long);
        return card.flags.iter().find(|f| f.long == long).map(flag_slot);
    }
    let name = shown.trim_start_matches('<').split('>').next().unwrap_or(shown);
    card.positionals.iter().find(|p| p.value_name == name).map(|p| MissingSlot {
        is_flag: false,
        name: p.id.clone(),
        label: p.value_name.clone(),
        kind: p.kind,
        possible: p.possible_values.clone(),
        shown: format!("<{}>", p.value_name),
    })
}

fn flag_slot(f: &super::schema::FlagSpec) -> MissingSlot {
    let kind = f.kind.unwrap_or(Kind::Text);
    MissingSlot {
        is_flag: true,
        name: f.long.clone(),
        label: format!("--{}", f.long),
        kind,
        possible: f.possible_values.clone(),
        shown: format!("--{} <{}>", f.long, f.value_name),
    }
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or("").trim_start_matches("error: ").to_string()
}
