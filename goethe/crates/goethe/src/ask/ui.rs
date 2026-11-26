// goethe ask output outside the Ask screen: the plan as numbered commands, each with a short why.

use super::planner::PlanState;
use super::schema::Risk;
use goethe_core::term::{self, Color};

pub fn print_plan(plan: &PlanState) {
    for (i, line) in plan.lines.iter().enumerate() {
        eprintln!("{}. {} {}", i + 1, risk_badge(line.risk), format_argv(&line.argv));
        if !line.why.is_empty() {
            eprintln!("   {}", term::paint(&line.why, Color::Gray));
        }
    }
}

pub fn print_abstain(reason: &str) {
    eprintln!("{} {reason}", term::paint("ask:", Color::Yellow));
}

fn risk_badge(risk: Risk) -> String {
    let tag = format!("[{}]", risk.label());
    match risk {
        Risk::ReadOnly => term::paint(&tag, Color::Green),
        Risk::Mutating | Risk::Download => term::paint(&tag, Color::Blue),
        Risk::Network => term::paint(&tag, Color::Yellow),
        Risk::Destructive => term::paint(&tag, Color::Red),
    }
}

pub fn format_argv(argv: &[String]) -> String {
    argv.iter()
        .map(|a| if a.contains(' ') || a.is_empty() { format!("\"{a}\"") } else { a.clone() })
        .collect::<Vec<_>>()
        .join(" ")
}
