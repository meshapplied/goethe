// Tab completion: the live values the shell asks goethe for, and goethe completion to set it up or take it out.

use clap::builder::StyledStr;
use clap_complete::engine::CompletionCandidate;
use goethe_core::completion::{self, Dirs, Installed, Shell, SHELLS};
use goethe_core::paths::goethe_home;

fn candidate(value: &str, help: &str) -> CompletionCandidate {
    CompletionCandidate::new(value).help(Some(StyledStr::from(help.to_string())))
}

pub fn model_candidates(ids: &[String], aliases: &[(String, String)]) -> Vec<CompletionCandidate> {
    // Each installed model by full id and by the name after its organisation (also in lowercase, hidden until typed), then the aliases.
    let mut out: Vec<CompletionCandidate> = ids.iter().map(|id| candidate(id, "installed model")).collect();
    out.extend(ids.iter().filter_map(|id| id.split_once('/').map(|(_, name)| candidate(name, id))));
    let lower = ids.iter().filter_map(|id| id.rsplit('/').next().map(|n| (n.to_lowercase(), id))).filter(|(n, id)| n != id.rsplit('/').next().unwrap_or(""));
    out.extend(lower.map(|(n, id)| candidate(&n, id).hide(true)));
    out.extend(aliases.iter().map(|(a, m)| candidate(a, &format!("alias for {m}"))));
    out
}

pub fn models() -> Vec<CompletionCandidate> {
    let mut ids = goethe_core::models::installed_ids(&goethe_home());
    ids.sort();
    model_candidates(&ids, &goethe_core::alias::list().unwrap_or_default())
}

pub fn models_or_all() -> Vec<CompletionCandidate> {
    let mut out = vec![candidate("all", "every model")];
    out.extend(models());
    out
}

pub fn aliases() -> Vec<CompletionCandidate> {
    goethe_core::alias::list().unwrap_or_default().iter().map(|(a, m)| candidate(a, m)).collect()
}

pub fn gpus() -> Vec<CompletionCandidate> {
    // From the saved hardware profile, so a Tab never waits for the driver.
    goethe_core::hw::load_hardware_state(&goethe_home())
        .map(|hw| hw.gpus.iter().map(|g| candidate(&format!("gpu{}", g.index), &g.name)).collect())
        .unwrap_or_default()
}

pub fn interfaces() -> Vec<CompletionCandidate> {
    let routes = goethe_core::netif::routes();
    goethe_core::netif::candidates(&goethe_core::netif::interfaces(&routes))
        .iter()
        .map(|i| candidate(&i.name, &i.addr.to_string()))
        .collect()
}

pub fn binds() -> Vec<CompletionCandidate> {
    let mut out = vec![candidate("127.0.0.1", "this machine only"), candidate("0.0.0.0", "every address")];
    out.extend(interfaces());
    out
}

fn keys() -> Vec<(String, String)> {
    goethe_core::security::key_list_detailed()
        .unwrap_or_default()
        .iter()
        .map(|k| {
            let field = |f: &str| k.get(f).and_then(|v| v.as_str()).unwrap_or("").to_string();
            (field("id"), field("name"))
        })
        .collect()
}

pub fn key_ids() -> Vec<CompletionCandidate> {
    keys().iter().map(|(id, name)| candidate(id, name)).collect()
}

pub fn key_names() -> Vec<CompletionCandidate> {
    keys().iter().map(|(id, name)| candidate(name, id)).collect()
}

fn script(shell: Shell) -> Result<String, String> {
    // The shell's registration script, as goethe itself prints it with COMPLETE set; it calls goethe from PATH at each Tab.
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    let out = std::process::Command::new(exe).env("COMPLETE", shell.name()).output().map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    if !out.status.success() || text.trim().is_empty() {
        return Err(format!("could not make the {} completion script", shell.name()));
    }
    Ok(text)
}

fn shell_of(arg: Option<&str>) -> Result<Shell, String> {
    let name = arg.map(str::to_string).or_else(|| std::env::var("SHELL").ok()).unwrap_or_default();
    Shell::parse(&name).ok_or_else(|| format!("Tab completion covers bash, zsh and fish; pass --shell (your shell is {:?})", name))
}

fn shown(p: &std::path::Path, dirs: &Dirs) -> String {
    p.strip_prefix(&dirs.home).map(|r| format!("~/{}", r.display())).unwrap_or_else(|_| p.display().to_string())
}

pub fn run(action: Option<&str>, shell: Option<&str>, auto: bool) -> Result<(), String> {
    // install (the default when a shell is named), remove, or the state of each shell.
    let dirs = Dirs::from_env()?;
    match action.unwrap_or("status") {
        "install" => {
            let sh = shell_of(shell)?;
            match completion::install(&dirs, sh, &script(sh)?, !auto)? {
                Installed::Loaded(p) => println!("Tab completion is on for {} ({}). Open a new terminal to use it.", sh.name(), shown(&p, &dirs)),
                Installed::Sourced(p, rc) => println!(
                    "Tab completion is on for {} ({} sources {}). Open a new terminal, or run: source {}",
                    sh.name(),
                    shown(&rc, &dirs),
                    shown(&p, &dirs),
                    shown(&rc, &dirs)
                ),
                Installed::NeedsRc(rc) => println!("Tab completion for {}: run goethe completion install (it adds a short marked block to {}).", sh.name(), shown(&rc, &dirs)),
            }
        }
        "remove" => {
            let shells: Vec<Shell> = match shell {
                Some(s) => vec![shell_of(Some(s))?],
                None => SHELLS.to_vec(),
            };
            let changed: Vec<String> = shells.iter().map(|s| completion::remove(&dirs, *s)).collect::<Result<Vec<_>, _>>()?.concat().iter().map(|p| shown(p, &dirs)).collect();
            if changed.is_empty() {
                println!("Tab completion was not set up; nothing to remove.");
            } else {
                println!("Tab completion removed ({}).", changed.join(", "));
            }
        }
        _ => {
            let current = shell_of(shell).ok();
            for s in SHELLS {
                let state = if completion::installed(&dirs, s) { format!("on ({})", shown(&dirs.script(s), &dirs)) } else { "off".to_string() };
                let mark = if Some(s) == current { "  (your shell)" } else { "" };
                println!("{:<5} {state}{mark}", s.name());
            }
            if let Some(s) = current.filter(|s| !completion::installed(&dirs, *s)) {
                print!("\n{}", goethe_core::term::commands(&format!("Turn it on for {}:", s.name()), &["goethe completion install"]));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_complete_by_id_by_name_and_by_alias() {
        let ids = vec!["meshapplied/Qwen3.8-27B-NVFP4-NInfer".to_string(), "local-model".to_string()];
        let got: Vec<String> = model_candidates(&ids, &[("fast".into(), "local-model".into())])
            .iter()
            .map(|c| c.get_value().to_string_lossy().into_owned())
            .collect();
        assert_eq!(got, ["meshapplied/Qwen3.8-27B-NVFP4-NInfer", "local-model", "Qwen3.8-27B-NVFP4-NInfer", "qwen3.8-27b-nvfp4-ninfer", "fast"]);
    }
}
