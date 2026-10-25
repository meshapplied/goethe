// goethe start: a guided first run (install a model, serve it through the API, set up access) and an advanced track.

use crate::configure::{self, ApiOpts};
use crate::install::{self, InstallOpts};
use crate::paths::goethe_home;
use crate::term::{self, paint, Color};
use std::io::IsTerminal;

const STEPS: usize = 3;
const EXAMPLE_ID: &str = "deepseek-ai/DeepSeek-V4.1-Flash-1017";

enum Next {
    Serve(String),
    Stop(Option<String>),
}

pub fn run() -> Result<(), String> {
    // Start Goethe, then let the user pick the guided track (recommended) or the advanced one.
    if !(std::io::stdin().is_terminal() && std::io::stderr().is_terminal()) {
        print!("{}", no_terminal_text());
        return Ok(());
    }
    let (blocking, notes) = readiness();
    if !blocking.is_empty() {
        print!("{}", not_ready_text(&blocking));
        return Ok(());
    }
    crate::service::run_up(true)?;
    crate::placement::ensure_hw_profile(&goethe_home())?;
    let note = (!notes.is_empty()).then(|| notes.iter().map(|n| format!("{} {n}", paint("Note:", Color::Yellow))).collect::<Vec<_>>().join("\n"));
    screen("Serve a model through your own API in three steps", note.as_deref());
    match choose(&["Guided setup (Recommended)", "Advanced setup"], 0)? {
        Some(0) => guided(),
        Some(_) => advanced(),
        None => {
            print!("{}", later_text(None));
            Ok(())
        }
    }
}

fn readiness() -> (Vec<String>, Vec<String>) {
    // What stops this machine from serving (driver, GPUs that did not start, no supported GPU) and what only limits it.
    let mut blocking = Vec::new();
    let mut notes = Vec::new();
    let probe = crate::nvidia_driver::probe();
    if let Some(why) = crate::nvidia_driver::diagnose(&probe) {
        blocking.push(why);
    } else {
        match crate::nvidia_driver::unstarted(&probe, &crate::nvidia_driver::started_bus_ids()) {
            Some((true, why)) => blocking.push(why),
            Some((false, note)) => notes.push(note),
            None => {}
        }
        if blocking.is_empty() {
            if let Ok(hw) = crate::hw::profile(&goethe_home()) {
                let reasons: Vec<String> = hw
                    .gpus
                    .iter()
                    .filter_map(|g| crate::gpu_arch::unsupported_reason(&g.compute_capability).map(|w| format!("gpu{} {}: {w}", g.index, g.name)))
                    .collect();
                if !hw.gpus.is_empty() && reasons.len() == hw.gpus.len() {
                    blocking.extend(reasons);
                } else {
                    notes.extend(reasons.into_iter().map(|r| format!("{r} (it is skipped)")));
                }
            }
        }
    }
    if let crate::host_compiler::Verdict::Unusable(why) =
        crate::host_compiler::verdict(crate::host_compiler::find().as_ref(), &probe.distro)
    {
        notes.push(why);
    }
    (blocking, notes)
}

fn not_ready_text(blocking: &[String]) -> String {
    let list: Vec<String> = blocking.iter().map(|b| format!("  \u{2022} {b}")).collect();
    format!(
        "Goethe cannot serve models on this machine yet:\n\n{}\n\n{}",
        list.join("\n"),
        term::commands("Fix it, then check and start again:", &["goethe doctor", "goethe start"])
    )
}

fn guided() -> Result<(), String> {
    // Step 1 picks or installs a model; steps 2 and 3 serve it and open access, unless the user stops after step 1.
    let model = match step_install()? {
        Next::Serve(id) => id,
        Next::Stop(id) => {
            print!("{}", later_text(id.as_deref()));
            return Ok(());
        }
    };
    step_serve(&model)?;
    step_access(&model)
}

fn screen(title: &str, note: Option<&str>) {
    // A fresh screen: cleared, the title in bold, then an optional message (such as the last error) above the menu.
    print!("\x1b[2J\x1b[H{}\n\n", paint(title, Color::Bold));
    if let Some(n) = note {
        println!("{n}\n");
    }
}

fn heading(n: usize, title: &str, note: Option<&str>) {
    screen(&format!("Step {n}/{STEPS}  {title}"), note);
}

fn choose(options: &[&str], default: usize) -> Result<Option<usize>, String> {
    let owned: Vec<String> = options.iter().map(|o| o.to_string()).collect();
    term::select(&owned, default)
}

fn step_install() -> Result<Next, String> {
    // Install a new model (the default), serve one installed before, or run one already on this machine.
    let mut note: Option<String> = None;
    loop {
        heading(1, "Install a model", note.take().as_deref());
        let installed = installed_ids();
        let mut options = vec!["Install a new model", "Run your own model (already on this machine)"];
        if !installed.is_empty() {
            options.insert(1, "Serve an existing model");
        }
        let Some(i) = choose(&options, 0)? else {
            return Ok(Next::Stop(None));
        };
        let model = match options[i] {
            "Serve an existing model" => {
                heading(1, "Install a model", Some("Pick the model to serve:"));
                pick_installed(&installed)
            }
            "Run your own model (already on this machine)" => run_your_own(),
            _ => install_from_catalog(),
        };
        match model {
            Ok(Some(id)) => return after_install(id),
            Ok(None) => {}
            Err(e) => note = Some(paint(&e, Color::Red)),
        }
    }
}

fn after_install(id: String) -> Result<Next, String> {
    heading(1, "Install a model", Some(&format!("{} is ready to serve.", paint(&id, Color::Bold))));
    match choose(&["Serve it through an API now (Recommended)", "That's it for now"], 0)? {
        Some(0) => Ok(Next::Serve(id)),
        _ => Ok(Next::Stop(Some(id))),
    }
}

fn pick_installed(installed: &[String]) -> Result<Option<String>, String> {
    let options: Vec<&str> = installed.iter().map(String::as_str).collect();
    Ok(choose(&options, 0)?.map(|i| installed[i].clone()))
}

fn install_from_catalog() -> Result<Option<String>, String> {
    // Catalog models that fit the allowed GPUs, then a free Hugging Face id, then a gated or private one with a token.
    let home = goethe_home();
    let hw = crate::placement::load_hw(&home).ok_or("no hardware profile; run goethe hw profile")?;
    let view = crate::recommend::View::capacity(&home, hw);
    let reg = crate::registry::fetch(&crate::registry::registry_url())?;
    let mut fits = crate::recommend::recommend(&view, &reg).recommended;
    fits.sort_by(|a, b| (a.vram_per_gpu_gb * a.tp as f64).total_cmp(&(b.vram_per_gpu_gb * b.tp as f64)));
    let width = fits.iter().map(|f| f.model_id.len()).max().unwrap_or(0);
    let mut rows: Vec<String> = fits
        .iter()
        .map(|f| {
            let download = reg.models.iter().find(|m| m.id == f.model_id).and_then(|m| m.download_gb).unwrap_or(0.0);
            format!("{:<width$}  {:>4.0} GB VRAM  {:>4.0} GB download", f.model_id, f.vram_per_gpu_gb * f.tp as f64, download)
        })
        .collect();
    let other = rows.len() + 1;
    rows.extend([String::new(), "A model not on this list (from Hugging Face)".to_string(), "A private or gated model (needs a Hugging Face token)".to_string()]);
    let mut note: Option<String> = None;
    loop {
        let intro = format!("Models that fit your GPUs. {}", view.free_text());
        heading(1, "Install a model", Some(&note.take().map_or(intro.clone(), |e| format!("{e}\n\n{intro}"))));
        let Some(i) = term::select(&rows, 0)? else {
            return Ok(None);
        };
        let id = match i {
            i if i < fits.len() => fits[i].model_id.clone(),
            i if i == other => match ask_model_id()? {
                Some(id) => id,
                None => continue,
            },
            _ => match ask_token().and_then(|ok| if ok { ask_model_id() } else { Ok(None) })? {
                Some(id) => id,
                None => continue,
            },
        };
        let result = install::run(InstallOpts { model_id: id.clone(), ..Default::default() });
        crate::hf::set_token(None);
        match result {
            Ok(()) => return Ok(Some(id)),
            Err(e) => note = Some(format!("{}\nPick another model, or press q to go back.", paint(&e, Color::Red))),
        }
    }
}

fn ask_model_id() -> Result<Option<String>, String> {
    let id = term::ask(&format!("Hugging Face model id, for example {EXAMPLE_ID} (empty to go back):"), "")?;
    Ok(Some(id).filter(|i| !i.is_empty()))
}

fn ask_token() -> Result<bool, String> {
    // A token checked with Hugging Face and kept for this download only; accept the model's license on its page first.
    println!("Create a read token at https://huggingface.co/settings/tokens, and accept the model's license on its page.");
    println!("Goethe uses the token for this download only and does not save it.\n");
    loop {
        let token = term::ask_secret("Hugging Face token (empty to go back):")?;
        if token.is_empty() {
            return Ok(false);
        }
        match crate::hf::whoami(&token) {
            Ok(name) => {
                println!("Token accepted for {name}.\n");
                crate::hf::set_token(Some(token));
                return Ok(true);
            }
            Err(e) => println!("{}", paint(&e, Color::Red)),
        }
    }
}

fn run_your_own() -> Result<Option<String>, String> {
    // Models Goethe finds on disk (its models folder and the Hugging Face cache), or any folder the user names.
    let home = goethe_home();
    let found = crate::scan::found_on_this_machine(&home);
    let usable: Vec<&crate::scan::Found> = found.iter().filter(|f| f.problem.is_none()).collect();
    let before = installed_ids();
    if !usable.is_empty() {
        heading(1, "Install a model", Some("Models found on this machine:"));
        let place = |f: &crate::scan::Found| if f.in_models { "Goethe's models folder" } else { "Hugging Face cache" };
        let mut rows: Vec<String> = usable.iter().map(|f| format!("{}  ({})", f.id, place(f))).collect();
        rows.extend([String::new(), "Select a custom model directory on this machine".to_string()]);
        let Some(i) = term::select(&rows, 0)? else {
            return Ok(None);
        };
        if i < usable.len() {
            crate::scan::register_found(&home, usable[i])?;
            return Ok(new_model(&before).or_else(|| Some(usable[i].id.clone())));
        }
    }
    heading(1, "Install a model", Some("Select a custom model directory on this machine:"));
    let path = term::ask("Folder with the model's config.json and .safetensors files (empty to go back):", "")?;
    if path.is_empty() {
        return Ok(None);
    }
    install::run(InstallOpts { model_id: path, ..Default::default() })?;
    Ok(new_model(&before))
}

fn step_serve(model: &str) -> Result<(), String> {
    // Load the model on the GPUs that fit it; goethed waits until it answers.
    heading(2, "Serve through an API", None);
    println!("Starting {model} on your GPUs. The first start can take a few minutes while kernels are built.\n");
    let params = serde_json::json!({
        "model_id": model, "gpus": null, "ctx": null, "max_conc": null, "tp": null,
        "force": false, "yes": false, "replica": false, "extra_sglang_args": [], "async_finish": true,
    });
    install::run_load(params, false, false)
}

fn step_access(model: &str) -> Result<(), String> {
    // Who can reach the API, then a key for the first client, then the ready-to-use call with that key filled in.
    heading(3, "Configure API access", Some("Choose who can reach the API:"));
    let api = configure::apply_access(&ApiOpts { mode: None, vpn_ip: None, allow: Vec::new(), yes: false })?;
    let token = choose_key()?;
    print!("\x1b[2J\x1b[H{}", ready_text(&api, model, &token));
    Ok(())
}

fn choose_key() -> Result<String, String> {
    // Use a key that exists (listed by name and id) or create one; without keys, go straight to naming the first.
    let keys = crate::security::key_list_detailed()?;
    let field = |k: &serde_json::Value, f: &str| k.get(f).and_then(|v| v.as_str()).unwrap_or("").to_string();
    if !keys.is_empty() {
        heading(3, "Configure API access", Some("Every user needs an API key. Use one you have, or create a new one:"));
        let mut rows: Vec<String> = keys.iter().map(|k| format!("{}  (id {})", field(k, "name"), field(k, "id"))).collect();
        rows.extend([String::new(), "Create a new key".to_string()]);
        if let Some(i) = term::select(&rows, 0)?.filter(|i| *i < keys.len()) {
            let key = field(&keys[i], "key");
            if !key.is_empty() {
                return Ok(key);
            }
            return Ok(format!("{}... (an older key; Goethe kept only its start)", field(&keys[i], "prefix")));
        }
    }
    println!();
    let prompt = if keys.is_empty() { "Every user needs an API key. Name the first key [dev/my-app-name]:" } else { "Name the new key [dev/my-app-name]:" };
    let name = term::ask(prompt, "dev")?;
    crate::security::key_create(&name, true, None, None, None, false)
}

fn ready_text(api: &str, model: &str, token: &str) -> String {
    let curl = format!(
        "curl -k {api}/chat/completions -H \"Authorization: Bearer {token}\" -H \"Content-Type: application/json\" \
         -d '{{\"model\": \"{model}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'"
    );
    format!(
        "{}\n\n  Base URL   {api}\n  Model      {model}\n  API key    {token}\n\n{}\n\
         Any OpenAI SDK works with the base URL and the key. The certificate is self-signed: skip verification (curl -k)\n\
         or trust $GOETHE_HOME/nginx/certs/server.crt.\n\n{}",
        paint("Your API is ready", Color::Bold),
        term::commands("Try it:", &[&curl]),
        term::commands("See all models and keys:", &["goethe api"])
    )
}

fn advanced() -> Result<(), String> {
    // Each part on its own, in any order, until Finish; an error shows on the next screen.
    let mut note: Option<String> = None;
    loop {
        screen("Advanced setup", note.take().as_deref());
        let options = [
            "GPUs: choose which ones Goethe may use",
            "Install a model",
            "Run your own model (already on this machine)",
            "Serve an installed model",
            "Configure API access",
            "Create an API key",
            "",
            "Finish",
        ];
        let installed = |id: Option<String>| id.map(|i| format!("Installed {i}."));
        let result = match choose(&options, 0)? {
            Some(0) => gpus().map(|()| None),
            Some(1) => install_from_catalog().map(installed),
            Some(2) => run_your_own().map(installed),
            Some(3) => {
                screen("Serve an installed model", None);
                match pick_installed(&installed_ids())? {
                    Some(id) => step_serve(&id).map(|()| Some(format!("{id} is running."))),
                    None => Ok(None),
                }
            }
            Some(4) => {
                screen("Configure API access", Some("Choose who can reach the API:"));
                configure::apply_access(&ApiOpts { mode: None, vpn_ip: None, allow: Vec::new(), yes: false })
                    .map(|api| Some(format!("API endpoint: {api}")))
            }
            Some(5) => create_key().map(Some),
            _ => {
                print!("\n{}", term::commands("Check everything at any time with:", &["goethe status"]));
                return Ok(());
            }
        };
        note = match result {
            Ok(done) => done,
            Err(e) => Some(paint(&e, Color::Red)),
        };
    }
}

fn gpus() -> Result<(), String> {
    // Each GPU with its state; Enter switches it between allowed and denied.
    let home = goethe_home();
    loop {
        let con = crate::db::open(&home).map_err(|e| e.to_string())?;
        let rows = crate::gpu_policy::list_rows(&con, &home)?;
        let mut options: Vec<String> = rows
            .iter()
            .map(|r| {
                let used = (r.managed_mb + r.foreign_mb) as f64 / 1024.0;
                format!("gpu{}  {}  SM {}  {:.1} GB used  {}", r.index, r.name, r.compute_capability, used, r.status)
            })
            .collect();
        options.extend([String::new(), "Back".to_string()]);
        screen("GPUs", Some("Pick a GPU to allow or deny it. Denied GPUs get no new models; running ones stay."));
        match term::select(&options, 0)? {
            Some(i) if i < rows.len() => {
                let deny = rows[i].status == "ALLOWED";
                crate::gpu_policy::set_denied(&con, rows[i].index, deny)?;
            }
            _ => return Ok(()),
        }
    }
}

fn create_key() -> Result<String, String> {
    screen("Create an API key", None);
    let name = term::ask("Every user needs an API key. Name the key [dev/my-app-name]:", "dev")?;
    let token = crate::security::key_create(&name, true, None, None, None, false)?;
    Ok(format!("API key for {name}: {token}\ngoethe api shows it again at any time."))
}

fn installed_ids() -> Vec<String> {
    crate::db::open(&goethe_home())
        .ok()
        .and_then(|con| {
            let mut stmt = con.prepare("SELECT model_id FROM installed_models ORDER BY model_id").ok()?;
            let ids = stmt.query_map([], |r| r.get::<_, String>(0)).ok()?.filter_map(Result::ok).collect();
            Some(ids)
        })
        .unwrap_or_default()
}

fn new_model(before: &[String]) -> Option<String> {
    installed_ids().into_iter().find(|id| !before.contains(id))
}

fn later_text(model: Option<&str>) -> String {
    // How to pick up where the user stopped.
    let load = model.map(|m| format!("goethe load --model {m}"));
    let mut lines = vec!["goethe start"];
    if let Some(l) = load.as_deref() {
        lines.push(l);
    }
    format!("\n{}", term::commands("Continue whenever you like:", &lines))
}

fn no_terminal_text() -> String {
    // The same three steps as commands, for scripts and terminals without arrow keys.
    format!(
        "goethe start needs an interactive terminal. The same steps as commands:\n\n{}\n{}\n{}",
        term::commands("Step 1/3  Install a model:", &["goethe up", "goethe recommend", &format!("goethe install --model {EXAMPLE_ID}")]),
        term::commands("Step 2/3  Serve through an API:", &[&format!("goethe load --model {EXAMPLE_ID}")]),
        term::commands("Step 3/3  Configure API access:", &["goethe configure api", "goethe key --create my-app"])
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_ready_text_fills_in_the_real_key() {
        let text = ready_text("https://10.0.0.2:8443/v1", "org/m", "goethe_sk_abc");
        assert!(text.contains("Base URL   https://10.0.0.2:8443/v1\n  Model      org/m\n  API key    goethe_sk_abc"), "{text}");
        assert!(text.contains("-H \"Authorization: Bearer goethe_sk_abc\""), "{text}");
        assert!(text.contains("\"model\": \"org/m\""), "{text}");
        assert!(!text.contains("<key>"));
    }

    #[test]
    fn without_a_terminal_the_three_steps_are_printed_as_commands() {
        let text = no_terminal_text();
        for step in ["Step 1/3  Install a model", "Step 2/3  Serve through an API", "Step 3/3  Configure API access"] {
            assert!(text.contains(step), "{text}");
        }
        assert!(text.contains(&format!("goethe install --model {EXAMPLE_ID}")));
        assert_eq!(later_text(Some("org/m")).matches("\u{2192}").count(), 2);
    }
}
