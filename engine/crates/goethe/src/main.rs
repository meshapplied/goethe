// goethe CLI entrypoint for hardware, install, config, and load commands.

use clap::{Parser, Subcommand};
use goethe_core::alias;
use goethe_core::config::{self, ConfigOpts};
use goethe_core::db;
use goethe_core::gpu_policy;
use goethe_core::hw;
use goethe_core::install::{self, InstallOpts};
use goethe_core::models;
use goethe_core::nginx;
use goethe_core::paths::{ensure_layout, goethe_home, validate_model_id};
use goethe_core::registry;
use goethe_core::recommend;
use goethe_core::rpc;
use goethe_core::security;
use goethe_core::shutdown;
use serde_json::{json, Value};

const KEY_HELP: &str = "Create a key (goethe api lists it again later):

\t→ goethe key --create my-app

With an expiry (days like 30d, or a date like 2026-12-31), a rate limit and admin rights:

\t→ goethe key --create ops --expires 90d --rpm 300 --admin

Only some models, or just the key for a script:

\t→ goethe key --create bot --models deepseek-ai/DeepSeek-V4.1-Flash-1017
\t→ API_KEY=$(goethe key --create ci --raw)

List keys, or revoke one:

\t→ goethe key
\t→ goethe key --revoke <key-id>";

#[derive(Parser)]
#[command(
    name = "goethe",
    disable_version_flag = true,
    about = "Goethe: serve LLMs on your NVIDIA GPUs behind one OpenAI-compatible gateway",
    after_help = "Examples:\n\n\t→ goethe up\n\t→ goethe install --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --load\n\t→ goethe load --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --gpus 0,1,2,3 --ctx 65536\n\t→ goethe key --create my-app\n\t→ goethe api\n\t→ goethe down\n\n\
                  goethe down stops goethed but keeps loaded models for the next goethe up; goethe unload stops a model for good."
)]
struct Cli {
    #[arg(short = 'V', long = "version", help = "Print version, git commit and build date.")]
    version: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(about = "GPU detection and placement helpers.")]
    Hw {
        #[command(subcommand)]
        cmd: HwCommands,
    },
    #[command(about = "List GPUs, or allow or deny placement on one.")]
    Gpu {
        #[arg(long, num_args = 1.., value_name = "gpuN", value_parser = parse_gpu_name, conflicts_with = "deny", help = "Allow new models on these GPUs again, for example gpu0 gpu1.")]
        allow: Vec<u32>,
        #[arg(long, num_args = 1.., value_name = "gpuN", value_parser = parse_gpu_name, help = "Keep new models off these GPUs (running ones stay), for example gpu0 gpu1.")]
        deny: Vec<u32>,
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
    #[command(about = "List catalog models that fit the free VRAM of the allowed GPUs.", alias = "recommended")]
    Recommend {
        #[arg(long, help = "Emit JSON instead of a table.")]
        json: bool,
    },
    #[command(about = "Download, verify, and register a model (from Hugging Face or a local path). With --load, waits for RUNNING like goethe load unless --no-wait.")]
    Install {
        #[arg(long = "model", value_name = "ID_OR_PATH", help = "Hugging Face repo id or local directory path.")]
        model_id: String,
        #[arg(long, help = "Default context length for the model card.")]
        ctx: Option<u64>,
        #[arg(long = "max-conc", help = "Default max concurrent requests.")]
        max_conc: Option<u64>,
        #[arg(long, help = "Tensor parallel size (0 = auto).")]
        tp: Option<u64>,
        #[arg(long, help = "Comma-separated GPU indices.")]
        gpus: Option<String>,
        #[arg(long, group = "load_mode", help = "Load the model after install.")]
        load: bool,
        #[arg(long, group = "load_mode", help = "Install only; do not load.")]
        no_load: bool,
        #[arg(long, help = "With --load, return after spawn without waiting for RUNNING.")]
        no_wait: bool,
        #[arg(long, help = "Re-download (Hub) or replace an existing local install of another kind.")]
        force: bool,
        #[arg(long, help = "Rebuild the model card from defaults instead of merging into the existing one.")]
        reset: bool,
        #[arg(long = "trust-remote-code", help = "Let SGLang run Python code shipped with the model (config.json auto_map).")]
        trust_remote_code: bool,
        #[arg(long, help = "With --load, load next to processes goethe did not start without asking.")]
        yes: bool,
        #[arg(long, help = "Skip sha256 verification.")]
        no_verify: bool,
        #[arg(long = "hf-token", help = "Ask for a Hugging Face token (gated or private models); it is used for this download only.")]
        hf_token: bool,
        #[arg(long, help = "Registered model id for local installs.")]
        name: Option<String>,
        #[arg(long, help = "Copy local tree instead of symlinking.")]
        copy: bool,
        #[arg(long = "sglang-arg", allow_hyphen_values = true, help = "Extra SGLang CLI flag.")]
        sglang_arg: Vec<String>,
    },
    #[command(about = "Register model folders you downloaded yourself and moved into GOETHE_HOME/models.")]
    Scan {
        #[arg(long, value_name = "FOLDER", help = "One folder inside GOETHE_HOME/models (default: all of them).")]
        dir: Option<String>,
        #[arg(long, help = "Add every supported model without asking.")]
        yes: bool,
        #[arg(long, help = "Emit JSON rows (dir, model_id, status, reason, notes).")]
        json: bool,
    },
    #[command(about = "Validate, generate, or merge registry entries.")]
    Registry {
        #[command(subcommand)]
        cmd: RegistryCommands,
    },
    #[command(about = "List installed and registry-available models.", visible_alias = "list")]
    Models {
        #[arg(long, help = "Emit JSON.")]
        json: bool,
        #[arg(long, help = "Only installed models.")]
        installed: bool,
        #[arg(long, help = "Only registry models.")]
        available: bool,
    },
    #[command(about = "Remove an installed model: unload it, delete files goethe downloaded, or unregister a local folder.")]
    Remove {
        #[arg(long = "model", required = true, num_args = 1.., value_name = "ID", help = "Installed model ids to remove.")]
        model_id: Vec<String>,
        #[arg(long, help = "Do not ask; also unloads the model if it is loaded.")]
        yes: bool,
        #[arg(long, help = "When unloading first, skip draining in-flight requests.")]
        force: bool,
    },
    #[command(about = "Change default load settings stored on the model card.")]
    Config {
        #[arg(long = "model", value_name = "ID", help = "Installed model id to update.")]
        model_id: String,
        #[arg(long, help = "Default context length.")]
        ctx: Option<u64>,
        #[arg(long = "max-conc", help = "Default max concurrent requests.")]
        max_conc: Option<u64>,
        #[arg(long, help = "Default tensor parallel size.")]
        tp: Option<u64>,
        #[arg(long, help = "Preferred GPU indices.")]
        gpus: Option<String>,
        #[arg(long = "sglang-arg", allow_hyphen_values = true, help = "Append SGLang flags.")]
        sglang_arg: Vec<String>,
        #[arg(long, help = "Clear stored extra SGLang args.")]
        clear_sglang_args: bool,
    },
    #[command(
        about = "Load a model onto GPUs and wait until this instance is ready.",
        after_help = "Extra SGLang flags go after --, for example:\n\n\t→ goethe load --model deepseek-ai/DeepSeek-V4.1-Flash-1017 --ctx 32768 -- --chunked-prefill-size 4096"
    )]
    Load {
        #[arg(long = "model", value_name = "ID", help = "Model id to load.")]
        model_id: String,
        #[arg(long, help = "Comma-separated GPU indices.")]
        gpus: Option<String>,
        #[arg(long, help = "Context length for this load.")]
        ctx: Option<u64>,
        #[arg(long = "max-conc", help = "Max concurrent requests.")]
        max_conc: Option<u64>,
        #[arg(long, help = "Tensor parallel size.")]
        tp: Option<u64>,
        #[arg(long, help = "Expert parallel size (SGLang --ep-size).")]
        ep: Option<u64>,
        #[arg(long, help = "Load on busy GPUs without prompting.")]
        force: bool,
        #[arg(long, help = "Confirm foreign GPU use without prompting.")]
        yes: bool,
        #[arg(long, help = "Start another instance even when the model is already loaded.")]
        replica: bool,
        #[arg(long, help = "Return after spawn without waiting for RUNNING.")]
        no_wait: bool,
        #[arg(long, help = "Print the instance as JSON instead of status lines.")]
        json: bool,
        #[arg(last = true, help = "Extra SGLang flags, only after --.")]
        extra: Vec<String>,
    },
    #[command(about = "Stop instances and clear desired load rows for a model.")]
    Unload {
        #[arg(long = "model", num_args = 1.., value_name = "ID", conflicts_with = "all", help = "Model ids to unload (all their instances).")]
        model_id: Vec<String>,
        #[arg(long, help = "Unload every loaded model.")]
        all: bool,
        #[arg(long, help = "Skip connection draining and tear down immediately.")]
        force: bool,
    },
    #[command(about = "Show running engine instances and the public gateway URL.")]
    Ps {
        #[arg(long, help = "JSON rows with instance_id, slot_key, status, restarts.")]
        json: bool,
    },
    #[command(about = "Start goethed and the gateway, and start them at boot (loaded models from last time come back).")]
    Up {
        #[arg(long = "no-boot", help = "Start now without turning on start at boot.")]
        no_boot: bool,
    },
    #[command(about = "Show whether goethed is running, the API address and the API keys.")]
    Status,
    #[command(about = "Stop goethed and its engines and turn off start at boot (loaded models come back on the next goethe up).")]
    Down {
        #[arg(long = "keep-engines", help = "Leave loaded engines serving; the next goethed adopts them.")]
        keep_engines: bool,
    },
    #[command(about = "Choose who can reach the API.")]
    Configure {
        #[command(subcommand)]
        cmd: ConfigureCommands,
    },
    #[command(about = "Guided setup: install a model, serve it through the API and set up access.")]
    Start,
    #[command(about = "The API address, its keys and the models it serves.")]
    Api {
        #[arg(long, help = "JSON with api, goethed_running, keys and models.")]
        json: bool,
    },
    #[command(hide = true, about = "Print connection details (same as goethe api).")]
    Get {
        #[command(subcommand)]
        cmd: GetCommands,
    },
    #[command(about = "Enable or inspect autostart for goethed.")]
    Service {
        #[command(subcommand)]
        cmd: ServiceCommands,
    },
    #[command(about = "Run environment and connectivity checks with fix hints.")]
    Doctor {
        #[arg(long, help = "Emit JSON.")]
        json: bool,
        #[arg(long = "probe-public", help = "Probe public gateway reachability.")]
        probe_public: bool,
    },
    #[command(about = "Show where the public gateway listens, or change it.")]
    Gateway {
        #[arg(long, value_name = "ADDRESS", help = "Listen on 127.0.0.1 (this host only), 0.0.0.0 (all addresses) or one IP of this host.")]
        bind: Option<String>,
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
    #[command(about = "How to reach the API by a domain name, and a check that it works.")]
    Domain {
        #[arg(long, value_name = "DOMAIN", help = "Check that the domain points at the API and the gateway answers there.")]
        check: Option<String>,
    },
    #[command(about = "Tail engine, goethed, or gateway logs.")]
    Logs {
        #[arg(long = "model", value_name = "ID", help = "Model id or instance id whose engine log to show.")]
        target: Option<String>,
        #[arg(short = 'f', long, help = "Follow log output.")]
        follow: bool,
        #[arg(short = 'n', long, default_value = "100", help = "Number of lines to print.")]
        lines: usize,
        #[arg(long, help = "Show goethed.log.")]
        daemon: bool,
        #[arg(long, help = "Show gateway log.")]
        gateway: bool,
        #[arg(long, help = "Rotate and prune logs, then exit.")]
        prune: bool,
    },
    #[command(about = "Download and apply a signed goethe or engine update.")]
    Update {
        #[arg(long, help = "Print available versions only.")]
        check: bool,
        #[arg(long, help = "Update SGLang engine packages.")]
        engine: bool,
        #[arg(long, help = "Update ternary runtime packages.")]
        ternary: bool,
        #[arg(long, help = "Skip confirmation.")]
        yes: bool,
        #[arg(long = "allow-downgrade", help = "Install the published version even if it is older.")]
        allow_downgrade: bool,
    },
    #[command(about = "Remove goethe and the models it installed (folders of your own files stay).")]
    Uninstall {
        #[arg(long, help = "Keep the installed models in GOETHE_HOME/models.")]
        keep_models: bool,
        #[arg(long, hide = true, help = "Deleting the installed models is the default now.")]
        purge_models: bool,
        #[arg(long, help = "Skip confirmation.")]
        yes: bool,
    },
    #[command(about = "List the client addresses allowed to reach the gateway, or add or remove one.")]
    Allowlist {
        #[arg(long, num_args = 1.., value_name = "CIDR", conflicts_with = "remove", help = "Allow these client addresses or ranges, for example 203.0.113.7/32.")]
        add: Vec<String>,
        #[arg(long, value_name = "TEXT", requires = "add", help = "Note stored with the new entries.")]
        description: Option<String>,
        #[arg(long, num_args = 1.., value_name = "CIDR", help = "Remove these addresses or ranges.")]
        remove: Vec<String>,
    },
    #[command(about = "List API keys, or create or revoke one.", after_help = KEY_HELP)]
    Key {
        #[arg(long, value_name = "NAME", conflicts_with = "revoke", help = "Create a key with this name.")]
        create: Option<String>,
        #[arg(long, value_name = "WHEN", requires = "create", help = "Expiry: days from now like 30d, or a date like 2026-12-31.")]
        expires: Option<String>,
        #[arg(long, value_name = "N", requires = "create", help = "Requests per minute (default 120, 0 = unlimited).")]
        rpm: Option<i64>,
        #[arg(long, value_name = "IDS", requires = "create", help = "Only these models, comma-separated.")]
        models: Option<String>,
        #[arg(long, requires = "create", help = "Admin key: health endpoints and every model.")]
        admin: bool,
        #[arg(long, requires = "create", help = "Print only the key, for scripts.")]
        raw: bool,
        #[arg(long, num_args = 1.., value_name = "KEY_ID", help = "Revoke the keys with these ids (from goethe key).")]
        revoke: Vec<String>,
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
    #[command(about = "List model aliases, or set or remove one.")]
    Alias {
        #[arg(long, value_name = "ALIAS", conflicts_with = "remove", requires = "model", help = "Alias name clients may use as the model field.")]
        set: Option<String>,
        #[arg(long, value_name = "ID", requires = "set", help = "Installed model id the alias points to.")]
        model: Option<String>,
        #[arg(long, num_args = 1.., value_name = "ALIAS", help = "Remove these aliases.")]
        remove: Vec<String>,
    },
}

#[derive(Subcommand)]
enum ConfigureCommands {
    #[command(about = "Choose who can reach the API: local only, VPN (recommended) or public network.")]
    Api {
        #[arg(long, help = "local, vpn or public (asked on a terminal when omitted).")]
        mode: Option<String>,
        #[arg(long = "vpn-ip", help = "VPN address of this machine to listen on (default: the detected one).")]
        vpn_ip: Option<std::net::IpAddr>,
        #[arg(long = "allow", help = "Client CIDR to allow (repeatable; default for vpn: the VPN's range).")]
        allow: Vec<String>,
        #[arg(long, help = "Apply without questions (public mode still refuses 0.0.0.0/0).")]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum GetCommands {
    #[command(about = "The OpenAI-compatible API base URL, then the loaded models.")]
    Api {
        #[arg(long, help = "JSON with api, goethed_running and models.")]
        json: bool,
    },
}

#[derive(Subcommand)]
enum RegistryCommands {
    #[command(about = "Check a registry JSON file (local path or URL).")]
    Validate {
        #[arg(long, value_name = "PATH_OR_URL", help = "Registry JSON to check.")]
        source: String,
        #[arg(long, help = "Fetch remote sha256 metadata.")]
        online: bool,
        #[arg(long, help = "Allow non-org Hugging Face repos.")]
        allow_individuals: bool,
        #[arg(long, help = "Validate SGLang flag names.")]
        check_sglang: bool,
    },
    #[command(about = "Generate registry stub entries for repo ids.")]
    Entry {
        #[arg(long = "repo", value_name = "ID", help = "Hugging Face repo id (repeatable).")]
        repo_ids: Vec<String>,
        #[arg(long, help = "Runtime: sglang or ternary.")]
        runtime: Option<String>,
        #[arg(long, help = "Git revision or branch.")]
        revision: Option<String>,
        #[arg(long, help = "Read repo ids from a file.")]
        ids_file: Option<String>,
        #[arg(long, help = "Fail on validation warnings.")]
        strict: bool,
    },
    #[command(about = "Register models from a registry JSON file.")]
    Add {
        #[arg(long = "repo", value_name = "ID", help = "Hugging Face repo id to add from the file (repeatable).")]
        repo_ids: Vec<String>,
        #[arg(long, help = "Read repo ids from a file.")]
        ids_file: Option<String>,
        #[arg(long, help = "Path or URL to the registry JSON source.")]
        file: String,
        #[arg(long, help = "Optional overlay JSON path.")]
        overlay: Option<String>,
        #[arg(long, help = "Update existing entries.")]
        update: bool,
        #[arg(long, help = "Print merge result without writing.")]
        dry_run: bool,
        #[arg(long, help = "Fail on validation warnings.")]
        strict: bool,
    },
}

#[derive(Subcommand)]
enum ServiceCommands {
    #[command(about = "Install autostart unit or cron for goethed.")]
    Enable,
    #[command(about = "Remove goethed autostart.")]
    Disable,
    #[command(about = "Show whether autostart is enabled.")]
    Status,
    #[command(about = "Toggle restore of desired loads at goethed start.")]
    Restore {
        #[arg(long, conflicts_with = "on", help = "Disable restore on start.")]
        off: bool,
        #[arg(long, help = "Enable restore on start.")]
        on: bool,
    },
}

#[derive(Subcommand)]
enum HwCommands {
    #[command(about = "GPUs and links, CPU, RAM, disk, OS, container and cloud; writes hardware_state.json.")]
    Profile {
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
    #[command(about = "Show where goethe load would place a model now (same chooser, foreign processes included).")]
    PickGpus {
        #[arg(long, help = "Tensor parallel size (default: the model's, else 1).")]
        tp: Option<u64>,
        #[arg(long, help = "Installed model id whose VRAM budget to place.")]
        model: Option<String>,
    },
}

fn usage(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

fn installed_id(id: &str, strict: bool) -> String {
    // The installed model id the user meant (any case or its folder name); strict callers stop on an unknown id.
    match models::resolve_installed_id(&goethe_home(), checked_id(id)) {
        Ok(found) => {
            if found != id {
                eprintln!("using installed model {found}");
            }
            found
        }
        Err(e) if strict => {
            eprintln!("{e}");
            std::process::exit(1);
        }
        Err(_) => id.to_string(),
    }
}

fn corrected_command(args: &[String], stray: &str) -> Option<String> {
    // The command with the flag a stray value belongs to: --model for model commands, --verb for old "verb value" forms.
    let at = args.iter().position(|a| a == stray).filter(|&i| i >= 2 && !stray.starts_with('-'))?;
    let flag = match (args[1].as_str(), stray) {
        ("install" | "load" | "unload" | "remove" | "config" | "logs", _) => "--model".to_string(),
        ("scan", _) => "--dir".to_string(),
        ("domain", "check") | ("gateway", "bind") => format!("--{stray}"),
        ("domain", _) => "--check".to_string(),
        ("gateway", _) => "--bind".to_string(),
        ("gpu", "allow" | "deny") | ("key", "create" | "revoke") | ("allowlist" | "alias", "add" | "remove" | "set") => {
            format!("--{stray}")
        }
        _ => return None,
    };
    let mut out: Vec<String> = args[1..at].to_vec();
    if flag.trim_start_matches('-') == stray {
        out.push(flag);
    } else {
        out.extend([flag, stray.to_string()]);
    }
    out.extend(args[at + 1..].iter().cloned());
    Some(format!("goethe {}", out.join(" ")))
}

fn each<T>(items: Vec<T>, act: impl Fn(T) -> Result<(), String>) -> Result<(), String> {
    // Apply one action to every value given; all are tried, and the errors are reported together.
    let errors: Vec<String> = items.into_iter().filter_map(|item| act(item).err()).collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n"))
    }
}

fn checked_id(id: &str) -> &str {
    if let Err(e) = validate_model_id(id) {
        usage(e);
    }
    id
}

fn parse_gpu_name(raw: &str) -> Result<u32, String> {
    // "gpu2", as goethe gpu lists it; a bare index is accepted too.
    let t = raw.trim().to_ascii_lowercase();
    t.strip_prefix("gpu").unwrap_or(&t).parse().map_err(|_| format!("{raw} is not a GPU name like gpu2 (goethe gpu lists them)"))
}

fn parse_gpus(raw: &str) -> Result<Vec<u32>, String> {
    raw.split(',')
        .map(|s| s.trim().parse::<u32>().map_err(|_| format!("invalid gpu index: {s}")))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && (args[1] == "--version" || args[1] == "-V") {
        println!("{}", goethe_core::build_info::version_line());
        return;
    }
    let cli = Cli::try_parse().unwrap_or_else(|e| {
        let stray = e.get(clap::error::ContextKind::InvalidArg).map(|v| v.to_string());
        if let Some(fixed) = stray.and_then(|v| corrected_command(&args, &v)) {
            eprintln!("error: values go after a flag. Did you mean:\n\n{}\n", goethe_core::term::command(&fixed));
            std::process::exit(2);
        }
        e.exit()
    });
    if cli.version {
        println!("{}", goethe_core::build_info::version_line());
        return;
    }
    let result = match cli.command {
        Commands::Hw {
            cmd: HwCommands::Profile { json },
        } => hw::run_profile(json),
        Commands::Hw {
            cmd: HwCommands::PickGpus { tp, model },
        } => hw::run_pick_gpus(tp, model.as_deref().map(checked_id)),
        Commands::Gpu { allow, deny, json } => match (allow.is_empty(), deny.is_empty()) {
            (false, _) => each(allow, do_gpu_allow),
            (_, false) => each(deny, do_gpu_deny),
            _ => do_gpu_list(json),
        },
        Commands::Recommend { json } => recommend::run(json),
        Commands::Install {
            model_id,
            ctx,
            max_conc,
            tp,
            gpus,
            load,
            no_load: _,
            no_wait,
            force,
            reset,
            trust_remote_code,
            yes,
            no_verify,
            hf_token,
            name,
            copy,
            sglang_arg,
        } => {
            if hf_token {
                let token = goethe_core::term::ask_secret("Hugging Face token:").unwrap_or_else(|e| usage(e));
                let account = goethe_core::hf::whoami(&token).unwrap_or_else(|e| usage(e));
                eprintln!("token accepted for {account}");
                goethe_core::hf::set_token(Some(token));
            }
            let gpus_parsed = match gpus.as_deref().map(parse_gpus) {
                Some(Ok(v)) => Some(v),
                Some(Err(e)) => usage(e),
                None => None,
            };
            if !std::path::Path::new(&model_id).is_dir() && !model_id.starts_with(['.', '/', '~']) {
                checked_id(&model_id);
            }
            if let Some(n) = name.as_deref() {
                checked_id(n);
            }
            if let Some(msg) = goethe_core::sglang_infer::reserved_flag_error(&sglang_arg) {
                usage(msg);
            }
            install::run(InstallOpts {
                model_id,
                ctx,
                max_conc,
                tp,
                gpus: gpus_parsed,
                load,
                force,
                no_verify,
                name,
                copy_local: copy,
                sglang_args: sglang_arg,
                no_wait,
                trust_remote_code,
                yes,
                reset,
            })
        }
        Commands::Scan { dir, yes, json } => goethe_core::scan::run(goethe_core::scan::ScanOpts { yes, json, dir }),
        Commands::Registry { cmd } => match cmd {
            RegistryCommands::Validate {
                source,
                online,
                allow_individuals,
                check_sglang,
            } => registry::run_validate(
                &source,
                &registry::ValidateOpts {
                    online,
                    allow_individuals,
                    check_sglang,
                },
            ),
            RegistryCommands::Entry {
                repo_ids,
                runtime,
                revision,
                ids_file,
                strict,
            } => do_registry_entry(repo_ids, runtime, revision, ids_file, strict),
            RegistryCommands::Add {
                repo_ids,
                ids_file,
                file,
                overlay,
                update,
                dry_run,
                strict,
            } => do_registry_add(repo_ids, ids_file, file, overlay, update, dry_run, strict),
        },
        Commands::Models {
            json,
            installed,
            available,
        } => models::run_list(json, installed, available),
        Commands::Remove {
            model_id,
            yes,
            force,
        } => each(model_id, |id| models::run_remove(&installed_id(&id, false), yes, force)),
        Commands::Config {
            model_id,
            ctx,
            max_conc,
            tp,
            gpus,
            sglang_arg,
            clear_sglang_args,
        } => {
            let gpus_parsed = match gpus.as_deref().map(parse_gpus) {
                Some(Ok(v)) => Some(v),
                Some(Err(e)) => usage(e),
                None => None,
            };
            if let Some(msg) = goethe_core::sglang_infer::reserved_flag_error(&sglang_arg) {
                usage(msg);
            }
            let model_id = installed_id(&model_id, true);
            if let Err(e) = config::validate_limits(&goethe_home(), &model_id, ctx, max_conc) {
                usage(e);
            }
            config::run(ConfigOpts {
                model_id,
                ctx,
                max_conc,
                tp,
                gpus: gpus_parsed,
                sglang_args: sglang_arg,
                clear_sglang_args,
            })
        }
        Commands::Load {
            model_id,
            gpus,
            ctx,
            max_conc,
            tp,
            ep,
            force,
            yes,
            replica,
            no_wait,
            json,
            extra,
        } => do_load(DoLoadArgs {
            model_id: installed_id(&model_id, true),
            gpus,
            ctx,
            max_conc,
            tp,
            ep,
            force,
            yes,
            replica,
            extra,
            no_wait,
            json,
        }),
        Commands::Unload { model_id, all, force } => match (model_id.is_empty(), all) {
            (true, false) => usage("give --model <id> or --all"),
            (true, true) => do_unload(None, force),
            _ => each(model_id, |id| do_unload(Some(&installed_id(&id, false)), force)),
        },
        Commands::Ps { json } => do_ps(json),
        Commands::Up { no_boot } => do_up(!no_boot),
        Commands::Status => do_status(),
        Commands::Get { cmd: GetCommands::Api { json } } | Commands::Api { json } => do_api(json),
        Commands::Start => goethe_core::onboard::run(),
        Commands::Configure { cmd: ConfigureCommands::Api { mode, vpn_ip, allow, yes } } => {
            let mode = mode.as_deref().map(goethe_core::configure::Mode::parse).transpose().unwrap_or_else(|e| usage(e));
            goethe_core::configure::run_api(goethe_core::configure::ApiOpts { mode, vpn_ip, allow, yes })
        }
        Commands::Down { keep_engines: false } => shutdown::run_down(),
        Commands::Down { keep_engines: true } => goethe_core::service::stop_keeping_engines().map(|(before, kept)| {
            if before > 0 {
                println!("{}", goethe_core::service::kept_text(before, kept));
            }
        }),
        Commands::Service { cmd } => match cmd {
            ServiceCommands::Enable => goethe_core::service::run_enable(),
            ServiceCommands::Disable => goethe_core::service::run_disable(),
            ServiceCommands::Status => goethe_core::service::run_status(),
            ServiceCommands::Restore { off, on } => {
                if off {
                    goethe_core::autostart::set_restore_enabled(false)
                } else if on {
                    goethe_core::autostart::set_restore_enabled(true)
                } else {
                    usage("goethe service restore needs --on or --off")
                }
            }
        },
        Commands::Doctor { json, probe_public } => goethe_core::doctor::run(json, probe_public),
        Commands::Gateway { bind, json } => match bind {
            Some(addr) => do_gateway_bind(&addr),
            None => do_gateway_status(json),
        },
        Commands::Domain { check } => match check {
            Some(domain) => goethe_core::domain::check(&domain).map(|text| print!("{text}")),
            None => goethe_core::domain::explain().map(|text| print!("{text}")),
        },
        Commands::Logs {
            target,
            follow,
            lines,
            daemon,
            gateway,
            prune,
        } => {
            if let Some(msg) = logs_target_error(target.as_deref(), daemon, gateway, prune) {
                usage(msg);
            }
            goethe_core::logs::run_logs(target.as_deref(), follow, lines, daemon, gateway, prune)
        }
        Commands::Update {
            check,
            engine,
            ternary,
            yes,
            allow_downgrade,
        } => goethe_core::update::run(goethe_core::update::UpdateOpts {
            check_only: check,
            engine,
            ternary,
            yes,
            allow_downgrade,
        }),
        Commands::Uninstall {
            keep_models,
            purge_models: _,
            yes,
        } => goethe_core::uninstall::run(goethe_core::uninstall::UninstallOpts { keep_models, yes }),
        Commands::Allowlist { add, description, remove } => match (add.is_empty(), remove.is_empty()) {
            (false, _) => each(add, |cidr| {
                security::allowlist_add(&cidr, description.as_deref())
                    .map(|()| println!("Allowed {}", security::canonical_cidr(&cidr).unwrap_or(cidr)))
            }),
            (_, false) => each(remove, |cidr| {
                security::allowlist_remove(&cidr).map(|()| println!("Removed {}", security::canonical_cidr(&cidr).unwrap_or(cidr)))
            }),
            _ => security::allowlist_list().map(|rows| print!("{}", allowlist_text(&rows))),
        },
        Commands::Key { create: Some(name), expires, rpm, models, admin, raw, .. } => {
            if name.trim().is_empty() {
                usage("--create needs a key name");
            }
            if let Some(Err(e)) = models.as_deref().map(security::model_scope_list) {
                usage(e);
            }
            security::key_create(&name, true, rpm, expires.as_deref(), models.as_deref(), admin)
                .map(|token| print!("{}", key_create_stdout(&token, &key_id_for_token(&token), &name, raw)))
        }
        Commands::Key { revoke, .. } if !revoke.is_empty() => {
            each(revoke, |id| security::key_revoke(&id).map(|()| println!("Revoked key {id}")))
        }
        Commands::Key { json, .. } => security::key_list_detailed().map(|rows| {
            if json {
                println!("{}", serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into()));
            } else {
                print!("{}", key_list_text(&rows));
            }
        }),
        Commands::Alias { set: Some(a), model: Some(model), .. } => {
            alias::set(&a, &model).map(|()| println!("Alias {a} now points to {model}"))
        }
        Commands::Alias { remove, .. } if !remove.is_empty() => {
            each(remove, |a| alias::remove(&a).map(|()| println!("Removed alias {a}")))
        }
        Commands::Alias { .. } => alias::list().map(|rows| print!("{}", alias_text(&rows))),
    };
    match result {
        Ok(()) => {}
        Err(msg) => fail(msg),
    }
}

struct DoLoadArgs {
    model_id: String,
    gpus: Option<String>,
    ctx: Option<u64>,
    max_conc: Option<u64>,
    tp: Option<u64>,
    ep: Option<u64>,
    force: bool,
    yes: bool,
    replica: bool,
    extra: Vec<String>,
    no_wait: bool,
    json: bool,
}

fn do_load(args: DoLoadArgs) -> Result<(), String> {
    // Validate locally, warn about host RAM for offloading cards, then load with the foreign-process prompt.
    checked_id(&args.model_id);
    if matches!(args.ep, Some(0)) {
        usage("--ep must be at least 1");
    }
    if let Err(e) = config::validate_limits(&goethe_home(), &args.model_id, args.ctx, args.max_conc) {
        usage(e);
    }
    if let (Some(ep), Some(tp)) = (args.ep, args.tp) {
        if ep > 0 && tp % ep != 0 {
            usage(format!("--tp {tp} is not divisible by --ep {ep}"));
        }
    }
    if let Some(msg) = goethe_core::sglang_infer::reserved_flag_error(&args.extra) {
        usage(msg);
    }
    if let Some(warning) = models::host_ram_warning(&goethe_home(), &args.model_id) {
        eprintln!("{warning}");
    }
    let mut extra = args.extra;
    if let Some(ep) = args.ep {
        if ep > 1 {
            extra.push("--ep-size".to_string());
            extra.push(ep.to_string());
        }
    }
    let params = json!({
        "model_id": args.model_id,
        "gpus": args.gpus,
        "ctx": args.ctx,
        "max_conc": args.max_conc,
        "tp": args.tp,
        "force": args.force,
        "yes": args.yes,
        "replica": args.replica,
        "extra_sglang_args": extra,
        "async_finish": true,
    });
    install::run_load(params, args.no_wait, args.json)
}

fn key_id_for_token(token: &str) -> String {
    // The stored key id of a token just created (ids are random, so look it up by the token hash).
    db::open(&goethe_home())
        .ok()
        .and_then(|con| {
            con.query_row("SELECT key_id FROM api_keys WHERE token_hash = ?1", [security::token_hash(token)], |r| {
                r.get::<_, String>(0)
            })
            .ok()
        })
        .unwrap_or_else(|| "?".to_string())
}

fn rpm_text(rpm: i64) -> String {
    if rpm <= 0 {
        "unlimited".to_string()
    } else {
        rpm.to_string()
    }
}

fn key_create_stdout(token: &str, id: &str, name: &str, raw: bool) -> String {
    // --raw prints exactly the token; otherwise the token once, its name and id, and a store-it-now note.
    if raw {
        return format!("{token}\n");
    }
    format!(
        "API key for {name} (id {id}):\n\n\t{token}\n\ngoethe api shows it again at any time.\n"
    )
}

fn create_first_key() -> String {
    goethe_core::term::commands("Create your first API key:", &["goethe key --create my-app"])
}

fn key_list_text(rows: &[Value]) -> String {
    // One row per key with its limits and today's use, or how to create the first one.
    if rows.is_empty() {
        return create_first_key();
    }
    let field = |r: &Value, k: &str| r.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let mut table = goethe_core::term::Table::new(&["NAME", "KEY ID", "ADMIN", "RPM", "EXPIRES", "MODELS", "TODAY"]);
    for r in rows {
        let admin = r.get("admin").and_then(Value::as_bool).unwrap_or(false);
        let today = format!(
            "{} req, {} tok",
            r.get("requests_today").and_then(Value::as_i64).unwrap_or(0),
            r.get("tokens_today").and_then(Value::as_i64).unwrap_or(0)
        );
        let models = field(r, "models");
        table.row(vec![
            field(r, "name"),
            field(r, "id"),
            if admin { "yes" } else { "no" }.to_string(),
            rpm_text(r.get("rpm").and_then(Value::as_i64).unwrap_or(0)),
            Some(field(r, "expires")).filter(|e| !e.is_empty()).unwrap_or_else(|| "never".to_string()),
            if models.is_empty() { "all".to_string() } else { models },
            today,
        ]);
    }
    table.render()
}

fn api_keys_text(rows: &[Value]) -> String {
    // Each key in full with its name, admin and expiry; keys made before keys were stored show their first characters.
    if rows.is_empty() {
        return create_first_key();
    }
    let field = |r: &Value, k: &str| r.get(k).and_then(Value::as_str).unwrap_or("").to_string();
    let mut table = goethe_core::term::Table::new(&["NAME", "API KEY", "ADMIN", "EXPIRES"]);
    for r in rows {
        let admin = r.get("admin").and_then(Value::as_bool).unwrap_or(false);
        let key = match (field(r, "key"), field(r, "prefix")) {
            (k, _) if !k.is_empty() => k,
            (_, p) if !p.is_empty() => format!("{p}... (older key, only its start is kept)"),
            _ => "-".to_string(),
        };
        table.row(vec![
            field(r, "name"),
            key,
            if admin { "yes" } else { "no" }.to_string(),
            Some(field(r, "expires")).filter(|e| !e.is_empty()).unwrap_or_else(|| "never".to_string()),
        ]);
    }
    table.render()
}

fn allowlist_text(rows: &[String]) -> String {
    // The allowed client ranges, or why an empty list refuses everyone and how to add a client.
    if rows.is_empty() {
        return goethe_core::term::commands(
            "No client may reach the gateway yet. Allow one with:",
            &["goethe allowlist --add <client-ip>/32"],
        );
    }
    let mut table = goethe_core::term::Table::new(&["ALLOWED CLIENTS"]);
    for cidr in rows {
        table.row(vec![cidr.clone()]);
    }
    table.render()
}

fn alias_text(rows: &[(String, String)]) -> String {
    if rows.is_empty() {
        return goethe_core::term::commands("No aliases yet. Set one with:", &["goethe alias --set <alias> --model <model-id>"]);
    }
    let mut table = goethe_core::term::Table::new(&["ALIAS", "MODEL"]);
    for (a, m) in rows {
        table.row(vec![a.clone(), m.clone()]);
    }
    table.render()
}

fn rpc_daemon_err(e: String) -> String {
    goethe_core::paths::daemon_err(e)
}

fn do_gpu_list(json: bool) -> Result<(), String> {
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    let con = db::open(&home).map_err(|e| e.to_string())?;
    let rows = gpu_policy::list_rows(&con, &home)?;
    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?
        );
        return Ok(());
    }
    let mut table = goethe_core::term::Table::new(&["GPU", "NAME", "SM", "VRAM TOTAL", "GOETHE", "FOREIGN", "STATUS"]);
    let gb = |mb: u64| format!("{:.1} GB", mb as f64 / 1024.0);
    for r in &rows {
        table.row(vec![
            format!("gpu{}", r.index),
            r.name.clone(),
            r.compute_capability.clone(),
            gb(r.vram_total_mb),
            gb(r.managed_mb),
            gb(r.foreign_mb),
            goethe_core::term::status(&r.status),
        ]);
    }
    table.print();
    let notes: Vec<String> = rows
        .iter()
        .filter_map(|r| goethe_core::gpu_arch::unsupported_reason(&r.compute_capability).map(|why| format!("gpu{}: {why}", r.index)))
        .collect();
    if !notes.is_empty() {
        println!("\nGoethe will not place models on these GPUs:\n");
        notes.iter().for_each(|n| println!("\t{n}"));
    }
    Ok(())
}

fn do_gpu_deny(id: u32) -> Result<(), String> {
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    gpu_policy::validate_gpu_index(&home, id)?;
    let con = db::open(&home).map_err(|e| e.to_string())?;
    gpu_policy::set_denied(&con, id, true)?;
    println!("gpu{id} denied: new models will not be placed on it");
    let running = gpu_policy::instances_on_gpu(&con, id)?;
    if !running.is_empty() {
        println!(
            "{} instance(s) keep running on gpu{id}",
            running.len()
        );
    }
    Ok(())
}

fn do_gpu_allow(id: u32) -> Result<(), String> {
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    gpu_policy::validate_gpu_index(&home, id)?;
    let con = db::open(&home).map_err(|e| e.to_string())?;
    gpu_policy::set_denied(&con, id, false)?;
    println!("gpu{id} allowed");
    Ok(())
}

fn do_unload(model_id: Option<&str>, force: bool) -> Result<(), String> {
    // Unload one model (or all), then confirm how many instances stopped; a model goethe knows nothing about exits 1.
    let (live, known) = unload_counts(model_id);
    let reply = rpc::call(
        "unload",
        json!({"model_id": model_id, "all": model_id.is_none(), "force": force}),
    )
    .map_err(rpc_daemon_err)?;
    if reply.code != 0 {
        eprintln!("{}", reply.stderr);
        std::process::exit(reply.code);
    }
    let what = model_id.unwrap_or("all models");
    match (live, known) {
        (_, 0) if model_id.is_some() => Err(format!("{what} is not loaded; goethe ps lists loaded models")),
        (_, 0) => {
            println!("nothing was loaded");
            Ok(())
        }
        (0, _) => {
            println!("unloaded {what} (no running instance; it will not be restored)");
            Ok(())
        }
        (n, _) => {
            println!("unloaded {what} ({n} instance{})", if n == 1 { "" } else { "s" });
            Ok(())
        }
    }
}

fn unload_counts(model_id: Option<&str>) -> (i64, i64) {
    // Live instances, and every instance or desired row, for one model or all, read before the unload.
    let home = goethe_home();
    if !home.join("state.db").is_file() {
        return (0, 0);
    }
    let Ok(con) = db::open(&home) else {
        return (0, 0);
    };
    let count = |sql: &str| con.query_row(sql, [model_id], |r| r.get::<_, i64>(0)).unwrap_or(0);
    let only = "(?1 IS NULL OR model_id = ?1)";
    let live = count(&format!(
        "SELECT COUNT(*) FROM loaded_instances WHERE status IN ('STARTING', 'RUNNING', 'RESTARTING') AND {only}"
    ));
    let rows = count(&format!("SELECT COUNT(*) FROM loaded_instances WHERE {only}"))
        + count(&format!("SELECT COUNT(*) FROM desired_loads WHERE {only}"));
    (live, rows)
}

fn ps_gpus_csv(row: &Value) -> String {
    row.get("gpu_indices")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64())
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default()
}

fn ps_status(row: &Value, now: i64) -> String {
    // RESTARTING rows (crash backoff) show the coming attempt and the seconds until it.
    let status = row.get("status").and_then(|v| v.as_str()).unwrap_or("");
    if status != "RESTARTING" {
        return status.to_string();
    }
    let attempt = row.get("restarts").and_then(|v| v.as_i64()).unwrap_or(0) + 1;
    match row.get("next_attempt_at").and_then(|v| v.as_i64()) {
        Some(at) => format!("RESTARTING (attempt {attempt}, next in {}s)", (at - now).max(0)),
        None => format!("RESTARTING (attempt {attempt})"),
    }
}

fn print_ps_table(rows: &[Value]) {
    // Loaded instances as an aligned table with colored status, or a hint when nothing is loaded.
    if rows.is_empty() {
        print!("{}", no_models_text());
        return;
    }
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    let mut table = goethe_core::term::Table::new(&[
        "INSTANCE", "MODEL", "SLOT", "STATUS", "GPUS", "PORT", "VRAM_GB", "RESTARTS", "IN_FLIGHT", "TOK_PER_S",
    ]);
    for row in rows {
        let inst = row.get("instance_id").and_then(|v| v.as_str()).unwrap_or("");
        let model = row.get("model_id").and_then(|v| v.as_str()).unwrap_or("");
        let slot = row.get("slot_key").and_then(|v| v.as_str()).unwrap_or("");
        let status = ps_status(row, now);
        let gpus = ps_gpus_csv(row);
        let port = row.get("internal_port").and_then(|v| v.as_i64()).unwrap_or(0);
        let vram = row
            .get("allocated_vram_gb_per_gpu")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let restarts = row.get("restarts").and_then(|v| v.as_i64()).unwrap_or(0);
        let inflight = row.get("in_flight").and_then(|v| v.as_u64()).unwrap_or(0);
        let tps = row
            .get("tokens_per_s_1m")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        table.row(vec![
            inst.to_string(),
            model.to_string(),
            slot.to_string(),
            goethe_core::term::status(&status),
            gpus,
            port.to_string(),
            format!("{vram:.2}"),
            restarts.to_string(),
            inflight.to_string(),
            format!("{tps:.1}"),
        ]);
    }
    table.print();
}

fn no_models_text() -> String {
    goethe_core::term::commands("No models loaded. Pick one and load it:", &["goethe models", "goethe load --model <model-id>"])
}

fn empty_allowlist_text() -> String {
    goethe_core::term::commands(
        "The allowlist is empty, so every client is refused. Allow one with:",
        &["goethe allowlist --add <client-ip>/32"],
    )
}

fn do_ps(json: bool) -> Result<(), String> {
    let reply = rpc::call("ps", json!(null)).map_err(rpc_daemon_err)?;
    if reply.code != 0 {
        eprintln!("{}", reply.stderr);
        std::process::exit(reply.code as i32);
    }
    if json {
        println!("{}", reply.stdout.trim());
    } else {
        if let Ok(Value::Array(rows)) = serde_json::from_str(reply.stdout.trim()) {
            print_ps_table(&rows);
        }
        print_gateway_line();
    }
    Ok(())
}

fn logs_target_error(target: Option<&str>, daemon: bool, gateway: bool, prune: bool) -> Option<&'static str> {
    if target.is_some_and(|t| t.trim().is_empty()) {
        return Some("the log target is empty: give --model <id>, --daemon or --gateway");
    }
    match [target.is_some(), daemon, gateway].iter().filter(|b| **b).count() {
        0 if !prune => Some("give --model <id> (a model or instance id), --daemon or --gateway"),
        n if n > 1 => Some("give one log target: --model <id>, --daemon or --gateway"),
        _ => None,
    }
}

fn do_gateway_bind(addr: &str) -> Result<(), String> {
    // Validate the address, persist it and apply it to the running gateway; remind about the allowlist.
    let ip: std::net::IpAddr = addr
        .trim()
        .parse()
        .unwrap_or_else(|_| usage(format!("invalid address {addr:?}: use 0.0.0.0, 127.0.0.1 or an IP address of this host")));
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    println!("{}", nginx::set_public_bind(&home, &ip.to_string())?);
    if !ip.is_loopback() && security::allowlist_list()?.is_empty() {
        print!("\n{}", empty_allowlist_text());
    }
    Ok(())
}

fn do_gateway_status(json: bool) -> Result<(), String> {
    // Bind, port, local URL, whether our nginx serves, allowlist size and provider endpoint hints.
    let home = goethe_home();
    let bind = nginx::public_bind(&home);
    let port = nginx::public_port_u16(&home);
    let url = format!("https://{}", nginx::gateway_connect_addr(&home));
    let running = nginx::format_gateway_line(&home).is_some();
    let allowlist = security::allowlist_list()?.len();
    let endpoints = goethe_core::host::probe_environment(port).public_endpoints;
    if json {
        let row = json!({"bind": bind, "port": port, "url": url, "running": running,
            "allowlist_entries": allowlist, "endpoints": endpoints});
        println!("{row}");
        return Ok(());
    }
    print!("{}", gateway_status_text(&bind, port, &url, running, allowlist, &endpoints));
    Ok(())
}

fn gateway_status_text(bind: &str, port: u16, url: &str, running: bool, allowlist: usize, endpoints: &[String]) -> String {
    // Bind and who can reach it, port, local URL, state, allowlist size, then provider endpoint hints.
    let loopback = bind.parse::<std::net::IpAddr>().map_or(true, |ip| ip.is_loopback());
    let scope = if loopback {
        "this host only; `goethe gateway --bind 0.0.0.0` accepts remote clients"
    } else {
        "remote clients reach it when allowlisted"
    };
    let state = if running { "running" } else { "not running (goethed starts it; see goethe logs --gateway)" };
    let list = match allowlist {
        0 => "empty, every client is refused (goethe allowlist --add <client-ip>/32)".to_string(),
        n => format!("{n} entr{}", if n == 1 { "y" } else { "ies" }),
    };
    let mut out = format!("bind: {bind} ({scope})\nport: {port}\nurl: {url}\nstate: {state}\nallowlist: {list}\n");
    for e in endpoints {
        out.push_str(&format!("endpoint: {e}\n"));
    }
    out
}

fn do_up(boot: bool) -> Result<(), String> {
    // Start goethed, give the gateway a few seconds to listen, then show the same summary as goethe status.
    goethe_core::service::run_up(boot)?;
    let home = goethe_home();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while nginx::format_gateway_line(&home).is_none() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    do_status()
}

fn do_status() -> Result<(), String> {
    // Running or not; when running, the API address and the keys clients can use.
    let home = goethe_home();
    if !goethe_core::service::goethed_running() {
        print!("{}", status_text(None, &[]));
        return Ok(());
    }
    let api = match nginx::format_gateway_line(&home) {
        Some(_) => format!("https://{}/v1", nginx::gateway_connect_addr(&home)),
        None => "not listening yet (goethe logs --gateway shows why)".to_string(),
    };
    let keys: Vec<(String, String)> = security::key_list_detailed()?
        .iter()
        .map(|r| {
            let field = |k: &str| r.get(k).and_then(Value::as_str).unwrap_or("").to_string();
            (field("name"), field("id"))
        })
        .collect();
    print!("{}", status_text(Some(&api), &keys));
    Ok(())
}

fn status_text(api: Option<&str>, keys: &[(String, String)]) -> String {
    // goethe up and goethe status: the state word in color, then the API, its keys and where to read more.
    use goethe_core::term::{commands, paint, Color, Table};
    let Some(api) = api else {
        return format!("goethe: {}\n\n{}", paint("not running", Color::Red), commands("Start it with:", &["goethe up"]));
    };
    let mut out = format!("goethe: {}\n\nAPI: {api}\n\n", paint("running", Color::Green));
    if keys.is_empty() {
        out.push_str(&create_first_key());
    } else {
        let mut table = Table::new(&["API KEY NAME", "KEY ID"]);
        for (name, id) in keys {
            table.row(vec![name.clone(), id.clone()]);
        }
        out.push_str(&table.render());
    }
    out.push('\n');
    out.push_str(&commands("For more details run:", &["goethe api"]));
    out
}

fn do_api(json: bool) -> Result<(), String> {
    // The endpoint, the models it serves, the keys that may call it, and one curl command to try it.
    let home = goethe_home();
    let api = format!("https://{}/v1", nginx::gateway_connect_addr(&home));
    let ps = rpc::call("ps", json!(null)).ok().filter(|r| r.code == 0);
    let rows: Vec<Value> = ps.as_ref().and_then(|r| serde_json::from_str(r.stdout.trim()).ok()).unwrap_or_default();
    let mut models: Vec<(String, Vec<String>, Vec<String>)> = Vec::new();
    for row in &rows {
        let id = row.get("model_id").and_then(Value::as_str).unwrap_or("").to_string();
        let status = row.get("status").and_then(Value::as_str).unwrap_or("").to_string();
        let gpus = ps_gpus_csv(row).split(',').filter(|g| !g.is_empty()).map(|g| format!("gpu{g}")).collect::<Vec<_>>().join(",");
        match models.iter_mut().find(|m| m.0 == id) {
            Some(m) => {
                m.1.push(status);
                m.2.push(gpus);
            }
            None => models.push((id, vec![status], vec![gpus])),
        }
    }
    let state = |s: &[String]| if s.iter().any(|x| x == "RUNNING") { "RUNNING".to_string() } else { s[0].clone() };
    let keys = security::key_list_detailed()?;
    if json {
        let list: Vec<Value> = models
            .iter()
            .map(|(id, st, g)| json!({"model": id, "status": state(st), "instances": st.len(), "gpus": g}))
            .collect();
        println!("{}", json!({"api": api, "schemas": ["openai", "anthropic", "litellm"], "goethed_running": ps.is_some(), "keys": keys, "models": list}));
        return Ok(());
    }
    use goethe_core::term::{commands, paint, Color, Table};
    println!("API endpoint: {api}");
    println!("Accepts: OpenAI, Anthropic (/v1/messages) and LiteLLM (routes with or without /v1, x-litellm-api-key)");
    for e in goethe_core::host::probe_environment(nginx::public_port_u16(&home)).public_endpoints {
        println!("Public endpoint: {e}");
    }
    println!("\n{}", paint("Models running", Color::Bold));
    if ps.is_none() {
        println!("{}", commands("goethed is not running, so no models are served. Start it with:", &["goethe up"]));
    } else if models.is_empty() {
        println!("{}", no_models_text());
    } else {
        let mut table = Table::new(&["MODEL", "STATUS", "INSTANCES", "GPUS"]);
        for (id, st, g) in &models {
            table.row(vec![id.clone(), goethe_core::term::status(&state(st)), st.len().to_string(), g.join(" ")]);
        }
        println!("{}", table.render());
    }
    println!("{}\n{}", paint("API keys", Color::Bold), api_keys_text(&keys));
    if security::allowlist_list()?.is_empty() {
        println!("{}", empty_allowlist_text());
    }
    print!("{}", commands("Query the API:", &[&curl_example(&api)]));
    Ok(())
}

fn curl_example(api: &str) -> String {
    format!(
        "curl -k {api}/chat/completions -H \"Authorization: Bearer <your_bearer_token>\" -H \"Content-Type: application/json\" \
         -d '{{\"model\": \"<model_name>\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'"
    )
}

fn print_gateway_line() {
    // On stderr, so stdout stays the data rows.
    if let Some(line) = nginx::format_gateway_line(&goethe_home()) {
        eprintln!("{line}");
    }
}

fn do_registry_entry(
    mut repo_ids: Vec<String>,
    runtime: Option<String>,
    revision: Option<String>,
    ids_file: Option<String>,
    strict: bool,
) -> Result<(), String> {
    if let Some(path) = ids_file {
        let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        for line in text.lines() {
            let t = line.trim();
            if !t.is_empty() && !t.starts_with('#') {
                repo_ids.push(t.to_string());
            }
        }
    }
    registry::run_entries(
        &repo_ids,
        runtime.as_deref(),
        revision.as_deref(),
        &registry::EntryOpts { strict },
    )
}

fn do_registry_add(
    mut repo_ids: Vec<String>,
    ids_file: Option<String>,
    file: String,
    overlay: Option<String>,
    update: bool,
    dry_run: bool,
    strict: bool,
) -> Result<(), String> {
    if let Some(path) = ids_file {
        let text = std::fs::read_to_string(&path).map_err(|e| e.to_string())?;
        for line in text.lines() {
            let t = line.trim();
            if !t.is_empty() && !t.starts_with('#') {
                repo_ids.push(t.to_string());
            }
        }
    }
    registry::run_add(&registry::AddOpts {
        repo_ids,
        file,
        overlay_path: overlay,
        update,
        dry_run,
        strict,
    })
}


fn fail(msg: String) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gateway_status_explains_scope_and_allowlist() {
        let local = gateway_status_text("127.0.0.1", 8443, "https://127.0.0.1:8443", true, 0, &[]);
        assert!(local.starts_with("bind: 127.0.0.1 (this host only; `goethe gateway --bind 0.0.0.0`"), "{local}");
        assert!(local.contains("port: 8443\nurl: https://127.0.0.1:8443\nstate: running\n"), "{local}");
        assert!(local.contains("allowlist: empty, every client is refused"), "{local}");
        let open = gateway_status_text("0.0.0.0", 9443, "https://127.0.0.1:9443", false, 2, &["tcp://1.2.3.4:4000 -> port 9443".into()]);
        assert!(open.contains("bind: 0.0.0.0 (remote clients reach it when allowlisted)"), "{open}");
        assert!(open.contains("state: not running"), "{open}");
        assert!(open.contains("allowlist: 2 entries\nendpoint: tcp://1.2.3.4:4000 -> port 9443\n"), "{open}");
    }

    #[test]
    fn status_says_running_with_the_api_and_keys_or_how_to_start() {
        let stopped = status_text(None, &[]);
        assert!(stopped.starts_with("goethe: ") && stopped.contains("not running"), "{stopped}");
        assert!(stopped.ends_with("Start it with:\n\n\t→ goethe up\n"), "{stopped}");
        let fresh = status_text(Some("https://10.0.0.2:8443/v1"), &[]);
        assert!(fresh.contains("running\n\nAPI: https://10.0.0.2:8443/v1\n\nCreate your first API key:\n\n\t→ goethe key --create my-app\n"), "{fresh}");
        assert!(fresh.ends_with("\nFor more details run:\n\n\t→ goethe api\n"), "{fresh}");
        let keyed = status_text(Some("https://h/v1"), &[("ops".into(), "a1b2c3d4e5f6".into())]);
        assert!(keyed.contains("ops") && keyed.contains("a1b2c3d4e5f6") && !keyed.contains("Create your first"), "{keyed}");
    }

    #[test]
    fn empty_lists_say_how_to_add_the_first_entry() {
        assert!(key_list_text(&[]).ends_with("\t→ goethe key --create my-app\n"));
        assert!(allowlist_text(&[]).ends_with("\t→ goethe allowlist --add <client-ip>/32\n"));
        assert!(alias_text(&[]).ends_with("\t→ goethe alias --set <alias> --model <model-id>\n"));
    }

    #[test]
    fn every_user_value_is_a_flag() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
        let parse = |v: &[&str]| Cli::try_parse_from(std::iter::once("goethe").chain(v.iter().copied()));
        assert!(parse(&["domain", "--check", "api.example.com"]).is_ok());
        assert!(parse(&["domain", "add", "api.example.com"]).is_err() && parse(&["domain", "--add", "x.y"]).is_err());
        assert!(parse(&["load", "--model", "org/m", "--gpus", "2"]).is_ok());
        assert!(parse(&["load", "org/m"]).is_err());
        assert!(parse(&["key", "--create", "ops", "--expires", "30d", "--admin"]).is_ok());
        assert!(parse(&["key", "--admin"]).is_err(), "--admin only goes with --create");
        assert!(parse(&["key", "--create", "a", "--revoke", "b"]).is_err());
        assert!(parse(&["gpu", "--allow", "gpu2"]).is_ok() && parse(&["alias", "--set", "fast"]).is_err());
        let Commands::Gpu { allow, .. } = parse(&["gpu", "--allow", "gpu0", "gpu1"]).unwrap().command else { panic!() };
        assert_eq!(allow, vec![0, 1]);
        assert!(parse(&["key", "--revoke", "a1", "b2"]).is_ok() && parse(&["allowlist", "--add", "10.0.0.1/32", "10.0.0.2/32"]).is_ok());
        assert!(parse(&["unload", "--model", "org/a", "org/b", "--force"]).is_ok() && parse(&["remove"]).is_err());
        assert_eq!(parse_gpu_name("gpu3"), Ok(3));
        assert_eq!(parse_gpu_name("GPU0"), Ok(0));
        assert!(parse_gpu_name("gpux").is_err());
    }

    #[test]
    fn values_without_their_flag_get_the_corrected_command() {
        let args = |v: &[&str]| std::iter::once("goethe").chain(v.iter().copied()).map(String::from).collect::<Vec<_>>();
        let unload = args(&["unload", "org/a", "org/b"]);
        assert_eq!(corrected_command(&unload, "org/a").as_deref(), Some("goethe unload --model org/a org/b"));
        let gpu = args(&["gpu", "allow", "gpu0", "gpu1"]);
        assert_eq!(corrected_command(&gpu, "allow").as_deref(), Some("goethe gpu --allow gpu0 gpu1"));
        let load = args(&["load", "org/a", "--gpus", "2"]);
        assert_eq!(corrected_command(&load, "org/a").as_deref(), Some("goethe load --model org/a --gpus 2"));
        assert_eq!(corrected_command(&args(&["domain", "api.example.com"]), "api.example.com").as_deref(), Some("goethe domain --check api.example.com"));
        assert_eq!(corrected_command(&args(&["status", "x"]), "x"), None);
    }

    #[test]
    fn logs_take_exactly_one_target() {
        assert!(logs_target_error(Some(""), false, false, false).is_some_and(|m| m.contains("empty")));
        assert!(logs_target_error(Some("  "), false, false, false).is_some());
        assert!(logs_target_error(Some("org/m"), true, false, false).is_some_and(|m| m.contains("one log target")));
        assert!(logs_target_error(None, true, true, false).is_some());
        assert!(logs_target_error(None, false, false, false).is_some());
        assert_eq!(logs_target_error(None, false, false, true), None);
        assert_eq!(logs_target_error(Some("org/m"), false, false, false), None);
        assert_eq!(logs_target_error(None, false, true, false), None);
    }

    #[test]
    fn restarting_rows_show_attempt_and_wait() {
        let row = json!({"status": "RESTARTING", "restarts": 2, "next_attempt_at": 1_000_030});
        assert_eq!(ps_status(&row, 1_000_000), "RESTARTING (attempt 3, next in 30s)");
        assert_eq!(ps_status(&row, 1_000_090), "RESTARTING (attempt 3, next in 0s)");
        assert_eq!(ps_status(&json!({"status": "RESTARTING", "restarts": 0}), 0), "RESTARTING (attempt 1)");
        assert_eq!(ps_status(&json!({"status": "RUNNING", "restarts": 4}), 0), "RUNNING");
    }

    #[test]
    fn key_create_raw_is_only_the_token() {
        let token = "goethe_sk_0123456789abcdef0123456789abcdef";
        assert_eq!(key_create_stdout(token, "a1b2c3d4e5f6", "ops", true), format!("{token}\n"));
        let human = key_create_stdout(token, "a1b2c3d4e5f6", "ops", false);
        assert!(human.contains(token));
        assert!(human.starts_with("API key for ops (id a1b2c3d4e5f6):\n\n\t"), "{human}");
        assert_eq!(rpm_text(0), "unlimited");
        assert_eq!(rpm_text(120), "120");
        assert!(human.contains("goethe api shows it again"));
        assert_eq!(human.matches(token).count(), 1);
    }
}
