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

#[derive(Parser)]
#[command(
    name = "goethe",
    disable_version_flag = true,
    about = "Goethe: serve LLMs on your NVIDIA GPUs behind one OpenAI-compatible gateway",
    after_help = "Examples:\n  goethe up\n  goethe install Qwen/Qwen3-8B --load\n  goethe scan\n  goethe load Qwen/Qwen3-8B --gpus 2,3 --ctx 65536\n  goethe get api\n  goethe key create --name ops --raw\n  goethe down\n\ngoethe down stops goethed but keeps loaded models for the next goethe up. \
                  goethe unload removes all instances of a model and it is not restored."
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
    #[command(about = "List GPUs and manage allow/deny policy.")]
    Gpu {
        #[command(subcommand)]
        cmd: GpuCommands,
    },
    #[command(about = "List registry models that fit this machine.")]
    Recommend {
        #[arg(long, help = "Emit JSON instead of a table.")]
        json: bool,
    },
    #[command(about = "Download, verify, and register a model (from Hugging Face or a local path). With --load, waits for RUNNING like goethe load unless --no-wait.")]
    Install {
        #[arg(help = "Hugging Face repo id or local directory path.")]
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
        #[arg(long, help = "Registered model id for local installs.")]
        name: Option<String>,
        #[arg(long, help = "Copy local tree instead of symlinking.")]
        copy: bool,
        #[arg(long = "sglang-arg", allow_hyphen_values = true, help = "Extra SGLang CLI flag.")]
        sglang_arg: Vec<String>,
    },
    #[command(about = "Register model folders you downloaded yourself and moved into GOETHE_HOME/models.")]
    Scan {
        #[arg(help = "One folder inside GOETHE_HOME/models (default: all of them).")]
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
        #[arg(help = "Installed model id to remove.")]
        model_id: String,
        #[arg(long, help = "Do not ask; also unloads the model if it is loaded.")]
        yes: bool,
        #[arg(long, help = "When unloading first, skip draining in-flight requests.")]
        force: bool,
    },
    #[command(about = "Change default load settings stored on the model card.")]
    Config {
        #[arg(help = "Installed model id to update.")]
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
        after_help = "Extra SGLang flags go after --, for example: goethe load org/model --ctx 32768 -- --chunked-prefill-size 4096"
    )]
    Load {
        #[arg(help = "Model id to load.")]
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
        #[arg(help = "Model id to unload (all slots), or omit with --all.")]
        model_id: Option<String>,
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
    #[command(about = "Stop goethed and its engines and turn off start at boot (loaded models come back on the next goethe up).")]
    Down {
        #[arg(long = "keep-engines", help = "Leave loaded engines serving; the next goethed adopts them.")]
        keep_engines: bool,
    },
    #[command(about = "Guided setup.")]
    Configure {
        #[command(subcommand)]
        cmd: ConfigureCommands,
    },
    #[command(about = "Print connection details.")]
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
    #[command(about = "Show or change where the public gateway listens.")]
    Gateway {
        #[command(subcommand)]
        cmd: GatewayCommands,
    },
    #[command(about = "Configure a public DNS name and TLS certificate.")]
    Domain {
        #[command(subcommand)]
        cmd: DomainCommands,
    },
    #[command(about = "Tail engine, goethed, or gateway logs.")]
    Logs {
        #[arg(help = "Model id, or omit with --daemon / --gateway.")]
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
    #[command(about = "Remove goethe binaries and optionally delete downloaded models.")]
    Uninstall {
        #[arg(long, help = "Also delete GOETHE_HOME/models.")]
        purge_models: bool,
        #[arg(long, help = "Skip confirmation.")]
        yes: bool,
    },
    #[command(about = "Manage CIDRs allowed to reach the public gateway.")]
    Allowlist {
        #[command(subcommand)]
        cmd: AllowlistCommands,
    },
    #[command(about = "Create, list, or revoke API keys for the gateway.")]
    Key {
        #[command(subcommand)]
        cmd: KeyCommands,
    },
    #[command(about = "Map gateway model aliases to installed model ids.")]
    Alias {
        #[command(subcommand)]
        cmd: AliasCommands,
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
enum GatewayCommands {
    #[command(about = "Listen address: 127.0.0.1 (default, this host only), 0.0.0.0 (all addresses) or one IP of this host.")]
    Bind {
        #[arg(help = "0.0.0.0, 127.0.0.1 or an IP address of this host.")]
        addr: String,
    },
    #[command(about = "Show the bind address, port, URL, state and allowlist size.")]
    Status {
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
}

#[derive(Subcommand)]
enum DomainCommands {
    #[command(about = "Obtain TLS for a domain and bind the gateway publicly.")]
    Add {
        #[arg(help = "DNS name (must resolve to this host).")]
        domain: String,
    },
}

#[derive(Subcommand)]
enum AllowlistCommands {
    #[command(about = "Allow a CIDR to reach the public gateway.")]
    Add {
        #[arg(help = "IPv4/IPv6 CIDR.")]
        cidr: String,
        #[arg(long, help = "Optional note stored with the entry.")]
        description: Option<String>,
    },
    #[command(about = "Remove a CIDR from the allowlist.")]
    Remove {
        #[arg(help = "CIDR to remove.")]
        cidr: String,
    },
    #[command(about = "List allowed CIDRs.")]
    List,
}

#[derive(Subcommand)]
#[command(after_help = "Without stream_options.include_usage, streaming usage counts SSE chunks with delta content as approximate completion tokens (prompt tokens 0).")]
enum KeyCommands {
    #[command(about = "Create an API key for the public gateway.")]
    Create {
        #[arg(long, help = "Human-readable label stored with the key.")]
        name: String,
        #[arg(long, help = "Print only the token on stdout (for API_KEY=$(goethe key create --raw ...)).")]
        raw: bool,
        #[arg(long, help = "Requests-per-minute limit.")]
        rpm: Option<i64>,
        #[arg(long, help = "Expiry like 30d or an ISO date.")]
        expires: Option<String>,
        #[arg(long, help = "Comma-separated model scope.")]
        models: Option<String>,
        #[arg(long, help = "Grant admin health and unrestricted scope when models is unset.")]
        admin: bool,
    },
    #[command(about = "Revoke an API key by id.")]
    Revoke {
        #[arg(help = "Key id from goethe key list.")]
        key_id: String,
    },
    #[command(about = "List API keys.")]
    List {
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
}

#[derive(Subcommand)]
enum AliasCommands {
    #[command(about = "Set an alias to an installed model id.")]
    Set {
        #[arg(help = "Alias name clients may use as the model field.")]
        alias: String,
        #[arg(help = "Installed model id.")]
        model: String,
    },
    #[command(about = "List model aliases.")]
    List,
    #[command(about = "Remove a model alias.")]
    Remove {
        #[arg(help = "Alias to remove.")]
        alias: String,
    },
}

#[derive(Subcommand)]
enum RegistryCommands {
    #[command(about = "Check a registry JSON file (local path or URL).")]
    Validate {
        #[arg(help = "Path or URL to registry JSON.")]
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
        #[arg(help = "Hugging Face repo ids.")]
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
        #[arg(help = "Hugging Face repo ids to add from the file.")]
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
enum GpuCommands {
    #[command(about = "Show GPUs with policy status.")]
    List {
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
    #[command(about = "Deny new placement on a GPU.")]
    Deny {
        #[arg(help = "GPU index.")]
        id: u32,
    },
    #[command(about = "Allow placement on a GPU.")]
    Allow {
        #[arg(help = "GPU index.")]
        id: u32,
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

fn checked_id(id: &str) -> &str {
    if let Err(e) = validate_model_id(id) {
        usage(e);
    }
    id
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
    let cli = Cli::parse();
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
        Commands::Gpu { cmd } => match cmd {
            GpuCommands::List { json } => do_gpu_list(json),
            GpuCommands::Deny { id } => do_gpu_deny(id),
            GpuCommands::Allow { id } => do_gpu_allow(id),
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
            name,
            copy,
            sglang_arg,
        } => {
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
        } => models::run_remove(&installed_id(&model_id, false), yes, force),
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
        Commands::Unload { model_id, all, force } => {
            let model_id = model_id.map(|id| installed_id(&id, false));
            match (&model_id, all) {
                (Some(_), false) => {}
                (None, false) => usage("give a model id or --all"),
                (Some(_), true) => usage("give either a model id or --all, not both"),
                (None, true) => {}
            }
            do_unload(model_id.as_deref(), force)
        }
        Commands::Ps { json } => do_ps(json),
        Commands::Up { no_boot } => do_up(!no_boot),
        Commands::Get { cmd: GetCommands::Api { json } } => do_get_api(json),
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
        Commands::Gateway { cmd } => match cmd {
            GatewayCommands::Bind { addr } => do_gateway_bind(&addr),
            GatewayCommands::Status { json } => do_gateway_status(json),
        },
        Commands::Domain { cmd } => match cmd {
            DomainCommands::Add { domain } => match goethe_core::domain::add(&domain) {
                Ok(url) => {
                    println!("{url}");
                    Ok(())
                }
                Err(e) => {
                    eprintln!("{e}");
                    std::process::exit(1);
                }
            },
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
            purge_models,
            yes,
        } => goethe_core::uninstall::run(goethe_core::uninstall::UninstallOpts {
            purge_models,
            yes,
        }),
        Commands::Allowlist { cmd } => match cmd {
            AllowlistCommands::Add { cidr, description } => {
                security::allowlist_add(&cidr, description.as_deref()).map(|()| {
                    println!("allowlist: added {}", security::canonical_cidr(&cidr).unwrap_or(cidr));
                })
            }
            AllowlistCommands::Remove { cidr } => security::allowlist_remove(&cidr).map(|()| {
                println!("allowlist: removed {}", security::canonical_cidr(&cidr).unwrap_or(cidr));
            }),
            AllowlistCommands::List => security::allowlist_list().map(|rows| {
                for cidr in &rows {
                    println!("{cidr}");
                }
                if rows.is_empty() {
                    eprintln!("the allowlist is empty, so the gateway refuses every client: goethe allowlist add <client-ip>/32");
                }
                print_gateway_line();
            }),
        },
        Commands::Key { cmd } => match cmd {
            KeyCommands::Create {
                name,
                raw,
                rpm,
                expires,
                models,
                admin,
            } => {
                if name.trim().is_empty() {
                    usage("--name must not be empty");
                }
                if let Some(Err(e)) = models.as_deref().map(security::model_scope_list) {
                    usage(e);
                }
                security::key_create(&name, true, rpm, expires.as_deref(), models.as_deref(), admin).map(|token| {
                    print!("{}", key_create_stdout(&token, &key_id_for_token(&token), &name, raw));
                    if let Some(line) = nginx::format_gateway_line(&goethe_home()) {
                        eprintln!("{line}");
                    }
                })
            }
            KeyCommands::Revoke { key_id } => security::key_revoke(&key_id).map(|()| println!("revoked key {key_id}")),
            KeyCommands::List { json } => {
                if json {
                    match security::key_list_detailed() {
                        Ok(rows) => {
                            println!("{}", serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into()));
                            Ok(())
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    match security::key_list_detailed() {
                        Ok(rows) => {
                            for row in rows {
                                let id = row.get("id").and_then(|v| v.as_str()).unwrap_or("");
                                let name = row.get("name").and_then(|v| v.as_str()).unwrap_or("");
                                let rpm = rpm_text(row.get("rpm").and_then(|v| v.as_i64()).unwrap_or(0));
                                let exp = row.get("expires").and_then(|v| v.as_str()).unwrap_or("-");
                                let models = row.get("models").and_then(|v| v.as_str()).unwrap_or("");
                                let admin = row
                                    .get("admin")
                                    .and_then(|v| v.as_bool())
                                    .unwrap_or(false);
                                let admin_s = if admin { "yes" } else { "no" };
                                let rt = row.get("requests_today").and_then(|v| v.as_i64()).unwrap_or(0);
                                let tt = row.get("tokens_today").and_then(|v| v.as_i64()).unwrap_or(0);
                                println!(
                                    "{id}\t{name}\tADMIN={admin_s}\trpm={rpm}\texpires={exp}\tmodels={models}\ttoday_req={rt}\ttoday_tok={tt}"
                                );
                            }
                            Ok(())
                        }
                        Err(e) => Err(e),
                    }
                }
            }
        },
        Commands::Alias { cmd } => match cmd {
            AliasCommands::Set { alias: a, model } => alias::set(&a, &model).map(|()| println!("alias {a} -> {model}")),
            AliasCommands::List => match alias::list() {
                Ok(rows) => {
                    for (a, model) in rows {
                        println!("{a}\t{model}");
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            },
            AliasCommands::Remove { alias: a } => alias::remove(&a).map(|()| println!("removed alias {a}")),
        },
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
    // --raw prints exactly the token; otherwise the token once, its id, and a store-it-now note.
    if raw {
        return format!("{token}\n");
    }
    format!(
        "API key: {token}\nKey id:  {id} (name {name})\nStore this key now: goethe keeps only its hash and cannot show it again.\n"
    )
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
    for r in &rows {
        if let Some(why) = goethe_core::gpu_arch::unsupported_reason(&r.compute_capability) {
            println!("gpu{}: {why}; Goethe will not place models on it", r.index);
        }
    }
    Ok(())
}

fn do_gpu_deny(id: u32) -> Result<(), String> {
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    gpu_policy::validate_gpu_index(&home, id)?;
    let con = db::open(&home).map_err(|e| e.to_string())?;
    gpu_policy::set_denied(&con, id, true)?;
    println!("GPU {id} denied");
    let running = gpu_policy::instances_on_gpu(&con, id)?;
    if !running.is_empty() {
        println!(
            "note: {} instance(s) still on GPU {id}; new loads will not use it",
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
    println!("GPU {id} allowed");
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
        (_, 0) if model_id.is_some() => Err(format!("{what} is not loaded (goethe ps lists loaded models)")),
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
        println!("no models loaded (goethe load <model> starts one; goethe models lists installed models)");
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
        return Some("the log target is empty: give a model id or instance id, or --daemon or --gateway");
    }
    match [target.is_some(), daemon, gateway].iter().filter(|b| **b).count() {
        0 if !prune => Some("give a model id or instance id, or --daemon or --gateway"),
        n if n > 1 => Some("give one log target: a model or instance id, --daemon, or --gateway"),
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
        println!("the allowlist is empty, so every client is refused: goethe allowlist add <client-ip>/32");
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
        "this host only; `goethe gateway bind 0.0.0.0` accepts remote clients"
    } else {
        "remote clients reach it when allowlisted"
    };
    let state = if running { "running" } else { "not running (goethed starts it; see goethe logs --gateway)" };
    let list = match allowlist {
        0 => "empty, every client is refused (goethe allowlist add <client-ip>/32)".to_string(),
        n => format!("{n} entr{}", if n == 1 { "y" } else { "ies" }),
    };
    let mut out = format!("bind: {bind} ({scope})\nport: {port}\nurl: {url}\nstate: {state}\nallowlist: {list}\n");
    for e in endpoints {
        out.push_str(&format!("endpoint: {e}\n"));
    }
    out
}

fn do_up(boot: bool) -> Result<(), String> {
    // Start goethed, give the gateway a few seconds to listen, then print where the API is.
    goethe_core::service::run_up(boot)?;
    let home = goethe_home();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(15);
    while nginx::format_gateway_line(&home).is_none() && std::time::Instant::now() < deadline {
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    match nginx::format_gateway_line(&home) {
        Some(_) => println!("API: https://{}/v1 (goethe get api lists the loaded models)", nginx::gateway_connect_addr(&home)),
        None => println!("the gateway is not listening yet (goethe logs --gateway shows why)"),
    }
    Ok(())
}

fn do_get_api(json: bool) -> Result<(), String> {
    // The base URL clients point an OpenAI SDK at, then one row per loaded model with its state and GPUs.
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
    if json {
        let list: Vec<Value> = models
            .iter()
            .map(|(id, st, g)| json!({"model": id, "status": state(st), "instances": st.len(), "gpus": g}))
            .collect();
        println!("{}", json!({"api": api, "goethed_running": ps.is_some(), "models": list}));
        return Ok(());
    }
    println!("API: {api}");
    println!("Auth: Authorization: Bearer <key>   (goethe key create --name <name>)");
    for e in goethe_core::host::probe_environment(nginx::public_port_u16(&home)).public_endpoints {
        println!("Public endpoint: {e}");
    }
    if security::allowlist_list()?.is_empty() {
        println!("note: the allowlist is empty, so every client is refused (goethe allowlist add 127.0.0.1/32)");
    }
    println!();
    if ps.is_none() {
        println!("goethed is not running, so no models are served (goethe up starts it)");
        return Ok(());
    }
    if models.is_empty() {
        println!("no models loaded (goethe load <model>; goethe models lists installed models)");
        return Ok(());
    }
    let mut table = goethe_core::term::Table::new(&["MODEL", "STATUS", "INSTANCES", "GPUS"]);
    for (id, st, g) in &models {
        table.row(vec![id.clone(), goethe_core::term::status(&state(st)), st.len().to_string(), g.join(" ")]);
    }
    table.print();
    Ok(())
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
        assert!(local.starts_with("bind: 127.0.0.1 (this host only; `goethe gateway bind 0.0.0.0`"), "{local}");
        assert!(local.contains("port: 8443\nurl: https://127.0.0.1:8443\nstate: running\n"), "{local}");
        assert!(local.contains("allowlist: empty, every client is refused"), "{local}");
        let open = gateway_status_text("0.0.0.0", 9443, "https://127.0.0.1:9443", false, 2, &["tcp://1.2.3.4:4000 -> port 9443".into()]);
        assert!(open.contains("bind: 0.0.0.0 (remote clients reach it when allowlisted)"), "{open}");
        assert!(open.contains("state: not running"), "{open}");
        assert!(open.contains("allowlist: 2 entries\nendpoint: tcp://1.2.3.4:4000 -> port 9443\n"), "{open}");
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
        assert!(human.contains("Key id:  a1b2c3d4e5f6 (name ops)"));
        assert_eq!(rpm_text(0), "unlimited");
        assert_eq!(rpm_text(120), "120");
        assert!(human.contains("Store this key now"));
        assert_eq!(human.matches(token).count(), 1);
    }
}
