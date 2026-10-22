// goethe CLI entrypoint for hardware, install, config, and load commands.

use clap::{Parser, Subcommand};
use goethe_core::alias;
use goethe_core::config::{self, ConfigOpts};
use goethe_core::db;
use goethe_core::gpu_policy;
use goethe_core::hw;
use goethe_core::hf;
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
    about = "MeshApplied inference engine",
    after_help = "Examples:\n  goethe install meshapplied/Qwen3-8B --load\n  goethe load my/model --gpus 2,3 --ctx 65536\n  goethe registry add org/model --file cdn/v1/registry.json\n  goethe key create --name ops --raw\n\ngoethe stop stops goethed but keeps desired loads for restore (every desired slot). \
                  goethe unload removes all instances of a model (all slots) and their desired rows."
)]
struct Cli {
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
    #[command(about = "Save a Hugging Face token for private or gated downloads.")]
    Login {
        #[arg(long, help = "Read the token from stdin.")]
        token_stdin: bool,
    },
    #[command(about = "Remove the stored Hugging Face token.")]
    Logout,
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
    #[command(about = "Load a model onto GPUs and wait until this instance is ready.")]
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
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, help = "Extra SGLang args after --.")]
        extra: Vec<String>,
        #[arg(long, help = "Return after spawn without waiting for RUNNING.")]
        no_wait: bool,
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
    #[command(about = "Ask goethed to shut down (desired loads are kept for restore).")]
    Stop,
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
        #[arg(long, help = "Disable restore on start.")]
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
    #[command(about = "Print GPU model, VRAM, and driver info.")]
    Profile {
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
    #[command(about = "Suggest GPU indices for a tensor-parallel size.")]
    PickGpus {
        #[arg(long, default_value = "1", help = "Tensor parallel size.")]
        tp: u64,
    },
}

fn usage(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
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
    let result = match cli.command {
        Commands::Hw {
            cmd: HwCommands::Profile { json },
        } => hw::run_profile(json),
        Commands::Hw {
            cmd: HwCommands::PickGpus { tp },
        } => goethe_core::placement::run_pick_gpus(tp),
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
        Commands::Login { token_stdin } => do_login(token_stdin),
        Commands::Logout => do_logout(),
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
        } => models::run_remove(checked_id(&model_id), yes, force),
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
            config::run(ConfigOpts {
                model_id: checked_id(&model_id).to_string(),
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
            extra,
            no_wait,
        } => do_load(DoLoadArgs {
            model_id,
            gpus,
            ctx,
            max_conc,
            tp,
            ep,
            force,
            yes,
            extra,
            no_wait,
        }),
        Commands::Unload { model_id, all, force } => {
            match (&model_id, all) {
                (Some(id), false) => {
                    checked_id(id);
                }
                (None, false) => usage("give a model id or --all"),
                (Some(_), true) => usage("give either a model id or --all, not both"),
                (None, true) => {}
            }
            do_unload(model_id, all, force)
        }
        Commands::Ps { json } => do_ps(json),
        Commands::Stop => shutdown::run_stop(),
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
        } => goethe_core::logs::run_logs(target.as_deref(), follow, lines, daemon, gateway, prune),
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
                security::allowlist_add(&cidr, description.as_deref())
            }
            AllowlistCommands::Remove { cidr } => security::allowlist_remove(&cidr),
            AllowlistCommands::List => match security::allowlist_list() {
                Ok(rows) => {
                    print_gateway_line();
                    for cidr in rows {
                        println!("{cidr}");
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            },
        },
        Commands::Key { cmd } => match cmd {
            KeyCommands::Create {
                name,
                raw,
                rpm,
                expires,
                models,
                admin,
            } => match security::key_create(&name, true, rpm, expires.as_deref(), models.as_deref(), admin) {
                Ok(token) => {
                    print!("{}", key_create_stdout(&token, &name, raw));
                    if let Some(line) = nginx::format_gateway_line(&goethe_home()) {
                        eprintln!("{line}");
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            },
            KeyCommands::Revoke { key_id } => security::key_revoke(&key_id),
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
                                let rpm = row.get("rpm").and_then(|v| v.as_i64()).unwrap_or(0);
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
            AliasCommands::Set { alias: a, model } => alias::set(&a, &model),
            AliasCommands::List => match alias::list() {
                Ok(rows) => {
                    for (a, model) in rows {
                        println!("{a}\t{model}");
                    }
                    Ok(())
                }
                Err(e) => Err(e),
            },
            AliasCommands::Remove { alias: a } => alias::remove(&a),
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
    extra: Vec<String>,
    no_wait: bool,
}

fn do_load(args: DoLoadArgs) -> Result<(), String> {
    // Validate locally, warn about host RAM for offloading cards, then load with the foreign-process prompt.
    checked_id(&args.model_id);
    if matches!(args.ep, Some(0)) {
        usage("--ep must be at least 1");
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
    let stdout = install::send_load(json!({
        "model_id": args.model_id,
        "gpus": args.gpus,
        "ctx": args.ctx,
        "max_conc": args.max_conc,
        "tp": args.tp,
        "force": args.force,
        "yes": args.yes,
        "extra_sglang_args": extra,
        "async_finish": true,
    }))?;
    if !stdout.is_empty() {
        print!("{stdout}");
    }
    goethe_core::load_wait::wait_after_load_stdout(&stdout, args.no_wait)
}

fn key_create_stdout(token: &str, name: &str, raw: bool) -> String {
    // --raw prints exactly the token; otherwise the token once, its id, and a store-it-now note.
    if raw {
        return format!("{token}\n");
    }
    let id = token.get(8..16).unwrap_or("?");
    format!(
        "API key: {token}\nKey id:  {id} (name {name})\nStore this key now: goethe keeps only its hash and cannot show it again.\n"
    )
}

fn rpc_daemon_err(e: String) -> String {
    rpc::map_daemon_err(e)
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
    println!(
        "INDEX\tNAME\tCC\tVRAM_TOTAL_MB\tMANAGED_MB\tFOREIGN_MB\tSTATUS"
    );
    for r in rows {
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            r.index,
            r.name,
            r.compute_capability,
            r.vram_total_mb,
            r.managed_mb,
            r.foreign_mb,
            r.status
        );
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

fn do_unload(model_id: Option<String>, all: bool, force: bool) -> Result<(), String> {
    let reply = rpc::call(
        "unload",
        json!({"model_id": model_id, "all": all, "force": force}),
    )
    .map_err(rpc_daemon_err)?;
    if reply.code != 0 {
        eprintln!("{}", reply.stderr);
        std::process::exit(reply.code as i32);
    }
    Ok(())
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

fn print_ps_table(rows: &[Value]) {
    if rows.is_empty() {
        return;
    }
    println!(
        "INSTANCE\tMODEL\tSLOT\tSTATUS\tGPUS\tPORT\tVRAM_GB\tRESTARTS\tIN_FLIGHT\tTOK_PER_S"
    );
    for row in rows {
        let inst = row.get("instance_id").and_then(|v| v.as_str()).unwrap_or("");
        let model = row.get("model_id").and_then(|v| v.as_str()).unwrap_or("");
        let slot = row.get("slot_key").and_then(|v| v.as_str()).unwrap_or("");
        let status = row.get("status").and_then(|v| v.as_str()).unwrap_or("");
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
        println!(
            "{inst}\t{model}\t{slot}\t{status}\t{gpus}\t{port}\t{vram:.2}\t{restarts}\t{inflight}\t{tps:.1}"
        );
    }
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
        print_gateway_line();
        if let Ok(Value::Array(rows)) = serde_json::from_str(reply.stdout.trim()) {
            print_ps_table(&rows);
        }
    }
    Ok(())
}

fn print_gateway_line() {
    if let Some(line) = nginx::format_gateway_line(&goethe_home()) {
        println!("{line}");
    }
}

fn do_login(token_stdin: bool) -> Result<(), String> {
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    let token = if token_stdin {
        hf::read_token_stdin()?
    } else {
        hf::read_token_tty()?
    };
    hf::login(&home, &token)?;
    println!("logged in to Hugging Face");
    Ok(())
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

fn do_logout() -> Result<(), String> {
    let home = goethe_home();
    hf::clear_token(&home)?;
    println!("logged out of Hugging Face");
    Ok(())
}

fn fail(msg: String) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_create_raw_is_only_the_token() {
        let token = "goethe_sk_0123456789abcdef0123456789abcdef";
        assert_eq!(key_create_stdout(token, "ops", true), format!("{token}\n"));
        let human = key_create_stdout(token, "ops", false);
        assert!(human.contains(token));
        assert!(human.contains("Key id:  k_012345"));
        assert!(human.contains("Store this key now"));
        assert_eq!(human.matches(token).count(), 1);
    }
}
