// mesh CLI entrypoint for hardware, install, config, and load commands.

use clap::{Parser, Subcommand};
use mesh_core::alias;
use mesh_core::config::{self, ConfigOpts};
use mesh_core::db;
use mesh_core::gpu_policy;
use mesh_core::hw;
use mesh_core::hf;
use mesh_core::install::{self, InstallOpts};
use mesh_core::models;
use mesh_core::nginx;
use mesh_core::paths::{ensure_layout, mesh_home};
use mesh_core::registry;
use mesh_core::recommend;
use mesh_core::rpc;
use mesh_core::security;
use mesh_core::shutdown;
use serde_json::{json, Value};
use std::io::IsTerminal;

#[derive(Parser)]
#[command(
    name = "mesh",
    disable_version_flag = true,
    about = "MeshApplied inference engine",
    after_help = "Examples:\n  mesh install meshapplied/Qwen3-8B --load\n  mesh load my/model --gpus 2,3 --ctx 65536\n  mesh registry add org/model --file cdn/v1/registry.json\n  mesh key create --name ops --raw\n\nmesh stop stops meshd but keeps desired loads for restore (every desired slot). \
                  mesh unload removes all instances of a model (all slots) and their desired rows."
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
    #[command(about = "Download, verify, and register a model (from Hugging Face or a local path). With --load, waits for RUNNING like mesh load unless --no-wait.")]
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
        #[arg(long, help = "Overwrite an existing install.")]
        force: bool,
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
    #[command(about = "Delete an installed model from disk and the local registry.")]
    Remove {
        #[arg(help = "Installed model id to remove.")]
        model_id: String,
        #[arg(long, help = "Skip confirmation.")]
        yes: bool,
        #[arg(long, help = "Remove even if loaded.")]
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
    #[command(about = "Ask meshd to shut down (desired loads are kept for restore).")]
    Stop,
    #[command(about = "Enable or inspect autostart for meshd.")]
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
    #[command(about = "Tail engine, meshd, or gateway logs.")]
    Logs {
        #[arg(help = "Model id, or omit with --daemon / --gateway.")]
        target: Option<String>,
        #[arg(short = 'f', long, help = "Follow log output.")]
        follow: bool,
        #[arg(short = 'n', long, default_value = "100", help = "Number of lines to print.")]
        lines: usize,
        #[arg(long, help = "Show meshd.log.")]
        daemon: bool,
        #[arg(long, help = "Show gateway log.")]
        gateway: bool,
        #[arg(long, help = "Rotate and prune logs, then exit.")]
        prune: bool,
    },
    #[command(about = "Download and apply a signed mesh or engine update.")]
    Update {
        #[arg(long, help = "Print available versions only.")]
        check: bool,
        #[arg(long, help = "Update SGLang engine packages.")]
        engine: bool,
        #[arg(long, help = "Update ternary runtime packages.")]
        ternary: bool,
        #[arg(long, help = "Skip confirmation.")]
        yes: bool,
    },
    #[command(about = "Remove mesh binaries and optionally delete downloaded models.")]
    Uninstall {
        #[arg(long, help = "Also delete MESH_HOME/models.")]
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
        #[arg(long, help = "Print the raw token once.")]
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
        #[arg(help = "Key id from mesh key list.")]
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
    #[command(about = "Install autostart unit or cron for meshd.")]
    Enable,
    #[command(about = "Remove meshd autostart.")]
    Disable,
    #[command(about = "Show whether autostart is enabled.")]
    Status,
    #[command(about = "Toggle restore of desired loads at meshd start.")]
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

fn parse_gpus(raw: &str) -> Result<Vec<u32>, String> {
    raw.split(',')
        .map(|s| s.trim().parse::<u32>().map_err(|_| format!("invalid gpu index: {s}")))
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && (args[1] == "--version" || args[1] == "-V") {
        println!("{}", mesh_core::build_info::version_line());
        return;
    }
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Hw {
            cmd: HwCommands::Profile { json },
        } => hw::run_profile(json),
        Commands::Hw {
            cmd: HwCommands::PickGpus { tp },
        } => mesh_core::placement::run_pick_gpus(tp),
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
            no_verify,
            name,
            copy,
            sglang_arg,
        } => {
            let gpus_parsed = match gpus.as_deref().map(parse_gpus) {
                Some(Ok(v)) => Some(v),
                Some(Err(e)) => fail(e),
                None => None,
            };
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
        } => models::run_remove(&model_id, yes, force),
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
                Some(Err(e)) => fail(e),
                None => None,
            };
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
        Commands::Unload { model_id, all, force } => do_unload(model_id, all, force),
        Commands::Ps { json } => do_ps(json),
        Commands::Stop => shutdown::run_stop(),
        Commands::Service { cmd } => match cmd {
            ServiceCommands::Enable => mesh_core::service::run_enable(),
            ServiceCommands::Disable => mesh_core::service::run_disable(),
            ServiceCommands::Status => mesh_core::service::run_status(),
            ServiceCommands::Restore { off, on } => {
                if off {
                    mesh_core::autostart::set_restore_enabled(false)
                } else if on {
                    mesh_core::autostart::set_restore_enabled(true)
                } else {
                    Err("usage: mesh service restore --on|--off".to_string())
                }
            }
        },
        Commands::Doctor { json, probe_public } => mesh_core::doctor::run(json, probe_public),
        Commands::Domain { cmd } => match cmd {
            DomainCommands::Add { domain } => match mesh_core::domain::add(&domain) {
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
        } => mesh_core::logs::run_logs(target.as_deref(), follow, lines, daemon, gateway, prune),
        Commands::Update {
            check,
            engine,
            ternary,
            yes,
        } => mesh_core::update::run(mesh_core::update::UpdateOpts {
            check_only: check,
            engine,
            ternary,
            yes,
        }),
        Commands::Uninstall {
            purge_models,
            yes,
        } => mesh_core::uninstall::run(mesh_core::uninstall::UninstallOpts {
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
            } => match security::key_create(
                &name,
                raw,
                rpm,
                expires.as_deref(),
                models.as_deref(),
                admin,
            ) {
                Ok(out) => {
                    println!("{out}");
                    print_gateway_line();
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
    if matches!(args.ep, Some(0)) {
        return Err("--ep must be at least 1".to_string());
    }
    if let (Some(ep), Some(tp)) = (args.ep, args.tp) {
        if ep > 0 && tp % ep != 0 {
            return Err(format!("--tp {tp} is not divisible by --ep {ep}"));
        }
    }
    let mut extra = args.extra;
    if let Some(ep) = args.ep {
        if ep > 1 {
            extra.push("--ep-size".to_string());
            extra.push(ep.to_string());
        }
    }
    let reply = rpc::call(
        "load",
        json!({
            "model_id": args.model_id,
            "gpus": args.gpus,
            "ctx": args.ctx,
            "max_conc": args.max_conc,
            "tp": args.tp,
            "force": args.force,
            "yes": args.yes,
            "is_tty": std::io::stdin().is_terminal(),
            "extra_sglang_args": extra,
            "async_finish": true,
        }),
    )
    .map_err(rpc_daemon_err)?;
    if !reply.stdout.is_empty() {
        print!("{}", reply.stdout);
    }
    if reply.code != 0 {
        if !reply.stderr.is_empty() {
            eprint!("{}", reply.stderr);
        }
        std::process::exit(reply.code as i32);
    }
    mesh_core::load_wait::wait_after_load_stdout(&reply.stdout, args.no_wait)
}

fn rpc_daemon_err(e: String) -> String {
    rpc::map_daemon_err(e)
}

fn do_gpu_list(json: bool) -> Result<(), String> {
    let home = mesh_home();
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
    let home = mesh_home();
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
    let home = mesh_home();
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
    if let Some(line) = nginx::format_gateway_line(&mesh_home()) {
        println!("{line}");
    }
}

fn do_login(token_stdin: bool) -> Result<(), String> {
    let home = mesh_home();
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
    let home = mesh_home();
    hf::clear_token(&home)?;
    println!("logged out of Hugging Face");
    Ok(())
}

fn fail(msg: String) -> ! {
    eprintln!("{msg}");
    std::process::exit(1);
}
