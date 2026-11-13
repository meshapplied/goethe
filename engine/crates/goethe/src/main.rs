// goethe CLI entrypoint for hardware, install, config, and load commands.

mod complete;
use clap::{CommandFactory, Parser, Subcommand};
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
use goethe_core::learn;
use goethe_core::trace;
use serde_json::{json, Value};

const KEY_HELP: &str = "Add, scope or delete keys in a menu:

\t→ goethe api manage

Create a key (goethe api lists it again later):

\t→ goethe api keys --create my-app

With an expiry (days like 30d, or a date like 2026-12-31), a rate limit and admin rights:

\t→ goethe api keys --create ops --expires 90d --rpm 300 --admin

Only some models, or just the key for a script:

\t→ goethe api keys --create bot --models deepseek-ai/DeepSeek-V4.1-Flash-1017
\t→ API_KEY=$(goethe api keys --create ci --raw)

List keys, or revoke one:

\t→ goethe api keys
\t→ goethe api keys --revoke <key-id>";

const LEARN_HELP: &str = "A session keeps what compaction would drop: the user's rules and corrections, facts from chat, the goal, decisions and mistakes. A small note goes into the prompt after each compaction. Sessions are told apart by the client's X-Session-Id header.

Every model:
\t→ goethe api learn on
\t→ goethe api learn off

One model:
\t→ goethe api learn on --model my-model
\t→ goethe api learn off --model my-model

goethe api learn alone shows the current scope. Changes apply to the next request. Details are in docs/learn.md.";

const TRACE_HELP: &str = "Record every generation request and reply, grouped into sessions, as a corpus for training later:

\t→ goethe api trace on

Only some API keys (by name or id), every key but one, or back to every key:

\t→ goethe api trace on --key my-app
\t→ goethe api trace off --key ci
\t→ goethe api trace on --all-keys

goethe api trace off stops recording and keeps the keys chosen; goethe api trace on resumes with them.

Another folder, or a size limit (default 20 GB) at which recording turns itself off:

\t→ goethe api trace on --dir /data/traces --max-gb 200
\t→ goethe api trace --max-gb 100

Stop recording:

\t→ goethe api trace off

goethe api trace alone shows what is recorded and how much space it takes. The folder format is in docs/traces.md.";

#[derive(Parser)]
#[command(
    name = "goethe",
    disable_version_flag = true,
    about = "Goethe: serve LLMs on your NVIDIA GPUs behind one OpenAI-compatible gateway",
    after_help = "Examples:\n\n\t→ goethe up\n\t→ goethe install deepseek-ai/DeepSeek-V4.1-Flash-1017 --load\n\t→ goethe load deepseek-ai/DeepSeek-V4.1-Flash-1017 --gpu gpu0 gpu1 gpu2 gpu3 --ctx 65536\n\t→ goethe api keys --create my-app\n\t→ goethe api\n\t→ goethe down\n\n\
                  goethe down stops goethed but keeps loaded models for the next goethe up; goethe unload stops a model for good."
)]
struct Cli {
    #[arg(short = 'V', long = "version", help = "Print the version, and how to upgrade when a newer one is out (--build-info adds commit and build date).")]
    version: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(display_order = 14, about = "Hardware profile and GPU placement helpers.")]
    Hw {
        #[command(subcommand)]
        cmd: HwCommands,
    },
    #[command(display_order = 13, about = "GPUs with memory, processes and bandwidth, live; allow or deny one.")]
    Gpu {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::gpus), long, num_args = 1.., value_name = "gpuN", value_parser = parse_gpu_name, conflicts_with = "deny", help = "Allow new models on these GPUs again, for example gpu0 gpu1.")]
        allow: Vec<u32>,
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::gpus), long, num_args = 1.., value_name = "gpuN", value_parser = parse_gpu_name, help = "Keep new models off these GPUs (running ones stay), for example gpu0 gpu1.")]
        deny: Vec<u32>,
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
    #[command(hide = true, about = "List catalog models that fit the free VRAM of the allowed GPUs.", alias = "recommended")]
    Recommend(RecommendArgs),
    #[command(display_order = 6, about = "Download a model from Hugging Face, or add a local folder; --load serves it too.")]
    Install {
        #[arg(value_name = "MODEL", num_args = 1.., required_unless_present = "model_id", help = "Hugging Face id (org/name), a local folder, or words from a catalog model's name.")]
        model: Vec<String>,
        #[arg(long = "model", value_name = "ID_OR_PATH", hide = true, conflicts_with = "model", help = "Older form of the model argument.")]
        model_id: Option<String>,
        #[arg(long, help = "Default context length for the model card.")]
        ctx: Option<u64>,
        #[arg(long = "max-conc", help = "Default max concurrent requests.")]
        max_conc: Option<u64>,
        #[arg(long, help = "Tensor parallel size (0 = auto).")]
        tp: Option<u64>,
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::gpus), long = "gpu", num_args = 1.., value_delimiter = ',', value_name = "GPU", value_parser = parse_gpu_name, help = "GPUs to load on after installing, as goethe gpu lists them: --gpu gpu2 gpu3.")]
        gpu: Vec<u32>,
        #[arg(long, hide = true, conflicts_with = "gpu", help = "Comma-separated GPU indices (older form of --gpu).")]
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
    #[command(hide = true, about = "Register model folders you downloaded yourself and moved into GOETHE_HOME/models.")]
    Scan(ScanArgs),
    #[command(hide = true, about = "Validate, generate, or merge registry entries.")]
    Registry {
        #[command(subcommand)]
        cmd: RegistryCommands,
    },
    #[command(display_order = 11, 
        about = "Installed and catalog models; also recommend, scan, config and alias.",
        visible_alias = "list",
        args_conflicts_with_subcommands = true
    )]
    Models {
        #[command(subcommand)]
        cmd: Option<ModelsCommands>,
        #[arg(long, help = "Emit JSON.")]
        json: bool,
        #[arg(long, help = "Only installed models.")]
        installed: bool,
        #[arg(long, help = "Only registry models.")]
        available: bool,
    },
    #[command(display_order = 7, about = "Unload and delete an installed model (a folder of your own is only unregistered).")]
    Remove {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), value_name = "MODEL", required_unless_present = "model_id", help = "Installed models to remove: ids, aliases or words from a name, one argument each.")]
        model: Vec<String>,
        #[arg(long = "model", num_args = 1.., value_name = "ID", hide = true, conflicts_with = "model", help = "Older form of the model argument.")]
        model_id: Vec<String>,
        #[arg(long, help = "Do not ask; also unloads the model if it is loaded.")]
        yes: bool,
        #[arg(long, help = "When unloading first, skip draining in-flight requests.")]
        force: bool,
        #[arg(long, conflicts_with = "keep_traces", help = "Also delete the traces recorded with the model, without asking.")]
        delete_traces: bool,
        #[arg(long, help = "Keep the traces recorded with the model, without asking (the default with --yes).")]
        keep_traces: bool,
    },
    #[command(hide = true, about = "Change default load settings stored on the model card.")]
    Config(ConfigArgs),
    #[command(display_order = 8, 
        about = "Serve an installed model on the GPUs and wait until it answers.",
        after_help = "Extra SGLang flags go after --, for example:\n\n\t→ goethe load deepseek-ai/DeepSeek-V4.1-Flash-1017 --ctx 32768 -- --chunked-prefill-size 4096"
    )]
    Load {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), value_name = "MODEL", num_args = 1.., required_unless_present = "model_id", help = "Installed model: id, alias or words from its name, such as qwen 27b.")]
        model: Vec<String>,
        #[arg(long = "model", value_name = "ID", hide = true, conflicts_with = "model", help = "Older form of the model argument.")]
        model_id: Option<String>,
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::gpus), long = "gpu", num_args = 1.., value_delimiter = ',', value_name = "GPU", value_parser = parse_gpu_name, help = "GPUs to load on, as goethe gpu lists them: --gpu gpu3, or --gpu gpu2 gpu3 to split the model across both (the number of GPUs is the tensor-parallel size).")]
        gpu: Vec<u32>,
        #[arg(long, hide = true, conflicts_with = "gpu", help = "Comma-separated GPU indices (older form of --gpu).")]
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
    #[command(display_order = 9, about = "Stop serving a model.")]
    Unload {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), value_name = "MODEL", conflicts_with = "all", help = "Models to unload (every instance): ids, aliases or words from a name, one argument each.")]
        model: Vec<String>,
        #[arg(long = "model", num_args = 1.., value_name = "ID", hide = true, conflicts_with_all = ["all", "model"], help = "Older form of the model argument.")]
        model_id: Vec<String>,
        #[arg(long, help = "Unload every loaded model.")]
        all: bool,
        #[arg(long, help = "Skip connection draining and tear down immediately.")]
        force: bool,
    },
    #[command(display_order = 5, about = "Running models: GPUs, port, memory, requests and speed.")]
    Ps {
        #[arg(long, help = "JSON rows with instance_id, slot_key, status, restarts.")]
        json: bool,
    },
    #[command(display_order = 2, about = "Start Goethe now and at every boot; models loaded last time come back.")]
    Up {
        #[arg(long = "no-boot", help = "Start now without turning on start at boot.")]
        no_boot: bool,
    },
    #[command(display_order = 4, about = "Whether Goethe runs, the API address and its keys.")]
    Status,
    #[command(display_order = 3, about = "Stop Goethe and its models, and stop starting at boot (models come back on the next up).")]
    Down {
        #[arg(long = "keep-engines", help = "Leave loaded engines serving; the next goethed adopts them.")]
        keep_engines: bool,
    },
    #[command(hide = true, about = "Older form of goethe api configure.")]
    Configure {
        #[command(subcommand)]
        cmd: ConfigureCommands,
    },
    #[command(display_order = 1, about = "Guided setup: install a model, serve it and open the API.")]
    Start,
    #[command(display_order = 12, 
        about = "The API address, keys and models; also configure, manage, keys, allowlist, gateway, domain, learn and trace.",
        args_conflicts_with_subcommands = true
    )]
    Api {
        #[arg(long, help = "JSON with api, goethed_running, keys and models.")]
        json: bool,
        #[command(subcommand)]
        cmd: Option<ApiCommands>,
    },
    #[command(hide = true, about = "Print connection details (same as goethe api).")]
    Get {
        #[command(subcommand)]
        cmd: GetCommands,
    },
    #[command(hide = true, about = "Enable or inspect autostart for goethed.")]
    Service {
        #[command(subcommand)]
        cmd: ServiceCommands,
    },
    #[command(display_order = 15, about = "Check the machine and the setup, with a fix for each problem.")]
    Doctor {
        #[arg(long, help = "Emit JSON.")]
        json: bool,
        #[arg(long = "probe-public", help = "Probe public gateway reachability.")]
        probe_public: bool,
    },
    #[command(hide = true, about = "Show where the public gateway listens, or change it.")]
    Gateway(GatewayArgs),
    #[command(hide = true, about = "How to reach the API by a domain name, and a check that it works.")]
    Domain(DomainArgs),
    #[command(display_order = 10, about = "Model, goethed or gateway logs.")]
    Logs {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), value_name = "MODEL", num_args = 1.., help = "Model (id, alias or words from its name) or instance id whose engine log to show.")]
        model: Vec<String>,
        #[arg(long = "model", value_name = "ID", hide = true, conflicts_with = "model", help = "Older form of the model argument.")]
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
    #[command(display_order = 16, about = "Install a signed update of Goethe or its engine.")]
    Update {
        #[arg(long, help = "Print available versions only.")]
        check: bool,
        #[arg(long, help = "Update the installed engines (SGLang, goethe-engine); one not installed yet is left out.")]
        engine: bool,
        #[arg(long, help = "Update ternary runtime packages.")]
        ternary: bool,
        #[arg(long, help = "Skip confirmation.")]
        yes: bool,
        #[arg(long = "allow-downgrade", help = "Install the published version even if it is older.")]
        allow_downgrade: bool,
    },
    #[command(display_order = 17, about = "Remove Goethe and the models it installed (folders of your own stay).")]
    Uninstall {
        #[arg(long, help = "Keep the installed models in GOETHE_HOME/models.")]
        keep_models: bool,
        #[arg(long, hide = true, help = "Deleting the installed models is the default now.")]
        purge_models: bool,
        #[arg(long, help = "Skip confirmation.")]
        yes: bool,
    },
    #[command(hide = true, about = "List the client addresses allowed to reach the gateway, or add or remove one.")]
    Allowlist(AllowlistArgs),
    #[command(hide = true, about = "List API keys, or create or revoke one.", after_help = KEY_HELP)]
    Key(KeyArgs),
    #[command(hide = true, about = "List model aliases, or set or remove one.")]
    Alias(AliasArgs),
    #[command(hide = true, about = "Let sessions keep what context compaction would drop: turn it on or off, for every model or for some.", after_help = LEARN_HELP)]
    Learn(LearnArgs),
    #[command(hide = true, about = "Record requests and replies as training traces: turn it on or off, per API key, and see the size.", after_help = TRACE_HELP)]
    Trace(TraceArgs),
    #[command(display_order = 19, about = "Link this node to the Mesh platform (optional): a browser login, or --authkey for headless machines.")]
    Login {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::platform_urls), long = "platform-url", value_name = "URL", help = "Platform to link to (default: the GOETHE_PLATFORM_URL setting, else https://platform.meshapplied.com).")]
        platform_url: Option<String>,
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::authkey_stdin), long, value_name = "KEY", help = "Link without a browser using an auth key (msh_ak_...); - reads it from stdin.")]
        authkey: Option<String>,
    },
    #[command(display_order = 20, about = "Unlink this node from the Mesh platform and remove its local credentials.")]
    Logout,
    #[command(display_order = 18, about = "Tab completion for bash, zsh and fish: models, keys, GPUs and interfaces complete too.")]
    Completion {
        #[arg(value_parser = ["install", "remove", "status"], help = "install, remove, or status (the default).")]
        action: Option<String>,
        #[arg(long, value_parser = ["bash", "zsh", "fish"], help = "The shell (default: yours, from $SHELL).")]
        shell: Option<String>,
    },
}

#[derive(Subcommand)]
enum ApiCommands {
    #[command(about = "Add, scope, show or delete API keys in a menu (arrow keys, Enter).")]
    Manage,
    #[command(about = "Choose who can reach the API: local only, a VPN or private network (recommended), or a public network.")]
    Configure(AccessArgs),
    #[command(about = "List API keys, or create or revoke one (for scripts; goethe api manage is the menu).", after_help = KEY_HELP)]
    Keys(KeyArgs),
    #[command(about = "The client addresses allowed to reach the API, or add or remove one.")]
    Allowlist(AllowlistArgs),
    #[command(about = "Where the API listens, or change it.")]
    Gateway(GatewayArgs),
    #[command(about = "How to reach the API by a domain name, and a check that it works.")]
    Domain(DomainArgs),
    #[command(about = "Let sessions keep what context compaction would drop: on or off, for every model or for some.", after_help = LEARN_HELP)]
    Learn(LearnArgs),
    #[command(about = "Record requests and replies as training traces: on or off, per API key, and the size.", after_help = TRACE_HELP)]
    Trace(TraceArgs),
}

#[derive(Subcommand)]
enum ModelsCommands {
    #[command(about = "Catalog models that fit the free VRAM of the allowed GPUs.", alias = "recommended")]
    Recommend(RecommendArgs),
    #[command(about = "Register model folders you downloaded yourself and moved into GOETHE_HOME/models.")]
    Scan(ScanArgs),
    #[command(about = "Change a model's default load settings.")]
    Config(ConfigArgs),
    #[command(about = "Short model names for clients, or set or remove one.")]
    Alias(AliasArgs),
}

#[derive(clap::Args)]
struct RecommendArgs {
    #[arg(long, help = "Emit JSON instead of a table.")]
    json: bool,
}

#[derive(clap::Args)]
struct ScanArgs {
    #[arg(long, value_name = "FOLDER", help = "One folder inside GOETHE_HOME/models (default: all of them).")]
    dir: Option<String>,
    #[arg(long, help = "Add every supported model without asking.")]
    yes: bool,
    #[arg(long, help = "Emit JSON rows (dir, model_id, status, reason, notes).")]
    json: bool,
}

#[derive(clap::Args)]
struct ConfigArgs {
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), value_name = "MODEL", num_args = 1.., required_unless_present = "model_id", help = "Installed model: id, alias or words from its name.")]
    model: Vec<String>,
    #[arg(long = "model", value_name = "ID", hide = true, conflicts_with = "model", help = "Older form of the model argument.")]
    model_id: Option<String>,
    #[arg(long, help = "Default context length.")]
    ctx: Option<u64>,
    #[arg(long = "max-conc", help = "Default max concurrent requests.")]
    max_conc: Option<u64>,
    #[arg(long, help = "Default tensor parallel size.")]
    tp: Option<u64>,
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::gpus), long = "gpu", num_args = 1.., value_delimiter = ',', value_name = "GPU", value_parser = parse_gpu_name, help = "Default GPUs for this model, as goethe gpu lists them: --gpu gpu2 gpu3.")]
    gpu: Vec<u32>,
    #[arg(long, hide = true, conflicts_with = "gpu", help = "Comma-separated GPU indices (older form of --gpu).")]
    gpus: Option<String>,
    #[arg(long = "sglang-arg", allow_hyphen_values = true, help = "Append SGLang flags.")]
    sglang_arg: Vec<String>,
    #[arg(long, help = "Clear stored extra SGLang args.")]
    clear_sglang_args: bool,
}

#[derive(clap::Args)]
struct GatewayArgs {
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::binds), long, value_name = "ADDRESS", help = "Listen on 127.0.0.1 (this host only), 0.0.0.0 (all addresses), one IP of this host, or an interface such as wg0 (its address is followed).")]
    bind: Option<String>,
    #[arg(long, help = "Emit JSON.")]
    json: bool,
}

#[derive(clap::Args)]
struct DomainArgs {
    #[arg(long, value_name = "DOMAIN", help = "Check that the domain points at the API and the gateway answers there.")]
    check: Option<String>,
}

#[derive(clap::Args)]
struct AllowlistArgs {
    #[arg(long, num_args = 1.., value_name = "CIDR", conflicts_with = "remove", help = "Allow these client addresses or ranges, for example 203.0.113.7/32.")]
    add: Vec<String>,
    #[arg(long, value_name = "TEXT", requires = "add", help = "Note stored with the new entries.")]
    description: Option<String>,
    #[arg(long, num_args = 1.., value_name = "CIDR", help = "Remove these addresses or ranges.")]
    remove: Vec<String>,
}

#[derive(clap::Args)]
struct KeyArgs {
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
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::key_ids), long, num_args = 1.., value_name = "KEY_ID", help = "Revoke the keys with these ids (from goethe api keys).")]
    revoke: Vec<String>,
    #[arg(long, help = "Emit JSON.")]
    json: bool,
}

#[derive(clap::Args)]
struct AliasArgs {
    #[arg(long, value_name = "ALIAS", conflicts_with = "remove", requires = "model", help = "Alias name clients may use as the model field.")]
    set: Option<String>,
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), long, value_name = "ID", requires = "set", help = "Installed model id the alias points to.")]
    model: Option<String>,
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::aliases), long, num_args = 1.., value_name = "ALIAS", help = "Remove these aliases.")]
    remove: Vec<String>,
}

#[derive(clap::Args)]
struct LearnArgs {
    #[arg(value_parser = ["on", "off", "status"], help = "on, off, or status (the default).")]
    action: Option<String>,
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models_or_all), long, num_args = 1.., value_name = "MODEL", help = "Only these models (id or alias); all, or leaving it out, means every model.")]
    model: Vec<String>,
    #[arg(long, hide = true, value_name = "SESSION", conflicts_with_all = ["action", "model"])]
    print: Option<String>,
    #[arg(long, help = "Emit JSON.")]
    json: bool,
}

#[derive(clap::Args)]
struct TraceArgs {
    #[arg(value_parser = ["on", "off", "status"], help = "on, off, or status (the default).")]
    action: Option<String>,
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::key_names), long, num_args = 1.., value_name = "KEY", help = "Add these API keys (name or id) with on, or leave them out with off.")]
    key: Vec<String>,
    #[arg(long = "all-keys", conflicts_with = "key", help = "With on: record every API key, clearing the chosen keys and exceptions.")]
    all_keys: bool,
    #[arg(long, value_name = "PATH", help = "Folder for the traces (default ~/.goethe/traces).")]
    dir: Option<String>,
    #[arg(long = "max-gb", value_name = "GB", help = "Size limit: recording turns off when the folder reaches it (default 20).")]
    max_gb: Option<f64>,
    #[arg(long, help = "Emit JSON.")]
    json: bool,
}


#[derive(Subcommand)]
enum ConfigureCommands {
    #[command(about = "Older form of goethe api configure.")]
    Api(AccessArgs),
}

#[derive(clap::Args)]
struct AccessArgs {
    #[arg(long, help = "local, vpn or public (asked on a terminal when omitted).")]
    mode: Option<String>,
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::interfaces), long, value_name = "NAME", conflicts_with = "vpn_ip", help = "Network interface to serve, e.g. wg0 (any VPN or private network); the gateway follows its address.")]
    interface: Option<String>,
    #[arg(long = "vpn-ip", help = "A fixed address of this machine to listen on, instead of following an interface.")]
    vpn_ip: Option<std::net::IpAddr>,
    #[arg(long = "allow", help = "Client CIDR to allow (repeatable; default for vpn: the ranges the network reaches).")]
    allow: Vec<String>,
    #[arg(long, help = "Apply without questions (public mode still refuses 0.0.0.0/0).")]
    yes: bool,
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
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), long, help = "Installed model id whose VRAM budget to place.")]
        model: Option<String>,
    },
}

fn usage(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

fn model_query(words: Vec<String>, flag: Option<String>) -> String {
    // The model as typed: the plain words joined, or the older --model value.
    match flag {
        Some(f) if words.is_empty() => f,
        _ => words.join(" "),
    }
}

fn pick_model(query: &str, hits: Vec<String>, what: &str) -> Result<Option<String>, String> {
    // One match is used; several are offered in a menu on a terminal and listed otherwise; none is None.
    match hits.len() {
        0 => Ok(None),
        1 => {
            if hits[0] != query {
                eprintln!("using {what} {}", hits[0]);
            }
            Ok(hits.into_iter().next())
        }
        _ if std::io::IsTerminal::is_terminal(&std::io::stdin()) && std::io::IsTerminal::is_terminal(&std::io::stderr()) => {
            eprintln!("{query} matches {} {what}s; which one?", hits.len());
            match goethe_core::term::select(&hits, 0)? {
                Some(i) => Ok(Some(hits[i].clone())),
                None => Err("cancelled".to_string()),
            }
        }
        _ => Err(format!("{query} matches several {what}s: {}; name one of them", hits.join(", "))),
    }
}

fn installed_id(id: &str, strict: bool) -> String {
    // The installed model the user meant: exact id, any case, folder name or alias, else every model whose id has all the words; strict callers stop when nothing matches.
    let home = goethe_home();
    let exact = (validate_model_id(id).is_ok()).then(|| models::resolve_installed_id(&home, id));
    let near = match exact {
        Some(Ok(found)) => {
            if found != id {
                eprintln!("using installed model {found}");
            }
            return found;
        }
        Some(Err(e)) => Some(e),
        None => None,
    };
    if let Some((_, target)) = alias::list().unwrap_or_default().into_iter().find(|(a, _)| a.eq_ignore_ascii_case(id)) {
        eprintln!("using installed model {target} (alias {id})");
        return target;
    }
    match pick_model(id, models::word_matches(id, &models::installed_ids(&home)), "installed model") {
        Ok(Some(found)) => found,
        Ok(None) if strict => {
            eprintln!("{}", near.unwrap_or_else(|| format!("no installed model matches {id} (goethe models lists them)")));
            std::process::exit(1);
        }
        Ok(None) => id.to_string(),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(2);
        }
    }
}

fn install_target(query: &str) -> String {
    // A local folder or an org/name id as given; otherwise words from a catalog or installed model's name.
    if std::path::Path::new(query).is_dir() || query.starts_with(['.', '/', '~']) || validate_model_id(query).is_ok() {
        return query.to_string();
    }
    let mut ids = models::installed_ids(&goethe_home());
    if let Ok(reg) = goethe_core::registry::fetch(&goethe_core::registry::registry_url()) {
        ids.extend(reg.models.into_iter().map(|m| m.id));
    }
    match pick_model(query, models::word_matches(query, &ids), "catalog model") {
        Ok(Some(id)) => id,
        Ok(None) => usage(format!("no catalog model matches {query}; give a Hugging Face id such as org/name (goethe models lists the catalog)")),
        Err(e) => usage(e),
    }
}

fn corrected_command(args: &[String], stray: &str) -> Option<String> {
    // The command with the flag a stray value belongs to (--verb for "verb value" forms, --check, --bind, --dir), in its grouped form.
    let nested = matches!(args.get(1).map(String::as_str), Some("api" | "models")) && args.len() > 2;
    let (start, cmd) = if nested { (3, args[2].as_str()) } else { (2, args.get(1)?.as_str()) };
    let at = args.iter().position(|a| a == stray).filter(|&i| i >= start && !stray.starts_with('-'))?;
    let flag = match (cmd, stray) {
        ("scan", _) => "--dir".to_string(),
        ("domain", "check") | ("gateway", "bind") => format!("--{stray}"),
        ("domain", _) => "--check".to_string(),
        ("gateway", _) => "--bind".to_string(),
        ("gpu", "allow" | "deny") | ("key" | "keys", "create" | "revoke") | ("allowlist" | "alias", "add" | "remove" | "set") => {
            format!("--{stray}")
        }
        _ => return None,
    };
    let mut out: Vec<String> = match (nested, cmd) {
        (true, _) => args[1..3].to_vec(),
        (false, "key") => vec!["api".into(), "keys".into()],
        (false, "allowlist" | "gateway" | "domain") => vec!["api".into(), cmd.into()],
        (false, "scan" | "alias") => vec!["models".into(), cmd.into()],
        (false, _) => vec![cmd.into()],
    };
    out.extend(args[start..at].iter().cloned());
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

fn gpu_list(gpu: Vec<u32>, gpus: Option<String>) -> Option<Vec<u32>> {
    // --gpu gpu2 gpu3 (or the older --gpus 2,3) as indices; None when neither was given.
    if !gpu.is_empty() {
        return Some(gpu);
    }
    let raw = gpus?;
    Some(raw.split(',').map(|s| parse_gpu_name(s).unwrap_or_else(|e| usage(e))).collect())
}

fn print_version() {
    // The version; on a terminal also whether a newer one is published, with the command to get it.
    println!("{}", goethe_core::build_info::version_line());
    if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        if let Some(newer) = goethe_core::update::newer_published() {
            println!("\nUpdate available: goethe {newer}\n\n{}", goethe_core::term::command("goethe update"));
        }
    }
}

fn main() {
    clap_complete::CompleteEnv::with_factory(Cli::command).bin("goethe").complete();
    let args: Vec<String> = std::env::args().collect();
    if args.len() == 2 && (args[1] == "--version" || args[1] == "-V") {
        print_version();
        return;
    }
    if args.len() == 2 && args[1] == "--build-info" {
        println!("{}", goethe_core::build_info::build_line());
        return;
    }
    if args.len() == 1 && std::io::IsTerminal::is_terminal(&std::io::stdin()) && std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        if let Err(e) = goethe_core::home::run() {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
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
        print_version();
        return;
    }
    let result = match route(cli.command) {
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
        Commands::Recommend(RecommendArgs { json }) => recommend::run(json),
        Commands::Install {
            model,
            model_id,
            ctx,
            max_conc,
            tp,
            gpu,
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
            let gpus_parsed = gpu_list(gpu, gpus);
            let model_id = install_target(&model_query(model, model_id));
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
        Commands::Scan(ScanArgs { dir, yes, json }) => goethe_core::scan::run(goethe_core::scan::ScanOpts { yes, json, dir }),
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
            cmd: None,
        } => models::run_list(json, installed, available),
        Commands::Models { cmd: Some(_), .. } => unreachable!("goethe models subcommands are routed first"),
        Commands::Remove {
            model,
            model_id,
            yes,
            force,
            delete_traces,
            keep_traces,
        } => {
            let traces = (delete_traces || keep_traces).then_some(delete_traces);
            each(if model.is_empty() { model_id } else { model }, |id| models::run_remove(&installed_id(&id, false), yes, force, traces))
        }
        Commands::Config(ConfigArgs {
            model,
            model_id,
            ctx,
            max_conc,
            tp,
            gpu,
            gpus,
            sglang_arg,
            clear_sglang_args,
        }) => {
            let gpus_parsed = gpu_list(gpu, gpus);
            if let Some(msg) = goethe_core::sglang_infer::reserved_flag_error(&sglang_arg) {
                usage(msg);
            }
            let model_id = installed_id(&model_query(model, model_id), true);
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
            model,
            model_id,
            gpu,
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
            model_id: installed_id(&model_query(model, model_id), true),
            gpus: gpu_list(gpu, gpus).map(|v| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",")),
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
        Commands::Unload { model, model_id, all, force } => {
            let models = if model.is_empty() { model_id } else { model };
            match (models.is_empty(), all) {
                (true, false) => usage("name a model to unload (goethe unload qwen-27b), or give --all"),
                (true, true) => do_unload(None, force),
                _ => each(models, |id| do_unload(Some(&installed_id(&id, false)), force)),
            }
        }
        Commands::Ps { json } => do_ps(json),
        Commands::Up { no_boot } => do_up(!no_boot),
        Commands::Status => do_status(),
        Commands::Get { cmd: GetCommands::Api { json } } | Commands::Api { json, cmd: None } => do_api(json),
        Commands::Api { cmd: Some(ApiCommands::Manage), .. } => goethe_core::keys_menu::run(),
        Commands::Api { cmd: Some(ApiCommands::Configure(a)), .. } => configure_access(a),
        Commands::Api { cmd: Some(_), .. } => unreachable!("goethe api subcommands are routed first"),
        Commands::Start => goethe_core::onboard::run(),
        Commands::Login { platform_url, authkey } => goethe_core::platform_cli::login(platform_url.as_deref(), authkey.as_deref()),
        Commands::Logout => goethe_core::platform_cli::logout(),
        Commands::Completion { action, shell } => complete::run(action.as_deref(), shell.as_deref()),
        Commands::Configure { cmd: ConfigureCommands::Api(a) } => {
            eprintln!("note: goethe configure api is now goethe api configure");
            configure_access(a)
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
        Commands::Gateway(GatewayArgs { bind, json }) => match bind {
            Some(addr) => do_gateway_bind(&addr),
            None => do_gateway_status(json),
        },
        Commands::Domain(DomainArgs { check }) => match check {
            Some(domain) => goethe_core::domain::check(&domain).map(|text| print!("{text}")),
            None => goethe_core::domain::explain().map(|text| print!("{text}")),
        },
        Commands::Logs {
            model,
            target,
            follow,
            lines,
            daemon,
            gateway,
            prune,
        } => {
            let target = if model.is_empty() { target } else { Some(model.join(" ")) };
            let target = target.map(|t| if t.starts_with("inst-") { t } else { installed_id(&t, false) });
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
        Commands::Allowlist(AllowlistArgs { add, description, remove }) => match (add.is_empty(), remove.is_empty()) {
            (false, _) => each(add, |cidr| {
                security::allowlist_add(&cidr, description.as_deref())
                    .map(|()| println!("Allowed {}", security::canonical_cidr(&cidr).unwrap_or(cidr)))
            }),
            (_, false) => each(remove, |cidr| {
                security::allowlist_remove(&cidr).map(|()| println!("Removed {}", security::canonical_cidr(&cidr).unwrap_or(cidr)))
            }),
            _ => security::allowlist_list().map(|rows| print!("{}", allowlist_text(&rows))),
        },
        Commands::Key(KeyArgs { create: Some(name), expires, rpm, models, admin, raw, .. }) => {
            if name.trim().is_empty() {
                usage("--create needs a key name");
            }
            if let Some(Err(e)) = models.as_deref().map(security::model_scope_list) {
                usage(e);
            }
            security::key_create(&name, true, rpm, expires.as_deref(), models.as_deref(), admin)
                .map(|token| print!("{}", key_create_stdout(&token, &key_id_for_token(&token), &name, raw)))
        }
        Commands::Key(KeyArgs { revoke, .. }) if !revoke.is_empty() => {
            each(revoke, |id| security::key_revoke(&id).map(|()| println!("Revoked key {id}")))
        }
        Commands::Key(KeyArgs { json, .. }) => security::key_list_detailed().map(|rows| {
            if json {
                println!("{}", serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into()));
            } else {
                print!("{}", key_list_text(&rows));
            }
        }),
        Commands::Alias(AliasArgs { set: Some(a), model: Some(model), .. }) => {
            alias::set(&a, &model).map(|()| println!("Alias {a} now points to {model}"))
        }
        Commands::Alias(AliasArgs { remove, .. }) if !remove.is_empty() => {
            each(remove, |a| alias::remove(&a).map(|()| println!("Removed alias {a}")))
        }
        Commands::Alias(AliasArgs { .. }) => alias::list().map(|rows| print!("{}", alias_text(&rows))),
        Commands::Trace(TraceArgs {
            action,
            key,
            all_keys,
            dir,
            max_gb,
            json,
        }) => do_trace(action.as_deref(), &key, all_keys, dir.as_deref(), max_gb, json),
        Commands::Learn(LearnArgs { action, model, print, json }) => do_learn(action.as_deref(), &model, print.as_deref(), json),
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
    goethe_core::term::commands("Create your first API key:", &["goethe api manage", "goethe api keys --create my-app"])
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
            &["goethe api allowlist --add <client-ip>/32"],
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
        return goethe_core::term::commands("No aliases yet. Set one with:", &["goethe models alias --set <alias> --model <model-id>"]);
    }
    let mut table = goethe_core::term::Table::new(&["ALIAS", "MODEL"]);
    for (a, m) in rows {
        table.row(vec![a.clone(), m.clone()]);
    }
    table.render()
}

fn do_learn(action: Option<&str>, models: &[String], print: Option<&str>, json: bool) -> Result<(), String> {
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    if let Some(session) = print {
        println!("{}", learn::print_session(&home, session, json)?);
        return Ok(());
    }
    match action {
        Some(a @ ("on" | "off")) => {
            let small = learn::set(&home, a == "on", models)?;
            for (id, ctx) in &small {
                eprintln!("Warning: {id} has a context of {ctx} tokens. Under {} tokens, learning can cause more compactions and use more input tokens (docs/learn.md).", learn::SMALL_CONTEXT);
            }
            if a == "on" && !json {
                eprintln!("Notes hold text from your sessions; only your user can read the folder. Updates are extra model calls, which no API key is charged for.");
            }
        }
        _ if !models.is_empty() => usage("--model goes with on or off"),
        _ => {}
    }
    let st = learn::status(&home)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&st).unwrap_or_default());
    } else {
        print!("{}", learn_text(&st));
    }
    Ok(())
}

fn learn_text(st: &Value) -> String {
    let mut out = String::new();
    for (name, value) in learn::status_rows(st) {
        out.push_str(&format!("{name:<9}{value}\n"));
    }
    for row in st["recent"].as_array().into_iter().flatten() {
        out.push_str(&format!("  {}  {} entries, {} compactions, {}\n", row["session"].as_str().unwrap_or(""), row["entries"], row["compactions"], row["updated_at"].as_str().unwrap_or("")));
    }
    if st["on"].as_bool() != Some(true) {
        out.push_str(&goethe_core::term::commands("Turn it on with:", &["goethe api learn on"]));
    }
    out
}

fn do_trace(action: Option<&str>, keys: &[String], all_keys: bool, dir: Option<&str>, max_gb: Option<f64>, json: bool) -> Result<(), String> {
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    match action {
        Some("off") if all_keys => usage("--all-keys goes with on"),
        Some(a @ ("on" | "off")) => {
            trace::set(&home, Some(a == "on"), keys, all_keys, dir, max_gb)?;
            if a == "on" && !json {
                eprintln!("Traces hold whole prompts and replies, including anything secret clients send; only your user can read the folder.");
            }
        }
        _ if !keys.is_empty() || all_keys => usage("--key and --all-keys go with on or off"),
        _ if dir.is_some() || max_gb.is_some() => trace::set(&home, None, keys, false, dir, max_gb)?,
        _ => {}
    }
    let st = trace::status(&home)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&st).unwrap_or_default());
    } else {
        print!("{}", trace_text(&st));
    }
    Ok(())
}

fn trace_text(st: &Value) -> String {
    let names = |v: &Value| -> String {
        v.as_array()
            .into_iter()
            .flatten()
            .map(|k| match k["name"].as_str().filter(|n| !n.is_empty()) {
                Some(n) => format!("{n} ({})", k["id"].as_str().unwrap_or("")),
                None => k["id"].as_str().unwrap_or("").to_string(),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let scope = if st["all_keys"].as_bool().unwrap_or(false) {
        match names(&st["except"]) {
            e if e.is_empty() => "every API key".to_string(),
            e => format!("every API key except {e}"),
        }
    } else {
        match names(&st["keys"]) {
            k if k.is_empty() => String::new(),
            k => format!("only {k}"),
        }
    };
    let recording = if st["recording"].as_bool().unwrap_or(false) {
        format!("on, {scope}")
    } else {
        let why = st["stopped"].as_str().map(|w| format!(", turned off automatically: {w}")).unwrap_or_default();
        let resumes = if scope.is_empty() { String::new() } else { format!("; on resumes {scope}") };
        format!("off{why}{resumes}")
    };
    let mut rows = vec![("Recording", recording), ("Folder", st["dir"].as_str().unwrap_or("").to_string())];
    let free = st["free_bytes"].as_u64().map(|b| format!(", {} free on the disk", trace::human_bytes(b))).unwrap_or_default();
    rows.push(("Size", format!("{} of a {} GB limit{free}", trace::human_bytes(st["bytes"].as_u64().unwrap_or(0)), st["max_gb"])));
    let count = |v: &Value, word: &str| match v.as_u64().unwrap_or(0) {
        1 => format!("1 {word}"),
        n => format!("{n} {word}s"),
    };
    rows.push((
        "Sessions",
        format!("{}, {}, {}", count(&st["sessions"], "session"), count(&st["turns"], "turn"), count(&st["models"], "model version")),
    ));
    if let Some(t) = st["last_turn_at"].as_str() {
        rows.push(("Last turn", t.to_string()));
    }
    if let Some(p) = st["paused"].as_str().filter(|_| st["recording"].as_bool().unwrap_or(false)) {
        rows.push(("Paused", format!("{p}; recording resumes when space is freed")));
    }
    let d = &st["dropped"];
    let dropped: u64 = ["queue_full", "paused", "errors"].iter().filter_map(|k| d[*k].as_u64()).sum();
    if dropped > 0 {
        rows.push(("Dropped", format!("{dropped} (queue full {}, while paused {}, write errors {})", d["queue_full"], d["paused"], d["errors"])));
    }
    let mut out: String = rows.iter().map(|(k, v)| format!("{k:<11} {v}\n")).collect();
    if st["stopped"].is_string() {
        let more = format!("goethe api trace on --max-gb {}", (st["max_gb"].as_f64().unwrap_or(trace::DEFAULT_MAX_GB) * 2.0).ceil());
        out.push_str(&goethe_core::term::commands("Raise the limit and turn it back on, or delete traces first:", &[&more]));
    } else if !st["recording"].as_bool().unwrap_or(false) {
        out.push_str(&goethe_core::term::commands("Turn it on with:", &["goethe api trace on"]));
    }
    out
}

fn rpc_daemon_err(e: String) -> String {
    goethe_core::paths::daemon_err(e)
}

fn do_gpu_list(json: bool) -> Result<(), String> {
    // JSON once, or the table redrawn every half second on a terminal until q.
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    if json {
        let con = db::open(&home).map_err(|e| e.to_string())?;
        let rows = gpu_policy::list_rows(&con, &home)?;
        println!("{}", serde_json::to_string_pretty(&rows).map_err(|e| e.to_string())?);
        return Ok(());
    }
    goethe_core::term::live(std::time::Duration::from_millis(500), || {
        let con = db::open(&home).map_err(|e| e.to_string())?;
        Ok(gpu_list_text(&gpu_policy::list_rows(&con, &home)?))
    })
}

fn gpu_list_text(rows: &[gpu_policy::GpuListRow]) -> String {
    // VRAM, policy, process ids (goethe's green, others red) and memory bandwidth use per GPU, then unsupported GPUs.
    use goethe_core::term::{bar, paint, Color};
    let mut table = goethe_core::term::Table::new(&["GPU", "NAME", "VRAM TOTAL", "GOETHE", "FOREIGN", "STATUS", "PROCESSES", "BANDWIDTH"]);
    let gb = |mb: u64| format!("{:.1} GB", mb as f64 / 1024.0);
    for r in rows {
        let pids: Vec<String> = r
            .processes
            .iter()
            .map(|p| paint(&p.pid.to_string(), if p.goethe { Color::Green } else { Color::Red }))
            .collect();
        table.row(vec![
            format!("gpu{}", r.index),
            r.name.clone(),
            gb(r.vram_total_mb),
            gb(r.managed_mb),
            gb(r.foreign_mb),
            goethe_core::term::status(&r.status),
            if pids.is_empty() { "-".to_string() } else { pids.join(" ") },
            r.memory_util_percent.map_or("-".to_string(), bar),
        ]);
    }
    let mut out = table.render();
    let notes: Vec<String> = rows
        .iter()
        .filter_map(|r| goethe_core::gpu_arch::unsupported_reason(&r.compute_capability).map(|why| format!("gpu{}: {why}", r.index)))
        .collect();
    if !notes.is_empty() {
        out.push_str("\nGoethe will not place models on these GPUs:\n\n");
        notes.iter().for_each(|n| out.push_str(&format!("\t{n}\n")));
    }
    out
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
        "INSTANCE", "MODEL", "SLOT", "STATUS", "GPUS", "PORT", "VRAM_GB", "RESTARTS", "IN_FLIGHT", "TOK_PER_S", "PREFIX",
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
        let prefix = match (
            row.get("prefix_cache_hit_tokens").and_then(|v| v.as_u64()),
            row.get("prompt_tokens").and_then(|v| v.as_u64()),
        ) {
            (Some(hit), Some(prompt)) if prompt > 0 => format!("{:.0}%", 100.0 * hit as f64 / prompt as f64),
            _ => "-".to_string(),
        };
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
            prefix,
        ]);
    }
    table.print();
}

fn no_models_text() -> String {
    goethe_core::term::commands("No models loaded. Pick one and load it:", &["goethe models", "goethe load <model-id>"])
}

fn empty_allowlist_text() -> String {
    goethe_core::term::commands(
        "The allowlist is empty, so every client is refused. Allow one with:",
        &["goethe api allowlist --add <client-ip>/32"],
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

fn route(command: Commands) -> Commands {
    // goethe api ... and goethe models ... map onto the commands that run them; the older top-level forms still work and say where they moved.
    let moved = |old: &str, new: &str| eprintln!("note: goethe {old} is now goethe {new}");
    match command {
        Commands::Api { cmd: Some(ApiCommands::Keys(a)), .. } => Commands::Key(a),
        Commands::Api { cmd: Some(ApiCommands::Allowlist(a)), .. } => Commands::Allowlist(a),
        Commands::Api { cmd: Some(ApiCommands::Gateway(a)), .. } => Commands::Gateway(a),
        Commands::Api { cmd: Some(ApiCommands::Domain(a)), .. } => Commands::Domain(a),
        Commands::Api { cmd: Some(ApiCommands::Learn(a)), .. } => Commands::Learn(a),
        Commands::Api { cmd: Some(ApiCommands::Trace(a)), .. } => Commands::Trace(a),
        Commands::Models { cmd: Some(ModelsCommands::Recommend(a)), .. } => Commands::Recommend(a),
        Commands::Models { cmd: Some(ModelsCommands::Scan(a)), .. } => Commands::Scan(a),
        Commands::Models { cmd: Some(ModelsCommands::Config(a)), .. } => Commands::Config(a),
        Commands::Models { cmd: Some(ModelsCommands::Alias(a)), .. } => Commands::Alias(a),
        c => {
            match &c {
                Commands::Key(_) => moved("key", "api keys"),
                Commands::Allowlist(_) => moved("allowlist", "api allowlist"),
                Commands::Gateway(_) => moved("gateway", "api gateway"),
                Commands::Domain(_) => moved("domain", "api domain"),
                Commands::Learn(_) => moved("learn", "api learn"),
                Commands::Trace(_) => moved("trace", "api trace"),
                Commands::Recommend(_) => moved("recommend", "models recommend"),
                Commands::Scan(_) => moved("scan", "models scan"),
                Commands::Config(_) => moved("config", "models config"),
                Commands::Alias(_) => moved("alias", "models alias"),
                _ => {}
            }
            c
        }
    }
}

fn configure_access(a: AccessArgs) -> Result<(), String> {
    let mode = a.mode.as_deref().map(goethe_core::configure::Mode::parse).transpose().unwrap_or_else(|e| usage(e));
    goethe_core::configure::run_api(goethe_core::configure::ApiOpts { mode, vpn_ip: a.vpn_ip, interface: a.interface, allow: a.allow, yes: a.yes })
}

fn do_gateway_bind(addr: &str) -> Result<(), String> {
    // An address, or an interface name whose address the gateway follows; persist, apply, and remind about the allowlist.
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    let loopback = match addr.trim().parse::<std::net::IpAddr>() {
        Ok(ip) => {
            println!("{}", nginx::set_public_bind(&home, &ip.to_string())?);
            ip.is_loopback()
        }
        Err(_) if goethe_core::netif::valid_name(addr.trim()) => {
            println!("{}", nginx::set_public_bind_interface(&home, addr.trim())?);
            false
        }
        Err(_) => usage(format!("invalid address {addr:?}: use 0.0.0.0, 127.0.0.1, an IP address of this host or an interface name")),
    };
    if !loopback && security::allowlist_list()?.is_empty() {
        print!("\n{}", empty_allowlist_text());
    }
    Ok(())
}

fn do_gateway_status(json: bool) -> Result<(), String> {
    // Bind, port, local URL, whether our nginx serves, allowlist size and provider endpoint hints.
    let home = goethe_home();
    let bind = nginx::public_bind(&home);
    let interface = nginx::bind_interface(&home);
    let waiting = nginx::bind_waiting(&home).is_some();
    let port = nginx::public_port_u16(&home);
    let url = format!("https://{}", nginx::gateway_connect_addr(&home));
    let running = nginx::format_gateway_line(&home).is_some();
    let allowlist = security::allowlist_list()?.len();
    let endpoints = goethe_core::host::probe_environment(port).public_endpoints;
    if json {
        let row = json!({"bind": bind, "interface": interface, "waiting": waiting, "port": port, "url": url, "running": running,
            "allowlist_entries": allowlist, "endpoints": endpoints});
        println!("{row}");
        return Ok(());
    }
    let bind_line = match (&interface, waiting) {
        (Some(n), true) => format!("127.0.0.1 (waiting for {n}: it has no IPv4 address, so only this host can connect)"),
        (Some(n), false) => format!("{bind} (follows {n})"),
        (None, _) => bind.clone(),
    };
    print!("{}", gateway_status_text(&bind_line, port, &url, running, allowlist, &endpoints));
    Ok(())
}

fn gateway_status_text(bind: &str, port: u16, url: &str, running: bool, allowlist: usize, endpoints: &[String]) -> String {
    // Bind and who can reach it, port, local URL, state, allowlist size, then provider endpoint hints.
    let loopback = bind.split_whitespace().next().and_then(|b| b.parse::<std::net::IpAddr>().ok()).is_none_or(|ip| ip.is_loopback());
    let scope = if loopback {
        "this host only; `goethe api gateway --bind 0.0.0.0` accepts remote clients"
    } else {
        "remote clients reach it when allowlisted"
    };
    let state = if running { "running" } else { "not running (goethed starts it; see goethe logs --gateway)" };
    let list = match allowlist {
        0 => "empty, every client is refused (goethe api allowlist --add <client-ip>/32)".to_string(),
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
        print_platform_lines(&home, false);
        return Ok(());
    }
    let api = match nginx::format_gateway_line(&home) {
        Some(_) => match nginx::bind_waiting(&home) {
            Some(n) => format!("https://{}/v1 (waiting for {n}; only this machine can connect until it has an address)", nginx::gateway_connect_addr(&home)),
            None => format!("https://{}/v1", nginx::gateway_connect_addr(&home)),
        },
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
    print_platform_lines(&home, true);
    Ok(())
}

fn print_platform_lines(home: &std::path::Path, running: bool) {
    // The platform link and the last heartbeat, after the rest of goethe status.
    println!("\n{}", goethe_core::platform_cli::status_lines(home, running).join("\n"));
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
    println!("SSL_CERT_FILE={}", cert_path());
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

fn cert_path() -> String {
    goethe_home().join("nginx/certs/server.crt").display().to_string()
}

fn curl_example(api: &str) -> String {
    format!(
        "curl --cacert {} {api}/chat/completions -H \"Authorization: Bearer <your_bearer_token>\" -H \"Content-Type: application/json\" \
         -d '{{\"model\": \"<model_name>\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'",
        cert_path()
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
        assert!(local.starts_with("bind: 127.0.0.1 (this host only; `goethe api gateway --bind 0.0.0.0`"), "{local}");
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
        assert!(fresh.contains("running\n\nAPI: https://10.0.0.2:8443/v1\n\nCreate your first API key:\n\n\t→ goethe api manage\n\t→ goethe api keys --create my-app\n"), "{fresh}");
        assert!(fresh.ends_with("\nFor more details run:\n\n\t→ goethe api\n"), "{fresh}");
        let keyed = status_text(Some("https://h/v1"), &[("ops".into(), "a1b2c3d4e5f6".into())]);
        assert!(keyed.contains("ops") && keyed.contains("a1b2c3d4e5f6") && !keyed.contains("Create your first"), "{keyed}");
    }

    #[test]
    fn empty_lists_say_how_to_add_the_first_entry() {
        assert!(key_list_text(&[]).ends_with("\t→ goethe api manage\n\t→ goethe api keys --create my-app\n"));
        assert!(allowlist_text(&[]).ends_with("\t→ goethe api allowlist --add <client-ip>/32\n"));
        assert!(alias_text(&[]).ends_with("\t→ goethe models alias --set <alias> --model <model-id>\n"));
    }

    #[test]
    fn every_user_value_is_a_flag() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
        let parse = |v: &[&str]| Cli::try_parse_from(std::iter::once("goethe").chain(v.iter().copied()));
        assert!(parse(&["domain", "--check", "api.example.com"]).is_ok());
        assert!(parse(&["domain", "add", "api.example.com"]).is_err() && parse(&["domain", "--add", "x.y"]).is_err());
        assert!(parse(&["load", "--model", "org/m", "--gpus", "2"]).is_ok());
        assert!(parse(&["load", "--model", "org/m", "--gpu", "gpu2", "gpu3"]).is_ok());
        assert!(parse(&["load", "--model", "org/m", "--gpu", "gpu2,gpu3", "--ctx", "4096"]).is_ok());
        assert!(parse(&["load", "--model", "org/m", "--gpu", "gpux"]).is_err());
        assert!(parse(&["load", "--model", "org/m", "--gpu", "gpu2", "--gpus", "3"]).is_err());
        assert_eq!(gpu_list(vec![2, 3], None), Some(vec![2, 3]));
        assert_eq!(gpu_list(Vec::new(), Some("gpu1,0".into())), Some(vec![1, 0]));
        assert_eq!(gpu_list(Vec::new(), None), None);
        let Commands::Load { model, extra, .. } = parse(&["load", "qwen", "27b", "--ctx", "4096", "--", "--chunked-prefill-size", "8"]).unwrap().command else { panic!() };
        assert_eq!((model_query(model, None), extra), ("qwen 27b".to_string(), vec!["--chunked-prefill-size".to_string(), "8".to_string()]), "words are the model, -- starts engine flags");
        assert!(parse(&["load", "org/m", "--model", "org/n"]).is_err(), "one model, not both forms");
        assert!(parse(&["load"]).is_err() && parse(&["install"]).is_err());
        assert!(parse(&["key", "--create", "ops", "--expires", "30d", "--admin"]).is_ok());
        assert!(parse(&["key", "--admin"]).is_err(), "--admin only goes with --create");
        assert!(parse(&["key", "--create", "a", "--revoke", "b"]).is_err());
        assert!(parse(&["gpu", "--allow", "gpu2"]).is_ok() && parse(&["alias", "--set", "fast"]).is_err());
        let Commands::Gpu { allow, .. } = parse(&["gpu", "--allow", "gpu0", "gpu1"]).unwrap().command else { panic!() };
        assert_eq!(allow, vec![0, 1]);
        assert!(parse(&["key", "--revoke", "a1", "b2"]).is_ok() && parse(&["allowlist", "--add", "10.0.0.1/32", "10.0.0.2/32"]).is_ok());
        assert!(parse(&["unload", "--model", "org/a", "org/b", "--force"]).is_ok() && parse(&["remove"]).is_err());
        let Commands::Unload { model, .. } = parse(&["unload", "qwen-27b", "org/b"]).unwrap().command else { panic!() };
        assert_eq!(model, ["qwen-27b", "org/b"], "unload and remove take one model per argument");
        assert!(parse(&["unload", "org/a", "--all"]).is_err());
        assert_eq!(parse_gpu_name("gpu3"), Ok(3));
        assert_eq!(parse_gpu_name("GPU0"), Ok(0));
        assert!(parse_gpu_name("gpux").is_err());
    }

    #[test]
    fn grouped_commands_route_to_the_same_handlers_and_old_forms_still_parse() {
        let parse = |v: &[&str]| Cli::try_parse_from(std::iter::once("goethe").chain(v.iter().copied())).map(|c| route(c.command));
        assert!(matches!(parse(&["api", "keys", "--create", "ops"]), Ok(Commands::Key(KeyArgs { create: Some(_), .. }))));
        assert!(matches!(parse(&["key", "--create", "ops"]), Ok(Commands::Key(KeyArgs { create: Some(_), .. }))), "the old form still works");
        assert!(matches!(parse(&["api", "trace", "on", "--all-keys"]), Ok(Commands::Trace(TraceArgs { all_keys: true, .. }))));
        assert!(matches!(parse(&["api", "learn", "off"]), Ok(Commands::Learn(_))));
        assert!(matches!(parse(&["api", "allowlist", "--add", "10.0.0.1/32"]), Ok(Commands::Allowlist(_))));
        assert!(matches!(parse(&["api", "gateway", "--bind", "wg0"]), Ok(Commands::Gateway(GatewayArgs { bind: Some(_), .. }))));
        assert!(matches!(parse(&["api", "domain", "--check", "a.example.com"]), Ok(Commands::Domain(_))));
        assert!(matches!(parse(&["models", "recommend", "--json"]), Ok(Commands::Recommend(RecommendArgs { json: true }))));
        assert!(matches!(parse(&["models", "scan", "--yes"]), Ok(Commands::Scan(_))));
        assert!(matches!(parse(&["models", "alias", "--remove", "fast"]), Ok(Commands::Alias(_))));
        let Ok(Commands::Config(c)) = parse(&["models", "config", "qwen", "27b", "--ctx", "8192"]) else { panic!() };
        assert_eq!((model_query(c.model, c.model_id), c.ctx), ("qwen 27b".to_string(), Some(8192)));
        assert!(matches!(parse(&["models", "--installed"]), Ok(Commands::Models { installed: true, cmd: None, .. })));
        assert!(parse(&["models", "--json", "scan"]).is_err() && parse(&["api", "--json", "keys"]).is_err());
        for old in [&["recommend"][..], &["scan"], &["config", "org/m"], &["alias"], &["gateway"], &["domain"], &["allowlist"], &["learn"], &["trace"], &["service", "status"]] {
            assert!(parse(old).is_ok(), "{old:?} still parses");
        }
    }

    #[test]
    fn values_without_their_flag_get_the_corrected_command() {
        let args = |v: &[&str]| std::iter::once("goethe").chain(v.iter().copied()).map(String::from).collect::<Vec<_>>();
        let gpu = args(&["gpu", "allow", "gpu0", "gpu1"]);
        assert_eq!(corrected_command(&gpu, "allow").as_deref(), Some("goethe gpu --allow gpu0 gpu1"));
        assert_eq!(corrected_command(&args(&["domain", "api.example.com"]), "api.example.com").as_deref(), Some("goethe api domain --check api.example.com"));
        assert_eq!(corrected_command(&args(&["api", "domain", "api.example.com"]), "api.example.com").as_deref(), Some("goethe api domain --check api.example.com"));
        assert_eq!(corrected_command(&args(&["key", "create", "ops"]), "create").as_deref(), Some("goethe api keys --create ops"), "old forms are corrected to the grouped one");
        assert_eq!(corrected_command(&args(&["api", "keys", "revoke", "a1"]), "revoke").as_deref(), Some("goethe api keys --revoke a1"));
        assert_eq!(corrected_command(&args(&["models", "scan", "mydir"]), "mydir").as_deref(), Some("goethe models scan --dir mydir"));
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
