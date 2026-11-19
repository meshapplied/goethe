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

const KEY_HELP: &str = "On a terminal, goethe api keys alone opens a menu to add, scope, show or delete keys.

Create a key (goethe api lists it again later):

\t→ goethe api keys create my-app

With an expiry (days like 30d, or a date like 2026-12-31), a rate limit and admin rights:

\t→ goethe api keys create ops --expires 90d --rpm 300 --admin

Only some models, or just the key for a script:

\t→ goethe api keys create bot --models deepseek-ai/DeepSeek-V4.1-Flash-1017
\t→ API_KEY=$(goethe api keys create ci --raw)

Revoke one:

\t→ goethe api keys revoke <key-id>";

const LEARN_HELP: &str = "A session keeps what compaction would drop: the user's rules and corrections, facts from chat, the goal, decisions and mistakes. A small note goes into the prompt after each compaction. Sessions are told apart by the client's X-Session-Id header.

Every model:
\t→ goethe learn on
\t→ goethe learn off

One model:
\t→ goethe learn on --model my-model
\t→ goethe learn off --model my-model

goethe learn alone shows the current scope. Changes apply to the next request. Details are in docs/learn.md.";

const TRACE_HELP: &str = "Record every generation request and reply, grouped into sessions, as a corpus for training later:

\t→ goethe trace on

Only some API keys (by name or id), every key but one, or back to every key:

\t→ goethe trace on --key my-app
\t→ goethe trace off --key ci
\t→ goethe trace on --all-keys

goethe trace off stops recording and keeps the keys chosen; goethe trace on resumes with them.

Another folder, or a size limit (default 20 GB) at which recording turns itself off:

\t→ goethe trace on --dir /data/traces --max-gb 200
\t→ goethe trace --max-gb 100

goethe trace alone shows what is recorded and how much space it takes. The folder format is in docs/traces.md.";

const EXPOSE_HELP: &str = "Who can reach the API:

\t→ goethe api expose local            (this machine only)
\t→ goethe api expose vpn --interface wg0
\t→ goethe api expose public --allow 203.0.113.7/32 --yes

Public always needs --allow <their-ip>/32 and a confirm (--yes, or y on a terminal). It never opens on a leftover allowlist alone.

goethe api expose alone shows where the API listens and who may connect.";

const SHARE_HELP: &str = "Examples:

\t→ goethe api share 5h --key my-app --allow <their-ip>/32
\t→ goethe api share 5h --key my-app --dangerously-allow-all-inbound
\t→ goethe api share             (status)
\t→ goethe api share stop

--allow is who may connect (their public IP: ask them, or https://ifconfig.me on their device). This machine's IP is only the URL goethe prints. Reopening the same keys keeps their last addresses. --dangerously-allow-all-inbound opens to every address (rate limits still apply). Closes when the last key ends.";

#[derive(Parser)]
#[command(
    name = "goethe",
    disable_version_flag = true,
    about = "Goethe: serve models through OpenAI and Anthropic compatible gateways"
)]
struct Cli {
    #[arg(short = 'V', long = "version", help = "Print the version, and how to upgrade when a newer one is out (--build-info adds commit and build date).")]
    version: bool,
    #[arg(short = 'v', long, global = true, help = "Show the output of every underlying process (also GOETHE_VERBOSE=1); it is always kept in <home>/logs.")]
    verbose: bool,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    #[command(display_order = 1, about = "Start Goethe now and at every boot; models loaded last time come back.")]
    Up {
        #[arg(long = "no-boot", help = "Start now without turning on start at boot.")]
        no_boot: bool,
    },
    #[command(display_order = 2, about = "Stop Goethe and its models, and stop starting at boot (models come back on the next up).")]
    Down {
        #[arg(long = "keep-engines", help = "Leave loaded engines serving; the next goethed adopts them.")]
        keep_engines: bool,
    },
    #[command(display_order = 3, about = "Whether Goethe runs, the API address, its keys and the models it serves.")]
    Status,
    #[command(display_order = 4, about = "Running models in detail: GPUs, port, memory, requests and speed.")]
    Ps {
        #[arg(long, help = "JSON rows with instance_id, slot_key, status, restarts.")]
        json: bool,
    },
    #[command(
        display_order = 5,
        about = "Download a model from Hugging Face, or add a local folder; --load serves it too.",
        after_help = "Extra engine flags go after --, and are stored as the model's defaults:\n\n\t→ goethe install org/name --load -- --chunked-prefill-size 4096"
    )]
    Install {
        #[arg(value_name = "MODEL", num_args = 1.., required = true, help = "Hugging Face id (org/name), a local folder, or words from a catalog model's name; a :QUANT suffix (org/name:Q8_0) picks that GGUF file or kernel pack variant.")]
        model: Vec<String>,
        #[arg(long, help = "Load the model once it is installed.")]
        load: bool,
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::gpus), long = "gpu", num_args = 1.., value_delimiter = ',', value_name = "GPU", value_parser = parse_gpu_name, help = "Default GPUs, as goethe gpu lists them: --gpu gpu2 gpu3.")]
        gpu: Vec<u32>,
        #[arg(long, help = "Default context length.")]
        ctx: Option<u64>,
        #[arg(long = "max-conc", help = "Default max concurrent requests.")]
        max_conc: Option<u64>,
        #[arg(long, help = "With --load, return once the engine starts instead of waiting until it answers.")]
        no_wait: bool,
        #[arg(long, help = "Download again (Hub), or replace an install of another kind, and rebuild the model's settings from defaults.")]
        force: bool,
        #[arg(long = "trust-remote-code", help = "Let SGLang run Python code shipped with the model (config.json auto_map).")]
        trust_remote_code: bool,
        #[arg(long, help = "Do not ask: download without confirming, and with --load use GPUs that are busy or hold processes goethe did not start.")]
        yes: bool,
        #[arg(long = "hf-token", help = "Ask for a Hugging Face token (gated or private models); it is used for this download only.")]
        hf_token: bool,
        #[arg(long, help = "Model id to register a local folder under.")]
        name: Option<String>,
        #[arg(long, help = "Copy a local folder instead of linking to it.")]
        copy: bool,
        #[arg(long, help = "If the model's kernel pack cannot run here, serve it with the plain runtime (llama.cpp or SGLang) instead of failing; also GOETHE_ALLOW_FALLBACK=1.")]
        fallback: bool,
        #[arg(last = true, help = "Extra engine flags, only after --.")]
        extra: Vec<String>,
    },
    #[command(display_order = 6, about = "Unload and delete an installed model (a folder of your own is only unregistered).")]
    Remove {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), value_name = "MODEL", required = true, help = "Installed models to remove: ids, aliases or words from a name, one argument each.")]
        model: Vec<String>,
        #[arg(long, help = "Do not ask; also unloads the model if it is loaded.")]
        yes: bool,
        #[arg(long, help = "When unloading first, skip draining in-flight requests.")]
        force: bool,
        #[arg(long, value_name = "keep|delete", value_parser = ["keep", "delete"], help = "What happens to the traces recorded with the model (asked on a terminal; keep with --yes).")]
        traces: Option<String>,
    },
    #[command(
        display_order = 7,
        about = "Serve an installed model on the GPUs and wait until it answers.",
        after_help = "Extra engine flags go after --, for example:\n\n\t→ goethe load deepseek-ai/DeepSeek-V4.1-Flash-1017 --ctx 32768 -- --chunked-prefill-size 4096\n\n\
                      --save keeps --gpu, --ctx, --max-conc and the flags after -- as the model's defaults (a bare -- clears the stored flags).\n\
                      --dry-run shows where the model would go without loading it; with --save it only stores the defaults."
    )]
    Load {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), value_name = "MODEL", num_args = 1.., required = true, help = "Installed model: id, alias or words from its name, such as qwen 27b.")]
        model: Vec<String>,
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::gpus), long = "gpu", num_args = 1.., value_delimiter = ',', value_name = "GPU", value_parser = parse_gpu_name, help = "GPUs to load on, as goethe gpu lists them: --gpu gpu3, or --gpu gpu2 gpu3 to split the model across both.")]
        gpu: Vec<u32>,
        #[arg(long, value_name = "N", conflicts_with = "gpu", help = "Split the model across N GPUs that goethe picks (with --gpu, the number of GPUs given is used).")]
        tp: Option<u64>,
        #[arg(long, help = "Context length for this load.")]
        ctx: Option<u64>,
        #[arg(long = "max-conc", help = "Max concurrent requests.")]
        max_conc: Option<u64>,
        #[arg(long, help = "Expert parallel size (SGLang --ep-size).")]
        ep: Option<u64>,
        #[arg(long, help = "Keep --gpu, --ctx, --max-conc and the flags after -- as the model's defaults.")]
        save: bool,
        #[arg(long = "dry-run", help = "Show where the model would be placed now, and load nothing.")]
        dry_run: bool,
        #[arg(long, help = "Do not ask: use GPUs that are busy or hold processes goethe did not start.")]
        yes: bool,
        #[arg(long, help = "Start another instance even when the model is already loaded.")]
        replica: bool,
        #[arg(long, help = "If the model's kernel pack fails to open, serve it with the plain runtime (llama.cpp or SGLang) instead of failing; also GOETHE_ALLOW_FALLBACK=1.")]
        fallback: bool,
        #[arg(long, help = "Return once the engine starts instead of waiting until it answers.")]
        no_wait: bool,
        #[arg(long, help = "Print the instance as JSON instead of status lines.")]
        json: bool,
        #[arg(last = true, help = "Extra engine flags, only after --.")]
        extra: Vec<String>,
    },
    #[command(display_order = 8, about = "Stop serving a model.")]
    Unload {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), value_name = "MODEL", conflicts_with = "all", required_unless_present = "all", help = "Models to unload (every instance): ids, aliases or words from a name, one argument each.")]
        model: Vec<String>,
        #[arg(long, help = "Unload every loaded model.")]
        all: bool,
        #[arg(long, help = "Skip connection draining and tear down immediately.")]
        force: bool,
    },
    #[command(display_order = 9, about = "Try a model by hand: a chat screen on a terminal without INPUT, else one reply, an embedding summary or an audio file's transcript. Loads an installed model that is not loaded and unloads it again.")]
    Run {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::models), value_name = "MODEL", num_args = 0..=2, help = "[MODEL] [INPUT]: MODEL may be left out when one model is loaded.")]
        args: Vec<String>,
        #[arg(long, help = "Leave a model loaded that this command loaded (it is unloaded when run ends otherwise).")]
        keep: bool,
        #[arg(long, help = "No chat screen: the plain prompt, as when input or output is not a terminal.")]
        plain: bool,
        #[arg(long, help = "Show the model's reasoning (dimmed) with the answer.")]
        thinking: bool,
    },
    #[command(display_order = 10,
        about = "Installed and catalog models; also recommend, scan and alias.",
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
        #[arg(long, help = "Only catalog models.")]
        available: bool,
        #[arg(long, help = "Every line; the default stops after 40.")]
        all: bool,
    },
    #[command(display_order = 11, about = "Logs of a model, of goethed or of the gateway.")]
    Logs {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::log_targets), value_name = "TARGET", num_args = 1.., required = true, help = "A model (id, alias or words from its name), an instance id, goethed, or gateway.")]
        target: Vec<String>,
        #[arg(short = 'f', long, help = "Follow log output.")]
        follow: bool,
        #[arg(short = 'n', long, default_value = "100", help = "Number of lines to print.")]
        lines: usize,
    },
    #[command(display_order = 12,
        about = "The API address and how to connect; also keys, expose, allow, domain and share.",
        args_conflicts_with_subcommands = true
    )]
    Api {
        #[arg(long, help = "JSON with api, goethed_running, keys and models.")]
        json: bool,
        #[command(subcommand)]
        cmd: Option<ApiCommands>,
    },
    #[command(display_order = 13, about = "Let sessions keep what context compaction would drop: on or off, for every model or for some.", after_help = LEARN_HELP)]
    Learn(LearnArgs),
    #[command(display_order = 14, about = "Record requests and replies as training traces: on or off, per API key, and the size.", after_help = TRACE_HELP)]
    Trace(TraceArgs),
    #[command(display_order = 15, about = "GPUs with memory, processes and bandwidth, live; allow or deny one.")]
    Gpu {
        #[arg(value_parser = ["allow", "deny"], help = "allow or deny, followed by GPUs (goethe gpu deny gpu2): deny keeps new models off them, running ones stay.")]
        action: Option<String>,
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::gpus), value_name = "GPU", requires = "action", value_parser = parse_gpu_name, help = "GPUs as goethe gpu lists them, for example gpu0 gpu1.")]
        gpus: Vec<u32>,
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
    #[command(display_order = 16, about = "Check the machine and the setup, with a fix for each problem.")]
    Doctor {
        #[arg(long, help = "Also check that the API is reachable from the internet.")]
        public: bool,
        #[arg(long, help = "Print the hardware profile instead: GPUs and links, CPU, RAM, disk, OS, container and cloud.")]
        hardware: bool,
        #[arg(long, hide = true, help = "Print the heartbeat goethed would send the platform now, without sending it.")]
        snapshot: bool,
        #[arg(long, help = "Fix what doctor finds on this machine (NVIDIA driver, CUDA forward compatibility, a C++ compiler), asking before each change.")]
        fix: bool,
        #[arg(long, requires = "fix", help = "With --fix, make every change without asking.")]
        yes: bool,
        #[arg(long, help = "Emit JSON.")]
        json: bool,
    },
    #[command(display_order = 17, about = "Install a signed update of Goethe, its engines or kernel packs; rollback puts the previous kernel packs back.")]
    Update {
        #[arg(value_parser = ["engine", "rollback"], help = "engine updates the installed engines only; rollback puts back the previous kernel pack of every model that has one.")]
        what: Option<String>,
        #[arg(long, help = "Print available versions only.")]
        check: bool,
        #[arg(long, help = "Skip confirmation.")]
        yes: bool,
        #[arg(long = "allow-downgrade", help = "Install the published goethe and goethed even if older than this build (a build ahead of the release otherwise keeps its binaries).")]
        allow_downgrade: bool,
    },
    #[command(display_order = 18, about = "Remove Goethe and the models it installed (folders of your own stay).")]
    Uninstall {
        #[arg(long, help = "Keep the installed models in GOETHE_HOME/models.")]
        keep_models: bool,
        #[arg(long, help = "Skip confirmation.")]
        yes: bool,
    },
    #[command(display_order = 19, about = "Link this node to the Goethe platform (optional): a browser login, or --authkey for headless machines.")]
    Login {
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::platform_urls), long = "platform-url", value_name = "URL", help = "Platform to link to (default: the GOETHE_PLATFORM_URL setting, else https://goethe.sh).")]
        platform_url: Option<String>,
        #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::authkey_stdin), long, value_name = "KEY", help = "Link without a browser using an auth key (msh_ak_...); - reads it from stdin.")]
        authkey: Option<String>,
        #[arg(long, value_name = "WHEN", help = "How long the login lasts: never, or 12h, 30d, 4w; on a linked node it only changes that.")]
        lifetime: Option<String>,
    },
    #[command(display_order = 20, about = "Unlink this node from the Goethe platform and remove its local credentials.")]
    Logout,
    #[command(hide = true, about = "Finish an install: run by install.sh once it verified this goethe; safe to run again to repair or upgrade.")]
    Setup {
        #[arg(long = "pkg-url", value_name = "URL", help = "The package host to keep using (default pkg.goethe.sh).")]
        pkg_url: Option<String>,
        #[arg(long, value_name = "URL", help = "Where this run downloads from, for example one release's folder.")]
        from: Option<String>,
        #[arg(long = "keep-file-url", help = "Keep a file:// package host for later updates.")]
        keep_file_url: bool,
        #[arg(long = "platform-url", value_name = "URL", help = "Platform to log in to.")]
        platform_url: Option<String>,
        #[arg(long = "no-link", help = "Do not link goethe and goethed onto PATH.")]
        no_link: bool,
    },
    #[command(display_order = 21, about = "Tab completion for bash, zsh and fish: models, keys, GPUs and interfaces complete too.")]
    Completion {
        #[arg(value_parser = ["install", "remove", "status"], help = "install, remove, or status (the default).")]
        action: Option<String>,
        #[arg(long, value_parser = ["bash", "zsh", "fish"], help = "The shell (default: yours, from $SHELL).")]
        shell: Option<String>,
    },
    #[command(hide = true, about = "Catalog maintenance: validate, generate or merge registry entries.")]
    Registry {
        #[command(subcommand)]
        cmd: RegistryCommands,
    },
}

#[derive(Subcommand)]
enum ApiCommands {
    #[command(about = "List API keys, create or revoke one; on a terminal without arguments, a menu.", after_help = KEY_HELP)]
    Keys(KeyArgs),
    #[command(about = "Who can reach the API: local, vpn, public, or one address or interface.", after_help = EXPOSE_HELP)]
    Expose(ExposeArgs),
    #[command(about = "The client addresses allowed to reach the API; add or remove some.")]
    Allow(AllowArgs),
    #[command(about = "How to reach the API by a domain name; check DOMAIN tests that it works.")]
    Domain(DomainArgs),
    #[command(about = "Open the API on a public address for a while, for chosen API keys; stop closes it.", after_help = SHARE_HELP)]
    Share(ShareArgs),
}

#[derive(Subcommand)]
enum ModelsCommands {
    #[command(about = "Catalog models that fit the free VRAM of the allowed GPUs.")]
    Recommend {
        #[arg(long, help = "Emit JSON instead of a table.")]
        json: bool,
    },
    #[command(about = "Register model folders you downloaded yourself and moved into GOETHE_HOME/models.")]
    Scan {
        #[arg(value_name = "FOLDER", help = "One folder inside GOETHE_HOME/models (default: all of them).")]
        dir: Option<String>,
        #[arg(long, help = "Add every supported model without asking.")]
        yes: bool,
        #[arg(long, help = "Emit JSON rows (dir, model_id, status, reason, notes).")]
        json: bool,
    },
    #[command(about = "Short model names for clients: list them, set one, or remove some.")]
    Alias(AliasArgs),
}

#[derive(clap::Args)]
struct DomainArgs {
    #[arg(value_parser = ["check"], help = "check, followed by the domain: whether it points at the API and the gateway answers there.")]
    action: Option<String>,
    #[arg(value_name = "DOMAIN", requires = "action")]
    domain: Option<String>,
}

#[derive(clap::Args)]
struct AllowArgs {
    #[arg(value_parser = ["add", "remove"], help = "add or remove, followed by client addresses or ranges such as 203.0.113.7/32.")]
    action: Option<String>,
    #[arg(value_name = "CIDR", requires = "action")]
    cidrs: Vec<String>,
    #[arg(long, value_name = "TEXT", help = "With add: a note stored with the new entries.")]
    note: Option<String>,
}

#[derive(clap::Args)]
struct KeyArgs {
    #[arg(value_parser = ["create", "revoke"], help = "create NAME, or revoke KEY_ID…; nothing lists the keys (a menu on a terminal).")]
    action: Option<String>,
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::key_ids), value_name = "NAME_OR_ID", requires = "action")]
    values: Vec<String>,
    #[arg(long, value_name = "WHEN", help = "With create: expiry, days from now like 30d, or a date like 2026-12-31.")]
    expires: Option<String>,
    #[arg(long, value_name = "N", help = "With create: requests per minute (default 120, 0 = unlimited).")]
    rpm: Option<i64>,
    #[arg(long, value_name = "IDS", help = "With create: only these models, comma-separated.")]
    models: Option<String>,
    #[arg(long, help = "With create: an admin key, for health endpoints and every model.")]
    admin: bool,
    #[arg(long, help = "With create: print only the key, for scripts.")]
    raw: bool,
    #[arg(long, help = "List as JSON.")]
    json: bool,
}

#[derive(clap::Args)]
struct AliasArgs {
    #[arg(value_parser = ["set", "remove"], help = "set ALIAS MODEL, or remove ALIAS…; nothing lists the aliases.")]
    action: Option<String>,
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::aliases), value_name = "ALIAS", requires = "action")]
    values: Vec<String>,
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

#[derive(clap::Args)]
struct ExposeArgs {
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::expose_targets), value_name = "WHERE", help = "local, vpn or public; or an address (127.0.0.1, an IP of this host) or an interface such as wg0 to listen on.")]
    target: Option<String>,
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::interfaces), long, value_name = "NAME", conflicts_with = "ip", help = "With vpn: the network interface to serve, e.g. wg0; the gateway follows its address.")]
    interface: Option<String>,
    #[arg(long, value_name = "IP", hide = true, help = "Advanced: with vpn, a fixed address of this machine to listen on, instead of following an interface.")]
    ip: Option<std::net::IpAddr>,
    #[arg(long, value_name = "CIDR", num_args = 1.., value_delimiter = ',', help = "With vpn or public: client ranges to allow (required for public; default for vpn: the ranges the network reaches).")]
    allow: Vec<String>,
    #[arg(long, help = "Apply without questions. Public still needs --allow <their-ip>/32 (or --dangerously-allow-all-inbound).")]
    yes: bool,
    #[arg(
        long,
        hide = true,
        help = "Advanced / public only: allow every client address (0.0.0.0/0). Needs --yes. Not with --allow."
    )]
    dangerously_allow_all_inbound: bool,
    #[arg(long, help = "With no WHERE: the status as JSON.")]
    json: bool,
}

#[derive(clap::Args)]
struct ShareArgs {
    #[arg(value_name = "DURATION", help = "How long to stay open (30m, 5h, 1d; at most 7d), stop to close now, or nothing for the status.")]
    duration: Option<String>,
    #[arg(add = clap_complete::engine::ArgValueCandidates::new(complete::key_names), long, value_name = "NAME_OR_ID", num_args = 1.., value_delimiter = ',', help = "API keys to share (not admin keys), each until DURATION; with stop, the keys to stop sharing.")]
    key: Vec<String>,
    #[arg(long, value_name = "CIDR", num_args = 1.., value_delimiter = ',', help = "Who may connect (required to open a share): their addresses or ranges, such as --allow 203.0.113.7 198.51.100.0/24; reopening the same keys keeps their last ones.")]
    allow: Vec<String>,
    #[arg(long, help = "Open the public gateway to every address (0.0.0.0/0) on purpose, instead of --allow. Rate limits still apply.")]
    dangerously_allow_all_inbound: bool,
    #[arg(long, help = "The public port (default 8444, or the last one used).")]
    port: Option<u16>,
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

fn usage(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
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

fn quant_suffix(query: &str) -> (String, Option<String>) {
    // `model:QUANT` (Hugging Face style): the model, and the quant after the last ':'; a folder named with a ':' stays whole.
    match query.rsplit_once(':') {
        Some((m, q)) if !std::path::Path::new(query).is_dir() => (m.to_string(), Some(q.to_string()).filter(|q| !q.is_empty())),
        _ => (query.to_string(), None),
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

fn print_version() {
    // The version; on a terminal also whether a newer one is published, with the command to get it.
    println!("{}", goethe_core::build_info::version_line());
    if std::io::IsTerminal::is_terminal(&std::io::stdout()) {
        if let Some(newer) = goethe_core::update::newer_published() {
            println!("\nUpdate available: goethe {newer}\n\n{}", goethe_core::term::command("goethe update"));
        }
    }
}

const HELP_SECTIONS: &[(&str, &[(&str, &str)])] = &[
    ("Start and stop", &[
        ("goethe", "the menu: quick setup, install models, API keys and gateway, status"),
        ("goethe up", "start Goethe now and at every boot"),
        ("goethe down", "stop Goethe; loaded models come back on the next up"),
        ("goethe status", "running or not, the API address, keys and served models"),
        ("goethe ps", "running models in detail: GPUs, memory, requests, speed"),
    ]),
    ("Models", &[
        ("goethe install MODEL --load", "download from Hugging Face (or add a folder) and serve it"),
        ("goethe load MODEL --gpu gpu2 gpu3", "serve an installed model on these GPUs"),
        ("goethe load MODEL --ctx 32768 --save", "the same, and keep the settings as its defaults"),
        ("goethe load MODEL --dry-run", "show where it would go, load nothing"),
        ("goethe unload MODEL", "stop serving it (--all stops every model)"),
        ("goethe remove MODEL", "unload and delete it"),
        ("goethe run MODEL", "try it by hand: chat, one reply, embedding or transcript"),
        ("goethe models", "installed and catalog models"),
        ("goethe models recommend", "catalog models that fit your GPUs"),
        ("goethe models scan FOLDER", "register a folder you put in GOETHE_HOME/models"),
        ("goethe models alias set fast MODEL", "a short name clients can use (remove fast drops it)"),
        ("goethe logs MODEL -f", "follow a model's log (also goethed or gateway)"),
    ]),
    ("API and access", &[
        ("goethe api", "the address, the keys and a curl command to try it"),
        ("goethe api keys", "list the keys (a menu on a terminal)"),
        ("goethe api keys create my-app", "a new key (also --expires 30d --rpm 300 --models ID --admin)"),
        ("goethe api keys revoke KEY_ID", "revoke a key"),
        ("goethe api expose", "where the API listens and who may connect"),
        ("goethe api expose vpn --interface wg0", "serve a VPN or private network (also local or public)"),
        ("goethe api allow add 203.0.113.7/32", "let a client address in (remove takes it out)"),
        ("goethe api domain check DOMAIN", "check that a domain reaches the API"),
        ("goethe api share 5h --key my-app --allow <their-ip>/32", "open a public address for a while (share stop closes it)"),
    ]),
    ("Learning and traces", &[
        ("goethe learn on", "sessions keep what compaction drops (--model M for one model)"),
        ("goethe trace on", "record requests and replies for training (--key K for one key)"),
    ]),
    ("Machine", &[
        ("goethe gpu", "GPUs live: memory, processes and bandwidth"),
        ("goethe gpu deny gpu2", "keep new models off a GPU (allow gpu2 undoes it)"),
        ("goethe doctor", "check the machine and the setup, with a fix for each problem"),
        ("goethe doctor --hardware", "the hardware profile: GPUs and links, CPU, RAM, disk, OS"),
        ("goethe update", "signed update of Goethe, its engines and kernel packs"),
        ("goethe uninstall", "remove Goethe and the models it installed"),
        ("goethe login", "link this node to the Goethe platform (optional)"),
        ("goethe logout", "unlink it and remove its local credentials"),
        ("goethe completion install", "tab completion for your shell"),
    ]),
];

fn top_help(model: &str) -> String {
    // goethe help: one example line per command, grouped, with what it does; goethe help COMMAND has the details.
    use goethe_core::term::{paint, Color};
    let width = HELP_SECTIONS.iter().flat_map(|(_, rows)| rows.iter()).map(|(c, _)| c.chars().count()).max().unwrap_or(0) + 2;
    let mut out = String::from("Goethe: serve models through OpenAI and Anthropic compatible gateways\n");
    for (title, rows) in HELP_SECTIONS {
        out.push_str(&format!("\n{}\n", paint(title, Color::Bold)));
        for (cmd, what) in *rows {
            out.push_str(&format!("  {cmd:<width$}{}\n", paint(what, Color::Gray)));
        }
    }
    out.push_str(&format!(
        "\nMODEL is a Hugging Face id, an installed id, an alias or words from a name, for example:\n\n\t→ goethe install {model} --load\n"
    ));
    out
}

fn home_screen() -> Result<(), String> {
    // goethe alone on a terminal: the main menu.
    goethe_core::menu::run()
}

fn opt_gpus(gpu: Vec<u32>) -> Option<Vec<u32>> {
    (!gpu.is_empty()).then_some(gpu)
}

fn extras_given(args: &[String]) -> bool {
    // Whether the command line has a `--`, so `goethe load m --save --` (no flags after it) clears the stored engine flags.
    args.iter().any(|a| a == "--")
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
        if let Err(e) = home_screen() {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.len() == 1 || (args.len() == 2 && matches!(args[1].as_str(), "-h" | "--help" | "help")) {
        print!("{}", top_help(&goethe_core::onboard::suggested(true)));
        return;
    }
    let cli = Cli::command().try_get_matches_from(&args).and_then(|m| <Cli as clap::FromArgMatches>::from_arg_matches(&m)).unwrap_or_else(|e| e.exit());
    if cli.version {
        print_version();
        return;
    }
    goethe_core::step::set_verbose(cli.verbose);
    goethe_core::step::begin(&goethe_core::paths::goethe_home(), args.iter().skip(1).find(|a| !a.starts_with('-')).map_or("goethe", String::as_str));
    let result = match cli.command {
        Commands::Up { no_boot } => do_up(!no_boot),
        Commands::Down { keep_engines: false } => shutdown::run_down(),
        Commands::Down { keep_engines: true } => goethe_core::service::stop_keeping_engines().map(|(before, kept)| {
            if before > 0 {
                println!("{}", goethe_core::service::kept_text(before, kept));
            }
        }),
        Commands::Status => do_status(),
        Commands::Ps { json } => do_ps(json),
        Commands::Install {
            model,
            load,
            gpu,
            ctx,
            max_conc,
            no_wait,
            force,
            trust_remote_code,
            yes,
            hf_token,
            name,
            copy,
            fallback,
            extra,
        } => {
            if hf_token {
                let token = goethe_core::term::ask_secret("Hugging Face token:").unwrap_or_else(|e| usage(e));
                let account = goethe_core::hf::whoami(&token).unwrap_or_else(|e| usage(e));
                eprintln!("token accepted for {account}");
                goethe_core::hf::set_token(Some(token));
            }
            let (query, mut quant) = quant_suffix(&model.join(" "));
            let model_id = install_target(&query);
            if let Some(q) = quant.take_if(|_| std::path::Path::new(&model_id).is_dir()) {
                eprintln!("note: {model_id} is a local folder; :{q} is ignored");
            }
            if let Some(n) = name.as_deref() {
                checked_id(n);
            }
            if let Some(msg) = goethe_core::sglang_infer::reserved_flag_error(&extra) {
                usage(msg);
            }
            install::run(InstallOpts {
                model_id,
                ctx,
                max_conc,
                tp: None,
                gpus: opt_gpus(gpu),
                load,
                force,
                no_verify: false,
                name,
                copy_local: copy,
                sglang_args: extra,
                no_wait,
                trust_remote_code,
                yes,
                reset: force,
                quant,
                fallback,
            })
        }
        Commands::Remove { model, yes, force, traces } => {
            let traces = traces.map(|t| t == "delete");
            each(model, |id| models::run_remove(&installed_id(&id, false), yes, force, traces))
        }
        Commands::Load {
            model,
            gpu,
            tp,
            ctx,
            max_conc,
            ep,
            save,
            dry_run,
            yes,
            replica,
            fallback,
            no_wait,
            json,
            extra,
        } => {
            let model_id = installed_id(&model.join(" "), true);
            let tp = if gpu.is_empty() { tp } else { Some(gpu.len() as u64) };
            let saved = save.then(|| {
                if let Some(msg) = goethe_core::sglang_infer::reserved_flag_error(&extra) {
                    usage(msg);
                }
                if let Err(e) = config::validate_limits(&goethe_home(), &model_id, ctx, max_conc) {
                    usage(e);
                }
                config::run(ConfigOpts {
                    model_id: model_id.clone(),
                    ctx,
                    max_conc,
                    tp: (!gpu.is_empty()).then_some(gpu.len() as u64),
                    gpus: opt_gpus(gpu.clone()),
                    sglang_args: extra.clone(),
                    clear_sglang_args: extras_given(&args),
                })
            });
            match (saved, dry_run) {
                (Some(Err(e)), _) => Err(e),
                (_, true) => hw::run_pick_gpus(tp, Some(checked_id(&model_id))),
                _ => do_load(DoLoadArgs {
                    model_id,
                    gpus: opt_gpus(gpu).map(|v| v.iter().map(u32::to_string).collect::<Vec<_>>().join(",")),
                    ctx,
                    max_conc,
                    tp,
                    ep,
                    yes,
                    replica,
                    fallback,
                    extra: if save { Vec::new() } else { extra },
                    no_wait,
                    json,
                }),
            }
        }
        Commands::Unload { model, all, force } => match (model.is_empty(), all) {
            (true, _) => do_unload(None, force),
            _ => each(model, |id| do_unload(Some(&installed_id(&id, false)), force)),
        },
        Commands::Run { args, keep, plain, thinking } => goethe_core::run::run(&args, goethe_core::run::Opts { keep, plain, thinking }),
        Commands::Models {
            json,
            installed,
            available,
            all,
            cmd: None,
        } => models::run_list(json, installed, available, all),
        Commands::Models { cmd: Some(ModelsCommands::Recommend { json }), .. } => recommend::run(json),
        Commands::Models { cmd: Some(ModelsCommands::Scan { dir, yes, json }), .. } => goethe_core::scan::run(goethe_core::scan::ScanOpts { yes, json, dir }),
        Commands::Models { cmd: Some(ModelsCommands::Alias(a)), .. } => do_alias(a),
        Commands::Logs { target, follow, lines } => {
            let target = target.join(" ");
            match target.as_str() {
                "goethed" | "daemon" => goethe_core::logs::run_logs(None, follow, lines, true, false, false),
                "gateway" => goethe_core::logs::run_logs(None, follow, lines, false, true, false),
                t => {
                    let t = if t.starts_with("inst-") { t.to_string() } else { installed_id(t, false) };
                    goethe_core::logs::run_logs(Some(&t), follow, lines, false, false, false)
                }
            }
        }
        Commands::Api { json, cmd: None } => do_api(json),
        Commands::Api { cmd: Some(ApiCommands::Keys(k)), .. } => do_keys(k),
        Commands::Api { cmd: Some(ApiCommands::Expose(e)), .. } => do_expose(e),
        Commands::Api { cmd: Some(ApiCommands::Allow(a)), .. } => do_allow(a),
        Commands::Api { cmd: Some(ApiCommands::Domain(DomainArgs { domain, .. })), .. } => match domain {
            Some(domain) => goethe_core::domain::check(&domain).map(|text| print!("{text}")),
            None => goethe_core::domain::explain().map(|text| print!("{text}")),
        },
        Commands::Api {
            cmd: Some(ApiCommands::Share(ShareArgs {
                duration,
                key,
                allow,
                dangerously_allow_all_inbound,
                port,
            })),
            ..
        } => do_share(
            duration.as_deref().unwrap_or("status"),
            &key,
            &allow,
            dangerously_allow_all_inbound,
            port,
        ),
        Commands::Learn(LearnArgs { action, model, print, json }) => do_learn(action.as_deref(), &model, print.as_deref(), json),
        Commands::Trace(TraceArgs {
            action,
            key,
            all_keys,
            dir,
            max_gb,
            json,
        }) => do_trace(action.as_deref(), &key, all_keys, dir.as_deref(), max_gb, json),
        Commands::Gpu { action, gpus, json } => match action.as_deref() {
            Some("allow") if !gpus.is_empty() => each(gpus, do_gpu_allow),
            Some("deny") if !gpus.is_empty() => each(gpus, do_gpu_deny),
            Some(a) => usage(format!("name the GPUs to {a}, for example goethe gpu {a} gpu2")),
            None => do_gpu_list(json),
        },
        Commands::Doctor { snapshot: true, .. } => {
            let snap = goethe_core::snapshot::Collector::new(goethe_core::snapshot::now_rfc3339()).build(&goethe_home(), "interval", None);
            goethe_core::snapshot::encode(&snap).map(|data| println!("{}", String::from_utf8_lossy(&data)))
        }
        Commands::Doctor { fix: true, yes, .. } => goethe_core::fix::run(yes),
        Commands::Doctor { public, hardware, json, .. } => match hardware {
            true => hw::run_profile(json),
            false => goethe_core::doctor::run(json, public),
        },
        Commands::Update {
            what,
            check,
            yes,
            allow_downgrade,
        } => match what.as_deref() {
            Some("rollback") => goethe_core::pack_update::run_rollback(),
            _ => {
                let engine = what.is_some();
                let update = goethe_core::update::run(goethe_core::update::UpdateOpts {
                    check_only: check,
                    engine,
                    ternary: engine && goethe_core::update::ternary_installed(&goethe_home()),
                    yes,
                    allow_downgrade,
                });
                match (update, goethe_core::pack_update::run_all(check)) {
                    (Err(a), Err(b)) => Err(format!("{a}; {b}")),
                    (Err(e), Ok(())) | (Ok(()), Err(e)) => Err(e),
                    (Ok(()), Ok(())) => Ok(()),
                }
            }
        },
        Commands::Uninstall { keep_models, yes } => goethe_core::uninstall::run(goethe_core::uninstall::UninstallOpts { keep_models, yes }),
        Commands::Login { platform_url, authkey, lifetime } => {
            let linked = goethe_core::platform_cli::is_linked(&goethe_home());
            let login = if linked && lifetime.is_some() { Ok(()) } else { goethe_core::platform_cli::login(platform_url.as_deref(), authkey.as_deref()) };
            match (login, lifetime) {
                (Ok(()), Some(l)) if goethe_core::platform_cli::is_linked(&goethe_home()) => goethe_core::platform_cli::token_lifetime(&l),
                (r, _) => r,
            }
        }
        Commands::Logout => goethe_core::platform_cli::logout(),
        Commands::Setup { pkg_url, from, keep_file_url, platform_url, no_link } => goethe_core::setup::run(goethe_core::setup::Opts {
            pkg: pkg_url.or_else(|| std::env::var("GOETHE_PKG_URL").ok().filter(|v| !v.is_empty())).unwrap_or_else(|| goethe_core::setup::DEFAULT_PKG_URL.to_string()),
            from,
            keep_file_url,
            platform_url,
            no_link,
        }),
        Commands::Completion { action, shell } => complete::run(action.as_deref(), shell.as_deref()),
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
    };
    match result {
        Ok(()) => {}
        Err(msg) => fail(msg),
    }
}

fn do_keys(k: KeyArgs) -> Result<(), String> {
    // goethe api keys [create NAME | revoke ID…]: nothing lists them, or opens the menu on a terminal.
    let KeyArgs { action, values, expires, rpm, models, admin, raw, json } = k;
    let create_only = expires.is_some() || rpm.is_some() || models.is_some() || admin || raw;
    if create_only && action.as_deref() != Some("create") {
        usage("--expires, --rpm, --models, --admin and --raw go with goethe api keys create NAME");
    }
    match action.as_deref() {
        Some("create") => {
            let name = match values.as_slice() {
                [n] if !n.trim().is_empty() => n.clone(),
                [] => usage("name the key: goethe api keys create my-app"),
                _ => usage("one key name at a time: goethe api keys create my-app"),
            };
            if let Some(Err(e)) = models.as_deref().map(security::model_scope_list) {
                usage(e);
            }
            security::key_create(&name, true, rpm, expires.as_deref(), models.as_deref(), admin)
                .map(|token| print!("{}", key_create_stdout(&token, &key_id_for_token(&token), &name, raw)))
        }
        Some(_) if values.is_empty() => usage("name the keys to revoke by id: goethe api keys revoke <key-id>"),
        Some(_) => each(values, |id| security::key_revoke(&id).map(|()| println!("Revoked key {id}"))),
        None if !json && std::io::IsTerminal::is_terminal(&std::io::stdin()) && std::io::IsTerminal::is_terminal(&std::io::stdout()) => {
            goethe_core::keys_menu::run()
        }
        None => security::key_list_detailed().map(|rows| {
            if json {
                println!("{}", serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".into()));
            } else {
                print!("{}", key_list_text(&rows));
            }
        }),
    }
}

fn do_alias(a: AliasArgs) -> Result<(), String> {
    // goethe models alias [set ALIAS MODEL | remove ALIAS…]: nothing lists them.
    match (a.action.as_deref(), a.values.as_slice()) {
        (Some("set"), [alias, model]) => {
            let model = installed_id(model, true);
            alias::set(alias, &model).map(|()| println!("Alias {alias} now points to {model}"))
        }
        (Some("set"), _) => usage("goethe models alias set <alias> <model>"),
        (Some(_), []) => usage("name the aliases to remove: goethe models alias remove <alias>"),
        (Some(_), _) => each(a.values, |x| alias::remove(&x).map(|()| println!("Removed alias {x}"))),
        (None, _) => alias::list().map(|rows| print!("{}", alias_text(&rows))),
    }
}

fn do_allow(a: AllowArgs) -> Result<(), String> {
    // goethe api allow [add|remove CIDR…]: nothing lists the allowed clients.
    if a.note.is_some() && a.action.as_deref() != Some("add") {
        usage("--note goes with goethe api allow add");
    }
    match a.action.as_deref() {
        Some(act) if a.cidrs.is_empty() => usage(format!("name the addresses to {act}, for example goethe api allow {act} 203.0.113.7/32")),
        Some("add") => each(a.cidrs, |cidr| {
            security::allowlist_add(&cidr, a.note.as_deref()).map(|()| println!("Allowed {}", security::canonical_cidr(&cidr).unwrap_or(cidr)))
        }),
        Some(_) => each(a.cidrs, |cidr| {
            security::allowlist_remove(&cidr).map(|()| println!("Removed {}", security::canonical_cidr(&cidr).unwrap_or(cidr)))
        }),
        None => security::allowlist_list().map(|rows| print!("{}", allowlist_text(&rows))),
    }
}

fn do_expose(e: ExposeArgs) -> Result<(), String> {
    // goethe api expose: the status; local|vpn|public runs the access setup; an address or interface is listened on directly.
    let ExposeArgs { target, interface, ip, allow, yes, dangerously_allow_all_inbound, json } = e;
    let mode_only = interface.is_some() || ip.is_some() || !allow.is_empty() || yes || dangerously_allow_all_inbound;
    match target.as_deref() {
        None if mode_only => usage("give local, vpn or public with --interface, --ip, --allow or --yes"),
        None => do_gateway_status(json),
        Some(m @ ("local" | "vpn" | "public")) => {
            let mode = goethe_core::configure::Mode::parse(m).unwrap_or_else(|e| usage(e));
            if dangerously_allow_all_inbound && mode != goethe_core::configure::Mode::Public {
                usage("--dangerously-allow-all-inbound is only for public");
            }
            goethe_core::configure::run_api(goethe_core::configure::ApiOpts {
                mode: Some(mode),
                vpn_ip: ip,
                interface,
                allow,
                yes,
                dangerously_allow_all_inbound,
            })
        }
        Some(_) if mode_only => usage("--interface, --ip, --allow and --yes go with local, vpn or public"),
        Some(addr) => do_gateway_bind(addr),
    }
}

struct DoLoadArgs {
    model_id: String,
    gpus: Option<String>,
    ctx: Option<u64>,
    max_conc: Option<u64>,
    tp: Option<u64>,
    ep: Option<u64>,
    yes: bool,
    replica: bool,
    fallback: bool,
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
        "force": args.yes,
        "yes": args.yes,
        "replica": args.replica,
        "fallback": args.fallback,
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
    goethe_core::term::commands("Create your first API key:", &["goethe api keys create my-app"])
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
            &["goethe api allow add <client-ip>/32"],
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
        return goethe_core::term::commands("No aliases yet. Set one with:", &["goethe models alias set <alias> <model>"]);
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
        out.push_str(&goethe_core::term::commands("Turn it on with:", &["goethe learn on"]));
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
        let more = format!("goethe trace on --max-gb {}", (st["max_gb"].as_f64().unwrap_or(trace::DEFAULT_MAX_GB) * 2.0).ceil());
        out.push_str(&goethe_core::term::commands("Raise the limit and turn it back on, or delete traces first:", &[&more]));
    } else if !st["recording"].as_bool().unwrap_or(false) {
        out.push_str(&goethe_core::term::commands("Turn it on with:", &["goethe trace on"]));
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
    goethe_core::status_view::run_gpus()
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
        "INSTANCE", "MODEL", "RUNTIME", "SLOT", "STATUS", "GPUS", "PORT", "VRAM_GB", "RESTARTS", "IN_FLIGHT", "TOK_PER_S", "PREFIX",
    ]);
    for row in rows {
        let inst = row.get("instance_id").and_then(|v| v.as_str()).unwrap_or("");
        let model = row.get("model_id").and_then(|v| v.as_str()).unwrap_or("");
        let runtime = row.get("runtime").and_then(|v| v.as_str()).unwrap_or("-");
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
            runtime.to_string(),
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
        &["goethe api allow add <client-ip>/32"],
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
    }
    Ok(())
}

fn do_gateway_bind(addr: &str) -> Result<(), String> {
    // An address or interface; unspecified binds go through expose public with its guards.
    let home = goethe_home();
    ensure_layout(&home).map_err(|e| format!("layout: {e}"))?;
    let loopback = match addr.trim().parse::<std::net::IpAddr>() {
        Ok(ip) if ip.is_unspecified() => usage(
            "listening on every address needs: goethe api expose public --allow <their-ip>/32 --yes",
        ),
        Ok(ip) => {
            println!("{}", nginx::set_public_bind(&home, &ip.to_string())?);
            ip.is_loopback()
        }
        Err(_) if goethe_core::netif::valid_name(addr.trim()) => {
            println!("{}", nginx::set_public_bind_interface(&home, addr.trim())?);
            false
        }
        Err(_) => usage(format!(
            "invalid address {addr:?}: use 127.0.0.1, an IP address of this host or an interface name"
        )),
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
        "this host only; `goethe api expose vpn` or `goethe api expose public` accepts remote clients"
    } else {
        "remote clients reach it when allowlisted"
    };
    let state = if running { "running" } else { "not running (goethed starts it; see goethe logs gateway)" };
    let list = match allowlist {
        0 => "empty, every client is refused (goethe api allow add <client-ip>/32)".to_string(),
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
        None => "not listening yet (goethe logs gateway shows why)".to_string(),
    };
    let keys: Vec<(String, String)> = security::key_list_detailed()?
        .iter()
        .map(|r| {
            let field = |k: &str| r.get(k).and_then(Value::as_str).unwrap_or("").to_string();
            (field("name"), field("id"))
        })
        .collect();
    print!("{}", status_text(Some(&api), &keys));
    let (answered, models) = served_models();
    print!("\n{}", models_text(answered, &models));
    if let Some(line) = goethe_core::public_share::status_line(&home) {
        println!("\n{line}");
    }
    print_platform_lines(&home, true);
    Ok(())
}

fn do_share(
    public: &str,
    keys: &[String],
    allow: &[String],
    dangerously_allow_all_inbound: bool,
    port: Option<u16>,
) -> Result<(), String> {
    // goethe api share <duration> | stop | (status).
    use goethe_core::public_share as share;
    let home = goethe_home();
    match public {
        "stop" | "shutdown" | "close" | "off" => {
            if !allow.is_empty() || port.is_some() {
                usage("goethe api share stop takes only --key");
            }
            if !keys.is_empty() {
                for k in keys {
                    match share::stop_key(&home, k)? {
                        true => println!("{k} is no longer shared; the public gateway stays open for the other keys, if any."),
                        false => println!("{k} is not shared on the public gateway."),
                    }
                }
                return Ok(());
            }
            match share::shutdown(&home)? {
                true => println!("Public gateway closed. The keys stay as they are and keep working on the normal API address."),
                false => println!("The public gateway is not open."),
            }
            Ok(())
        }
        "status" => {
            if !keys.is_empty() || !allow.is_empty() || port.is_some() {
                usage("give a duration to open the public gateway, such as goethe api share 5h --key my-app --allow <their-ip>/32");
            }
            match share::active(&home) {
                Some(s) => {
                    let ip = goethe_core::platform::lookup_public_ip(&home).map(|i| i.to_string()).unwrap_or_else(|_| "<public IP>".into());
                    println!("Public gateway: open until {} ({} left)", share::local_time(s.until), left(s.until - share::now()));
                    println!("URL:  https://{}/v1", host_port(&ip, s.port));
                    for (name, g) in share::shared(&home) {
                        let from = if g.allow.is_empty() { "no addresses (unreachable)".to_string() } else { g.allow.join(", ") };
                        println!("Key:  {name} until {} ({} left), from {from}", share::local_time(g.until), left(g.until - share::now()));
                    }
                    println!("\nStop one key with: goethe api share stop --key <name>; close it with: goethe api share stop");
                }
                None => println!("The public gateway is closed. Open it with: goethe api share 5h --key <name> --allow <their-ip>/32"),
            }
            Ok(())
        }
        dur => {
            let secs = share::parse_duration(dur)?;
            let o = share::open(
                &home,
                share::OpenArgs {
                    secs,
                    keys,
                    allow,
                    port,
                    dangerously_allow_all_inbound,
                },
            )?;
            let ip = o.public_ip.map(|i| i.to_string()).unwrap_or_else(|| "<public IP>".into());
            let url = format!("https://{}/v1", host_port(&ip, o.share.port));
            println!("Shared until {} ({}); the public gateway is open until {}\n", share::local_time(share::now() + secs), left(secs), share::local_time(o.share.until));
            println!("URL:  {url}");
            for k in &o.keys {
                let models = k.models.as_deref().filter(|m| !m.is_empty()).unwrap_or("every model");
                match k.token.as_deref() {
                    Some(token) => println!("Key:  {} {token}  models: {models}", k.name),
                    None => println!("Key:  {} (id {})  models: {models}", k.name, k.id),
                }
            }
            let clients: Vec<String> = o.share.allow.clone();
            if clients.iter().any(|c| c.ends_with("/0")) {
                println!("Only these clients: every address (--dangerously-allow-all-inbound)");
            } else if !clients.is_empty() {
                println!("Only these clients: {}", clients.join(", "));
            }
            println!("\nTLS:  self-signed for this address. Send {} with the key (it holds no secret); the recipient", o.cert.display());
            println!("      verifies the gateway with it, for example:");
            let token = o.keys.first().and_then(|k| k.token.as_deref()).unwrap_or("<your_bearer_token>");
            println!("      curl --cacert public.crt {url}/models -H \"Authorization: Bearer {token}\"");
            if let Some(f) = &o.fingerprint {
                println!("      SHA-256 fingerprint: {f}");
            }
            for w in &o.warnings {
                println!("\nnote: {w}");
            }
            println!("\nIf nobody can connect, forward TCP port {} to this machine or allow it in the cloud firewall.", o.share.port);
            println!("Close it early with: goethe api share stop");
            Ok(())
        }
    }
}

fn host_port(ip: &str, port: u16) -> String {
    if ip.contains(':') { format!("[{ip}]:{port}") } else { format!("{ip}:{port}") }
}

fn left(secs: i64) -> String {
    let s = secs.max(0);
    match (s / 86_400, s % 86_400 / 3600, s % 3600 / 60) {
        (0, 0, m) => format!("{m}m"),
        (0, h, m) => format!("{h}h {m}m"),
        (d, h, _) => format!("{d}d {h}h"),
    }
}

fn print_platform_lines(home: &std::path::Path, running: bool) {
    // The platform link and the last heartbeat, after the rest of goethe status.
    println!("{}", goethe_core::platform_cli::status_lines(home, running).join("\n"));
}

fn status_text(api: Option<&str>, keys: &[(String, String)]) -> String {
    // goethe up and goethe status: the state word in color, the API, and its keys or how to create one.
    use goethe_core::term::{paint, Color, Table};
    let Some(api) = api else {
        return format!("goethe: {} (goethe up starts it)\n", paint("not running", Color::Red));
    };
    let mut out = format!("goethe: {}\nAPI: {api}\n", paint("running", Color::Green));
    if keys.is_empty() {
        out.push_str("keys: none, create one with: goethe api keys create my-app\n");
    } else {
        let mut table = Table::new(&["API KEY NAME", "KEY ID"]);
        for (name, id) in keys {
            table.row(vec![name.clone(), id.clone()]);
        }
        out.push_str(&table.render());
    }
    out
}

fn do_api(json: bool) -> Result<(), String> {
    // The endpoint, the models it serves, the keys that may call it, and one curl command to try it.
    let home = goethe_home();
    let api = format!("https://{}/v1", nginx::gateway_connect_addr(&home));
    let (ps, models) = served_models();
    let state = |s: &[String]| if s.iter().any(|x| x == "RUNNING") { "RUNNING".to_string() } else { s[0].clone() };
    let keys = security::key_list_detailed()?;
    if json {
        let list: Vec<Value> = models
            .iter()
            .map(|(id, st, g)| json!({"model": id, "status": state(st), "instances": st.len(), "gpus": g}))
            .collect();
        println!("{}", json!({"api": api, "schemas": ["openai", "anthropic", "litellm"], "goethed_running": ps, "keys": keys, "models": list}));
        return Ok(());
    }
    use goethe_core::term::{commands, paint, Color};
    println!("API endpoint: {api}");
    println!("SSL_CERT_FILE={}", cert_path());
    println!("Accepts: OpenAI, Anthropic (/v1/messages) and LiteLLM (routes with or without /v1, x-litellm-api-key)");
    for e in goethe_core::host::probe_environment(nginx::public_port_u16(&home)).public_endpoints {
        println!("Public endpoint: {e}");
    }
    println!("\n{}\n{}", paint("API keys", Color::Bold), api_keys_text(&keys));
    if security::allowlist_list()?.is_empty() {
        println!("{}", empty_allowlist_text());
    }
    print!("{}", commands("Query the API:", &[&curl_example(&api)]));
    Ok(())
}

type ServedModel = (String, Vec<String>, Vec<String>);

fn served_models() -> (bool, Vec<ServedModel>) {
    // Whether goethed answered, and each served model with the status and GPUs of its instances.
    let ps = rpc::call("ps", json!(null)).ok().filter(|r| r.code == 0);
    let rows: Vec<Value> = ps.as_ref().and_then(|r| serde_json::from_str(r.stdout.trim()).ok()).unwrap_or_default();
    let mut models: Vec<ServedModel> = Vec::new();
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
    (ps.is_some(), models)
}

fn models_text(running: bool, models: &[ServedModel]) -> String {
    // The models part of goethe status: one row per served model, or what to do when there are none.
    use goethe_core::term::Table;
    if !running {
        return String::new();
    }
    if models.is_empty() {
        return format!("{}\n", no_models_text());
    }
    let state = |s: &[String]| if s.iter().any(|x| x == "RUNNING") { "RUNNING".to_string() } else { s[0].clone() };
    let mut table = Table::new(&["MODEL", "STATUS", "INSTANCES", "GPUS"]);
    for (id, st, g) in models {
        table.row(vec![id.clone(), goethe_core::term::status(&state(st)), st.len().to_string(), g.join(" ")]);
    }
    table.render()
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
    fn a_quant_suffix_is_split_off_the_model() {
        assert_eq!(quant_suffix("org/m-GGUF:Q8_0"), ("org/m-GGUF".to_string(), Some("Q8_0".to_string())));
        assert_eq!(quant_suffix("qwen 8b:q4_k_m"), ("qwen 8b".to_string(), Some("q4_k_m".to_string())));
        assert_eq!(quant_suffix("org/m"), ("org/m".to_string(), None));
        assert_eq!(quant_suffix("org/m:"), ("org/m".to_string(), None));
        let odd = std::env::temp_dir().join(format!("goethe-test-{}-m:v2", std::process::id()));
        std::fs::create_dir_all(&odd).unwrap();
        assert_eq!(quant_suffix(odd.to_str().unwrap()), (odd.to_str().unwrap().to_string(), None), "a folder named with a ':'");
        let _ = std::fs::remove_dir(&odd);
    }

    #[test]
    fn gateway_status_explains_scope_and_allowlist() {
        let local = gateway_status_text("127.0.0.1", 8443, "https://127.0.0.1:8443", true, 0, &[]);
        assert!(local.starts_with("bind: 127.0.0.1 (this host only; `goethe api expose vpn`"), "{local}");
        assert!(local.contains("port: 8443\nurl: https://127.0.0.1:8443\nstate: running\n"), "{local}");
        assert!(local.contains("allowlist: empty, every client is refused (goethe api allow add"), "{local}");
        let open = gateway_status_text("0.0.0.0", 9443, "https://127.0.0.1:9443", false, 2, &["tcp://1.2.3.4:4000 -> port 9443".into()]);
        assert!(open.contains("bind: 0.0.0.0 (remote clients reach it when allowlisted)"), "{open}");
        assert!(open.contains("state: not running"), "{open}");
        assert!(open.contains("allowlist: 2 entries\nendpoint: tcp://1.2.3.4:4000 -> port 9443\n"), "{open}");
    }

    #[test]
    fn status_says_running_with_the_api_and_keys_or_how_to_start() {
        let stopped = status_text(None, &[]);
        assert!(stopped.starts_with("goethe: ") && stopped.contains("not running"), "{stopped}");
        assert!(stopped.ends_with("(goethe up starts it)\n"), "{stopped}");
        let fresh = status_text(Some("https://10.0.0.2:8443/v1"), &[]);
        assert!(fresh.contains("running\nAPI: https://10.0.0.2:8443/v1\nkeys: none, create one with: goethe api keys create my-app\n"), "{fresh}");
        assert_eq!(fresh.lines().count(), 3, "{fresh}");
        let keyed = status_text(Some("https://h/v1"), &[("ops".into(), "a1b2c3d4e5f6".into())]);
        assert!(keyed.contains("ops") && keyed.contains("a1b2c3d4e5f6") && !keyed.contains("Create your first"), "{keyed}");
    }

    #[test]
    fn empty_lists_say_how_to_add_the_first_entry() {
        assert!(key_list_text(&[]).ends_with("Create your first API key:\n\n\t→ goethe api keys create my-app\n"));
        assert!(allowlist_text(&[]).ends_with("\t→ goethe api allow add <client-ip>/32\n"));
        assert!(alias_text(&[]).ends_with("\t→ goethe models alias set <alias> <model>\n"));
    }

    fn parse(v: &[&str]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(std::iter::once("goethe").chain(v.iter().copied()))
    }

    #[test]
    fn actions_are_words_and_flags_only_change_how() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
        assert!(matches!(parse(&["api", "keys", "create", "ops", "--expires", "30d", "--admin"]).unwrap().command,
            Commands::Api { cmd: Some(ApiCommands::Keys(KeyArgs { action: Some(_), admin: true, .. })), .. }));
        let Commands::Api { cmd: Some(ApiCommands::Keys(k)), .. } = parse(&["api", "keys", "revoke", "a1", "b2"]).unwrap().command else { panic!() };
        assert_eq!((k.action.as_deref(), k.values), (Some("revoke"), vec!["a1".to_string(), "b2".to_string()]));
        assert!(parse(&["api", "keys", "a1"]).is_err(), "a value needs its action");
        let Commands::Gpu { action, gpus, .. } = parse(&["gpu", "deny", "gpu0", "gpu1"]).unwrap().command else { panic!() };
        assert_eq!((action.as_deref(), gpus), (Some("deny"), vec![0, 1]));
        assert!(parse(&["gpu", "gpu2"]).is_err() && parse(&["gpu", "--allow", "gpu2"]).is_err());
        assert!(parse(&["api", "allow", "add", "10.0.0.1/32", "10.0.0.2/32", "--note", "office"]).is_ok());
        assert!(parse(&["api", "domain", "check", "api.example.com"]).is_ok() && parse(&["api", "domain", "add", "x.y"]).is_err());
        assert!(parse(&["api", "expose", "vpn", "--interface", "wg0"]).is_ok() && parse(&["api", "expose", "wg0"]).is_ok());
        assert!(matches!(parse(&["api", "share", "5h", "--key", "a,b"]).unwrap().command,
            Commands::Api { cmd: Some(ApiCommands::Share(ShareArgs { duration: Some(_), .. })), .. }));
        let Commands::Api { cmd: Some(ApiCommands::Share(s)), .. } =
            parse(&["api", "share", "5h", "--key", "my-app", "--dangerously-allow-all-inbound"]).unwrap().command
        else {
            panic!()
        };
        assert!(s.dangerously_allow_all_inbound && s.allow.is_empty());
        assert!(parse(&["models", "alias", "set", "fast", "org/m"]).is_ok() && parse(&["models", "scan", "mydir"]).is_ok());
        assert!(parse(&["models", "--json", "scan"]).is_err() && parse(&["api", "--json", "keys"]).is_err());
        assert!(parse(&["learn", "on", "--model", "m"]).is_ok() && parse(&["trace", "on", "--all-keys"]).is_ok());
        assert!(matches!(parse(&["update", "engine"]).unwrap().command, Commands::Update { what: Some(_), .. }));
        assert!(parse(&["update", "--engine"]).is_err() && parse(&["update", "ternary"]).is_err());
        assert!(matches!(parse(&["doctor", "--hardware"]).unwrap().command, Commands::Doctor { hardware: true, .. }));
        assert!(parse(&["login", "--lifetime", "30d"]).is_ok());
    }

    #[test]
    fn the_help_shows_every_command_with_an_example_that_parses() {
        use clap::CommandFactory;
        let text = top_help("org/m");
        for c in Cli::command().get_subcommands().filter(|c| !c.is_hide_set()) {
            assert!(text.contains(&format!("goethe {}", c.get_name())), "{} is missing from goethe help", c.get_name());
        }
        for (_, rows) in HELP_SECTIONS {
            for (cmd, _) in *rows {
                let words: Vec<&str> = cmd.split_whitespace().skip(1).map(|w| match w { "MODEL" => "org/m", "FOLDER" => "f", "KEY_ID" => "a1", "DOMAIN" => "a.example.com", w => w }).collect();
                if !words.is_empty() {
                    assert!(parse(&words).is_ok(), "{cmd} does not parse");
                }
            }
        }
    }

    #[test]
    fn old_forms_are_gone() {
        for old in [
            &["start"][..], &["serve", "--public", "5h"], &["hw", "profile"], &["key"], &["allowlist"], &["gateway"], &["domain"],
            &["recommend"], &["scan"], &["config", "org/m"], &["alias"], &["configure", "api"], &["get", "api"], &["service", "status"],
            &["token-lifetime", "30d"], &["api", "manage"], &["api", "configure"], &["api", "gateway"], &["api", "allowlist"],
            &["api", "learn"], &["api", "trace"], &["models", "config", "org/m"], &["logs", "--daemon"], &["load", "--model", "org/m"],
            &["load", "org/m", "--gpus", "2"], &["load", "org/m", "--force"], &["install", "org/m", "--sglang-arg", "--x"],
            &["install", "org/m", "--no-load"], &["install", "org/m", "--no-verify"], &["remove", "org/m", "--keep-traces"],
            &["doctor", "--probe-public"], &["uninstall", "--purge-models"],
        ] {
            assert!(parse(old).is_err(), "{old:?} should no longer parse");
        }
    }

    #[test]
    fn models_are_words_and_gpus_are_names() {
        assert!(parse(&["load", "org/m", "--gpu", "gpu2", "gpu3"]).is_ok());
        assert!(parse(&["load", "org/m", "--gpu", "gpu2,gpu3", "--ctx", "4096"]).is_ok());
        assert!(parse(&["load", "org/m", "--gpu", "gpux"]).is_err());
        assert!(parse(&["load", "org/m", "--gpu", "gpu2", "--tp", "2"]).is_err(), "the GPUs given are the split");
        let Commands::Load { model, extra, .. } = parse(&["load", "qwen", "27b", "--ctx", "4096", "--", "--chunked-prefill-size", "8"]).unwrap().command else { panic!() };
        assert_eq!((model.join(" "), extra), ("qwen 27b".to_string(), vec!["--chunked-prefill-size".to_string(), "8".to_string()]), "words are the model, -- starts engine flags");
        let Commands::Install { extra, .. } = parse(&["install", "org/m", "--load", "--", "--x", "1"]).unwrap().command else { panic!() };
        assert_eq!(extra, ["--x", "1"]);
        assert!(parse(&["load"]).is_err() && parse(&["install"]).is_err() && parse(&["remove"]).is_err() && parse(&["unload"]).is_err());
        let Commands::Unload { model, .. } = parse(&["unload", "qwen-27b", "org/b"]).unwrap().command else { panic!() };
        assert_eq!(model, ["qwen-27b", "org/b"], "unload and remove take one model per argument");
        assert!(parse(&["unload", "org/a", "--all"]).is_err() && parse(&["unload", "--all"]).is_ok());
        assert!(matches!(parse(&["remove", "org/m", "--traces", "delete"]).unwrap().command, Commands::Remove { traces: Some(_), .. }));
        assert!(parse(&["logs", "gateway", "-f"]).is_ok() && parse(&["logs"]).is_err());
        assert_eq!(parse_gpu_name("gpu3"), Ok(3));
        assert_eq!(parse_gpu_name("GPU0"), Ok(0));
        assert!(parse_gpu_name("gpux").is_err());
        assert!(extras_given(&["goethe".into(), "load".into(), "m".into(), "--".into()]) && !extras_given(&["goethe".into(), "load".into()]));
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
