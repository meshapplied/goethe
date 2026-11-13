# Goethe command reference

Every `goethe` command and its options. For a walkthrough, see the [user guide](user-guide.md).

```text
goethe <command> [options]
goethe <command> --help
```

## Conventions

- The model is a plain argument: `goethe load qwen 27b`, `goethe unload qwen-27b`. Give its id, an alias, or words from its name; when several models match, a menu asks which one (without a terminal, they are listed). `--model <id>` still works.
- Every other value goes after its flag: `goethe gpu --allow gpu0 gpu1`. A value typed without its flag prints the corrected command instead of running anything.
- Commands are grouped: what concerns the API is under `goethe api` (configure, manage, keys, allowlist, gateway, domain, learn, trace), and model housekeeping under `goethe models` (recommend, scan, config, alias). The older top-level forms such as `goethe key` still work and say where they moved.
- GPUs are named as `goethe gpu` lists them: `gpu0`, `gpu1`, and so on. Flags that take GPUs accept several: `--gpu gpu2 gpu3`.
- Model ids are Hugging Face repo names such as `zai-org/GLM-5.3-Flash`. Case doesn't matter, and the folder form `zai-org__GLM-5.3-Flash` works too. A mistyped id gets a suggestion.
- Commands that list things accept `--json` for scripts.

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | Error |
| `2` | Wrong usage |
| `3` | A confirmation was needed but there was no terminal to ask on; add `--yes` |

## Contents

- [Start and stop](#start-and-stop): `goethe`, `start`, `up`, `status`, `down`
- [Models](#models): `install`, `models`, `models recommend`, `models scan`, `models config`, `models alias`, `remove`
- [Running models](#running-models): `load`, `unload`, `ps`, `logs`
- [API access](#api-access): `api`, `api configure`, `api manage`, `api keys`, `api allowlist`, `api gateway`, `api domain`, `api learn`, `api trace`
- [GPUs and system](#gpus-and-system): `gpu`, `hw`, `doctor`, `update`, `uninstall`, `completion`, `--version`
- [Platform](#platform): `login`, `logout`

## Start and stop

### `goethe`

On a terminal, plain `goethe` opens the home screen: the sections on the left (Models, API access, API keys, GPUs, Learning & traces, Platform, Done), their state and what can be done on the right, refreshed every 5 seconds. ↑↓ move, → opens a section, ← goes back, `q` leaves. Enter on a model loads or unloads it; the access, key, learning and trace rows run `goethe api configure`, `goethe api manage`, `goethe api learn` and `goethe api trace` in the normal terminal and come back on Enter; Enter on a GPU allows or denies it; the Platform rows run `goethe login` or `goethe logout`. Without a terminal, `goethe` prints the commands.

### `goethe start`

Guided setup. Starts Goethe, then walks through three steps: install a model, serve it through the API, and configure who can reach it. An advanced mode does each part on its own. Without a terminal, it prints the equivalent commands. Step 3 offers only API keys that may call the model just served, and the advanced setup opens `goethe api manage` for keys.

### `goethe up`

Starts Goethe and the API gateway, and starts them at every boot from now on. Models that were loaded last time come back.

| Option | Description |
|---|---|
| `--no-boot` | Start now without turning on start at boot |

### `goethe status`

Whether Goethe is running, the API address, and the names and ids of your API keys. The last lines show the [platform](#platform) link (`Linked (workspace X)`, `Not linked`, or `Revoked`) and, when linked, the last heartbeat (`OK (2 s ago)` or `failing: <reason>`).

### `goethe down`

Stops Goethe and its models, and turns off start at boot. Loaded models come back on the next `goethe up`.

| Option | Description |
|---|---|
| `--keep-engines` | Keep models serving while Goethe itself restarts |

### `goethe service`

Not listed in `goethe --help`: `goethe up` already turns on start at boot and `goethe down` turns it off. It stays for the restore setting.

| Command | Description |
|---|---|
| `goethe service enable` | Start Goethe at boot |
| `goethe service disable` | Don't start Goethe at boot |
| `goethe service status` | Whether Goethe starts at boot |
| `goethe service restore --on` / `--off` | Whether loaded models come back when Goethe starts |

## Models

### `goethe models recommend`

The catalog models that fit the GPUs Goethe may use, grouped by the memory they take, with the GPUs they would load on and their download size.

| Option | Description |
|---|---|
| `--json` | JSON instead of a table |

### `goethe install <model or folder>`

Downloads a model from Hugging Face, or registers a model folder on this machine. When the platform has optimised kernels for the model and this machine's GPU, a logged-in node installs them and serves the model with goethe-engine; a node that is not logged in is asked to log in (on a terminal) or continues with SGLang. A model neither engine serves is refused before anything is downloaded.

| Option | Description |
|---|---|
| `<model>` | A Hugging Face id (`org/name`), a local folder, or words from a catalog model's name such as `glm flash` |
| `--load` | Load the model after installing it |
| `--no-load` | Install only |
| `--no-wait` | With `--load`, return without waiting until it's ready |
| `--ctx <N>` | Default context length (default: the model's full context window; a load uses the largest that fits in VRAM) |
| `--max-conc <N>` | Default number of requests served at the same time |
| `--tp <N>` | Default number of GPUs to split it across (`0`: automatic) |
| `--gpu <gpuN>...` | GPUs to load on |
| `--trust-remote-code` | Allow Python code that ships with the model; only for models you trust |
| `--hf-token` | Ask for a Hugging Face token, for gated or private models; used for this download only, never stored |
| `--yes` | With `--load`, load next to other programs on the GPU without asking |
| `--force` | Download again, or replace an existing install |
| `--reset` | Forget this model's saved settings |
| `--no-verify` | Skip checking the downloaded files |
| `--name <id>` | Id for a local folder |
| `--copy` | Copy a local folder instead of linking to it |
| `--sglang-arg <flag>` | An extra engine option saved with the model |

### `goethe models scan`

Adds model folders you placed in Goethe's `models` folder yourself. It checks each one and asks before adding it; a folder that can't be served is named with the reason.

| Option | Description |
|---|---|
| `--dir <folder>` | Only this folder |
| `--yes` | Add every supported model without asking |
| `--json` | JSON output |

### `goethe models`

Installed models, then catalog models, with whether each fits this machine and its size on disk. Alias: `goethe list`.

| Option | Description |
|---|---|
| `--installed` | Only installed models |
| `--available` | Only catalog models |
| `--json` | JSON output |

### `goethe models config <model>`

Changes the default settings a model loads with.

| Option | Description |
|---|---|
| `--ctx <N>` | Default context length (default: the model's full context window; a load uses the largest that fits in VRAM) |
| `--max-conc <N>` | Default number of requests served at the same time |
| `--tp <N>` | Default number of GPUs |
| `--gpu <gpuN>...` | Default GPUs |
| `--sglang-arg <flag>` | Add an engine option |
| `--clear-sglang-args` | Remove the saved engine options |

### `goethe remove <model>...`

Unloads a model and deletes the files Goethe downloaded. For a folder you registered in place, it only unregisters it. If [traces](traces.md) were recorded with the model, it asks whether to delete them too; the default is to keep them.

| Option | Description |
|---|---|
| `--yes` | Don't ask; unload it first if it's loaded; keep its traces |
| `--force` | Don't wait for requests in progress |
| `--delete-traces` | Also delete the model's traces, without asking |
| `--keep-traces` | Keep the model's traces, without asking |

## Running models

### `goethe load <model> [-- <engine options>]`

Loads a model and waits until it answers. Goethe picks the GPUs and sizes the memory unless you choose.

| Option | Description |
|---|---|
| `--gpu <gpuN>...` | GPUs to load on; several split the model across them |
| `--tp <N>` | Number of GPUs to split it across |
| `--ep <N>` | Expert-parallel size, for mixture-of-experts models |
| `--ctx <N>` | Context length |
| `--max-conc <N>` | Requests served at the same time |
| `--replica` | Start another copy, to serve more traffic |
| `--yes` | Load next to other programs on the GPU without asking |
| `--force` | Load on busy GPUs without asking |
| `--no-wait` | Return without waiting until it's ready |
| `--json` | Print the instance as JSON |
| `-- <options>` | Extra engine options, after `--` |

```bash
goethe load zai-org/GLM-5.3-Flash --gpu gpu1
goethe load zai-org/GLM-5.3-Flash --gpu gpu0 gpu1 gpu2 gpu3 --ctx 65536
goethe load zai-org/GLM-5.3-Flash -- --chunked-prefill-size 4096
```

### `goethe unload`

Stops a model. Requests in progress finish first.

| Option | Description |
|---|---|
| `<model>...` | The models to stop, one argument each |
| `--all` | Every loaded model |
| `--force` | Stop right away |

### `goethe ps`

Running models with their GPUs, port, memory, restarts, requests in progress and speed.

| Option | Description |
|---|---|
| `--json` | JSON output |

### `goethe logs`

| Option | Description |
|---|---|
| `<model>` | A model's log (also takes an instance id) |
| `-f`, `--follow` | Keep printing new lines |
| `-n`, `--lines <N>` | Number of lines (default 100) |
| `--daemon` | Goethe's own log |
| `--gateway` | The API gateway's log |
| `--prune` | Clean up old logs |

## API access

### `goethe api configure`

Chooses who can reach the API: this machine only, a VPN or other private network (recommended), or a public address. For a VPN it lists the networks this machine is on, with the ranges each one reaches, and works with any VPN that has a network interface. It then asks whether to restrict access to specific IP addresses. The gateway follows the chosen interface: when its address changes, it moves within a few seconds, and while the interface has no address only this machine can connect.

| Option | Description |
|---|---|
| `--mode local\|vpn\|public` | The access mode (asked when omitted) |
| `--interface <name>` | The network interface to serve, such as `wg0`; needed without a terminal when there are several |
| `--vpn-ip <ip>` | A fixed address of this machine instead of an interface |
| `--allow <cidr>` | A client address or range to allow; repeatable |
| `--yes` | Apply without questions |

### `goethe api`

The API address, how to authenticate, the models it serves, your API keys and a `curl` command to try it.

| Option | Description |
|---|---|
| `--json` | JSON output |

### `goethe api manage`

A full-screen key manager in two columns, like ranger. The left column picks what to work on: **+ Create new key**, **Done**, then your keys. The right column shows the hovered entry and is where things change. For a key it shows `API key:` with the key itself, its name and id (fixed), then what can change: the allowed models (**Every model**, or ticked models), the rate limit, the expiry, **Save changes** and **Delete key**. For **+ Create new key** it is the same form with a name, ending in **Create key**.

| Key | In the left column | In the right column |
|---|---|---|
| ↑ ↓ (or `k` `j`) | Move between entries | Move between fields |
| → (or `l`, Enter) | Open the entry on the right; on Done, leave | |
| ← (or `h`, Esc) | | Back to the left; asks first when there are unsaved changes |
| Space | | Tick a model or Every model |
| Enter | | Edit a value in place (Enter keeps it, Esc cancels), save, create, or delete (press twice) |
| `q` | Leave | Leave at once; unsaved changes are not saved, and a note says so |

On Rate limit or Expires, the bottom line shows the values they take: a whole number of requests per minute (`0` for unlimited), and `never`, days such as `30d` or a date such as `2027-01-31`. Changes apply only when saved and reach the gateway on the next request. The screen is restored on leaving, and keys created in the session are printed then. Admin keys are made with `goethe api keys --admin`; scripts use `goethe api keys`, since the menu needs a terminal (at least 64×12).

### `goethe api keys`

Lists API keys with their limits and today's usage, or creates or revokes one.

| Option | Description |
|---|---|
| `--create <name>` | Create a key |
| `--expires <when>` | Expiry: days from now such as `30d`, or a date such as `2027-01-31` |
| `--rpm <N>` | Requests per minute (default 120, `0`: unlimited) |
| `--models <a,b>` | Only these models |
| `--admin` | Admin key: health endpoints and every model |
| `--raw` | Print only the key, for scripts |
| `--revoke <key id>...` | Revoke keys, by the id `goethe api keys` shows |
| `--json` | JSON output |

```bash
goethe api keys --create my-app
goethe api keys --create ops --expires 90d --rpm 300 --admin
API_KEY=$(goethe api keys --create ci --raw)
```

### `goethe api allowlist`

Lists the client addresses allowed to reach the API, or adds or removes one.

| Option | Description |
|---|---|
| `--add <cidr>...` | Allow addresses or ranges, such as `203.0.113.7` or `192.168.1.0/24` |
| `--description <text>` | A note stored with the new entries |
| `--remove <cidr>...` | Remove addresses or ranges |

### `goethe models alias`

Short model names for clients. Changing what an alias points to moves every client that uses it.

| Option | Description |
|---|---|
| `--set <alias>` | The alias clients send as the model |
| `--model <id>` | The installed model it points to |
| `--remove <alias>...` | Remove aliases |

### `goethe api trace [on | off]`

Records every generation request and reply, grouped into sessions, as a corpus for training later. `off` stops recording and keeps the chosen keys; `on` resumes with them (every key the first time). Without `on` or `off` it shows what is recorded, the folder, its size and the free disk. Changes apply to the next request. The format is in [traces.md](traces.md).

| Option | Description |
|---|---|
| `--key <name or id>...` | With `on`: add these keys to the ones recorded. With `off`: leave them out |
| `--all-keys` | With `on`: record every key, clearing the chosen keys and exceptions |
| `--dir <path>` | Folder for the traces (default `~/.goethe/traces`) |
| `--max-gb <N>` | Size limit: recording turns off when the folder reaches it (default 20). Works without `on` or `off` |
| `--json` | JSON output |

```bash
goethe api trace on
goethe api trace on --key my-app
goethe api trace off
```

### `goethe api learn [on | off]`

Lets a session keep what context compaction would drop, in a small note that goes into the prompt after each compaction. Sessions are told apart by the client's `X-Session-Id` header. Without `on` or `off` it shows the scope and the sessions kept. Changes apply to the next request. `on` warns about models with a context under 100k tokens. See [learn.md](learn.md).

| Option | Description |
|---|---|
| `--model <id or alias>...` | Only these models. `all`, or leaving it out, means every model |
| `--print <session id>` | Show the note of a session |
| `--json` | JSON output |

```bash
goethe api learn on
goethe api learn on --model my-model
goethe api learn off --model my-model
goethe api learn --print my-session-id
```

### `goethe api gateway`

Where the API listens, and how many client addresses are allowed.

| Option | Description |
|---|---|
| `--bind <address>` | Listen on `127.0.0.1` (this machine only), `0.0.0.0` (all addresses), one IP of this machine, or an interface such as `wg0`, whose address is followed |
| `--json` | JSON output |

### `goethe api domain`

The DNS record that serves the API under your own domain name, with this machine's address filled in.

| Option | Description |
|---|---|
| `--check <domain>` | Check that the name points at the API and that it answers there |

## GPUs and system

### `goethe gpu`

GPUs with their memory use, whether Goethe may use them, the process ids on each (Goethe's in green, others in red) and a bar of memory bandwidth use. The bar turns blue past the line at 70 %. On a terminal the table updates every half second until you press `q`; piped or with `--json` it prints once. The JSON adds `processes` (`pid`, `goethe`) and `memory_util_percent`.

| Option | Description |
|---|---|
| `--deny <gpuN>...` | Keep new models off these GPUs; models already on them stay |
| `--allow <gpuN>...` | Allow these GPUs again |
| `--json` | JSON output |

### `goethe hw`

| Command | Description |
|---|---|
| `goethe hw profile` | GPUs, CPU, memory, disk, OS, container and cloud provider. GPU links, NUMA nodes and processes are in `--json` and `hardware_state.json`; `goethe gpu` shows the processes |
| `goethe hw pick-gpus [--model <id>] [--tp <N>]` | Where `goethe load` would place a model right now, and the memory it needs |

### `goethe doctor`

Checks the machine and the setup: GPU and driver, compiler, Python environment, gateway, ports and keys. Each problem comes with its fix.

| Option | Description |
|---|---|
| `--json` | JSON output |
| `--probe-public` | Also test that the public address answers |

### `goethe update`

Installs the latest Goethe. Loaded models keep serving.

| Option | Description |
|---|---|
| `--check` | Only show what's available |
| `--engine` | Also update the inference engine |
| `--yes` | Don't ask |
| `--allow-downgrade` | Install the published version even if it's older |

### `goethe uninstall`

Removes Goethe and the models it installed. Folders of your own that you registered stay. A node linked to the platform is logged out first (best effort) and its credentials go with the rest.

| Option | Description |
|---|---|
| `--keep-models` | Keep the installed models |
| `--yes` | Don't ask |

### `goethe completion [install | remove]`

Tab completion for bash, zsh and fish. Commands, subcommands and flags complete, and so do live values: installed models (by full id, by the name after the organisation, in any case, and by alias), GPU names, API key ids and names, network interfaces and bind addresses. The shell asks `goethe` for them at each Tab, so they are always current.

| Command | Description |
|---|---|
| `goethe completion` | Whether it is on for each shell |
| `goethe completion install` | Turn it on for your shell (`--shell bash\|zsh\|fish` for another) |
| `goethe completion remove` | Turn it off again |

fish and bash with the bash-completion package load the script from their own completion folders; for zsh, and bash without bash-completion, `install` adds a short marked block to `~/.zshrc` or `~/.bashrc`, and `remove` takes only that block out. The installer turns it on for your shell (`GOETHE_NO_COMPLETION=1` skips that), and `goethe uninstall` removes it.

### `goethe --version`

Prints the version, and the upgrade command when a newer version is available. `goethe --build-info` adds the commit and build date.

## Platform

Linking is optional. A node that is not linked never contacts the platform. A linked node's goethed sends a health snapshot every 10 seconds, and right after a model loads or unloads: hostname, OS, environment (bare metal, VM, container, cloud), network addresses, the gateway address, GPUs with memory use, and installed and loaded models. It never sends API keys, engine keys, prompts, traces, file paths, environment variables or command lines. Inference never depends on the platform.

### `goethe login`

Links this node to a workspace. Prints a link and a code, then waits until the login is approved in the browser (Ctrl-C cancels; the code expires after 15 minutes). Logging in again replaces the current link and logs the old node out. A running goethed sends its first heartbeat within a second; otherwise it starts with `goethe up`.

| Option | Description |
|---|---|
| `--authkey <msh_ak_...>` | Link without a browser, with an auth key from the platform; `--authkey -` reads it from stdin so it stays out of the process list |
| `--platform-url <url>` | Another platform (default: `GOETHE_PLATFORM_URL`, else `https://platform.meshapplied.com`). Must be HTTPS; plain HTTP only for `localhost` and `*.localhost` |

The installer takes the same flags: `curl -fsSL .../install.sh | bash -s -- --authkey msh_ak_...` (or `GOETHE_AUTHKEY=...`).

The credentials and the heartbeat state live in `GOETHE_HOME/platform/` (directory 0700, files 0600). Goethe refuses a credentials file or folder other users can read, a symlink, or one you do not own, and says how to fix it; it also refuses a `GOETHE_HOME` that every user can write to (a group-writable one is fine). When the platform revokes the node (for example after it is removed there), heartbeats stop and `goethe status` says `Revoked` until the next `goethe login`.

### `goethe logout`

Removes this node from its workspace and deletes the local credentials. When the platform cannot be reached, the local link is still removed and the node shows offline on the platform until you remove it there.
