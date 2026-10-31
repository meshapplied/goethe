# Goethe command reference

Every `goethe` command and its options. For a walkthrough, see the [user guide](user-guide.md).

```text
goethe <command> [options]
goethe <command> --help
```

## Conventions

- Every value goes after its flag: `goethe unload --model <model>`, `goethe gpu --allow gpu0 gpu1`. A value typed without its flag prints the corrected command instead of running anything.
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

- [Start and stop](#start-and-stop): `start`, `up`, `status`, `down`, `service`
- [Models](#models): `recommend`, `install`, `scan`, `models`, `config`, `remove`
- [Running models](#running-models): `load`, `unload`, `ps`, `logs`
- [API access](#api-access): `configure api`, `api`, `key`, `allowlist`, `alias`, `trace`, `gateway`, `domain`
- [GPUs and system](#gpus-and-system): `gpu`, `hw`, `doctor`, `update`, `uninstall`, `--version`

## Start and stop

### `goethe start`

Guided setup. Starts Goethe, then walks through three steps: install a model, serve it through the API, and configure who can reach it. An advanced mode does each part on its own. Without a terminal, it prints the equivalent commands.

### `goethe up`

Starts Goethe and the API gateway, and starts them at every boot from now on. Models that were loaded last time come back.

| Option | Description |
|---|---|
| `--no-boot` | Start now without turning on start at boot |

### `goethe status`

Whether Goethe is running, the API address, and the names and ids of your API keys.

### `goethe down`

Stops Goethe and its models, and turns off start at boot. Loaded models come back on the next `goethe up`.

| Option | Description |
|---|---|
| `--keep-engines` | Keep models serving while Goethe itself restarts |

### `goethe service`

| Command | Description |
|---|---|
| `goethe service enable` | Start Goethe at boot |
| `goethe service disable` | Don't start Goethe at boot |
| `goethe service status` | Whether Goethe starts at boot |
| `goethe service restore --on` / `--off` | Whether loaded models come back when Goethe starts |

## Models

### `goethe recommend`

The catalog models that fit the GPUs Goethe may use, grouped by the memory they take, with the GPUs they would load on and their download size.

| Option | Description |
|---|---|
| `--json` | JSON instead of a table |

### `goethe install --model <id or folder>`

Downloads a model from Hugging Face, or registers a model folder on this machine.

| Option | Description |
|---|---|
| `--model <id or folder>` | Hugging Face repo id, or a local folder |
| `--load` | Load the model after installing it |
| `--no-load` | Install only |
| `--no-wait` | With `--load`, return without waiting until it's ready |
| `--ctx <N>` | Default context length |
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

### `goethe scan`

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

### `goethe config --model <id>`

Changes the default settings a model loads with.

| Option | Description |
|---|---|
| `--ctx <N>` | Default context length |
| `--max-conc <N>` | Default number of requests served at the same time |
| `--tp <N>` | Default number of GPUs |
| `--gpu <gpuN>...` | Default GPUs |
| `--sglang-arg <flag>` | Add an engine option |
| `--clear-sglang-args` | Remove the saved engine options |

### `goethe remove --model <id>...`

Unloads a model and deletes the files Goethe downloaded. For a folder you registered in place, it only unregisters it. If [traces](traces.md) were recorded with the model, it asks whether to delete them too; the default is to keep them.

| Option | Description |
|---|---|
| `--yes` | Don't ask; unload it first if it's loaded; keep its traces |
| `--force` | Don't wait for requests in progress |
| `--delete-traces` | Also delete the model's traces, without asking |
| `--keep-traces` | Keep the model's traces, without asking |

## Running models

### `goethe load --model <id> [-- <engine options>]`

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
goethe load --model zai-org/GLM-5.3-Flash --gpu gpu1
goethe load --model zai-org/GLM-5.3-Flash --gpu gpu0 gpu1 gpu2 gpu3 --ctx 65536
goethe load --model zai-org/GLM-5.3-Flash -- --chunked-prefill-size 4096
```

### `goethe unload`

Stops a model. Requests in progress finish first.

| Option | Description |
|---|---|
| `--model <id>...` | The models to stop |
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
| `--model <id>` | A model's log (also takes an instance id) |
| `-f`, `--follow` | Keep printing new lines |
| `-n`, `--lines <N>` | Number of lines (default 100) |
| `--daemon` | Goethe's own log |
| `--gateway` | The API gateway's log |
| `--prune` | Clean up old logs |

## API access

### `goethe configure api`

Chooses who can reach the API: this machine only, your VPN (Tailscale, WireGuard or ZeroTier; recommended), or a public address. It then asks whether to restrict access to specific IP addresses.

| Option | Description |
|---|---|
| `--mode local\|vpn\|public` | The access mode (asked when omitted) |
| `--vpn-ip <ip>` | This machine's VPN address to listen on (default: the detected one) |
| `--allow <cidr>` | A client address or range to allow; repeatable |
| `--yes` | Apply without questions |

### `goethe api`

The API address, how to authenticate, the models it serves, your API keys and a `curl` command to try it.

| Option | Description |
|---|---|
| `--json` | JSON output |

### `goethe key`

Lists API keys with their limits and today's usage, or creates or revokes one.

| Option | Description |
|---|---|
| `--create <name>` | Create a key |
| `--expires <when>` | Expiry: days from now such as `30d`, or a date such as `2027-01-31` |
| `--rpm <N>` | Requests per minute (default 120, `0`: unlimited) |
| `--models <a,b>` | Only these models |
| `--admin` | Admin key: health endpoints and every model |
| `--raw` | Print only the key, for scripts |
| `--revoke <key id>...` | Revoke keys, by the id `goethe key` shows |
| `--json` | JSON output |

```bash
goethe key --create my-app
goethe key --create ops --expires 90d --rpm 300 --admin
API_KEY=$(goethe key --create ci --raw)
```

### `goethe allowlist`

Lists the client addresses allowed to reach the API, or adds or removes one.

| Option | Description |
|---|---|
| `--add <cidr>...` | Allow addresses or ranges, such as `203.0.113.7` or `192.168.1.0/24` |
| `--description <text>` | A note stored with the new entries |
| `--remove <cidr>...` | Remove addresses or ranges |

### `goethe alias`

Short model names for clients. Changing what an alias points to moves every client that uses it.

| Option | Description |
|---|---|
| `--set <alias>` | The alias clients send as the model |
| `--model <id>` | The installed model it points to |
| `--remove <alias>...` | Remove aliases |

### `goethe trace [on | off]`

Records every generation request and reply, grouped into sessions, as a corpus for training later. Without `on` or `off` it shows what is recorded, the folder, its size and the free disk. Changes apply to the next request. The format is in [traces.md](traces.md).

| Option | Description |
|---|---|
| `--key <name or id>...` | Only these keys. With `on` and recording off: record just them. With `off` and every key on: leave them out |
| `--dir <path>` | Folder for the traces (default `~/.goethe/traces`) |
| `--max-gb <N>` | Size limit: recording turns off when the folder reaches it (default 20). Works without `on` or `off` |
| `--json` | JSON output |

```bash
goethe trace on
goethe trace on --key my-app
goethe trace off
```

### `goethe gateway`

Where the API listens, and how many client addresses are allowed.

| Option | Description |
|---|---|
| `--bind <address>` | Listen on `127.0.0.1` (this machine only), `0.0.0.0` (all addresses) or one IP of this machine |
| `--json` | JSON output |

### `goethe domain`

The DNS record that serves the API under your own domain name, with this machine's address filled in.

| Option | Description |
|---|---|
| `--check <domain>` | Check that the name points at the API and that it answers there |

## GPUs and system

### `goethe gpu`

GPUs with their memory use and whether Goethe may use them.

| Option | Description |
|---|---|
| `--deny <gpuN>...` | Keep new models off these GPUs; models already on them stay |
| `--allow <gpuN>...` | Allow these GPUs again |
| `--json` | JSON output |

### `goethe hw`

| Command | Description |
|---|---|
| `goethe hw profile` | GPUs and their links, CPU, memory, disk, OS, container and cloud provider |
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

Removes Goethe and the models it installed. Folders of your own that you registered stay.

| Option | Description |
|---|---|
| `--keep-models` | Keep the installed models |
| `--yes` | Don't ask |

### `goethe --version`

Prints the version, and the upgrade command when a newer version is available. `goethe --build-info` adds the commit and build date.
