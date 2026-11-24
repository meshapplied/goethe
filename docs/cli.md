# Goethe command reference

Every `goethe` command and its options. For a walkthrough, see the [user guide](user-guide.md).

```text
goethe <command> [options]
goethe <command> --help
```

## Conventions

- The model is a plain argument: `goethe load qwen 27b`, `goethe unload qwen-27b`. Give its id, an alias, or words from its name; when several models match, a menu asks which one (without a terminal, they are listed).
- Every action is a word, and flags only change how it runs: `goethe gpu allow gpu0 gpu1`, `goethe api keys create my-app`, `goethe models alias set fast org/model`.
- Commands are grouped: what concerns the API is under `goethe api` (keys, expose, allow, domain, share), and model housekeeping under `goethe models` (recommend, scan, alias). Learning and traces have their own commands, `goethe learn` and `goethe trace`.
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
- [Models](#models): `install`, `models`, `models recommend`, `models scan`, `models alias`, `remove`
- [Running models](#running-models): `load`, `unload`, `run`, `ps`, `logs`
- [API access](#api-access): `api`, `api keys`, `api expose`, `api allow`, `api domain`, `api share`
- [Learning and traces](#learning-and-traces): `learn`, `trace`
- [GPUs and system](#gpus-and-system): `gpu`, `doctor`, `update`, `uninstall`, `completion`, `--version`
- [Console](#console): `login`, `logout`

## Start and stop

### `goethe`

On a terminal, plain `goethe` opens the menu on its own screen:

| Entry | What it does |
|---|---|
| Quick setup | Install a model, serve it through the API and choose who can reach it, step by step |
| Load/Unload | The installed models, loaded ones marked. Enter unloads a loaded model (Space marks several to unload together). On one that is not loaded, Enter asks where: Auto (goethe picks, as `goethe load` does) or the GPUs you mark with Space, several splitting one copy (`goethe load --gpu`); each GPU shows its free memory and whether the model fits on it alone, GPUs that are not allowed are gray and skipped by the arrows, and the line under them is the plan `goethe load` would use (tp, context and GB per GPU). Enter loads, with a bar beside the model. Under the list, a line names `goethe gpu allow/deny gpuN` for freeing up or denying GPUs |
| Install models | A table of the catalog models that fit this machine's GPUs, with their size and the VRAM they need, under the machine's VRAM and storage. Space selects a model to install, or an installed one (filled dot) to uninstall, whose dot then turns the theme's third color. Above the keys the screen shows what Enter will do (`install: ...`, `uninstall: ...`); Enter uninstalls first, then installs, each one in turn, and goes on when one fails. A custom Hugging Face id and a token for gated models can be typed below the table. The table comes back after, with the new state |
| API keys and gateway | API keys (create, scope, new secret or delete), network exposure (local, VPN or private network, public) and the public address ([`goethe api share`](#goethe-api-share-duration)) |
| Status | Whether Goethe runs, the environment, then live tables of the GPUs (VRAM, processes, allowed or denied, memory bandwidth), CPU and memory |
| Metrics | From the [usage log](#usage-log), over the last hour, 24 hours, 7 days, this month (from the 1st), 30 or 90 days (↑ the next, longer range, ↓ back): per model the requests, errors (client and server errors, rate limits, cancelled), prompt and generated tokens, average time to first token, average decode speed and the 95th-percentile latency, with a total row; per GPU the average and peak utilisation, memory bandwidth, average VRAM, average power against its limit, peak temperature and any throttling, with the host's CPU, RAM and free disk; and per API key the requests and tokens, by whole days. Requests for a model Goethe does not serve and requests refused by a key's rate limit have their own rows |

↑↓ move, Enter opens, ← goes back, `q` quits. On the main menu 1, 2 and 3 pick a color theme (default, sea, earth); the choice is remembered. Without a terminal, `goethe` prints the commands.

### `goethe start`

Quick setup on its own, the menu's first entry: install a model, serve it through the API and choose who can reach it, step by step, then back to the shell. It needs a terminal; scripts use `goethe up`, `goethe install <model> --load` and `goethe api keys create <name>`.

### `goethe up`

Starts Goethe and the API gateway, and starts them at every boot from now on. Models that were loaded last time come back.

| Option | Description |
|---|---|
| `--no-boot` | Start now without turning on start at boot |

### `goethe status`

Whether Goethe is running, the API address, the names and ids of your API keys, the models it serves, and the public gateway while it is open ([`goethe api share`](#goethe-api-share-duration)). The last lines show the [console](#console) link (`Linked (workspace X)`, `Not linked`, or `Revoked`) and, when linked, the last heartbeat (`OK (2 s ago)` or `failing: <reason>`).

### `goethe down`

Stops Goethe and its models, and turns off start at boot. Loaded models come back on the next `goethe up`.

| Option | Description |
|---|---|
| `--keep-engines` | Keep models serving while Goethe itself restarts |

## Models

### `goethe models recommend`

The catalog models that fit the GPUs Goethe may use, grouped by the memory they take, with the GPUs they would load on and their download size.

| Option | Description |
|---|---|
| `--json` | JSON instead of a table |

### `goethe install <model or folder> [-- <engine options>]`

Downloads a model from Hugging Face, or registers a model folder on this machine. Models with optimised kernels for this GPU use them when the node is logged in (`goethe login`); otherwise goethe asks to log in or continues with SGLang. A model that cannot be served is refused before anything is downloaded.

| Option | Description |
|---|---|
| `<model>` | A Hugging Face id (`org/name`), a local folder, or words from a catalog model's name such as `glm flash` |
| `--load` | Load the model after installing it |
| `--no-wait` | With `--load`, return without waiting until it's ready |
| `--ctx <N>` | Default context length (default: the model's full context window; a load uses the largest that fits in VRAM) |
| `--max-conc <N>` | Default number of requests served at the same time |
| `--gpu <gpuN>...` | Default GPUs; several split the model across them |
| `--trust-remote-code` | Allow Python code that ships with the model; only for models you trust |
| `--hf-token` | Ask for a Hugging Face token, for gated or private models; used for this download only, never stored |
| `--yes` | Don't ask: download without confirming, and with `--load` load next to other programs on the GPU |
| `--force` | Download again, or replace an existing install, and rebuild the model's saved settings from the defaults |
| `--fallback` | Serve the model with its plain runtime (llama.cpp or SGLang) when its kernel pack cannot run here, even with `GOETHE_ALLOW_FALLBACK=0` (the plain runtime takes over by default) |
| `--name <id>` | Id for a local folder |
| `--copy` | Copy a local folder instead of linking to it |
| `-- <options>` | Extra engine options, after `--`, saved with the model |

### `goethe models scan [folder]`

Adds model folders you placed in Goethe's `models` folder yourself. It checks each one and asks before adding it; a folder that can't be served is named with the reason.

| Option | Description |
|---|---|
| `<folder>` | Only this folder (default: all of them) |
| `--yes` | Add every supported model without asking |
| `--json` | JSON output |

### `goethe models`

Installed models, then catalog models, with whether each fits this machine and its size on disk. Alias: `goethe list`.

| Option | Description |
|---|---|
| `--installed` | Only installed models |
| `--available` | Only catalog models |
| `--json` | JSON output |

### `goethe models alias`

Short model names for clients. Changing what an alias points to moves every client that uses it.

| Command | Description |
|---|---|
| `goethe models alias` | List the aliases |
| `goethe models alias set <alias> <model>` | Point an alias at an installed model; clients send the alias as the model |
| `goethe models alias remove <alias>...` | Remove aliases |

To change the settings a model loads with, use [`goethe load`](#goethe-load---) with `--save`.

### `goethe remove <model>...`

Unloads a model and deletes the files Goethe downloaded. For a folder you registered in place, it only unregisters it. If [traces](traces.md) were recorded with the model, it asks whether to delete them too; the default is to keep them.

| Option | Description |
|---|---|
| `--yes` | Don't ask; unload it first if it's loaded; keep its traces unless `--traces delete` |
| `--force` | When unloading first, don't wait for requests in progress |
| `--traces keep\|delete` | What happens to the model's traces, without asking |

## Running models

### `goethe load <model> [-- <engine options>]`

Loads a model and waits until it answers. Goethe picks the GPUs and sizes the memory unless you choose. On a terminal it shows the context window, the concurrency and the VRAM it will hold, then a bar that fills as the model reaches VRAM; piped, or with `--verbose`, it prints the plan and a final `RUNNING:` line.

| Option | Description |
|---|---|
| `--gpu <gpuN>...` | GPUs to load on; the number given is the split |
| `--tp <N>` | Split the model across N GPUs that Goethe picks (not with `--gpu`) |
| `--ep <N>` | Expert-parallel size, for mixture-of-experts models |
| `--ctx <N>` | Context length |
| `--max-conc <N>` | Requests served at the same time |
| `--save` | Keep `--gpu`, `--ctx`, `--max-conc` and the options after `--` as the model's defaults; a bare `--` clears the saved options |
| `--dry-run` | Show where the model would be placed now and the memory it needs, and load nothing; with `--save` it only stores the defaults |
| `--replica` | Start another copy, to serve more traffic |
| `--yes` | Don't ask: load on busy GPUs, or next to other programs on the GPU |
| `--fallback` | Serve the model with its plain runtime when its kernel pack fails to open, even with `GOETHE_ALLOW_FALLBACK=0` |
| `--no-wait` | Return without waiting until it's ready |
| `--json` | Print the instance as JSON |
| `-- <options>` | Extra engine options, after `--` |

```bash
goethe load zai-org/GLM-5.3-Flash --gpu gpu1
goethe load zai-org/GLM-5.3-Flash --gpu gpu0 gpu1 gpu2 gpu3 --ctx 65536
goethe load zai-org/GLM-5.3-Flash -- --chunked-prefill-size 4096
goethe load zai-org/GLM-5.3-Flash --ctx 32768 --save --dry-run   # only store the defaults
goethe load zai-org/GLM-5.3-Flash --dry-run                      # where would it go?
```

### `goethe unload`

Stops a model. Requests in progress finish first.

| Option | Description |
|---|---|
| `<model>...` | The models to stop, one argument each |
| `--all` | Every loaded model |
| `--force` | Stop right away |

### `goethe run [MODEL] [INPUT]`

Tries a model by hand. Without MODEL or INPUT on a terminal, a list of the installed models opens at the top of the screen (loaded ones first, marked `loaded`): ↑↓ move, Enter runs the chosen one (one that is not loaded loads right there, its bar beside it), `q` or ← leaves. With text it prints one reply (a text model), an embedding summary (`dims= norm= [first 4]`) or, for an audio file, its transcript. With no INPUT on a terminal it opens an ephemeral chat screen; with no terminal, or `--plain`, a plain `> ` prompt (a speech model takes an audio file as INPUT). An installed model that is not loaded is loaded first, like `goethe load` but with a 100,000-token context (or the model's maximum when smaller; reduced when the GPU cannot fit it), and unloaded quietly in the background when run ends unless `--keep`. A model that was loaded (or loading) before run started stays loaded; run never installs. It talks to the model's own engine on 127.0.0.1 with its engine key, not through the gateway, and creates no API key.

| Option | Description |
|---|---|
| `--keep` | Leave a model loaded that run loaded |
| `--plain` | No chat screen |
| `--thinking` | Show the model's reasoning, dimmed, with the answer (hidden by default) |

Chat screen: the model on the top line, the conversation from the top row down (it follows the reply as it streams), your line and the keys at the bottom. Each message starts with a `➤ User` or `➤ Model` line in the theme's colors. Once a reply is in, the keys line also shows this chat's generated tokens and average decode speed (tokens over the time from first to last token), with the last reply's when there is room; these chats are not in the usage log and the numbers go when it closes. Typing: Enter sends, Esc opens the command keys; you can type while a reply streams, and Enter then sends the line after it (Esc stops the reply). Commands: `m` pick another model (one that is not loaded loads on the list; Esc cancels; a new conversation starts), `i` or Enter back to typing, `q` quit at once (Ctrl-C asks first). Up and Down or `k` and `j` scroll a line, Page Up and Page Down (or Ctrl-U and Ctrl-D, also while typing) a page, `g` and `G` jump to the top and the end; the screen follows the stream again at the end or on the next send. `/audio PATH` sends an audio file to a speech model; an embedding model shows `dims=` and `norm=` for each line. Nothing is written.

### `goethe metrics`

The menu's Metrics page as a command: model usage, API key usage per model, and GPU and host use from the [usage log](#usage-log). On a terminal ↑ moves to the next, longer range and ↓ back (1 h, 24 h, 7 d, this month, 30 d, 90 d; 24 h first) and ← or `q` leaves; when the output is not a terminal it prints the last 24 hours once.

### `goethe ps`

Running models with the runtime serving each (goethe-engine runs a kernel pack; llama.cpp and sglang are the plain runtimes), their GPUs, port, memory, restarts, requests in progress and speed. `goethe models --json` gives installed models the same `runtime`.

| Option | Description |
|---|---|
| `--json` | JSON output |

### `goethe logs <target>`

| Option | Description |
|---|---|
| `<target>` | A model (id, alias or words from its name), an instance id, `goethed` for Goethe's own log, or `gateway` for the API gateway's log |
| `-f`, `--follow` | Keep printing new lines |
| `-n`, `--lines <N>` | Number of lines (default 100) |

Goethe rotates its logs every hour.

## API access

### `goethe api`

The API address, how to authenticate, your API keys and a `curl` command to try it. The models it serves are in `goethe status`.

| Option | Description |
|---|---|
| `--json` | JSON output (`api`, `goethed_running`, `keys`, `models`) |

### `goethe api keys`

Lists API keys with their limits and today's usage, or creates, rotates or revokes one. On a terminal, `goethe api keys` alone opens a full-screen key manager (see below). Goethe stores only a hash of each key, so the key itself is shown once, when it is made: store it then. A lost key gets a new secret with `rotate`; its id, name, models and limits stay, and the old secret stops working on the next request.

| Command | Description |
|---|---|
| `goethe api keys` | List the keys (`--json` for scripts) |
| `goethe api keys create <name>` | Create a key |
| `goethe api keys rotate <key id>` | A new secret for a key; the old one stops working |
| `goethe api keys revoke <key id>...` | Revoke keys, by the id `goethe api keys` shows |

| Option | Description |
|---|---|
| `--expires <when>` | With `create`: expiry, days from now such as `30d`, or a date such as `2027-01-31` |
| `--rpm <N>` | With `create`: requests per minute (default 120, `0`: unlimited) |
| `--models <a,b>` | With `create`: only these models |
| `--admin` | With `create`: admin key, for health endpoints and every model |
| `--raw` | With `create` or `rotate`: print only the key, for scripts |
| `--json` | List as JSON |

```bash
goethe api keys create my-app
goethe api keys create ops --expires 90d --rpm 300 --admin
API_KEY=$(goethe api keys create ci --raw)
API_KEY=$(goethe api keys rotate <key-id> --raw)
goethe api keys revoke <key-id>
```

The key manager is in two columns, like ranger. The left column picks what to work on: **+ Create new key**, **Done**, then your keys. The right column shows the hovered entry and is where things change. For a key it shows `API key:` with its first characters (the whole key only for one made or renewed in this session), its name and id (fixed), then what can change: the allowed models (**Every model**, or ticked models), the rate limit, the expiry, **Save changes**, **New secret** and **Delete key**. For **+ Create new key** it is the same form with a name, ending in **Create key**.

| Key | In the left column | In the right column |
|---|---|---|
| ↑ ↓ (or `k` `j`) | Move between entries | Move between fields |
| → (or `l`, Enter) | Open the entry on the right; on Done, leave | |
| ← (or `h`, Esc) | | Back to the left; asks first when there are unsaved changes |
| Space | | Tick a model or Every model |
| Enter | | Edit a value in place (Enter keeps it, Esc cancels), save, create, or make a new secret or delete (press twice) |
| `q` | Leave | Leave at once; unsaved changes are not saved, and a note says so |

On Rate limit or Expires, the bottom line shows the values they take: a whole number of requests per minute (`0` for unlimited), and `never`, days such as `30d` or a date such as `2027-01-31`. Changes apply only when saved and reach the gateway on the next request. The screen is restored on leaving, and keys created or renewed in the session are printed then. Scripts use the commands above, since the menu needs a terminal (at least 64×12).

### `goethe api expose [where]`

Who can reach the API. Without an argument it shows where the API listens and how many client addresses are allowed.

| Command | Description |
|---|---|
| `goethe api expose` | Show where the API listens and who may connect (`--json` for scripts) |
| `goethe api expose local` | This machine only |
| `goethe api expose vpn` | A VPN or other private network (recommended) |
| `goethe api expose public` | A public address (needs `--allow <their-ip>/32` and `--yes`, or `y` on a terminal) |
| `goethe api expose <address or interface>` | Listen on `127.0.0.1`, one IP of this machine, or an interface such as `wg0` (followed). Not `0.0.0.0`: use `expose public` |

For a VPN it lists the networks this machine is on, with the ranges each one reaches, and works with any VPN that has a network interface. It then asks whether to restrict access to specific IP addresses. The gateway follows the chosen interface: when its address changes, it moves within a few seconds, and while the interface has no address only this machine can connect.

`expose public` never binds every interface on a leftover allowlist. Name the clients with `--allow`, confirm with `--yes` (or `y`), and only then does it listen on `0.0.0.0`. Opening every client address needs the hidden `--dangerously-allow-all-inbound --yes`.

| Option | Description |
|---|---|
| `--interface <name>` | With `vpn`: the network interface to serve, such as `wg0`; needed without a terminal when there are several |
| `--allow <cidr>` | With `vpn` or `public`: a client address or range to allow; **required for public**; repeatable |
| `--yes` | Apply without questions. Public still needs `--allow` (or `--dangerously-allow-all-inbound`) |
| `--json` | With no argument: the status as JSON |

```bash
goethe api expose local
goethe api expose vpn --interface wg0
goethe api expose public --allow 203.0.113.7/32 --yes
```

### `goethe api allow`

Lists the client addresses allowed to reach the API, or adds or removes one.

| Command | Description |
|---|---|
| `goethe api allow` | List the allowed addresses |
| `goethe api allow add <cidr>...` | Allow addresses or ranges, such as `203.0.113.7` or `192.168.1.0/24` |
| `goethe api allow remove <cidr>...` | Remove addresses or ranges |

| Option | Description |
|---|---|
| `--note <text>` | With `add`: a note stored with the new entries |

### `goethe api domain`

The DNS record that serves the API under your own domain name, with this machine's address filled in.

| Command | Description |
|---|---|
| `goethe api domain check <domain>` | Check that the name points at the API and that it answers there |

### `goethe api share [duration]`

Opens the API on a public address for a limited time, for API keys you choose, so you can give someone outside your network a URL and a key. The normal API address, local or VPN, stays as it is.

```
goethe api share 5h --key my-app --allow 203.0.113.7/32              # prints the URL, the key and the certificate to send
goethe api share 5h --key my-app --dangerously-allow-all-inbound     # every address (rate limits still apply)
goethe api share 5h --key my-app                                     # reopen: keeps addresses from the last share of those keys
goethe api share                                                     # status: URL, each key with its time left and addresses
goethe api share stop --key ci                                       # stop sharing one key
goethe api share stop                                                # close the public gateway
```

Every shared key uses the same public gateway, each until its own time and from its own addresses. The gateway stays open until the last key's time ends. A key whose time has ended is refused while the others keep working, and the addresses only it allowed are dropped from the gateway. Sharing a key again starts its time over (`5h` means 5 hours from now) and replaces its addresses; the other keys are not touched. Without `--key`, the keys of the last opening are shared again. Opening prints the URL, each key with the models it may use, the certificate to send along, and a `curl` command to try it. Goethe keeps only hashes of keys, so a key that is not shared yet gets a new secret, printed here: its old secret stops working, on the normal address too. A key that is already shared keeps its secret (`secret unchanged`), so a recipient can keep using the key you sent, and reopening without `--key` prints no secret. `goethe api keys rotate` makes a new one. Connections in progress are not interrupted. The menu's Public address page does the same with a form: the key, how long, and optional addresses. `goethe status` shows the gateway while it is open, and a node linked to the console shows the public URL and when it closes on its page.

| Option | Description |
|---|---|
| `<duration>` | How long to share the keys: minutes, hours or days, such as `30m`, `5h` or `1d`, at most `7d`. `stop` stops sharing; nothing shows the status |
| `--key <name or id>` | API keys to share, each until the duration; repeatable or comma-separated. With `stop`, the keys to stop sharing |
| `--allow <cidr>` | Only these client addresses or ranges may use the keys given, such as `203.0.113.7`; repeatable. Without it, addresses from the last share of those keys are kept; with none, Goethe refuses to open |
| `--dangerously-allow-all-inbound` | Allow every address (`0.0.0.0/0`). Rate limits still apply (10 requests/s per client, burst 20, 16 connections). Not with `--allow` |
| `--port <port>` | The public port (default `8444`, or the last one used) |

The recipient verifies the gateway with the certificate goethe prints, `GOETHE_HOME/nginx/certs/public.crt`. It holds no secret: `curl --cacert public.crt https://<address>:8444/v1/models -H "Authorization: Bearer <key>"`. Most OpenAI-compatible clients take it through `SSL_CERT_FILE` or their CA-bundle setting. If nobody can connect, forward the port to this machine on your router or allow it in your cloud firewall; goethe does not open ports on routers. The public listener uses IPv4.

**Security precautions**

- **Closed by default:** nothing listens on the public port unless a window is open. Shutdown or the end of the window removes the listener.
- **Time limit checked twice:** goethed closes the listener when the time runs out (it checks every 2 seconds, and on start it never reopens an expired window). In addition, every request on the public address checks the end time, so a request after it is refused even if closing failed or goethed was stopped.
- **Only chosen keys, each for its own time:** only shared keys work on the public address, each until its own time; every other key gets 403 there. Stopping a key, or `goethe api share stop`, applies to the next request.
- **No admin keys:** admin keys are refused when opening and on every request. Expired keys are refused when opening, and a key that expires during the window stops working when it expires. goethe warns about a key that may use every model; a key limited with `--models` is safer to share.
- **Model scope unchanged:** a key reaches only the models it is limited to, as on the normal address.
- **Separate listener:** public requests reach goethe through their own socket, `GOETHE_HOME/nginx/mux-public.sock`, that only nginx can open, so they can never be mistaken for requests from your own network, and the client address goethe checks is the one nginx saw.
- **API routes only:** health, readiness, admin and metrics endpoints are not served publicly.
- **HTTPS only:** TLS 1.2 or 1.3, with its own certificate for the public listener. Plain HTTP connections are dropped.
- **Stricter limits than the normal address:** 10 requests per second per client address (bursts of 20) and 16 connections per address, plus the same body size limit and header and body timeouts.
- **Client address required:** every shared key has an allowlist (`--allow`, or addresses kept from the last share of those keys). Nginx admits only that union and always ends with `deny all`. Every address (`0.0.0.0/0`) needs `--dangerously-allow-all-inbound`; this machine's own public IP is refused as `--allow`. Goethe also checks the client on every public request.
- **Checks before opening:** nothing changes unless every check passes: the gateway and goethed are running, the keys are valid, and the port is free and is not the normal gateway's port. If nginx does not accept the new listener, the previous state is restored.

## Learning and traces

### `goethe trace [on | off | status]`

Records every generation request and reply, grouped into sessions, as a corpus for training later. `off` stops recording and keeps the chosen keys; `on` resumes with them (every key the first time). `status` (or no action) shows what is recorded, the folder, its size and the free disk. Changes apply to the next request. The format is in [traces.md](traces.md).

| Option | Description |
|---|---|
| `--key <name or id>...` | With `on`: add these keys to the ones recorded. With `off`: leave them out |
| `--all-keys` | With `on`: record every key, clearing the chosen keys and exceptions |
| `--dir <path>` | Folder for the traces (default `~/.goethe/traces`) |
| `--max-gb <N>` | Size limit: recording turns off when the folder reaches it (default 20). Works without `on` or `off` |
| `--json` | JSON output |

```bash
goethe trace on
goethe trace on --key my-app
goethe trace off
goethe trace status
```

### `goethe learn [on | off | status]`

Lets a session keep what context compaction would drop, in a small note that goes into the prompt after each compaction. Sessions are told apart by the client's `X-Session-Id` header. `status` (or no action) shows the scope and the sessions kept. Changes apply to the next request. `on` warns about models with a context under 100k tokens. See [learn.md](learn.md).

| Option | Description |
|---|---|
| `--model <id or alias>...` | Only these models. `all`, or leaving it out, means every model |
| `--print <session id>` | Show the note of a session |
| `--json` | JSON output |

```bash
goethe learn on
goethe learn on --model my-model
goethe learn off --model my-model
goethe learn status
goethe learn --print my-session-id
```

## GPUs and system

### `goethe gpu`

The GPU table of the menu's Status page: per GPU the VRAM in use, a 10-block bar (Goethe's models in the theme's main color, other processes in white, free memory dotted) and the total; the processes, one per line (another program's as a red process id, a Goethe model as its process id and model, with the engine's other processes such as `└ scheduler` below it); ALLOWED or DENIED; and a bar of memory bandwidth use with its percent. A legend follows the table. On a terminal it updates every 100 ms until `q` or ←; piped or with `--json` it prints once. The JSON adds `processes` (`pid`, `goethe`) and `memory_util_percent`.

| Command | Description |
|---|---|
| `goethe gpu deny <gpuN>...` | Keep new models off these GPUs; models already on them stay |
| `goethe gpu allow <gpuN>...` | Allow these GPUs again |

| Option | Description |
|---|---|
| `--json` | JSON output |

### `goethe doctor`

Checks the machine and the setup: GPU and driver, compiler, FFmpeg, Python environment, gateway, ports and keys. Each problem comes with its fix. With `--hardware` it prints the hardware profile instead: GPUs, CPU, memory, disk, OS, container and cloud provider. GPU links, NUMA nodes and processes are in `--json` and `hardware_state.json`; `goethe gpu` shows the processes. To see where `goethe load` would place a model right now, and the memory it needs, use `goethe load <model> --dry-run`.

| Option | Description |
|---|---|
| `--json` | JSON output |
| `--public` | Also test that the public address answers |
| `--hardware` | Print the hardware profile instead of the checks |

### `goethe update [engine | rollback]`

Installs the latest Goethe. Loaded models keep serving. It then checks the kernel pack of each installed model against the signed index: when the pack changed, only the blocks the installed pack lacks are downloaded, the new pack is verified and swapped in, and the previous one is kept for one rollback. A loaded model keeps its old pack until reloaded. `goethe load` says once a day when a newer pack exists.

| Command | Description |
|---|---|
| `goethe update` | Update Goethe, bundled nginx when its sha differs, already-installed engines, and kernel packs |
| `goethe update engine` | Update only the installed engines |
| `goethe update rollback` | Put the previous kernel pack of every model that has one back |

CLI help text stays the source of truth for flags and examples. A later pass can regenerate the command tables from clap while keeping the prose paragraphs above hand-written.

| Option | Description |
|---|---|
| `--check` | Only show what would change: one line per component (goethe and goethed by version, nginx by version, SGLang by wheel, goethe-engine), then a summary such as `2 updates available (goethe, nginx): run goethe update` or `everything is up to date`. Changes nothing |
| `--yes` | Don't ask |
| `--allow-downgrade` | Install the published goethe and goethed even if older than this build (a build ahead of the release otherwise keeps its binaries); not with `engine` |

### `goethe uninstall`

Stops Goethe and its engines, then removes everything Goethe installed: the models it downloaded (also a `models` folder linked to another disk, whose own folder stays), the inference engines (the Python environment with goethe-engine and SGLang, and llama.cpp), the kernel packs, and the rest of the Goethe home. The NInfer weights that `goethe install` put in `~/.cache/goethe/nvfp4` go too. Before asking, it lists the models with their total size, the engines and the kernel packs it will delete.

Folders of your own files stay: for a model registered in place only Goethe's links go, and a folder of yours inside `models` loses only Goethe's card. A trace folder in the home stays. A node linked to the console is logged out first (best effort) and its credentials go with the rest. When something cannot be deleted, the rest still goes, the command names what is left and exits non-zero.

| Option | Description |
|---|---|
| `--keep-models` | Keep the installed models |
| `--yes` | Don't ask |

### `goethe completion [install | remove | status]`

Tab completion for bash, zsh and fish. Commands, subcommands and flags complete, and so do live values: installed models (by full id, by the name after the organisation, in any case, and by alias), GPU names, API key ids and names, network interfaces and listen addresses. The shell asks `goethe` for them at each Tab, so they are always current.

| Command | Description |
|---|---|
| `goethe completion status` | Whether it is on for each shell (also without an action) |
| `goethe completion install` | Turn it on for your shell (`--shell bash\|zsh\|fish` for another) |
| `goethe completion remove` | Turn it off again |

fish and bash with the bash-completion package load the script from their own completion folders; for zsh, and bash without bash-completion, `install` adds a short marked block to `~/.zshrc` or `~/.bashrc`, and `remove` takes only that block out. The installer turns it on for your shell (`GOETHE_NO_COMPLETION=1` skips that), and `goethe uninstall` removes it.

### `goethe --version`

Prints the version, and the upgrade command when a newer version is available. `goethe --build-info` adds the commit and build date.

## Console

Linking is optional. A node that is not linked never contacts the console. A linked node's goethed sends a health snapshot every 10 seconds, and right after a model loads or unloads: hostname, OS, environment (bare metal, VM, container, cloud), network addresses, the gateway address (and the public gateway's address and end time while [`goethe api share`](#goethe-api-share-duration) is open), GPUs with memory use, installed and loaded models, the node's API key inventory (id, name, prefix, model scope, expiry — never the raw key), and the finished minutes of the [usage log](#usage-log) the console has not confirmed yet (per model, GPU and host; never per API key, never prompt or reply text), and the error rows of failed installs, setup steps and doctor checks: a fixed code such as `MODEL_DOWNLOAD`, the Goethe version, the time and one scrubbed line (no paths, addresses, URLs, emails or typed values). Those rows are first written to `GOETHE_HOME/errors.log` (at most 64 KB; the same failure at most once a day), also when the node is not linked, and are sent once the node is linked. It never sends API key secrets, engine keys, prompts, traces, file paths, environment variables or command lines. Inference never depends on the console.

### Usage log

goethed keeps a usage log whether or not the node is linked. It holds counts and timings only, never a prompt, a reply or the text of a token. Every 5 seconds (10 on a Mac) it samples each GPU (utilisation, memory-bandwidth use, VRAM used by Goethe and by other processes, power and its limit, temperature, throttle reasons; on a Mac the GPU's busy % and memory in use) and the host (CPU %, RAM, free disk under `GOETHE_HOME/models`, and on a Mac the memory pressure level). Each finished request adds to its model's row for that clock minute: the requests by outcome (ok, client error, server error, rate-limited, cancelled), prompt and generated token counts, a latency and a time-to-first-token histogram, and the decode speed. The gateway streams every generation request (chat, completions, Anthropic messages, Responses) from the engine, so each reply is timed the same way: time to first token, and decode speed from the first to the last token. A request the client did not stream gets the reply rebuilt into the one body a non-streaming call returns. Token counts are the ones the engine reports; on the OpenAI routes the gateway asks for them (`stream_options.include_usage`) and passes the extra usage chunk on only to a client that asked for it.

Rows are per clock minute, kept 48 hours, and are rolled into hourly rows kept 90 days: counts are added, hardware readings keep their mean and peak. A minute with no requests has no model row. The tables (`usage_*`, `gpu_*`, `host_*`) live in `GOETHE_HOME/state.db` and take a few MB at most. A linked node adds the minute rows the console has not confirmed yet to its heartbeat (`usage`, per model, GPU and host, never per API key); per-key totals stay local.

### `goethe login`

Links this node to a workspace. Prints a link and a code, then waits until the login is approved in the browser (Ctrl-C cancels; the code expires after 15 minutes). Logging in again replaces the current link. Each install has its own key (`GOETHE_HOME/platform/install_key`, made on the first login, kept through `goethe logout` and `goethe update`), so logging in again to the same workspace links the same node, with its name, cluster, shares and analytics, instead of adding a new one. A node removed on the console, another workspace, or a new install (after `goethe uninstall`) gets a new node. A running goethed sends its first heartbeat within a second; otherwise it starts with `goethe up`.

| Option | Description |
|---|---|
| `--authkey <msh_ak_...>` | Link without a browser, with an auth key from the console; `--authkey -` reads it from stdin so it stays out of the process list |
| `--lifetime <when>` | How long the login lasts: `never`, or a time such as `12h`, `30d` or `4w`. On a linked node it only changes that |
| `--platform-url <url>` | Another console URL (default: `GOETHE_PLATFORM_URL`, else `https://goethe.sh`). Must be HTTPS; plain HTTP only for `localhost` and `*.localhost` |

The installer takes the same flags: `curl -fsSL .../install.sh | bash -s -- --authkey msh_ak_...` (or `GOETHE_AUTHKEY=...`).

The credentials and the heartbeat state live in `GOETHE_HOME/platform/` (directory 0700, files 0600). Goethe refuses a credentials file or folder other users can read, a symlink, or one you do not own, and says how to fix it; it also refuses a `GOETHE_HOME` that every user can write to (a group-writable one is fine). When the console revokes the node (for example after it is removed there), heartbeats stop and `goethe status` says `Revoked` until the next `goethe login`.

### `goethe logout`

Ends this node's login and deletes the local credentials. The console keeps the node, offline and with its token revoked, so the next `goethe login` to that workspace links the same node again; remove it on the console to delete it and its analytics. When the console cannot be reached, the local link is still removed and the node shows offline on the console.
