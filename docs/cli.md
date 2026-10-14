# goethe command

Start a model, check the machine, see what runs on it.

```bash
curl -fsSL https://raw.githubusercontent.com/meshapplied/Goethe/main/install.sh | bash
goethe doctor
goethe serve
```

The installer needs `python3` 3.11 or newer and `git`; it clones this repository under `~/.local/share/goethe/src` and writes a
launcher to `~/.local/bin/goethe`. Everything else is standard library. Run the installer again to update.

## Commands

- `goethe serve [model] [--profile agent|chat|long]` installs what is missing (NInfer build, model download, head patch from
  `nvfp4-kernels/`), then starts an OpenAI, Responses and Anthropic compatible server on `127.0.0.1:8080`.
  `--explain` prints every setting with where it came from and why; `--dry-run` also prints the commands and runs nothing.
  Arguments after `--` go to `ninfer-serve` unchanged. If a healthy server already answers on the port, it says so and stops.
- `goethe connect <harness>` runs a coding agent against the server for one command. It sets environment variables and runs nothing else; it does not edit a config file. `--print` shows the command, arguments after `--` go to the agent. The agent has to be installed already.
  claude, codex, aider, opencode, qwen, goose, kilo and mini-swe-agent work this way. Claude Code is started with thinking off and `--effort medium`, because the server rejects Claude's own defaults. Codex is started with web search off, because the server has no such tool. A headless `qwen` waits forever for approval unless you pass `-y`. Kilo reads a config file and has no env override, so connect writes one under the state directory and points that run at it.
  continue, zed, cursor, jan, roo and cline are refused: each is configured through its own file or settings screen.
- `goethe list` shows each model and whether this GPU can run it: `tested`, `compatible`, or `unsupported` with the reason.
  `--json` for scripts.
- `goethe doctor` checks driver, GPU, memory, disk, build tools, the engine's dynamic libraries and network, and changes nothing. Each problem comes with a fix.
  Exit code 1 if anything failed. `--json` for scripts.
- `goethe bench` replays the bundled agent traces against a server that is already running. One stream, temperature 1.
  Decode tok/s and acceptance come from the engine log (`--log`). The first two requests are not in the mean.
- `goethe update --check` compares the installed NInfer commit, the down-projection patch and the model file to `NINFER_PIN`.
  It hashes the model file and changes nothing. Exit code 1 if anything differs. `--json` for scripts.

## What runs where

NInfer builds only for Blackwell (compute capability 12.0) and the 27B NVFP4 model needs about 30 GiB of VRAM. So today:

- RTX 5090: `tested`.
- Other 12.0 GPUs with 30 GiB or more, such as the RTX PRO 6000: `compatible`, not tested by us.
- Everything else, including 24 GB Blackwell cards and all Ampere and Ada cards: `unsupported`, with the reason. Community
  forks of NInfer exist for 3090 and 4090; goethe does not build them.

## Profiles

`--profile` picks lanes (concurrent requests) and context length. Everything else is the measured configuration from
`nvfp4-kernels/RESULTS.md`: DFlash2 with 7 draft tokens, fp8 KV cache, prefill chunk 1024. `goethe serve --explain` shows the
numbers for your machine.

- `agent` (default): 4 lanes, 100000 tokens. The configuration that was benchmarked.
- `chat`: 1 lane, 32768 tokens.
- `long`: 1 lane, 131072 tokens. A context the engine cannot reserve memory for is stepped down until one boots (262144 lands on 212992 on a 5090) and the result is remembered. Not benchmarked.

Speed does not depend on the lane count, but the exact tokens at temperature 0 do: the same replay gives different output with
1 and with 4 lanes (269.9 tok/s both), so compare runs with the same `lanes`.

## Configure

Precedence, highest first: command-line flag, environment variable `GOETHE_<KEY>`, `config.toml`, the profile.
Keys: `host`, `port`, `lanes`, `context`, `spec` (`dflash2` or `mtp`), `draft_tokens`, `kv_dtype`, `api_key`.
`--set KEY=VALUE` is a flag for any of them; `--host` and `--port` are shortcuts.

`~/.config/goethe/config.toml` (or under `$XDG_CONFIG_HOME`), unknown keys are an error:

```toml
[serve]
port = 9000

[profiles.agent]
lanes = 2
```

A host other than loopback needs an API key. If none is configured goethe generates one in
`~/.local/state/goethe/api_key` (mode 0600) and starts the server with it.

Files live in `~/.local/share/goethe` (or `$XDG_DATA_HOME`); an existing `~/.cache/goethe/nvfp4` or `$NVFP4_HOME` from the
nvfp4-kernels scripts is reused. There is no telemetry; the network is used for the model download, the
NInfer clone and the reachability check in `doctor`.
