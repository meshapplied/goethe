# goethe

`goethe doctor` checks the machine. `goethe serve` starts a model. `goethe list` says what this GPU can run.

```bash
curl -fsSL https://raw.githubusercontent.com/meshapplied/Goethe/main/install.sh | bash
goethe doctor
goethe serve
```

The installer wants Python 3.11 or newer, and git. It clones this repo to `~/.local/share/goethe/src` and writes `~/.local/bin/goethe`. The command itself is the standard library. Run the installer again to update.

## Commands

- `goethe serve [model] [--profile agent|chat|long]` starts the model you name. No name means Qwen3.8-27B: build NInfer if the binary is missing, download the weights if the file is missing, patch the draft head, and listen on `127.0.0.1:8080` with OpenAI, Responses, and Anthropic routes.
  The other names run the SGLang command in `examples/`. Goethe does not install SGLang. Ternary names download the pack and the base first. `qwen-flash-next-nvfp4` needs `NVFP4_MODEL_DIR` pointed at the tree.
  On the 27B install, an executable `$NVFP4_HOME/bin/ninfer-serve` is used as-is, and `install.sh` is skipped. Unpack with `tar -C "$NVFP4_HOME" -xf ninfer-serve-*-cu13-linux-x86_64.tar` so `bin/libcudart.so.13` sits next to the binary. That tarball runs on Ubuntu 24.04 with driver 580. The image is the same binary plus Ubuntu 24.04's FFmpeg and libcurl, so a different distro only has to supply the driver.
  `nvfp4-kernels/image.sh` turns that tarball into the `goethe:pack` image, which runs the same binary. A container run has to mount the model directory, `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-modeset`, `/usr/bin/nvidia-smi`, `libnvidia-ml.so.1`, and `libcuda.so.1`. The driver files come from the host. They are not copied into the image.
  `--explain` prints every 27B setting, where it came from, and why. `--dry-run` prints the commands and runs nothing. Arguments after `--` are appended to the server command. If the 27B port already answers `/health`, serve says so and exits.
- `goethe connect <harness>` runs one coding agent against the server. It sets environment variables and execs the agent. It does not edit the agent's config, except Kilo, which has no environment override: connect writes a config under the state directory and points that run at it. `--print` shows the command. Arguments after `--` go to the agent. The agent has to already be installed.
  claude, codex, aider, opencode, qwen, goose, kilo, and mini-swe-agent work this way. Claude is started with thinking off and `--effort medium`, because the server rejects Claude's own defaults. Codex is started with web search off, because the server has no such tool. A headless `qwen` waits forever for approval unless you pass `-y`.
  continue, zed, cursor, jan, roo, and cline are refused. Each one is configured in its own file or settings screen.
- `goethe list` prints each model and `tested`, `compatible`, or `unsupported`, with the reason. `--json` for scripts.
- `goethe doctor` checks the driver, the GPU, memory, disk, the build tools, the engine's shared libraries, and whether huggingface.co answers. It changes nothing. A failure exits 1, and each failure names a fix. `--json` for scripts.
- `goethe bench` replays the bundled agent traces against a server that is already up. One stream, temperature 1. Decode tok/s is token-weighted. Cache hit, time to first token, and acceptance come from the engine log (`--log`). The first two requests are left out of the mean.
- `goethe update --check` compares the installed NInfer commit, both patches, and the model file to `NINFER_PIN`. It hashes the model file and changes nothing. It prints `GREEDY_DIGEST` and does not replay it. Exit 1 if anything differs. `--json` for scripts.

## What runs where

NInfer builds only for Blackwell, compute capability 12.0. The 27B file needs a GPU with 30 GiB.

- RTX 5090: `tested`.
- Another 12.0 GPU with 30 GiB or more, such as the RTX PRO 6000: `compatible`. We have not run it.
- Anything else, including 24 GB Blackwell cards and every Ampere and Ada card: `unsupported`, and the line says why. Community forks of NInfer exist for the 3090 and the 4090. Goethe does not build them.

The SGLang models are `compatible` on a machine that has a GPU. We have not run them from this command.

## Profiles

`--profile` is a 27B knob. It picks lanes (concurrent requests) and context length. The rest is the configuration measured in `nvfp4-kernels/RESULTS.md`: DFlash2, 7 draft tokens, fp8 KV, prefill chunk 1024. `goethe serve --explain` prints the numbers for this machine.

- `agent` (default): 4 lanes, 100000 tokens. This is the configuration that was benchmarked.
- `chat`: 1 lane, 32768 tokens.
- `long`: 1 lane, 131072 tokens. If the engine cannot reserve that context it steps down until a boot succeeds (262144 lands on 212992 on a 5090) and the result is cached. Not benchmarked.

Lane count does not change the tok/s on the replay we ran, but it does change the tokens at temperature 0. The same replay at 1 lane and at 4 lanes both did 269.9 tok/s and did not emit the same tokens. Compare runs that use the same `lanes`.

## Configure

Highest precedence first: the flag, then `GOETHE_<KEY>`, then `config.toml`, then the profile.
Keys: `host`, `port`, `lanes`, `context`, `spec` (`dflash2` or `mtp`), `draft_tokens`, `kv_dtype`, `api_key`.
`--set KEY=VALUE` sets any of them. `--host` and `--port` are shortcuts.

`~/.config/goethe/config.toml`, or under `$XDG_CONFIG_HOME`. An unknown key is an error.

```toml
[serve]
port = 9000

[profiles.agent]
lanes = 2
```

A host other than loopback needs an API key. If you did not set one, goethe writes `~/.local/state/goethe/api_key` (mode 0600) and starts the server with it.

Files live in `~/.local/share/goethe`, or `$XDG_DATA_HOME`. An existing `~/.cache/goethe/nvfp4`, or `$NVFP4_HOME` from the nvfp4-kernels scripts, is reused. Doctor asks huggingface.co whether it answers. Serve downloads a model that is not on disk, and the 27B path clones NInfer. That is the network traffic.
