# Goethe user guide

Goethe serves large language models from your own NVIDIA GPUs through one OpenAI-compatible address. You install a model, load it, and share it with your apps, your team or your other machines, all from the `goethe` command.

## Contents

- [What you can use it for](#what-you-can-use-it-for)
- [Quick start](#quick-start)
- [Install](#install)
- [Add models](#add-models)
- [Load and run models](#load-and-run-models)
- [Share the API](#share-the-api)
- [Use the API](#use-the-api)
- [Everyday commands](#everyday-commands)
- [What Goethe supports](#what-goethe-supports)
- [Command reference](#command-reference)
- [Settings](#settings)
- [Troubleshooting](#troubleshooting)

## What you can use it for

- **A private OpenAI-compatible API for your team.** Load a model on a GPU workstation and give each person or app its own OpenAI-compatible API key.
- **Your home lab or office over a VPN.** Serve from a GPU box and use it from your laptop anywhere through Tailscale, WireGuard or ZeroTier.
- **Local AI for your own apps.** Point code that uses the OpenAI SDK at your machine instead of a cloud service.
- **Coding assistants and chat UIs.** Use Goethe as the backend for tools that accept an OpenAI-compatible endpoint, such as Open WebUI and many coding assistants.
- **Comparing models.** Load several models at once and switch between them by name.
- **Cloud GPUs.** Run it on RunPod, Vast.ai, Lambda or any other GPU server. Goethe prints the public address to use.

## Quick start

```bash
curl -fsSL https://meshapplied.com/install.sh | bash   # 1. install
goethe up                                              #    start Goethe
goethe install Qwen/Qwen3-8B --load                    # 2. add and load a model
goethe configure api                                   # 3. choose who can reach it
goethe key create --name my-laptop                     #    make a key (shown once)
goethe get api                                         #    print the address and loaded models
```

Then send a request, replacing the address, port and key with your own:

```bash
curl -k https://127.0.0.1:8443/v1/chat/completions \
  -H "Authorization: Bearer goethe_sk_..." \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hello!"}]}'
```

## Install

### Requirements

- Linux on x86_64 or ARM64, with glibc 2.30 or newer: Ubuntu 20.04+, Debian 11+ and similar.
- A [supported NVIDIA GPU](#hardware) with driver **580 or newer**. Check with `nvidia-smi`.
- `curl`, `zstd` and `openssl`. Most systems have them; on Ubuntu run `sudo apt-get install -y curl zstd openssl`.
- Disk space for your models, plus about 10 GB for Goethe itself.

### If the GPU driver is missing

Goethe brings its own CUDA compiler, so you do not install the CUDA toolkit. It needs only the NVIDIA driver 580 or newer.

The installer checks the driver first. If it is there and new enough, nothing happens. If a GPU has no driver, or the driver is older than 580, the installer says so and asks whether to install it. It uses `ubuntu-drivers` on Ubuntu and NVIDIA's CUDA repository on Debian and on RHEL, Rocky, AlmaLinux and Fedora, with `sudo` when you are not root. It then loads the driver, or tells you to reboot. Set `GOETHE_INSTALL_DRIVER=1` to install without asking, or `GOETHE_INSTALL_DRIVER=0` to never install. Inside a container it never installs, because the driver belongs on the host.

On a machine where Goethe is already installed, run `goethe doctor`. If the driver is not usable, its hardware line names the cause and the fix:

| What `goethe doctor` finds | Fix it names |
|---|---|
| No NVIDIA GPU visible | Attach the GPU to the VM (PCI passthrough or vGPU), or start the container with `--gpus all` |
| A GPU but no driver loaded | The driver package for your distribution, for example `sudo ubuntu-drivers install --gpgpu nvidia:580-server-open`, then a reboot |
| A driver older than 580 | The same, to update it |
| A driver but no `/dev/nvidia*` | Pass the devices into the container, or `sudo nvidia-modprobe -u -c=0` on the host |
| A driver but no `libcuda.so.1` or `libnvidia-ml.so.1` | The NVIDIA Container Toolkit for containers, or the matching user-space driver (`libnvidia-compute-<version>`) |

### Install Goethe

```bash
curl -fsSL https://meshapplied.com/install.sh | bash
```

No root is needed. The installer:
- checks your system first;
- downloads Goethe and verifies its signature;
- sets up everything it needs in one folder;
- starts Goethe;
- prints the address of your API.

Open a new terminal if the `goethe` command isn't found yet. Then check your setup:

```bash
goethe doctor
```

Every line should say `OK` (green), `INFO` or `WARN` (yellow). Each `WARN` or `FAIL` line tells you how to fix it.

Where Goethe keeps its files (models, keys, logs), in order of preference:
- `/workspace/.goethe` on cloud machines with a persistent `/workspace` volume;
- `/var/lib/goethe` when run as root;
- `~/.goethe` otherwise.

`goethe doctor` shows the folder in use on its `goethe_home` line.

### Updating and removing Goethe

```bash
goethe update --check      # is there a new version?
goethe update              # install it; loaded models keep serving
goethe uninstall           # remove Goethe; add --purge-models to delete your models too
```

## Add models

### Find models that fit

```bash
goethe recommend
```

This lists models from Goethe's catalog that fit your GPUs right now, with their size on disk. You aren't limited to this list: any [supported model](#model-families) on Hugging Face works.

### Install from Hugging Face

```bash
goethe install Qwen/Qwen3-8B
goethe install nvidia/Llama-3.1-8B-Instruct-NVFP4 --load      # install and load in one step
```

The model id is the Hugging Face repo name. Downloads are checked file by file, and an interrupted download resumes where it left off when you run the command again.

Useful options:
- `--load`: load it right after installing.
- `--ctx 32768`, `--max-conc 16`: default context length and number of parallel requests for this model.
- `--trust-remote-code`: needed for models that ship their own Python code. Only use it with models you trust.

### Gated, private or already downloaded models

Goethe only downloads public models. For a gated or private model (for example a Llama model that asks you to accept a license), download it yourself and hand it to Goethe:

1. Download it with your own Hugging Face account, for example with `huggingface-cli download <org>/<model> --local-dir <org>__<model>`.
2. Move the folder into Goethe's `models` folder, for example `~/.goethe/models/`. If you name the folder `<org>__<model>`, the model gets the id `<org>/<model>`; any other folder name gives `local/<folder-name>`.
3. Run:
   ```bash
   goethe scan
   ```
   Goethe checks each new folder and asks `Model found: <name> ... Add to goethe? [y/N]`. If a model can't be served, it tells you why, for example an unsupported architecture or missing weight files.

You can also register a folder anywhere on disk without moving it:

```bash
goethe install /data/models/my-finetune --name me/my-finetune
```

### See and remove models

```bash
goethe models                  # installed and catalog models, and whether each fits
goethe remove Qwen/Qwen3-8B    # unloads it and deletes the files Goethe downloaded
```

For folders you registered in place, `remove` only unregisters the model and prints the command to delete the files yourself.

## Load and run models

```bash
goethe load Qwen/Qwen3-8B
```

Goethe picks the GPUs, sizes the memory and waits until the model is ready. It ends with a line like:

```
RUNNING: Qwen/Qwen3-8B instance inst-... on GPU 0 port 30001 (45.2 s)
```

The first load of a model on a new machine takes a few extra minutes while GPU kernels are built. Later loads are faster.

Choosing where and how:

```bash
goethe load Qwen/Qwen3-8B --gpus 1            # a specific GPU
goethe load big/model --tp 2                  # split across 2 GPUs
goethe load big/model --gpus 0,1              # split across GPUs 0 and 1
goethe load Qwen/Qwen3-8B --ctx 65536         # longer context
goethe load Qwen/Qwen3-8B --max-conc 32       # more requests at the same time
goethe load Qwen/Qwen3-8B --replica           # another copy, to serve more traffic
goethe load Qwen/Qwen3-8B -- --chunked-prefill-size 4096   # extra SGLang options after --
```

To make settings the default for a model, so you don't have to type them each time:

```bash
goethe config Qwen/Qwen3-8B --ctx 65536 --max-conc 32
```

Model names are forgiving: `qwen/qwen3-8b` and the folder name `Qwen__Qwen3-8B` both mean `Qwen/Qwen3-8B`. If you mistype one, Goethe suggests the closest installed model.

### Several models at once

Load as many models as your GPUs can hold. They all share the same API address, and each request picks its model by name. `goethe ps` shows what is running:

```bash
goethe ps
```

```
INSTANCE               MODEL            SLOT   STATUS   GPUS  PORT   VRAM_GB  RESTARTS  IN_FLIGHT  TOK_PER_S
inst-18e1347e003bffe2  Qwen/Qwen3-8B    ...    RUNNING  0     30001  21.99    0         0          0.0
```

### Stopping a model

```bash
goethe unload Qwen/Qwen3-8B      # finishes requests in progress, then stops
goethe unload --all              # every model
goethe unload Qwen/Qwen3-8B --force   # stop now, without waiting
```

### Sharing a GPU with other software

If a GPU you pick is already used by other programs, Goethe lists them and asks before loading next to them:

```
GPU 0 has unmanaged process PID 3127 (python, user alice) using 10.3 GB VRAM
Goethe never stops processes it did not start. To stop it yourself: kill 3127
Load alongside these processes? [y/N]
```

Goethe never stops programs it didn't start. In scripts, add `--yes` to load anyway. To keep Goethe off a GPU completely:

```bash
goethe gpu deny 0      # never use GPU 0
goethe gpu allow 0     # allow it again
goethe gpu list
```

## Share the API

### Choose who can connect

```bash
goethe configure api
```

Pick one of three options with the arrow keys and press Enter:
- **Local only:** only programs on this machine.
- **VPN (recommended):** devices on your Tailscale, WireGuard or ZeroTier network. Goethe detects the VPN and allows its addresses. Programs on this machine then also use the VPN address.
- **Public network (not recommended):** any address that can reach this machine. Goethe shows a warning and your public IP, then asks which client addresses to allow. You also need to open the port in your firewall or router yourself.

If no API key exists yet, Goethe offers to create one at the end, then prints the address of your API.

In scripts:

```bash
goethe configure api --mode vpn --yes
goethe configure api --mode public --allow 203.0.113.7 --yes
```

### API keys

Every request needs a key; without a valid key the API answers `401`.

```bash
goethe key create --name my-laptop                 # shows the key once: store it
goethe key create --name ci --rpm 600 --expires 30d
goethe key create --name chat-only --models Qwen/Qwen3-8B
goethe key list                                    # keys with limits and today's usage
goethe key revoke <key-id>                         # takes effect immediately
```

Key options:
- `--rpm N`: requests per minute; the default is 120, and `0` means unlimited.
- `--expires`: an expiry such as `30d` or a date like `2027-01-31`.
- `--models a,b`: the key can only use these models; other models look missing to it.
- `--raw`: prints only the key, handy for `KEY=$(goethe key create --name x --raw)`.

### Allowed addresses

On top of keys, only allowed client addresses can connect at all. `goethe configure api` sets this up for you. To change it by hand:

```bash
goethe allowlist list
goethe allowlist add 192.168.1.0/24 --description "office LAN"
goethe allowlist add 203.0.113.7          # one address
goethe allowlist remove 192.168.1.0/24
```

### The address to share

```bash
goethe get api
```

```
API: https://100.90.97.111:8443/v1
Auth: Authorization: Bearer <key>   (goethe key create --name <name>)

MODEL                               STATUS   INSTANCES  GPUS
nvidia/Llama-3.1-8B-Instruct-NVFP4  RUNNING  1          gpu0
```

The API uses HTTPS with a certificate Goethe creates for itself. Clients either skip certificate verification (`curl -k`, or `verify=False` in Python) or trust Goethe's certificate, found at `<goethe folder>/nginx/certs/server.crt`.

### Short model names

```bash
goethe alias set chat Qwen/Qwen3-8B      # clients can now send "model": "chat"
goethe alias list
goethe alias remove chat
```

Change the model behind an alias at any time, and every client follows without changing its code.

## Use the API

The API is OpenAI-compatible at `https://<address>:<port>/v1`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat, with streaming, tools and reasoning output |
| `POST /v1/completions` | Plain text completion |
| `POST /v1/embeddings` | Embeddings, for embedding models |
| `GET /v1/models` | Models you can use with your key |
| `GET /v1/models/{id}` | One model |
| `GET /health` | Is the API up (no key needed) |

### curl

```bash
curl -k https://<address>:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Write a haiku."}],"max_tokens":128}'
```

If a long command breaks when you paste it, put the parts in variables first:

```bash
URL=https://<address>:8443/v1/chat/completions
BODY='{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"Hi"}]}'
curl -k "$URL" -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" -d "$BODY"
```

### Python (OpenAI SDK)

```python
import httpx
from openai import OpenAI

client = OpenAI(
    base_url="https://<address>:8443/v1",
    api_key="goethe_sk_...",
    http_client=httpx.Client(verify=False),   # or verify="/path/to/server.crt"
)
stream = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Explain GPUs in one paragraph."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

### Other tools

In tools that accept a custom OpenAI endpoint, set:
- the base URL to `https://<address>:<port>/v1`;
- the API key to your Goethe key;
- the model to a name from `goethe get api`.

If a tool can't skip certificate checks, add Goethe's certificate to the system's trusted certificates.

### Limits to know

- Each client address can send up to 20 requests per second.
- Each key has its own per-minute limit, set with `--rpm`.
- Request bodies can be up to 10 MB.
- When you unload a model, requests in progress finish first, and new requests get `503` with a `Retry-After` header.

## Everyday commands

```bash
goethe up          # start Goethe; it also starts automatically at boot from now on
goethe down        # stop Goethe and its models; it won't start at boot until the next goethe up
goethe ps          # what is running
goethe get api     # the address and loaded models
goethe logs Qwen/Qwen3-8B -f      # follow a model's log
goethe logs --daemon              # Goethe's own log
goethe doctor      # check the setup
goethe hw profile  # your GPUs, CPU, memory, disk and cloud provider
```

Models that were loaded when Goethe stopped come back by themselves when it starts again, including after a reboot. To start without that, run `goethe service restore --off`.

## What Goethe supports

### Hardware

| GPU family | Examples | Compute capability | Status |
|---|---|---|---|
| GPU family | Examples | Compute capability | NVFP4 path | Status |
|---|---|---|---|---|
| Blackwell (consumer and workstation) | GeForce RTX 5090, RTX PRO 6000 Blackwell | 12.0 | Native FP4 plus Marlin for small batches | Supported and tested |
| Blackwell (DGX Spark) | GB10 | 12.1 | Native FP4 plus Marlin for small batches | Supported |
| Blackwell data center | B100, B200, GB200 | 10.0 | Native FP4 | Supported |
| Blackwell Ultra data center | B300, GB300 | 10.3 | Native FP4 | Supported |
| Hopper | H100, H200, GH200 | 9.0 | Marlin (4-bit weights, 16-bit math) | Supported |
| Ampere | GeForce RTX 3090, RTX A6000, A40, A10 | 8.6 | Marlin (4-bit weights, 16-bit math) | Supported |
| Ampere data center | A100, A30 | 8.0 | Marlin (4-bit weights, 16-bit math) | Supported |

- **Several GPUs:** a model can be split across several GPUs in one machine (`--tp`), and mixture-of-experts models can also use expert parallelism (`--ep`).
- **Mixed GPUs:** one machine can hold GPUs of different architectures, for example two RTX PRO 6000 and an H100. Each model loads on GPUs it runs on, and a model split across GPUs always uses GPUs of one architecture; `goethe load --gpus` refuses a set that mixes them. Kernels that SGLang builds on first use are compiled for each architecture separately (`sm_80`, `sm_86`, `sm_90a`, `sm_100a`, `sm_103a`, `sm_120a`, `sm_121a`) and cached under `$GOETHE_HOME/cache/jit/<target>`. `goethe doctor` lists the architectures it finds.
- **Catalog models:** a model validated on Ampere (8.6) also runs on A100 (8.0) and Hopper (9.0), which take the same NVFP4 path with more shared memory; a model validated on consumer Blackwell (12.x) also runs on data-center Blackwell (10.x).
- **Operating system:** Linux on x86_64 or ARM64 with glibc 2.30 or newer, and NVIDIA driver 580 or newer. Goethe brings its own CUDA compiler.
- **Where it runs:** bare metal, virtual machines and containers (Docker, Podman, Kubernetes, LXC). Cloud GPU providers are detected automatically: RunPod, Vast.ai, Lambda, AWS, Google Cloud, Azure, Oracle Cloud, CoreWeave and Paperspace.

Other NVIDIA GPUs, such as the L40S or RTX 4090 (8.9) and anything older than Ampere, are not supported by this version.

### Model families

Goethe serves models in the Hugging Face format (`config.json` plus `.safetensors` weights). It supports the architectures of its built-in SGLang, more than 200 in total, including:

| Family | Examples |
|---|---|
| Llama | Llama 3, 3.1, 3.2, 3.3, Llama 4 |
| Qwen | Qwen 2.5, Qwen 3, Qwen3-Next, Qwen 3.5 MoE, QwQ, Qwen-VL, Qwen3-VL |
| DeepSeek | DeepSeek V2, V3, R1, V4 |
| Mistral | Mistral, Ministral, Mixtral, Mistral Small, Pixtral |
| Gemma | Gemma 2, Gemma 3, Gemma 4 |
| Phi | Phi-3, Phi-4, Phi-4 multimodal |
| GLM | GLM-4, GLM-4.5, GLM-4V |
| Others | GPT-OSS, Kimi K2.5 and K3, MiniMax M, Nemotron-H, Granite, Command R, OLMo, InternLM, InternVL, MiMo, EXAONE, ERNIE, Baichuan, Yi, StarCoder |
| Embeddings | BERT-style and LLM-based embedding models |

Dense, mixture-of-experts, hybrid (linear attention and Mamba) and vision-language models are all supported. `goethe scan` and `goethe install` tell you straight away if a model's architecture isn't supported.

### Quantization formats

| Format | Typical use | GPUs |
|---|---|---|
| BF16 / FP16 | Original model weights | All supported GPUs |
| **NVFP4** (NVIDIA ModelOpt, compressed-tensors) | 4-bit, about 4x smaller than BF16 | All supported GPUs, with Goethe's fast kernels |
| FP8 (ModelOpt, compressed-tensors, block FP8) | 8-bit, about 2x smaller | Native on Blackwell; Ampere runs it more slowly |
| MXFP4 | GPT-OSS models | Blackwell |
| AWQ, GPTQ | 4-bit community quantizations | All supported GPUs |
| INT8 (W8A8) | 8-bit integer | All supported GPUs |
| Mixed precision (ModelOpt) | NVFP4 and FP8 in one model | Blackwell |

You don't need to tell Goethe a model's format; it's detected from the model files. GGUF files and `.bin` (PyTorch pickle) weights are not supported; use the safetensors release of the model.

## Command reference

Most listing commands accept `--json` for scripts.

Exit codes:
- `0`: success.
- `1`: error.
- `2`: wrong usage.
- `3`: a confirmation was needed but there was no terminal to ask on; add `--yes`.

### Start and stop

| Command | What it does | Options |
|---|---|---|
| `goethe up` | Start Goethe and the API, and start them at boot | `--no-boot`: start now only |
| `goethe down` | Stop Goethe and its models, and stop starting at boot | `--keep-engines`: keep models serving while Goethe restarts |
| `goethe service enable` | Start Goethe at boot | |
| `goethe service disable` | Don't start Goethe at boot | |
| `goethe service status` | Show whether Goethe starts at boot | |
| `goethe service restore` | Bring loaded models back when Goethe starts | `--on`, `--off` |

### Models

| Command | What it does | Arguments and options |
|---|---|---|
| `goethe recommend` | Catalog models that fit your GPUs, with their size | `--json` |
| `goethe install <model>` | Download a Hugging Face model, or register a local folder | `<model>`: repo id or folder path<br>`--load`: load after installing<br>`--no-load`: only install<br>`--no-wait`: with `--load`, don't wait for it to be ready<br>`--ctx N`: default context length<br>`--max-conc N`: default parallel requests<br>`--tp N`: default number of GPUs (0 = automatic)<br>`--gpus 0,1`: preferred GPUs<br>`--trust-remote-code`: allow model-provided code<br>`--yes`: with `--load`, share GPUs without asking<br>`--force`: download again<br>`--reset`: forget saved settings for this model<br>`--no-verify`: skip the download checksum check<br>`--name ID`: id for a local folder<br>`--copy`: copy a local folder instead of linking it<br>`--sglang-arg FLAG`: extra SGLang option to save with the model |
| `goethe scan [folder]` | Add model folders you put into Goethe's `models` folder | `[folder]`: one folder (default: all)<br>`--yes`: add all supported models without asking<br>`--json` |
| `goethe models` | Installed and catalog models, and whether each fits | `--installed`, `--available`, `--json` |
| `goethe config <model>` | Change a model's default settings | `--ctx N`, `--max-conc N`, `--tp N`, `--gpus 0,1`, `--sglang-arg FLAG`, `--clear-sglang-args` |
| `goethe remove <model>` | Unload and delete a model (or unregister a folder) | `--yes`: don't ask<br>`--force`: don't wait for requests in progress |

### Running

| Command | What it does | Arguments and options |
|---|---|---|
| `goethe load <model>` | Load a model and wait until it's ready | `--gpus 0,1`: which GPUs<br>`--tp N`: number of GPUs<br>`--ep N`: expert parallel size<br>`--ctx N`: context length<br>`--max-conc N`: parallel requests<br>`--replica`: start another copy<br>`--yes` / `--force`: share busy GPUs without asking<br>`--no-wait`: return right away<br>`--json`<br>`-- <SGLang options>`: extra engine options |
| `goethe unload [model]` | Stop a model, letting requests in progress finish | `--all`: every model<br>`--force`: stop right away |
| `goethe ps` | Running models with GPU, status and speed | `--json` |
| `goethe logs [model]` | Show a model's log | `-f`: follow<br>`-n N`: number of lines (default 100)<br>`--daemon`: Goethe's own log<br>`--gateway`: the API gateway's log<br>`--prune`: clean up old logs |

### Sharing

| Command | What it does | Arguments and options |
|---|---|---|
| `goethe configure api` | Guided setup: local, VPN or public | `--mode local\|vpn\|public`<br>`--vpn-ip IP`: VPN address to use<br>`--allow CIDR`: client address to allow (repeatable)<br>`--yes`: no questions |
| `goethe get api` | API address, how to authenticate, and loaded models | `--json` |
| `goethe key create` | Create an API key (shown once) | `--name NAME` (required)<br>`--rpm N`: requests per minute (default 120, 0 = unlimited)<br>`--expires 30d\|DATE`<br>`--models a,b`: limit to these models<br>`--admin`: access to admin health info<br>`--raw`: print only the key |
| `goethe key list` | Keys with limits and today's usage | `--json` |
| `goethe key revoke <id>` | Revoke a key | `<id>`: from `goethe key list` |
| `goethe allowlist add <cidr>` | Allow a client address or network | `--description TEXT` |
| `goethe allowlist remove <cidr>` | Remove an allowed address | |
| `goethe allowlist list` | Allowed addresses | |
| `goethe alias set <alias> <model>` | Give a model a short name for clients | |
| `goethe alias list` | Aliases | |
| `goethe alias remove <alias>` | Remove an alias | |
| `goethe gateway status` | Where the API listens and how many addresses are allowed | `--json` |
| `goethe gateway bind <addr>` | Listen on `127.0.0.1`, `0.0.0.0` or one IP of this machine | |
| `goethe domain add <domain>` | Use your own domain name with a real certificate (preview) | `<domain>`: must point to this machine |

### Hardware and system

| Command | What it does | Arguments and options |
|---|---|---|
| `goethe hw profile` | GPUs, CPU, memory, disk, OS, container and cloud | `--json` |
| `goethe hw pick-gpus` | Show which GPUs a load would use right now | `--tp N`, `--model ID` |
| `goethe gpu list` | GPUs with memory use and allow/deny status | `--json` |
| `goethe gpu deny <index>` | Keep Goethe off a GPU | |
| `goethe gpu allow <index>` | Allow a GPU again | |
| `goethe doctor` | Check the setup, with a fix for each problem | `--json`<br>`--probe-public`: also test the public address |
| `goethe update` | Install the latest Goethe | `--check`: only show what's available<br>`--yes`: don't ask<br>`--engine`: also update the inference engine<br>`--allow-downgrade` |
| `goethe uninstall` | Remove Goethe | `--purge-models`: also delete models<br>`--yes` |
| `goethe --version` | Version and build | |

## Settings

Most people never need these. Set them in your shell, or in `goethe.env` inside Goethe's folder so that Goethe's background service picks them up.

| Variable | What it changes |
|---|---|
| `GOETHE_HOME` | Where Goethe keeps its files |
| `GOETHE_PUBLIC_PORT` | API port; the default is 8443, and another port is used if it's taken |
| `GOETHE_NO_AUTOSTART=1` | Never set up start at boot |
| `GOETHE_PUBLIC_IP` | Your public IP, shown by `goethe configure api` in public mode |
| `GOETHE_MAX_BODY_MB` | Largest request size, in MB (default 10) |
| `NO_COLOR=1` | Turn off colored output |

## Troubleshooting

**`goethe: command not found` after installing.** Open a new terminal, or run `export PATH="$HOME/.local/bin:$PATH"`.

**`goethed is not running`.** Run `goethe up`.

**`model ... is not installed; did you mean ...?`** Use the id from `goethe models`.

**A load fails.** Goethe prints the reason and the end of the model's log. For the full log, run `goethe logs <model> -n 200`. Common causes:
- **`NVCC version must be at least 12.9`, or a CUDA compiler error:** Goethe gives each engine its own CUDA compiler unless `CUDA_HOME` or `CUDA_PATH` is set in goethed's environment or in the model card. Remove that setting so the bundled compiler is used, then run `rm -rf ~/.cache/sglang/.cache/flashinfer` and load again.
- **Not enough GPU memory:** use a smaller context (`--ctx`), fewer parallel requests (`--max-conc`), more GPUs (`--tp 2`), or a quantized version of the model.
- **The model needs its own code:** reinstall it with `--trust-remote-code`.

**The first request is slow.** The first request after a load can take a while, because GPU kernels are being prepared. Requests after that are fast.

**Requests fail:**

| Answer | Meaning | Fix |
|---|---|---|
| No connection | Wrong address or port, `http` instead of `https`, or a firewall | Use the address from `goethe get api` |
| `403 Forbidden` | Your address isn't allowed | `goethe allowlist add <your-ip>` |
| `401` | Missing or wrong key | Send `Authorization: Bearer <key>` |
| `404 Model Not Found` | Unknown model name, or your key isn't allowed that model | Use a name from `goethe get api` |
| `429` | Too many requests | Slow down, or raise the key's `--rpm` |
| `503` | The model isn't ready, or is being unloaded | Wait, or check `goethe ps` |

**Something else.** Run `goethe doctor` and follow the hints. `goethe logs --daemon` shows Goethe's own log.
