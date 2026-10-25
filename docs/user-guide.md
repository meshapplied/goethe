# Goethe user guide

Goethe serves large language models from your own NVIDIA GPUs through one OpenAI-compatible address. You install a model, load it, and share it with your apps, your team or your other machines, all from the `goethe` command.

## Contents

- [What you can use it for](#what-you-can-use-it-for)
- [Quick start](#quick-start)
- [Install](#install)
- [First run: goethe start](#first-run-goethe-start)
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
curl -fsSL https://meshapplied.com/install.sh | bash   # install
goethe start                                           # start Goethe and set it up step by step
```

`goethe start` walks you through installing a model, serving it and sharing the API (see [First run](#first-run-goethe-start)). The same with plain commands:

```bash
goethe up                                              # 1. start Goethe
goethe install --model Qwen/Qwen3-8B --load            # 2. add and load a model
goethe configure api                                   # 3. choose who can reach it
goethe key --create my-laptop                          #    make a key (goethe api lists it)
goethe api                                             #    print the address, keys and loaded models
```

`goethe status` tells you at any time whether Goethe is running, with the API address and the names and ids of your keys. `goethe up` prints the same summary.

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
- `curl`, `zstd`, `openssl` and a C++ compiler (`g++`). The installer installs any that are missing.
- Disk space for your models, plus about 12 GB for Goethe itself.

### What the installer takes care of

Goethe brings its own CUDA compiler, so you never install the CUDA toolkit. Everything else the installer checks before it downloads anything. It fixes what it can and names the fix for the rest; a GPU problem never stops the install.

| What it finds | What it does |
|---|---|
| Missing `curl`, `zstd`, `openssl` or CA certificates | Installs them with apt, dnf, yum or zypper |
| No C++ compiler (needed to build GPU kernels the first time a model loads) | Installs `build-essential` (Debian, Ubuntu) or `gcc-c++` (RHEL, Fedora, SUSE) |
| A GPU without a driver, or a driver older than 580 | Installs the open NVIDIA driver 580+ (`ubuntu-drivers` on Ubuntu, NVIDIA's CUDA repository on Debian and the RHEL family), replacing a driver installed from NVIDIA's `.run` file, then loads it or asks for a reboot |
| The driver is installed but not loaded | Loads it (`modprobe nvidia`); if Secure Boot blocks it, says how to enroll the key |
| NVIDIA's proprietary kernel module with a Blackwell GPU (RTX 50, RTX PRO 6000, B200) | Switches to the open kernel module, which Blackwell needs |
| A GPU that did not start (for example after the card crashed) | Explains: a full power cycle (on a VM, stop and start the instance), power cables, Above 4G decoding in the BIOS |
| Some GPUs did not start, or are older than Goethe supports | Says which ones are skipped |
| No NVIDIA GPU (a VM without passthrough, a container without `--gpus all`) | Says how to attach one; Goethe installs anyway |
| Not enough disk space | Stops before downloading, with the numbers |
| A new release still reaching the package host's cache | Waits and retries for up to 12 minutes |

Before installing packages or a driver, the installer asks (as root it installs the small prerequisites and the compiler without asking). `GOETHE_INSTALL_DEPS=1` and `GOETHE_INSTALL_DRIVER=1` say yes in advance, `=0` says no. Inside a container it never installs a driver, because the driver belongs on the host. `GOETHE_REQUIRE_GPU=1` makes a missing or broken GPU stop the install instead of warning.

On a machine where Goethe is already installed, `goethe doctor` names the cause and the fix for each of these, `goethe start` explains them before the first step, and a model that fails to load because of one of them says so:

| What `goethe doctor` finds | Fix it names |
|---|---|
| No NVIDIA GPU visible | Attach the GPU to the VM (PCI passthrough or vGPU), or start the container with `--gpus all` |
| A GPU but no driver loaded | The driver package for your distribution, for example `sudo ubuntu-drivers install --gpgpu nvidia:580-server-open`, then a reboot |
| A driver older than 580 | The same, to update it |
| The proprietary module with a Blackwell GPU | The open module (`nvidia-driver-580-server-open`, `nvidia-open`) |
| A GPU the driver did not start | A full power cycle, power cables, Above 4G decoding |
| A driver but no `/dev/nvidia*` | Pass the devices into the container, or `sudo nvidia-modprobe -u -c=0` on the host |
| A driver but no `libcuda.so.1` or `libnvidia-ml.so.1` | The NVIDIA Container Toolkit for containers, or the matching user-space driver (`libnvidia-compute-<version>`) |
| No C++ compiler, or gcc older than 9 | `sudo apt install build-essential` or `sudo dnf install gcc-c++` |
| gcc newer than 15 | Nothing: Goethe tells nvcc to accept it, and names `gcc-15` in case a build still fails |

### Install Goethe

```bash
curl -fsSL https://meshapplied.com/install.sh | bash
```

No root is needed. The installer:
- checks your system first;
- downloads Goethe and verifies its signature;
- sets up everything it needs in one folder;
- ends with the next step, `goethe start`.

The installer does not start Goethe. `goethe start` (or `goethe up`) does that. When you run the installer again to upgrade a Goethe that is running, it swaps in the new version and keeps your models serving.

Open a new terminal if the `goethe` command isn't found yet. To check your setup:

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
goethe uninstall           # remove Goethe and the models it installed; add --keep-models to keep them
```

## First run: goethe start

```bash
goethe start
```

`goethe start` starts Goethe, which from then on also starts at every boot, unless it was stopped with `goethe down` when the machine shut down. Then it offers two ways to go on.

**Guided setup (recommended)** takes you through three steps, choosing with the arrow keys and Enter:

1. **Install a model.** Pick one from the models that fit your GPUs, type any Hugging Face model id, or pick a gated or private model, for which you type a Hugging Face token. You can also run a model that is already on this machine: Goethe lists the ones it finds in its models folder and in the Hugging Face cache, or you name a folder.
2. **Serve through an API.** Goethe loads the model on the GPUs that fit it and waits until it answers.
3. **Configure API access.** Choose who can reach the API (this machine only, your VPN or a public address), then name the first API key. Goethe prints the base URL, the model name, the key and a `curl` command to try it.

You can stop after step 1 and continue later with `goethe start` or `goethe load --model <model>`.

**Advanced setup** lets you do each part on its own, in any order: choose which GPUs Goethe may use, install a model, run your own model, serve an installed model, configure API access and create API keys. Choose Finish when you are done.

Without an interactive terminal, for example in a script, `goethe start` prints the same three steps as commands.

## Add models

### Find models that fit

```bash
goethe recommend
```

This shows the VRAM of the GPUs Goethe may use, counted as if they were empty, then the catalog models they can run, grouped by the VRAM they take (up to 24 GB, 48 GB, 96 GB and so on), with their GPUs and download size. You aren't limited to this list: any [supported model](#model-families) on Hugging Face works.

### Install from Hugging Face

```bash
goethe install --model Qwen/Qwen3-8B
goethe install --model nvidia/Llama-3.1-8B-Instruct-NVFP4 --load      # install and load in one step
```

The model id is the Hugging Face repo name. Downloads are checked file by file, and an interrupted download resumes where it left off when you run the command again.

Useful options:
- `--load`: load it right after installing.
- `--ctx 32768`, `--max-conc 16`: default context length and number of parallel requests for this model.
- `--trust-remote-code`: needed for models that ship their own Python code. Only use it with models you trust.
- `--hf-token`: ask for your Hugging Face token, for [gated and private models](#gated-and-private-models).

### Gated and private models

A gated model (for example a Llama model that asks you to accept a license) or a private one needs your Hugging Face token:

1. Accept the model's license on its page on huggingface.co, and create a read token at https://huggingface.co/settings/tokens.
2. Install it, typing the token when asked (it is not shown on screen):
   ```bash
   goethe install --model <org>/<model> --hf-token
   ```
   In `goethe start`, choose "A private or gated model" instead.

Goethe checks the token with Hugging Face and uses it for that download only. It never saves the token.

### Models you downloaded yourself

To add a model folder you placed on the machine yourself, move it into Goethe's `models` folder, for example `~/.goethe/models/`. If you name the folder `<org>__<model>`, the model gets the id `<org>/<model>`; any other folder name gives `local/<folder-name>`. Then run:

```bash
goethe scan
```

Goethe checks each new folder and asks `Model found: <name> ... Add to goethe? [y/N]`. If a model can't be served, it tells you why, for example an unsupported architecture or missing weight files.

You can also register a folder anywhere on disk without moving it:

```bash
goethe install --model /data/models/my-finetune --name me/my-finetune
```

### See and remove models

```bash
goethe models                          # installed and catalog models, and whether each fits
goethe remove --model Qwen/Qwen3-8B    # unloads it and deletes the files Goethe downloaded
```

`goethe models` lists installed models first, then catalog models in the same VRAM groups, then those that are too large for the allowed GPUs or need another GPU architecture. Both lists show what the machine can run, not what fits next to the models loaded now; `goethe load` checks the free memory. The FIT ON THIS MACHINE column says `fits: N.N GB per GPU`, or `does not fit` with the reason. DISK is the size on disk.

For folders you registered in place, `remove` only unregisters the model and prints the command to delete the files yourself.

## Load and run models

```bash
goethe load --model Qwen/Qwen3-8B
```

Goethe picks the GPUs, sizes the memory and waits until the model is ready. It ends with a line like:

```
RUNNING: Qwen/Qwen3-8B instance inst-... on GPU 0 port 30001 (45.2 s)
```

The first load of a model on a new machine takes a few extra minutes while GPU kernels are built. Later loads are faster.

Choosing where and how:

```bash
goethe load --model Qwen/Qwen3-8B --gpus 1            # a specific GPU
goethe load --model big/model --tp 2                  # split across 2 GPUs
goethe load --model big/model --gpus 0,1              # split across GPUs 0 and 1
goethe load --model Qwen/Qwen3-8B --ctx 65536         # longer context
goethe load --model Qwen/Qwen3-8B --max-conc 32       # more requests at the same time
goethe load --model Qwen/Qwen3-8B --replica           # another copy, to serve more traffic
goethe load --model Qwen/Qwen3-8B -- --chunked-prefill-size 4096   # extra SGLang options after --
```

To make settings the default for a model, so you don't have to type them each time:

```bash
goethe config --model Qwen/Qwen3-8B --ctx 65536 --max-conc 32
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
goethe unload --model Qwen/Qwen3-8B           # finishes requests in progress, then stops
goethe unload --all                           # every model
goethe unload --model Qwen/Qwen3-8B --force   # stop now, without waiting
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
goethe gpu --deny gpu0 gpu1   # never use gpu0 and gpu1
goethe gpu --allow gpu0       # allow gpu0 again
goethe gpu               # GPUs with memory use and allow/deny status
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
goethe key --create my-laptop                       # shows the key once: store it
goethe key --create ci --rpm 600 --expires 30d
goethe key --create chat-only --models Qwen/Qwen3-8B
goethe key --create ops --expires 90d --rpm 300 --admin
goethe key                                          # keys with limits and today's usage
goethe key --revoke <key-id>                        # takes effect immediately
```

Options for `--create`:
- `--expires 30d`: expires in 30 days; or a date, such as `--expires 2027-01-31`.
- `--rpm N`: requests per minute; the default is 120, and `0` means unlimited.
- `--models a,b`: the key can only use these models; other models look missing to it.
- `--admin`: an admin key, which can read the health endpoints and use every model.
- `--raw`: prints only the key, for scripts:

```bash
KEY=$(goethe key --create ci --raw)
```

`goethe key --help` shows these options too.

### Allowed addresses

On top of keys, only allowed client addresses can connect at all. `goethe configure api` sets this up for you. To change it by hand:

```bash
goethe allowlist
goethe allowlist --add 192.168.1.0/24 --description "office LAN"
goethe allowlist --add 203.0.113.7          # one address
goethe allowlist --remove 192.168.1.0/24
```

### The address to share

```bash
goethe api
```

```
API endpoint: https://100.90.97.111:8443/v1

Models running
MODEL                               STATUS   INSTANCES  GPUS
nvidia/Llama-3.1-8B-Instruct-NVFP4  RUNNING  1          gpu0

API keys
NAME       API KEY                                    ADMIN  EXPIRES
my-laptop  goethe_sk_0123456789abcdef0123456789abcdef  no     never

Query the API:

	→ curl -k https://100.90.97.111:8443/v1/chat/completions -H "Authorization: Bearer <your_bearer_token>" ...
```

`goethe api` lists each key in full; keys created before Goethe 0.1.1 show only their first characters.

When there is no key yet, the allowlist is empty, Goethe is not running or no model is loaded, it says so instead and prints the command that fixes it.

The API uses HTTPS with a certificate Goethe creates for itself. Clients either skip certificate verification (`curl -k`, or `verify=False` in Python) or trust Goethe's certificate, found at `<goethe folder>/nginx/certs/server.crt`.

### Short model names

```bash
goethe alias --set chat --model Qwen/Qwen3-8B      # clients can now send "model": "chat"
goethe alias
goethe alias --remove chat
```

Change the model behind an alias at any time, and every client follows without changing its code.

### Your own domain name

To reach the API by a name such as `api.example.com` instead of its IP address, add a DNS A record that points the name at the address the API listens on. Goethe needs no setup for this. Run `goethe domain` to see the record with this machine's address filled in:

```text
api.example.com  A  100.70.155.114
```

The address can be public, or private such as a VPN or LAN address; a private one only resolves to clients on that network. Clients then use `https://api.example.com:8443/v1` with the same certificate settings as for the IP address. To check that the name works:

```bash
goethe domain --check api.example.com     # the name points at the API and the gateway answers there
```

## Use the API

The API is at `https://<address>:<port>/v1`. It speaks three schemas at the same time, so OpenAI, Anthropic and LiteLLM clients all work against the same address and key:

- **OpenAI:** the routes below.
- **Anthropic:** `POST /v1/messages`. Errors on this route come back in Anthropic's format.
- **LiteLLM proxy:** every route also works without `/v1` (for example `/chat/completions`), plus `/model/info`, `/model_group/info`, `/health/liveliness` and `/health/readiness`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat, with streaming, tools and reasoning output |
| `POST /v1/completions` | Plain text completion |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/messages` | Anthropic Messages API, with streaming |
| `POST /v1/messages/count_tokens` | Anthropic token count |
| `POST /v1/embeddings` | Embeddings, for embedding models |
| `POST /v1/rerank` | Reranking, for reranker models |
| `GET /v1/models` | Models you can use with your key |
| `GET /v1/models/{id}` | One model |
| `GET /model/info`, `GET /model_group/info` | Models in LiteLLM's format |
| `GET /health` | Is the API up (no key needed) |

Send the key in whichever header your client uses: `Authorization: Bearer <key>` (OpenAI), `x-api-key` (Anthropic), `x-litellm-api-key` (LiteLLM, with or without `Bearer`) or `api-key` (Azure-style).

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

### Anthropic SDK

```python
import httpx
from anthropic import Anthropic

client = Anthropic(
    base_url="https://<address>:8443",
    api_key="goethe_sk_...",
    http_client=httpx.Client(verify=False),
)
reply = client.messages.create(
    model="Qwen/Qwen3-8B",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.content[0].text)
```

### LiteLLM

Use Goethe as a LiteLLM proxy, with the `litellm_proxy/` prefix:

```python
import litellm

reply = litellm.completion(
    model="litellm_proxy/Qwen/Qwen3-8B",
    api_base="https://<address>:8443",
    api_key="goethe_sk_...",
    messages=[{"role": "user", "content": "Hello!"}],
    ssl_verify=False,
)
```

Or add it to a LiteLLM proxy's `config.yaml` as an OpenAI-compatible backend (`model: openai/Qwen/Qwen3-8B`, `api_base: https://<address>:8443/v1`).

### Other tools

In tools that accept a custom OpenAI endpoint, set:
- the base URL to `https://<address>:<port>/v1`;
- the API key to your Goethe key;
- the model to a name from `goethe api`.

If a tool can't skip certificate checks, add Goethe's certificate to the system's trusted certificates.

### Limits to know

- Each client address can send up to 20 requests per second.
- Each key has its own per-minute limit, set with `--rpm`.
- Request bodies can be up to 10 MB.
- When you unload a model, requests in progress finish first, and new requests get `503` with a `Retry-After` header.

## Everyday commands

```bash
goethe up          # start Goethe; it also starts automatically at boot from now on
goethe status      # running or not, the API address and your keys
goethe down        # stop Goethe and its models; it won't start at boot until the next goethe up
goethe ps          # what is running
goethe api         # the address, keys and loaded models
goethe logs --model Qwen/Qwen3-8B -f      # follow a model's log
goethe logs --daemon                      # Goethe's own log
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
| Ada | GeForce RTX 4090, RTX 6000 Ada, L40S, L4 | 8.9 | Marlin (4-bit weights, 16-bit math) | Supported |
| Ampere | GeForce RTX 3090, RTX A6000, A40, A10 | 8.6 | Marlin (4-bit weights, 16-bit math) | Supported |
| Ampere data center | A100, A30 | 8.0 | Marlin (4-bit weights, 16-bit math) | Supported |

- **Several GPUs:** a model can be split across several GPUs in one machine (`--tp`), and mixture-of-experts models can also use expert parallelism (`--ep`).
- **Mixed GPUs:** one machine can hold GPUs of different architectures, for example two RTX PRO 6000 and an H100. Each model loads on GPUs it runs on, and a model split across GPUs always uses GPUs of one architecture; `goethe load --gpus` refuses a set that mixes them. Kernels that SGLang builds on first use are compiled for each architecture separately (`sm_80`, `sm_86`, `sm_89`, `sm_90a`, `sm_100a`, `sm_103a`, `sm_120a`, `sm_121a`) and cached under `$GOETHE_HOME/cache/jit/<target>`. `goethe doctor` lists the architectures it finds.
- **Catalog models:** a model validated on Ampere (8.6) also runs on A100 (8.0), Ada (8.9) and Hopper (9.0), which take the same NVFP4 path with more shared memory; a model validated on consumer Blackwell (12.x) also runs on data-center Blackwell (10.x).
- **Operating system:** Linux on x86_64 or ARM64 with glibc 2.30 or newer, and NVIDIA driver 580 or newer. Goethe brings its own CUDA compiler.
- **Where it runs:** bare metal, virtual machines and containers (Docker, Podman, Kubernetes, LXC). Cloud GPU providers are detected automatically: RunPod, Vast.ai, Lambda, AWS, Google Cloud, Azure, Oracle Cloud, CoreWeave and Paperspace.

GPUs older than Ampere (Volta, Turing and earlier) are not supported; Goethe names them and skips them.

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

Every value goes after a flag: `goethe unload --model <model>`, `goethe gpu --allow gpu0 gpu1`. Type a value without its flag and Goethe prints the corrected command instead of running anything.

Exit codes:
- `0`: success.
- `1`: error.
- `2`: wrong usage.
- `3`: a confirmation was needed but there was no terminal to ask on; add `--yes`.

### Start and stop

| Command | What it does | Options |
|---|---|---|
| `goethe start` | Start Goethe and set it up step by step: install a model, serve it, configure API access | |
| `goethe up` | Start Goethe and the API, and start them at boot; prints the same summary as `goethe status` | `--no-boot`: start now only |
| `goethe status` | Whether Goethe is running, the API address, and your keys' names and ids | |
| `goethe down` | Stop Goethe and its models, and stop starting at boot | `--keep-engines`: keep models serving while Goethe restarts |
| `goethe service enable` | Start Goethe at boot | |
| `goethe service disable` | Don't start Goethe at boot | |
| `goethe service status` | Show whether Goethe starts at boot | |
| `goethe service restore` | Bring loaded models back when Goethe starts | `--on`, `--off` |

### Models

| Command | What it does | Arguments and options |
|---|---|---|
| `goethe recommend` | Catalog models that fit your GPUs, with their size | `--json` |
| `goethe install --model <model>` | Download a Hugging Face model, or register a local folder | `--model`: repo id or folder path<br>`--load`: load after installing<br>`--no-load`: only install<br>`--no-wait`: with `--load`, don't wait for it to be ready<br>`--ctx N`: default context length<br>`--max-conc N`: default parallel requests<br>`--tp N`: default number of GPUs (0 = automatic)<br>`--gpus 0,1`: preferred GPUs<br>`--trust-remote-code`: allow model-provided code<br>`--yes`: with `--load`, share GPUs without asking<br>`--force`: download again<br>`--reset`: forget saved settings for this model<br>`--no-verify`: skip the download checksum check<br>`--name ID`: id for a local folder<br>`--copy`: copy a local folder instead of linking it<br>`--sglang-arg FLAG`: extra SGLang option to save with the model<br>`--hf-token`: ask for a Hugging Face token, used for this download only |
| `goethe scan` | Add model folders you put into Goethe's `models` folder | `--dir FOLDER`: one folder (default: all)<br>`--yes`: add all supported models without asking<br>`--json` |
| `goethe models` | Installed and catalog models, and whether each fits | `--installed`, `--available`, `--json` |
| `goethe config --model <model>` | Change a model's default settings | `--ctx N`, `--max-conc N`, `--tp N`, `--gpus 0,1`, `--sglang-arg FLAG`, `--clear-sglang-args` |
| `goethe remove --model <model>` | Unload and delete a model (or unregister a folder) | `--yes`: don't ask<br>`--force`: don't wait for requests in progress |

### Running

| Command | What it does | Arguments and options |
|---|---|---|
| `goethe load --model <model>` | Load a model and wait until it's ready | `--gpus 0,1`: which GPUs<br>`--tp N`: number of GPUs<br>`--ep N`: expert parallel size<br>`--ctx N`: context length<br>`--max-conc N`: parallel requests<br>`--replica`: start another copy<br>`--yes` / `--force`: share busy GPUs without asking<br>`--no-wait`: return right away<br>`--json`<br>`-- <SGLang options>`: extra engine options |
| `goethe unload --model <model>` | Stop a model, letting requests in progress finish | `--all`: every model instead<br>`--force`: stop right away |
| `goethe ps` | Running models with GPU, status and speed | `--json` |
| `goethe logs --model <model>` | Show a model's log | `-f`: follow<br>`-n N`: number of lines (default 100)<br>`--daemon`: Goethe's own log<br>`--gateway`: the API gateway's log<br>`--prune`: clean up old logs |

### Sharing

| Command | What it does | Arguments and options |
|---|---|---|
| `goethe configure api` | Guided setup: local, VPN or public | `--mode local\|vpn\|public`<br>`--vpn-ip IP`: VPN address to use<br>`--allow CIDR`: client address to allow (repeatable)<br>`--yes`: no questions |
| `goethe api` | API address, how to authenticate, models running, your keys in full and a curl example | `--json` |
| `goethe key --create <name>` | Create an API key (goethe api lists it again) | `--rpm N`: requests per minute (default 120, 0 = unlimited)<br>`--expires 30d\|DATE`<br>`--models a,b`: limit to these models<br>`--admin`: access to admin health info<br>`--raw`: print only the key |
| `goethe key` | Keys with limits and today's usage | `--json` |
| `goethe key --revoke <id>` | Revoke a key | `<id>`: the KEY ID from `goethe key` |
| `goethe allowlist --add <cidr>` | Allow a client address or network | `--description TEXT` |
| `goethe allowlist --remove <cidr>` | Remove an allowed address | |
| `goethe allowlist` | Allowed addresses | |
| `goethe alias --set <alias> --model <model>` | Give a model a short name for clients | |
| `goethe alias` | Aliases | |
| `goethe alias --remove <alias>` | Remove an alias | |
| `goethe gateway` | Where the API listens and how many addresses are allowed | `--json` |
| `goethe gateway --bind <addr>` | Listen on `127.0.0.1`, `0.0.0.0` or one IP of this machine | |
| `goethe domain` | How to serve the API under your own domain name (preview) | |
| `goethe domain` | How to reach the API by a domain name (a DNS A record) | |
| `goethe domain --check <domain>` | Check that the name points at the API and the gateway answers | `<domain>` |
| `goethe domain --check <domain>` | Check that the domain points here and answers with a valid certificate | |

### Hardware and system

| Command | What it does | Arguments and options |
|---|---|---|
| `goethe hw profile` | GPUs, CPU, memory, disk, OS, container and cloud | `--json` |
| `goethe hw pick-gpus` | Show which GPUs a load would use right now | `--tp N`, `--model ID` |
| `goethe gpu` | GPUs with memory use and allow/deny status | `--json` |
| `goethe gpu --deny gpu<index>` | Keep Goethe off a GPU | |
| `goethe gpu --allow gpu<index>` | Allow a GPU again | |
| `goethe doctor` | Check the setup, with a fix for each problem | `--json`<br>`--probe-public`: also test the public address |
| `goethe update` | Install the latest Goethe | `--check`: only show what's available<br>`--yes`: don't ask<br>`--engine`: also update the inference engine<br>`--allow-downgrade` |
| `goethe uninstall` | Remove Goethe and the models it installed; folders of your own files that you registered stay | `--keep-models`: keep the installed models<br>`--yes` |
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

**`goethed is not running`.** Run `goethe up`. `goethe status` shows whether it is running.

**`model ... is not installed; did you mean ...?`** Use the id from `goethe models`.

**A load fails.** Goethe prints the reason and the end of the model's log. For the full log, run `goethe logs --model <model> -n 200`. Common causes:
- **`NVCC version must be at least 12.9`, or a CUDA compiler error:** Goethe gives each engine its own CUDA compiler unless `CUDA_HOME` or `CUDA_PATH` is set in goethed's environment or in the model card. Remove that setting so the bundled compiler is used, then run `rm -rf ~/.cache/sglang/.cache/flashinfer` and load again.
- **Not enough GPU memory:** use a smaller context (`--ctx`), fewer parallel requests (`--max-conc`), more GPUs (`--tp 2`), or a quantized version of the model.
- **The model needs its own code:** reinstall it with `--trust-remote-code`.

**The first request is slow.** The first request after a load can take a while, because GPU kernels are being prepared. Requests after that are fast.

**Requests fail:**

| Answer | Meaning | Fix |
|---|---|---|
| No connection | Wrong address or port, `http` instead of `https`, or a firewall | Use the address from `goethe api` |
| `403 Forbidden` | Your address isn't allowed | `goethe allowlist --add <your-ip>` |
| `401` | Missing or wrong key | Send `Authorization: Bearer <key>` |
| `404 Model Not Found` | Unknown model name, or your key isn't allowed that model | Use a name from `goethe api` |
| `429` | Too many requests | Slow down, or raise the key's `--rpm` |
| `503` | The model isn't ready, or is being unloaded | Wait, or check `goethe ps` |

**Something else.** Run `goethe doctor` and follow the hints. `goethe logs --daemon` shows Goethe's own log.
