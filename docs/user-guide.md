# Goethe user guide

Run and share your models on your own infrastructure in three steps, and run them fast.

Goethe turns a Linux machine into your own API, compatible with OpenAI, Anthropic and LiteLLM clients. Install it, add a model, and point your apps, your team or your other machines at one address. Your prompts, your data and your models stay on hardware you control.

## Contents

- [What you can use it for](#what-you-can-use-it-for)
- [Quick install](#quick-install)
- [API compatibility](#api-compatibility)
- [CLI](#cli)
- [Add models](#add-models)
- [Load and run](#load-and-run)
- [Share the API](#share-the-api)
- [Hardware requirements](#hardware-requirements)
- [Software requirements](#software-requirements)
- [Where it runs](#where-it-runs)
- [Supported models](#supported-models)
- [Settings](#settings)
- [Troubleshooting](#troubleshooting)

## What you can use it for

- **Easy to serve.** Every model you load is served at the same URL; a request picks its model by name.
- **Fast.** Use the custom kernel attachments to accelerate and optimise model capability.
- **A private API for your team.** One key per person or app; nothing reaches the API until you allow it.
- **Home or office over a VPN.** Serve from a GPU box and use it from your laptop through your VPN.
- **Your own apps and tools.** Point OpenAI, Anthropic or LiteLLM clients, Open WebUI, coding assistants and similar tools at hardware you control.
- **Many models, many GPUs.** Compare models by name on the same address, split large models across GPUs, or run on cloud GPUs when you need them.

## Quick install

```bash
curl -fsSL https://goethe.sh/install.sh | bash
goethe login
goethe
```

`goethe login` links the machine to the Goethe platform, which also unlocks optimised kernels; it is optional. `goethe` on its own starts Goethe and, while nothing is installed, walks you through three steps: install a model, serve it through the API, and configure who can reach it. You can stop after any step and continue later with `goethe`. If you prefer plain commands:

```bash
# 1. Start Goethe and load a model
goethe up
goethe install zai-org/GLM-5.3-Flash --load

# 2. Share it
goethe api expose vpn                # or local, or public: who can reach the API
goethe api keys                      # on a terminal: add, scope or delete API keys in a menu
goethe api                           # the address to give your apps
```

`goethe status` tells you whether Goethe is running, with the API address, your keys and the models it serves. Check the machine with `goethe doctor`: each `WARN` or `FAIL` line names its fix. See the [hardware](#hardware-requirements) and [software](#software-requirements) requirements before you install.

## API compatibility

The API is at `https://<address>:<port>/v1`. It speaks three schemas at once, so OpenAI, Anthropic and LiteLLM clients all work against the same address and key:

- **OpenAI:** chat, completions, responses, embeddings, rerank and models.
- **Anthropic:** `POST /v1/messages` and its token count. Errors on these routes use Anthropic's format.
- **LiteLLM proxy:** every route also works without `/v1`, plus `/model/info`, `/model_group/info`, `/health/liveliness` and `/health/readiness`.

| Endpoint | Purpose |
|---|---|
| `POST /v1/chat/completions` | Chat, with streaming, tools and reasoning |
| `POST /v1/completions` | Plain text completion |
| `POST /v1/responses` | OpenAI Responses API |
| `POST /v1/messages` | Anthropic Messages API, with streaming |
| `POST /v1/messages/count_tokens` | Anthropic token count |
| `POST /v1/embeddings` | Embeddings, for embedding models |
| `POST /v1/rerank` | Reranking, for reranker models |
| `POST /v1/audio/transcriptions` | Speech to text (OpenAI multipart upload), for transcription models; the body limit is `GOETHE_MAX_BODY_MB` (default 10 MB, about 5 minutes of 16 kHz WAV) |
| `GET /v1/models`, `GET /v1/models/{id}` | Models your key can use |
| `GET /model/info`, `GET /model_group/info` | Models in LiteLLM's format |
| `GET /health` | Up check, no key needed |

Send the key in whichever header your client uses: `Authorization: Bearer <key>` (OpenAI), `x-api-key` (Anthropic), `x-litellm-api-key` (LiteLLM) or `api-key` (Azure style).

The API uses HTTPS with a certificate Goethe creates for itself. Clients either skip verification (`curl -k`, `verify=False`) or trust `<goethe folder>/nginx/certs/server.crt`.

### curl

```bash
curl -k https://<address>:8443/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"zai-org/GLM-5.3-Flash","messages":[{"role":"user","content":"Hello!"}],"max_tokens":128}'
```

### OpenAI SDK

```python
import httpx
from openai import OpenAI

client = OpenAI(
    base_url="https://<address>:8443/v1",
    api_key="goethe_sk_...",
    http_client=httpx.Client(verify=False),   # or verify="/path/to/server.crt"
)
stream = client.chat.completions.create(
    model="zai-org/GLM-5.3-Flash",
    messages=[{"role": "user", "content": "Explain mixture-of-experts in one paragraph."}],
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
    model="zai-org/GLM-5.3-Flash",
    max_tokens=256,
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.content[0].text)
```

### LiteLLM

```python
import litellm

reply = litellm.completion(
    model="litellm_proxy/zai-org/GLM-5.3-Flash",
    api_base="https://<address>:8443",
    api_key="goethe_sk_...",
    messages=[{"role": "user", "content": "Hello!"}],
    ssl_verify=False,
)
```

Or add Goethe to a LiteLLM proxy's `config.yaml` as an OpenAI-compatible backend (`model: openai/zai-org/GLM-5.3-Flash`, `api_base: https://<address>:8443/v1`).

### Other tools

Open WebUI, LangChain, LlamaIndex, coding assistants and any tool that takes a custom OpenAI or Anthropic endpoint work the same way: set the base URL to `https://<address>:<port>/v1` (Anthropic clients: without `/v1`), your Goethe key, and a model name from `goethe api`. If a tool can't skip certificate checks, add Goethe's certificate to the system's trusted certificates.

### Limits

- Each client address can send up to 20 requests per second.
- Each key has its own per-minute limit, set with `--rpm`.
- Request bodies can be up to 10 MB.
- When a model unloads, requests in progress finish first; new requests get `503` with a `Retry-After` header.

## CLI

The model is a plain argument (`goethe load qwen 27b`: an id, an alias, or words from its name, with a menu when several match); every action is a word and flags only change how it runs (`goethe gpu allow gpu0`). API settings are under `goethe api`, model housekeeping under `goethe models`. Listing commands accept `--json`. Exit codes: `0` success, `1` error, `2` wrong usage, `3` a confirmation was needed without a terminal (add `--yes`). The [command reference](cli.md) lists every command and option.

### Everyday

```bash
goethe up          # start; also starts at boot from now on
goethe status      # running or not, API address, keys and models
goethe down        # stop; won't start at boot until the next up
goethe ps          # what is running
goethe api         # address, authentication and keys
goethe logs zai-org/GLM-5.3-Flash -f
goethe logs goethed
goethe doctor
goethe doctor --hardware
```

Models that were loaded when Goethe stopped come back when it starts again, including after a reboot.

### Start and stop

| Command | What it does |
|---|---|
| `goethe` | The menu on a terminal: quick setup, install models, API keys and gateway, live status; 1 2 3 pick a color theme |
| `goethe up` | Start Goethe and the API, and start at boot (`--no-boot` to skip) |
| `goethe status` | Running state, address and keys |
| `goethe down` | Stop Goethe and its models, and stop starting at boot |

### Models

| Command | What it does |
|---|---|
| `goethe models recommend` | Catalog models that fit your GPUs |
| `goethe install …` | Download from Hugging Face (`org/name:Q8_0` picks one quantization: a GGUF file or a kernel pack variant), or register a local folder |
| `goethe models scan` | Add folders you placed in the models directory |
| `goethe models` | Installed and catalog models, and whether each fits |
| `goethe load … --save` | Keep the context, concurrency and GPUs you give as the model's defaults (`--dry-run` to only store them) |
| `goethe remove …` | Unload and delete (or unregister a folder) |

### Running

| Command | What it does |
|---|---|
| `goethe load …` | Load and wait until ready (`--gpu`, `--tp`, `--ctx`, `--max-conc`, `--replica`, `--dry-run`) |
| `goethe unload …` | Stop a model (`--all`, `--force`) |
| `goethe run [MODEL] [INPUT]` | Try a loaded model by hand: text to a text model (a chat without INPUT, else one reply) or an embedding model (dims, norm, first values); an audio file (.wav, .flac, .mp3, .ogg, .m4a) to a speech model (its transcript; without INPUT it records from the microphone until Enter, with arecord, parecord or ffmpeg) |
| `goethe ps` | Running instances, GPUs and speed |
| `goethe logs <target>` | A model, `goethed` or `gateway`; `-f`, `-n` |

### Sharing

| Command | What it does |
|---|---|
| `goethe api expose` | Local, VPN or public access, or one address to listen on (`local`, `vpn`, `public`, `0.0.0.0`, `wg0`) |
| `goethe api` | Address, authentication, keys and a curl example |
| `goethe api keys` | List keys; `create`, `revoke`, `--rpm`, `--expires`, `--models`, `--admin`; on a terminal alone, a menu |
| `goethe api allow` | `add` / `remove` client addresses |
| `goethe models alias` | Short model names for clients (`set`, `remove`) |
| `goethe trace` | Record requests and replies as training traces (`on`, `off`, `--key`); see [traces.md](traces.md) |
| `goethe learn` | Let a session keep what compaction would drop (`on`, `off`, `--model`); see [learn.md](learn.md) |
| `goethe api domain` | DNS record for your own hostname (`check <name>`) |
| `goethe api share 5h --key <name> --allow <their-ip>/32` | Open the API on a public address for a while for chosen keys from their IP (`stop` closes it) |

### GPUs and system

| Command | What it does |
|---|---|
| `goethe gpu` | VRAM, processes and bandwidth per GPU, live until `q`; `allow` / `deny` GPUs |
| `goethe doctor` | Setup check with a fix for each problem; `--hardware` prints GPUs, CPU, memory, disk and cloud provider |
| `goethe update` | Install the latest Goethe (`--check` to only look; `engine`, `rollback`) |
| `goethe uninstall` | Remove Goethe and its models (`--keep-models` to keep them) |
| `goethe completion install` | Tab completion for bash, zsh or fish: commands, models, keys, GPUs |
| `goethe login` | Optional: link this node to the Goethe platform to see its health there (`--authkey` without a browser) |
| `goethe logout` | Unlink it again |

## Add models

```bash
goethe models recommend                                        # catalog models that fit your GPUs
goethe install zai-org/GLM-5.3-Flash            # download it
goethe install zai-org/GLM-5.3-Flash --load     # download and load in one step
```

The model id is the Hugging Face repo name. Any [supported model](#supported-models) on Hugging Face works, not only the catalog. Useful flags: `--load`, `--ctx`, `--max-conc`, `--trust-remote-code` (only for models you trust) and `--hf-token`.

**Optimised kernels:** some models have a kernel pack: the optimised kernels for every supported GPU microarchitecture, with the model's quantizations and configs (NVIDIA CUDA for now; other platforms such as AMD ROCm later). Packs are available to nodes logged in to the Goethe platform (`goethe login`), and `goethe install` uses them automatically. A node that is not logged in is asked to log in, or continues with SGLang. `goethe models` marks the models that have optimised kernels for your machine.

**GGUF models:** a Hugging Face GGUF repo (`org/model-GGUF`, or `org/model-GGUF:Q8_0` for one quant) is served by llama.cpp on one GPU; a GGUF with a mean, cls or last pooling type is served as an embedding model (`goethe run <model> "text"`). If a model's kernel pack cannot open or does not run on your GPU, goethe says why in one line and serves the model with its plain runtime instead (`GOETHE_ALLOW_FALLBACK=0` makes that an error). A model that has only a pack, and none for your GPU, is refused before anything is downloaded.

**Which engine serves a model:** goethe decides before downloading anything. On a node logged in to the platform, a goethe kernel pack for the model and your GPU is used automatically. Otherwise SGLang serves it when SGLang supports its architecture and quantization; else llama.cpp serves the repo's GGUF files, or the GGUF build Hugging Face lists for the model (the publisher's own `<model>-GGUF` repo first), and the name you typed becomes an alias of it. A model none of them serves is refused with each engine's reason. On a Mac llama.cpp serves every model. `goethe models` shows the engine of each model in its ENGINE column.

**Gated or private models:** accept the license on the model's Hugging Face page, create a read token at huggingface.co/settings/tokens, then run `goethe install <org>/<model> --hf-token` and type the token when asked. It is used for that download only and never stored. In the guided `goethe` setup, choose "A private or gated model".

**Models you already have:** move the folder into Goethe's `models` folder and run `goethe models scan`; it checks each folder and asks before adding it. A folder named `<org>__<model>` gets the id `<org>/<model>`. To register a folder where it is:

```bash
goethe install /data/models/my-finetune --name me/my-finetune
```

**List and remove:**

```bash
goethe models                                   # installed and catalog models, and whether each fits
goethe remove zai-org/GLM-5.3-Flash     # unload it and delete the files Goethe downloaded
```

For a folder you registered in place, `remove` only unregisters it.

## Load and run

```bash
goethe load zai-org/GLM-5.3-Flash
```

Goethe picks the GPUs, sizes the memory and waits until the model answers:

```
RUNNING: zai-org/GLM-5.3-Flash instance inst-... on GPU 0 port 30001 (45.2 s)
```

The first load on a new machine takes a few extra minutes; later loads are faster. `goethe load` shows how the GPU memory is used, and `goethe load <model> --dry-run` shows the same without loading.

```bash
goethe load zai-org/GLM-5.3-Flash --gpu gpu1                    # a specific GPU
goethe load zai-org/GLM-5.3-Flash --tp 2                        # split across 2 GPUs Goethe picks
goethe load zai-org/GLM-5.3-Flash --gpu gpu0 gpu1               # split across GPUs 0 and 1
goethe load zai-org/GLM-5.3-Flash --ctx 65536 --max-conc 32     # longer context, more parallel requests
goethe load zai-org/GLM-5.3-Flash --replica                     # another copy, for more traffic
goethe load zai-org/GLM-5.3-Flash -- --chunked-prefill-size 4096   # extra engine options after --
goethe load zai-org/GLM-5.3-Flash --ctx 65536 --max-conc 32 --save --dry-run   # make settings the default, load nothing
```

Model names are forgiving: `zai-org/glm-5.3-flash` and `zai-org__GLM-5.3-Flash` mean the same model, and a mistyped name gets a suggestion.

Load as many models as your GPUs can hold. They share one API address, each request picks its model by name, and `goethe ps` shows what is running.

```bash
goethe unload zai-org/GLM-5.3-Flash           # finish requests in progress, then stop
goethe unload --all                                   # every model
goethe unload zai-org/GLM-5.3-Flash --force   # stop now
```

**Sharing a GPU with other software:** if a GPU is already used by other programs, Goethe lists them and asks before loading next to them. It never stops processes it did not start. Add `--yes` in scripts. To keep Goethe off a GPU, use `goethe gpu deny gpu0`; `goethe gpu allow gpu0` allows it again.

## Share the API

```bash
goethe api expose vpn      # or local, or public; asks who may connect
```

- **Local only:** programs on this machine.
- **VPN or private network (recommended):** connect your VPN first, then pick it from the list. Any VPN with a network interface works, and so does a cloud VM's private network. Goethe lists tunnels and interfaces on private addresses, leaves out container bridges and public network cards, and shows the range each one reaches, taken from the system's routes. When a VPN routes each device separately, the private block its address is in (such as `100.64.0.0/10` or `10.0.0.0/8`) is used, so devices added later can connect too; you can narrow it to specific addresses. Default routes are never allowed. The gateway follows the interface: a new address after a reconnect or a reboot is picked up within a few seconds, and while the interface is down only this machine can connect (`goethe status` says so). The allowlist itself only changes when you run `goethe api expose` again; `goethe doctor` warns if it no longer lets the network's clients in. A VPN without an interface (userspace mode) can't be listened on directly: use local mode and the VPN's own port forwarding.
- **Public network:** any address that can reach this machine. You open the port in your firewall yourself.

It then asks whether to restrict access to specific IP addresses, and offers to create the first key. In scripts: `goethe api expose vpn --yes`, or `goethe api expose public --allow 203.0.113.7/32 --yes`. Public always needs `--allow` and a confirm (`--yes` or `y`); it will not open every interface on a leftover allowlist alone.

### Keys

```bash
goethe api keys                                                 # on a terminal, full-screen: create, scope, limit or delete keys
goethe api keys create my-laptop                                   # shows the key once
goethe api keys create ci --rpm 600 --expires 30d
goethe api keys create chat-only --models zai-org/GLM-5.3-Flash
goethe api keys                                                      # keys, limits and today's usage
goethe api keys revoke <key-id>                                    # takes effect immediately
KEY=$(goethe api keys create ci --raw)                             # only the key, for scripts
```

Every request needs a valid key; without one the API answers `401`. A key can expire (`--expires 30d` or a date), have its own rate limit (`--rpm`, default 120 per minute, `0` for unlimited), be limited to some models (`--models`), or be an admin key that can read the health endpoints (`--admin`).

### Allowed addresses

On top of keys, only allowed client addresses can connect. `goethe api expose` sets this up; to change it by hand:

```bash
goethe api allow
goethe api allow add 192.168.1.0/24 --note "office LAN"
goethe api allow add 203.0.113.7
goethe api allow remove 192.168.1.0/24
```

### Share for a while on a public address

To give someone outside your network access for a limited time, without changing how the API is normally reached:

```bash
goethe api share 5h --key my-app --allow 203.0.113.7/32   # prints the URL, the key and the certificate to send
goethe api share 1d --key ci --allow 203.0.113.7/32       # another key on the same address, for a day
goethe api share stop --key ci                            # stop sharing one key
goethe api share stop                                     # close early
```

The public address exists only while some key is shared, accepts only the keys you chose (never admin keys), each until its own time and from the client addresses you named with `--allow` (or kept from the last share of those keys), and closes by itself when the last one ends. Keys stay the same every time. The menu's Public address page does the same with a form. Details and the full list of security precautions: [`goethe api share`](cli.md#goethe-api-share-duration).

### Address, aliases and your own domain

```bash
goethe api                                            # the address, keys and a curl command
goethe models alias set chat zai-org/GLM-5.3-Flash  # clients can send "model": "chat"
goethe api domain                                         # the DNS record for your own hostname
goethe api domain check api.example.com                 # check that the name points at the API
```

Change the model behind an alias at any time and every client follows. For a hostname such as `api.example.com`, add the DNS A record `goethe api domain` shows; clients then use `https://api.example.com:8443/v1` with the same certificate settings as for the IP address.

## Hardware requirements

Goethe runs on NVIDIA GPUs from the Ampere generation onwards:

| Architecture | Compute capability | GPUs |
|---|---|---|
| Blackwell | 12.0, 12.1 | GeForce RTX 5090, RTX PRO 6000 Blackwell, DGX Spark (GB10) |
| Blackwell data center | 10.0, 10.3 | B200, GB200, B300, GB300 |
| Hopper | 9.0 | H100, H200, GH200 |
| Ada | 8.9 | GeForce RTX 4090, RTX 6000 Ada, L40S, L40, L4 |
| Ampere | 8.0, 8.6 | A100, A30, A40, A10, RTX A6000, GeForce RTX 3090 |

Older GPUs (Volta, Turing and earlier) are not supported; Goethe names them and skips them. A machine can mix generations; each model runs on GPUs of one generation. Large models can be split across GPUs with `--tp`, and mixture-of-experts models can also use `--ep`. 4-bit NVFP4 models run on every supported GPU.

Goethe ships its own CUDA 13 runtime, so the machine only needs the NVIDIA driver:

| NVIDIA driver | CUDA it offers | Goethe runs on |
|---|---|---|
| 580 or newer | 13.0 and newer | Every supported GPU |
| 535 to 579 | 12.2 to 12.9 | Data-center GPUs (A100, H100, H200, B200, L40S, L4 and similar), through CUDA forward compatibility |
| Older than 535 | | Not supported; the installer offers to update the driver |

GeForce and TITAN cards need driver 580 or newer. Check the driver with `nvidia-smi`. Enough GPU memory for the model is the other requirement; `goethe models recommend` lists what fits.

## Software requirements

- Linux on x86_64 or ARM64 with glibc 2.30 or newer, for example Ubuntu 20.04+ or Debian 11+, or a Mac with Apple Silicon and macOS 14+.
- `curl` and `openssl` to install (as root, the installer adds `openssl` and CA certificates when they are missing). Goethe brings its own nginx; the Python environment comes with the first model you install.
- Disk space for your models, plus about 2 GB for Goethe itself and about 8 GB more for SGLang, which installs once, with the first model that needs it.
- No CUDA toolkit: Goethe brings its own CUDA runtime and compiler.

`GOETHE_VERSION=0.3.3` (or `bash -s -- --version 0.3.3`) installs one release instead of the newest. Running the installer again repairs or upgrades an install.

The installer changes nothing outside Goethe's folder except the links to `goethe` and `goethed` (and, after asking, a PATH line in your shell's startup file). When the machine needs a driver or compiler fix, it runs `goethe doctor --fix`, which asks before each change; run it yourself any time. `GOETHE_INSTALL_DRIVER=1` and `GOETHE_INSTALL_DEPS=1` answer yes in advance, `=0` no. Inside a container it never touches the driver, which belongs to the host. A GPU problem never stops the install; `goethe doctor` names the fix afterwards:

| What `goethe doctor` finds | Fix it names |
|---|---|
| No NVIDIA GPU visible | Attach the GPU to the VM (PCI passthrough or vGPU), or start the container with `--gpus all` |
| A GPU but no driver loaded | The driver package for your distribution, then a reboot |
| A driver too old for this GPU | The same, to update it |
| The proprietary module with a Blackwell GPU | The open kernel module, which Blackwell needs |
| A GPU the driver did not start | A full power cycle, power cables, Above 4G decoding in the BIOS |
| A driver but no `/dev/nvidia*` | Pass the devices into the container, or `sudo nvidia-modprobe -u -c=0` on the host |
| No C++ compiler | `sudo apt install build-essential` or `sudo dnf install gcc-c++` |

After installing, open a new terminal if `goethe` is not found yet. Goethe keeps its files in `/workspace/.goethe` on cloud machines with a persistent volume, `/var/lib/goethe` as root, or `~/.goethe` otherwise.

```bash
goethe update --check      # is there a new version?
goethe update              # install it; loaded models keep serving
goethe uninstall           # remove Goethe and its models; add --keep-models to keep them
```

## Where it runs

| Environment | Examples |
|---|---|
| Your own hardware | Workstations and servers with one or more NVIDIA GPUs, including DGX Spark |
| Cloud GPU virtual machines | AWS, Google Cloud, Azure, Oracle Cloud, Lambda, CoreWeave, Paperspace |
| GPU containers | RunPod, Vast.ai, and Docker with the NVIDIA Container Toolkit |

Goethe recognises the cloud provider or container it runs in, keeps its files on the persistent volume when there is one, and shows the address to use from outside. In a container, the GPU and its driver come from the host.

## Supported models

Goethe serves models in the Hugging Face format (`config.json` plus `.safetensors` weights), across more than 200 architectures:

| Family | Examples |
|---|---|
| GLM | GLM-4, GLM-4.5, GLM-5.2, GLM-5.3 Flash |
| Qwen | Qwen 2.5, Qwen 3, Qwen3-Next, Qwen 3.5, Qwen3-VL |
| DeepSeek | DeepSeek V3, R1, V4 |
| Llama | Llama 3.x, Llama 4 |
| Mistral | Mistral, Mixtral, Mistral Small, Pixtral |
| Gemma | Gemma 2, 3 and 4 |
| Others | GPT-OSS, Kimi, MiniMax, Nemotron, Phi, Granite, Command R, OLMo, InternLM, InternVL, MiMo, EXAONE |
| Embeddings | BERT-style and LLM-based embedding and reranker models |

Dense, mixture-of-experts, hybrid (linear attention and Mamba) and vision-language models are supported. `goethe install` and `goethe models scan` say straight away if an architecture is not.

| Format | Typical use |
|---|---|
| BF16 / FP16 | Original weights |
| NVFP4 | 4-bit, about a quarter of BF16's size; every supported GPU |
| FP8 | 8-bit; native on Hopper and Blackwell |
| MXFP4 | GPT-OSS models |
| AWQ, GPTQ | 4-bit community quantizations |
| INT8 (W8A8) | 8-bit integer |
| Mixed precision | NVFP4 and FP8 in one model |

The format is detected from the model files. `.bin` (PyTorch pickle) weights are not supported; use the safetensors release of the model. GGUF models run with [optimised kernels](#add-models) (goethe-engine, logged-in nodes), or otherwise with llama.cpp when `llama-server` is in `engines/llama.cpp/bin` inside Goethe's folder (or named by `LLAMA_SERVER`): `goethe install` fetches one quantization (Q4_K_M, else Q8_0) and runs it on one GPU. `goethe install` says before downloading anything when a model is not supported.

## Settings

Optional. Set them in your shell, or in `goethe.env` inside Goethe's folder so the background service picks them up.

| Variable | What it changes |
|---|---|
| `GOETHE_HOME` | Where Goethe keeps its files |
| `GOETHE_PUBLIC_PORT` | API port (default 8443; another is used if it's taken) |
| `GOETHE_NO_AUTOSTART=1` | Never set up start at boot |
| `GOETHE_NO_COMPLETION=1` | Don't turn on Tab completion at install |
| `GOETHE_PUBLIC_IP` | Public IP shown by `goethe api expose public`  |
| `GOETHE_MAX_BODY_MB` | Largest request size in MB (default 10) |
| `GOETHE_GATEWAY_RATE_PER_IP` | Requests per second one client address may send (default 100, bursts up to twice that) |
| `GOETHE_GATEWAY_CONN_PER_IP` | Open API connections one client address may hold (default 512) |
| `GOETHE_SHARE_RATE_PER_IP` | Same for `goethe api share` (default 10, bursts up to twice that) |
| `GOETHE_SHARE_CONN_PER_IP` | Open share connections one client address may hold (default 16) |
| `GOETHE_PLATFORM_URL` | Platform `goethe login` links to (default `https://goethe.sh`) |
| `GOETHE_PUBLIC_IP_URL` | Where `goethe api expose`, `goethe api domain` and `goethe doctor` look up the public IP (default: the platform's `/api/ip`; a plain-text or `{"ip": ...}` answer both work) |
| `GOETHE_PUBLIC_IP_LOOKUP=0` | Never look up the public IP (set `GOETHE_PUBLIC_IP` to name it) |
| `NO_COLOR=1` | Turn off colored output |

## Troubleshooting

**`goethe: command not found`.** Open a new terminal, or run `export PATH="$HOME/.local/bin:$PATH"`.

**`goethed is not running`.** Run `goethe up`; `goethe status` shows whether it is running.

**Unknown model.** Use an id from `goethe models`.

**A load fails.** Goethe prints the reason and the end of the model's log; `goethe logs <model> -n 200` shows more. Common fixes: a smaller context (`--ctx`), fewer parallel requests (`--max-conc`), more GPUs (`--tp 2`), or a quantized version of the model; `--trust-remote-code` when the model needs its own code; `goethe doctor` for driver and GPU problems.

**The first request is slow.** The first request after a load can take a while; later requests are fast.

**Requests fail:**

| Answer | Meaning | Fix |
|---|---|---|
| No connection | Wrong address or port, `http` instead of `https`, or a firewall | Use the address from `goethe api` |
| `403` | Your address isn't allowed | `goethe api allow add <your-ip>` |
| `401` | Missing or wrong key | Send the key, for example `Authorization: Bearer <key>` |
| `404 Model Not Found` | Unknown model, or the key isn't allowed that model | Use a name from `goethe api` |
| `429` | Too many requests | Slow down, or raise the key's `--rpm` |
| `503` | The model isn't ready, or is unloading | Wait, or check `goethe ps` |

**Anything else.** Run `goethe doctor`, then `goethe logs goethed`.
