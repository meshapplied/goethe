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
- **Home or office over a VPN.** Serve from a GPU box and use it from your laptop through Tailscale, WireGuard or ZeroTier.
- **Your own apps and tools.** Point OpenAI, Anthropic or LiteLLM clients, Open WebUI, coding assistants and similar tools at hardware you control.
- **Many models, many GPUs.** Compare models by name on the same address, split large models across GPUs, or run on cloud GPUs when you need them.

## Quick install

```bash
curl -fsSL https://meshapplied.com/install.sh | bash
goethe start
```

`goethe start` starts Goethe and walks you through three steps: install a model, serve it through the API, and configure who can reach it. You can stop after any step and continue later with `goethe start`. If you prefer plain commands:

```bash
# 1. Start Goethe and load a model
goethe up
goethe install --model zai-org/GLM-5.3-Flash --load

# 2. Share it
goethe configure api                 # this machine only, your VPN (recommended), or a public address
goethe key --create my-laptop        # one key per app or person
goethe api                           # the address to give your apps
```

`goethe status` tells you whether Goethe is running, with the API address and your keys. Check the machine with `goethe doctor`: each `WARN` or `FAIL` line names its fix. See the [hardware](#hardware-requirements) and [software](#software-requirements) requirements before you install.

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

Every value goes after its flag: `goethe unload --model <model>`, `goethe gpu --allow gpu0`. Listing commands accept `--json`. Exit codes: `0` success, `1` error, `2` wrong usage, `3` a confirmation was needed without a terminal (add `--yes`). The [command reference](cli.md) lists every command and option.

### Everyday

```bash
goethe up          # start; also starts at boot from now on
goethe status      # running or not, API address and keys
goethe down        # stop; won't start at boot until the next up
goethe ps          # what is running
goethe api         # address, keys and loaded models
goethe logs --model zai-org/GLM-5.3-Flash -f
goethe logs --daemon
goethe doctor
goethe hw profile
```

Models that were loaded when Goethe stopped come back when it starts again, including after a reboot. `goethe service restore --off` turns that off.

### Start and stop

| Command | What it does |
|---|---|
| `goethe start` | Start, and guided setup: model, serve, API access |
| `goethe up` | Start Goethe and the API, and start at boot (`--no-boot` to skip) |
| `goethe status` | Running state, address and keys |
| `goethe down` | Stop Goethe and its models, and stop starting at boot |
| `goethe service` | `enable`, `disable`, `status`, `restore --on`/`--off` |

### Models

| Command | What it does |
|---|---|
| `goethe recommend` | Catalog models that fit your GPUs |
| `goethe install --model …` | Download from Hugging Face, or register a local folder |
| `goethe scan` | Add folders you placed in the models directory |
| `goethe models` | Installed and catalog models, and whether each fits |
| `goethe config --model …` | Default context, concurrency and GPUs for a model |
| `goethe remove --model …` | Unload and delete (or unregister a folder) |

### Running

| Command | What it does |
|---|---|
| `goethe load --model …` | Load and wait until ready (`--gpu`, `--tp`, `--ctx`, `--max-conc`, `--replica`) |
| `goethe unload --model …` | Stop a model (`--all`, `--force`) |
| `goethe ps` | Running instances, GPUs and speed |
| `goethe logs` | `--model`, `--daemon`, `--gateway`, `-f`, `-n` |

### Sharing

| Command | What it does |
|---|---|
| `goethe configure api` | Local, VPN or public access |
| `goethe api` | Address, authentication, models, keys and a curl example |
| `goethe key` | `--create`, `--revoke`, `--rpm`, `--expires`, `--models`, `--admin` |
| `goethe allowlist` | `--add` / `--remove` client addresses |
| `goethe alias` | Short model names for clients (`--set`, `--remove`) |
| `goethe gateway` | Where the API listens (`--bind 127.0.0.1`, `0.0.0.0` or an IP) |
| `goethe domain` | DNS record for your own hostname (`--check <name>`) |

### GPUs and system

| Command | What it does |
|---|---|
| `goethe gpu` | Memory use and allow/deny (`--deny`, `--allow`) |
| `goethe hw profile` | GPUs, CPU, memory, disk and cloud provider |
| `goethe doctor` | Setup check with a fix for each problem |
| `goethe update` | Install the latest Goethe (`--check` to only look) |
| `goethe uninstall` | Remove Goethe and its models (`--keep-models` to keep them) |

## Add models

```bash
goethe recommend                                        # catalog models that fit your GPUs
goethe install --model zai-org/GLM-5.3-Flash            # download it
goethe install --model zai-org/GLM-5.3-Flash --load     # download and load in one step
```

The model id is the Hugging Face repo name. Any [supported model](#supported-models) on Hugging Face works, not only the catalog. Useful flags: `--load`, `--ctx`, `--max-conc`, `--trust-remote-code` (only for models you trust) and `--hf-token`.

**Gated or private models:** accept the license on the model's Hugging Face page, create a read token at huggingface.co/settings/tokens, then run `goethe install --model <org>/<model> --hf-token` and type the token when asked. It is used for that download only and never stored. In `goethe start`, choose "A private or gated model".

**Models you already have:** move the folder into Goethe's `models` folder and run `goethe scan`; it checks each folder and asks before adding it. A folder named `<org>__<model>` gets the id `<org>/<model>`. To register a folder where it is:

```bash
goethe install --model /data/models/my-finetune --name me/my-finetune
```

**List and remove:**

```bash
goethe models                                   # installed and catalog models, and whether each fits
goethe remove --model zai-org/GLM-5.3-Flash     # unload it and delete the files Goethe downloaded
```

For a folder you registered in place, `remove` only unregisters it.

## Load and run

```bash
goethe load --model zai-org/GLM-5.3-Flash
```

Goethe picks the GPUs, sizes the memory and waits until the model answers:

```
RUNNING: zai-org/GLM-5.3-Flash instance inst-... on GPU 0 port 30001 (45.2 s)
```

The first load on a new machine takes a few extra minutes; later loads are faster. `goethe load` shows how the GPU memory is used, and `goethe hw pick-gpus --model <model>` shows the same without loading.

```bash
goethe load --model zai-org/GLM-5.3-Flash --gpu gpu1                    # a specific GPU
goethe load --model zai-org/GLM-5.3-Flash --tp 2                        # split across 2 GPUs Goethe picks
goethe load --model zai-org/GLM-5.3-Flash --gpu gpu0 gpu1               # split across GPUs 0 and 1
goethe load --model zai-org/GLM-5.3-Flash --ctx 65536 --max-conc 32     # longer context, more parallel requests
goethe load --model zai-org/GLM-5.3-Flash --replica                     # another copy, for more traffic
goethe load --model zai-org/GLM-5.3-Flash -- --chunked-prefill-size 4096   # extra engine options after --
goethe config --model zai-org/GLM-5.3-Flash --ctx 65536 --max-conc 32   # make settings the default
```

Model names are forgiving: `zai-org/glm-5.3-flash` and `zai-org__GLM-5.3-Flash` mean the same model, and a mistyped name gets a suggestion.

Load as many models as your GPUs can hold. They share one API address, each request picks its model by name, and `goethe ps` shows what is running.

```bash
goethe unload --model zai-org/GLM-5.3-Flash           # finish requests in progress, then stop
goethe unload --all                                   # every model
goethe unload --model zai-org/GLM-5.3-Flash --force   # stop now
```

**Sharing a GPU with other software:** if a GPU is already used by other programs, Goethe lists them and asks before loading next to them. It never stops processes it did not start. Add `--yes` in scripts. To keep Goethe off a GPU, use `goethe gpu --deny gpu0`; `goethe gpu --allow gpu0` allows it again.

## Share the API

```bash
goethe configure api
```

- **Local only:** programs on this machine.
- **VPN (recommended):** devices on your Tailscale, WireGuard or ZeroTier network. Goethe detects the VPN and allows its addresses.
- **Public network:** any address that can reach this machine. You open the port in your firewall yourself.

It then asks whether to restrict access to specific IP addresses, and offers to create the first key. In scripts: `goethe configure api --mode vpn --yes`, or `--mode public --allow 203.0.113.7 --yes`.

### Keys

```bash
goethe key --create my-laptop                                   # shows the key once
goethe key --create ci --rpm 600 --expires 30d
goethe key --create chat-only --models zai-org/GLM-5.3-Flash
goethe key                                                      # keys, limits and today's usage
goethe key --revoke <key-id>                                    # takes effect immediately
KEY=$(goethe key --create ci --raw)                             # only the key, for scripts
```

Every request needs a valid key; without one the API answers `401`. A key can expire (`--expires 30d` or a date), have its own rate limit (`--rpm`, default 120 per minute, `0` for unlimited), be limited to some models (`--models`), or be an admin key that can read the health endpoints (`--admin`).

### Allowed addresses

On top of keys, only allowed client addresses can connect. `goethe configure api` sets this up; to change it by hand:

```bash
goethe allowlist
goethe allowlist --add 192.168.1.0/24 --description "office LAN"
goethe allowlist --add 203.0.113.7
goethe allowlist --remove 192.168.1.0/24
```

### Address, aliases and your own domain

```bash
goethe api                                            # the address, models, keys and a curl command
goethe alias --set chat --model zai-org/GLM-5.3-Flash  # clients can send "model": "chat"
goethe domain                                         # the DNS record for your own hostname
goethe domain --check api.example.com                 # check that the name points at the API
```

Change the model behind an alias at any time and every client follows. For a hostname such as `api.example.com`, add the DNS A record `goethe domain` shows; clients then use `https://api.example.com:8443/v1` with the same certificate settings as for the IP address.

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

GeForce and TITAN cards need driver 580 or newer. Check the driver with `nvidia-smi`. Enough GPU memory for the model is the other requirement; `goethe recommend` lists what fits.

## Software requirements

- Linux on x86_64 or ARM64 with glibc 2.30 or newer, for example Ubuntu 20.04+ or Debian 11+.
- Disk space for your models, plus about 12 GB for Goethe itself.
- No CUDA toolkit: Goethe brings its own CUDA runtime and compiler.

The installer checks the machine first and installs missing pieces, such as a C++ compiler or the NVIDIA driver, after asking. `GOETHE_INSTALL_DEPS=1` and `GOETHE_INSTALL_DRIVER=1` answer yes in advance, `=0` no. Inside a container it never touches the driver, which belongs to the host. A GPU problem never stops the install; `goethe doctor` names the fix afterwards:

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

Dense, mixture-of-experts, hybrid (linear attention and Mamba) and vision-language models are supported. `goethe install` and `goethe scan` say straight away if an architecture is not.

| Format | Typical use |
|---|---|
| BF16 / FP16 | Original weights |
| NVFP4 | 4-bit, about a quarter of BF16's size; every supported GPU |
| FP8 | 8-bit; native on Hopper and Blackwell |
| MXFP4 | GPT-OSS models |
| AWQ, GPTQ | 4-bit community quantizations |
| INT8 (W8A8) | 8-bit integer |
| Mixed precision | NVFP4 and FP8 in one model |

The format is detected from the model files. GGUF and `.bin` (PyTorch pickle) weights are not supported; use the safetensors release of the model.

## Settings

Optional. Set them in your shell, or in `goethe.env` inside Goethe's folder so the background service picks them up.

| Variable | What it changes |
|---|---|
| `GOETHE_HOME` | Where Goethe keeps its files |
| `GOETHE_PUBLIC_PORT` | API port (default 8443; another is used if it's taken) |
| `GOETHE_NO_AUTOSTART=1` | Never set up start at boot |
| `GOETHE_PUBLIC_IP` | Public IP shown by `goethe configure api` in public mode |
| `GOETHE_MAX_BODY_MB` | Largest request size in MB (default 10) |
| `NO_COLOR=1` | Turn off colored output |

## Troubleshooting

**`goethe: command not found`.** Open a new terminal, or run `export PATH="$HOME/.local/bin:$PATH"`.

**`goethed is not running`.** Run `goethe up`; `goethe status` shows whether it is running.

**Unknown model.** Use an id from `goethe models`.

**A load fails.** Goethe prints the reason and the end of the model's log; `goethe logs --model <model> -n 200` shows more. Common fixes: a smaller context (`--ctx`), fewer parallel requests (`--max-conc`), more GPUs (`--tp 2`), or a quantized version of the model; `--trust-remote-code` when the model needs its own code; `goethe doctor` for driver and GPU problems.

**The first request is slow.** The first request after a load can take a while; later requests are fast.

**Requests fail:**

| Answer | Meaning | Fix |
|---|---|---|
| No connection | Wrong address or port, `http` instead of `https`, or a firewall | Use the address from `goethe api` |
| `403` | Your address isn't allowed | `goethe allowlist --add <your-ip>` |
| `401` | Missing or wrong key | Send the key, for example `Authorization: Bearer <key>` |
| `404 Model Not Found` | Unknown model, or the key isn't allowed that model | Use a name from `goethe api` |
| `429` | Too many requests | Slow down, or raise the key's `--rpm` |
| `503` | The model isn't ready, or is unloading | Wait, or check `goethe ps` |

**Anything else.** Run `goethe doctor`, then `goethe logs --daemon`.
