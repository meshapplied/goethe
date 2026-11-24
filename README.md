<h1 align="center">
  <img src="docs/assets/goethe-banner.svg" alt="Goethe" width="480">
</h1>

<p align="center">
  <img alt="NVIDIA CUDA 12.8 or newer" src="https://img.shields.io/badge/NVIDIA%20CUDA-%E2%89%A5%2012.8-76B900?logo=nvidia&logoColor=white">
  <img alt="GPU architectures sm_80 to sm_121" src="https://img.shields.io/badge/arch-sm__80%20%E2%80%93%20sm__121-76B900">
  <img alt="Written in Rust" src="https://img.shields.io/badge/language-Rust-B7410E?logo=rust&logoColor=white">
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <a href="https://github.com/meshapplied/goethe/actions/workflows/ci.yml"><img alt="CI status" src="https://github.com/meshapplied/goethe/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <br>
  <img alt="Linux" src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-000000?logo=apple&logoColor=white">
  <img alt="Windows via WSL" src="https://img.shields.io/badge/Windows%20(WSL)-0078D4?logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0wIDBoMTEuNHYxMS40SDB6TTEyLjYgMEgyNHYxMS40SDEyLjZ6TTAgMTIuNmgxMS40VjI0SDB6TTEyLjYgMTIuNkgyNFYyNEgxMi42eiIvPjwvc3ZnPg==">
  <br>
  <a href="https://goethe.sh/docs"><img alt="Documentation at goethe.sh/docs" src="https://img.shields.io/badge/docs-goethe.sh%2Fdocs-87D7FF?logo=readthedocs&logoColor=white"></a>
  <a href="https://discord.gg/fadnuFrm4"><img alt="Join us on Discord" src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white"></a>
</p>

<p align="center"><b>Run and share your models on your own infra in 3 steps, and run them fast.</b></p>

<p align="center">If Goethe is useful to you, a ⭐ on <a href="https://github.com/meshapplied/goethe">GitHub</a> helps others find it.</p>

Goethe turns a machine into your own API, compatible with OpenAI, Anthropic and LiteLLM clients. Install it, add a model, and point your apps, your team or your other machines at one address. Your prompts, your data and your models stay on hardware you control.

For Linux and MacOS systems run this command in the terminal:

```bash
curl -fsSL https://pkg.goethe.sh/install.sh | bash
```

For Windows 10/11 (x64), run this in PowerShell:

```powershell
irm https://pkg.goethe.sh/install.ps1 | iex
```

On a Linux GPU server or cloud GPU host with Docker ([setup without a terminal](docs/user-guide.md#docker-and-cloud-gpu-hosts)):

```bash
docker run -d --name goethe --gpus all --network host -v goethe:/workspace ghcr.io/meshapplied/goethe
```

Then:

```bash
goethe          # launch interactive menu
goethe start    # quick setup: install a model, serve it and open the gateway
```

Anything that speaks the OpenAI, Anthropic or LiteLLM API can use it right away, all at the same address: the OpenAI and Anthropic SDKs, LiteLLM, LangChain, LlamaIndex, Open WebUI, coding assistants, or plain `curl`.

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-goethe-address>:8443/v1", api_key="goethe_sk_...")  # from goethe api keys create, shown once
reply = client.chat.completions.create(
    model="nvidia/Llama-3.1-8B-Instruct-NVFP4",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)
```

## Why Goethe

- **Easy to serve.** Every model you load is served at the same URL; a request picks its model by name.
- **Fast.** Optimised kernel packs speed up supported models on supported GPUs, with the plain runtime for everything else.
- **Private by default.** Nothing reaches the API until you allow it. Keys are stored only as hashes; rate-limit, expire, scope, renew or revoke them at any time.
- **Measured.** `goethe metrics` shows requests, tokens, time to first token, decode speed and GPU use per model and key, kept on the machine.
- **Fits your network.** This machine only, a VPN, or a public IP. `goethe api expose` sets it.
- **Many models, many GPUs.** Run several models side by side, split large models across GPUs, and keep other GPU software running next to Goethe.
- **Always on.** Loaded models come back after a reboot or a crash.

## Where it runs

| Environment | Examples |
|---|---|
| Your own hardware | Workstations and servers with one or more NVIDIA GPUs, including DGX Spark |
| Cloud GPU virtual machines | AWS, Google Cloud, Azure, Oracle Cloud, Lambda, CoreWeave, Paperspace |
| GPU containers | RunPod, Vast.ai, and Docker with the NVIDIA Container Toolkit |

Linux on x86_64 or ARM64 (glibc 2.30 or newer, for example Ubuntu 20.04+ or Debian 11+), macOS on Apple Silicon (macOS 14+), or Windows 10/11 via WSL2. Goethe recognises the cloud provider or container it runs in and shows the address to use from outside.

## GPU and CUDA compatibility

| Architecture | Compute capability | GPUs |
|---|---|---|
| Blackwell | 12.0, 12.1 | GeForce RTX 5090, RTX PRO 6000 Blackwell, DGX Spark (GB10) |
| Blackwell data center | 10.0, 10.3 | B200, GB200, B300, GB300 |
| Hopper | 9.0 | H100, H200, GH200 |
| Ada | 8.9 | GeForce RTX 4090, RTX 6000 Ada, L40S, L40, L4 |
| Ampere | 8.0, 8.6 | A100, A30, A40, A10, RTX A6000, GeForce RTX 3090 |

Older GPUs (Volta, Turing) are not supported. A machine can mix GPU generations; each model runs on GPUs of one generation, and large models can be split across several GPUs.

Goethe ships its own CUDA 13 runtime and compiler, so no CUDA toolkit is needed on the machine. What it needs is the NVIDIA driver:

| NVIDIA driver | CUDA it offers | Goethe runs on |
|---|---|---|
| 580 or newer | 13.0 and newer | Every supported GPU |
| 535 to 579 | 12.2 to 12.9 | Data-center GPUs (A100, H100, H200, B200, L40S, L4 and similar), through NVIDIA's CUDA forward compatibility |
| Older than 535 | | Not supported; the installer offers to update the driver |

GeForce and TITAN cards need driver 580 or newer. Enough GPU memory for the model is the only other requirement; `goethe models recommend` lists the models that fit.

The llama.cpp build goethe downloads for GGUF models uses CUDA 12.8 libraries; NVIDIA lists the minimum driver of each CUDA release in its [CUDA compatibility table](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html). Drivers older than the 580 line above are unverified with goethe.

Supported [model families and quantization formats](docs/user-guide.md#supported-models) are in the user guide. GGUF models (single GPU) and GGUF embedding models are served through llama.cpp.

## Known limits

- Gated models need the license accepted on Hugging Face and a read token (install with `--hf-token`).
- Optimised kernel packs need `goethe login` on the node. If optimised kernels are not available for a given model, goethe defaults to SGlang, if SGlang does not support it either, it checks llama.cpp, displays model unsupported otherwise.
- If a model's kernel pack cannot run on your GPU, install or load says why in one line and serves the model with its plain runtime instead; `GOETHE_ALLOW_FALLBACK=0` makes that an error, and `--fallback` overrides it for one command.

## Documentation

- **[User guide](docs/user-guide.md):** install, add models, share the API, and troubleshooting.
- **[Command reference](docs/cli.md):** every `goethe` command and option.
- **Full documentation:** [goethe.sh/docs](https://goethe.sh/docs).

## Repository layout

| Path | What it is |
|------|------------|
| [`goethe/`](goethe/) | Goethe: the `goethe` command, the `goethed` service, the gateway, packaging and the installer |
| [`docs/`](docs/) | User guide, command reference |
| [`install.sh`](install.sh) | Builds Goethe from source; the normal install is `pkg.goethe.sh/install.sh` above |
| [`AGENTS.md`](AGENTS.md) | Instructions for AI agents |

## License

Goethe is licensed under the [Apache License 2.0](LICENSE). The license of each model you serve still applies.

Goethe is built on third-party software, among them SGLang, PyTorch, FlashInfer, Triton, Hugging Face Transformers, nginx and NVIDIA's CUDA libraries. [NOTICE](NOTICE) lists them with their licenses.
