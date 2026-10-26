<h1 align="center">
  <img src="docs/assets/goethe-banner.svg" alt="Goethe" width="480">
</h1>

<p align="center">
  <img alt="NVIDIA CUDA 12.8 or newer" src="https://img.shields.io/badge/NVIDIA%20CUDA-%E2%89%A5%2012.8-76B900?logo=nvidia&logoColor=white">
  <img alt="GPU architectures sm_86 to sm_120" src="https://img.shields.io/badge/arch-sm__86%20%E2%80%93%20sm__120-76B900">
  <img alt="Written in Rust" src="https://img.shields.io/badge/language-Rust-B7410E?logo=rust&logoColor=white">
  <img alt="Linux x86_64 and aarch64" src="https://img.shields.io/badge/OS-Linux%20x86__64%20%2F%20aarch64-FCC624?logo=linux&logoColor=black">
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <a href="https://github.com/meshapplied/goethe/actions/workflows/ci.yml"><img alt="CI status" src="https://github.com/meshapplied/goethe/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <br>
  <a href="https://meshapplied.com/docs"><img alt="Documentation at meshapplied.com/docs" src="https://img.shields.io/badge/docs-meshapplied.com%2Fdocs-87D7FF?logo=readthedocs&logoColor=white"></a>
  <a href="https://discord.gg/fadnuFrm4"><img alt="Join us on Discord" src="https://img.shields.io/badge/Discord-join-5865F2?logo=discord&logoColor=white"></a>
</p>

<p align="center"><b>Run and share your models on your own infra in 3 steps, and run them fast.</b></p>

Goethe turns a Linux machine with an NVIDIA GPU into your own API, compatible with OpenAI, Anthropic and LiteLLM clients. Install it, add a model, and point your apps, your team or your other machines at one address. Your prompts, your data and your models stay on hardware you control.

```bash
curl -fsSL https://meshapplied.com/install.sh | bash
goethe start
```

`goethe start` starts Goethe and walks you through three steps: install a model, serve it through the API, and configure who can reach it. If you prefer plain commands, these do the same:

```bash
# 1. Start Goethe and load a model
goethe up
goethe install --model Qwen/Qwen3-8B --load

# 2. Share it
goethe configure api                 # this machine only, your VPN (recommended), or a public address
goethe key --create my-laptop        # one key per app or person
goethe api                           # the address to give your apps
```

Anything that speaks the OpenAI, Anthropic or LiteLLM API can use it right away, all at the same address: the OpenAI and Anthropic SDKs, LiteLLM, LangChain, LlamaIndex, Open WebUI, coding assistants, or plain `curl`.

```python
from openai import OpenAI

client = OpenAI(base_url="https://<your-goethe-address>:8443/v1", api_key="goethe_sk_...")
reply = client.chat.completions.create(
    model="Qwen/Qwen3-8B",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(reply.choices[0].message.content)
```

## Why Goethe

- **Three steps, one address.** Every model you load is served at the same URL; a request picks its model by name.
- **Fast.** Models run on SGLang, with continuous batching, prefix caching and CUDA graphs. Goethe adds its own NVFP4 kernels for Blackwell, Hopper, Ada and Ampere GPUs, so 4-bit models use about a quarter of the memory of the full-precision version and stay quick.
- **Private by default.** Nothing can reach the API until you allow it, you can rate-limit, expire, scope or revoke keys at any time.
- **Fits your network.** Serve on your own infra only, over Tailscale, WireGuard or ZeroTier, or on a public IP. `goethe configure api` walks you through the choice.
- **Many models, many GPUs.** Run several models side by side, split large models across GPUs, and keep other GPU software running next to Goethe.
- **Always on.** Loaded models come back by themselves after a reboot or a crash.

## Where it runs

| Environment | Examples |
|---|---|
| Your own hardware | Workstations and servers with one or more NVIDIA GPUs, including DGX Spark |
| Cloud GPU virtual machines | AWS, Google Cloud, Azure, Oracle Cloud, Lambda, CoreWeave, Paperspace |
| GPU containers | RunPod, Vast.ai, and Docker with the NVIDIA Container Toolkit |

Linux on x86_64 or ARM64 (glibc 2.30 or newer, for example Ubuntu 20.04+ or Debian 11+). Goethe recognises the cloud provider or container it runs in and shows the address to use from outside.

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

GeForce and TITAN cards need driver 580 or newer. Enough GPU memory for the model is the only other requirement; `goethe recommend` lists the models that fit.

Supported [model families and quantization formats](docs/user-guide.md#what-goethe-supports) are in the user guide.

## Documentation

- **[User guide](docs/user-guide.md):** install, add models, share the API, and troubleshooting.
- **[Command reference](docs/cli.md):** every `goethe` command and option.
- **[Our models](docs/research-models.md):** the ternary and NVFP4 models we publish, and how fast they run.
- **Full documentation:** [meshapplied.com/docs](https://meshapplied.com/docs).

## Repository layout

| Path | What it is |
|------|------------|
| [`engine/`](engine/) | Goethe: the `goethe` command, the background service, the gateway and the installer |
| [`nvfp4-kernels/`](nvfp4-kernels/), [`ternary-kernels/`](ternary-kernels/) | GPU kernels behind our NVFP4 and ternary models |
| [`docs/`](docs/) | User guide, command reference and model notes |

Our models on [Hugging Face](https://huggingface.co/meshapplied) and at [meshapplied.com/models](https://meshapplied.com/models).

## License

Goethe is licensed under the [Apache License 2.0](LICENSE). The license of each model you serve still applies.

Goethe is built on third-party software, among them SGLang, PyTorch, FlashInfer, Triton, Hugging Face Transformers, nginx and NVIDIA's CUDA libraries. [NOTICE](NOTICE) names them and their licenses; [third_party/THIRD_PARTY_NOTICES.md](third_party/THIRD_PARTY_NOTICES.md) lists every component.
