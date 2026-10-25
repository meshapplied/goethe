<h1 align="center">
  <img src="docs/assets/goethe-banner.svg" alt="Own your inference → Goethe" width="100%">
</h1>

<p align="center">
  <img alt="NVIDIA CUDA 12.8 or newer" src="https://img.shields.io/badge/NVIDIA%20CUDA-%E2%89%A5%2012.8-76B900?logo=nvidia&logoColor=white">
  <img alt="GPU architectures sm_86 to sm_120" src="https://img.shields.io/badge/arch-sm__86%20%E2%80%93%20sm__120-76B900">
  <img alt="Written in Rust" src="https://img.shields.io/badge/language-Rust-B7410E?logo=rust&logoColor=white">
  <img alt="Linux x86_64 and aarch64" src="https://img.shields.io/badge/OS-Linux%20x86__64%20%2F%20aarch64-333333?logo=linux&logoColor=white">
  <a href="LICENSE"><img alt="Apache 2.0 license" src="https://img.shields.io/badge/license-Apache%202.0-blue"></a>
  <a href="https://github.com/meshapplied/goethe/actions/workflows/ci.yml"><img alt="CI status" src="https://github.com/meshapplied/goethe/actions/workflows/ci.yml/badge.svg"></a>
  <br>
  <a href="https://meshapplied.com/docs"><img alt="Documentation at meshapplied.com/docs" src="https://img.shields.io/badge/docs-meshapplied.com%2Fdocs-87D7FF?logo=readthedocs&logoColor=white"></a>
</p>

<p align="center"><b>Share the LLMs on your GPUs through one IP address, in three steps, and run them fast.</b></p>

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
- **Private by default.** Nothing can reach the API until you allow it, and every client needs its own key. You can rate-limit, expire, scope or revoke keys at any time.
- **Fits your network.** Serve on this machine only, over Tailscale, WireGuard or ZeroTier, or on a public IP. `goethe configure api` walks you through the choice.
- **Many models, many GPUs.** Run several models side by side, split large models across GPUs, and keep other GPU software running next to Goethe.
- **Always on.** Loaded models come back by themselves after a reboot or a crash.

## What you need

- Linux on x86_64 or ARM64.
- An NVIDIA GPU:
  - Blackwell: GeForce RTX 5090, RTX PRO 6000 Blackwell, DGX Spark (GB10), B200 or B300;
  - Hopper: H100, H200 or GH200;
  - Ada: GeForce RTX 4090, RTX 6000 Ada, L40S or L4;
  - or Ampere: A100, RTX A6000, GeForce RTX 3090, and similar.
  - GPUs of different kinds can share one machine; each model runs on GPUs of one kind.
- NVIDIA driver 580 or newer (data-center GPUs also run on 535 to 579 through NVIDIA's CUDA 13 forward compatibility). That is all: Goethe brings its own CUDA compiler, and the installer sets up the driver, a C++ compiler and the other tools it needs when they are missing (it asks first).
- Enough GPU memory for your model. `goethe recommend` lists the models that fit your machine.

Details on [supported hardware, model families and formats](docs/user-guide.md#what-goethe-supports) are in the user guide.

## Documentation

- **[User guide](docs/user-guide.md):** install, add models, share the API, every command with its options, and troubleshooting.
- **[Engine README](engine/README.md):** building Goethe from source and publishing releases.
- **[Research model packs](docs/research-models.md):** our published ternary and NVFP4 packs, the `goethe-lab` launcher and benchmark results.

## Repository layout

| Path | What it is |
|------|------------|
| [`engine/`](engine/) | Goethe: the `goethe` command, the background service, the gateway and the installer |
| [`nvfp4-kernels/`](nvfp4-kernels/), [`ternary-kernels/`](ternary-kernels/) | GPU kernels and model-pack tooling |
| [`goethe/`](goethe/), [`examples/`](examples/) | `goethe-lab`, the launcher for our research model packs |
| [`docs/`](docs/) | User guide and research notes |

Our models are on [Hugging Face](https://huggingface.co/meshapplied) and at [meshapplied.com/models](https://meshapplied.com/models).

## License

Apache-2.0 for this repository. The license of each model you serve still applies.

Goethe is built on open-source software, notably [SGLang](https://github.com/sgl-project/sglang) (Apache-2.0, shipped in a modified build), PyTorch, FlashInfer, Triton, Hugging Face Transformers, uv and nginx, and installs NVIDIA's CUDA libraries under NVIDIA's license. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) lists every component and its license; the installer places it, with the full license texts, in `$GOETHE_HOME`.
