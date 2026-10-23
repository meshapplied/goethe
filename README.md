# Goethe

**Share the LLMs on your GPUs through one IP address, in three steps, and run them fast.**

Goethe turns a Linux machine with an NVIDIA GPU into your own OpenAI-compatible API. Install it, add a model, and point your apps, your team or your other machines at one address. Your prompts, your data and your models stay on hardware you control.

```bash
# 1. Install
curl -fsSL https://meshapplied.com/install.sh | bash

# 2. Start Goethe and load a model
goethe up
goethe install Qwen/Qwen3-8B --load

# 3. Share it
goethe configure api                  # this machine only, your VPN (recommended), or a public address
goethe key create --name my-laptop    # one key per app or person
goethe get api                        # the address to give your apps
```

Anything that speaks the OpenAI API can use it right away: the OpenAI SDKs, LangChain, LlamaIndex, Open WebUI, coding assistants, or plain `curl`.

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
- **Fast.** Models run on SGLang, with continuous batching, prefix caching and CUDA graphs. Goethe adds its own NVFP4 kernels for Blackwell and Ampere GPUs, so 4-bit models use about a quarter of the memory of the full-precision version and stay quick.
- **Private by default.** Nothing can reach the API until you allow it, and every client needs its own key. You can rate-limit, expire, scope or revoke keys at any time.
- **Fits your network.** Serve on this machine only, over Tailscale, WireGuard or ZeroTier, or on a public IP. `goethe configure api` walks you through the choice.
- **Many models, many GPUs.** Run several models side by side, split large models across GPUs, and keep other GPU software running next to Goethe.
- **Always on.** Loaded models come back by themselves after a reboot or a crash.

## What you need

- Linux on x86_64 or ARM64.
- An NVIDIA GPU:
  - Blackwell: B100, B200, B300, GeForce RTX 5090, RTX PRO 6000 Blackwell, or DGX Spark (GB10);
  - or Ampere with compute capability 8.6: GeForce RTX 3090, RTX A6000, and similar.
- NVIDIA driver 580 or newer.
- The CUDA 13.0 toolkit, used to build GPU kernels on the first model load ([install steps](docs/user-guide.md#install)).
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
