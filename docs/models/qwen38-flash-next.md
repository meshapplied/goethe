# Qwen3.8-Flash-Next

Sparse MoE (125B total, 6B active per token) on [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Goethe ships a **ternary** expert pack on Hugging Face. **NVFP4** is served from upstream quant checkpoints with vLLM or SGLang; there is no meshapplied NVFP4 overlay in this repository yet.

Both variants keep attention, router, shared experts, MTP and the n-gram embedding table in the base checkpoint; only routed expert weights differ by compression path.

---

## Variant: ternary experts

### Hub

| Field | Value |
|-------|-------|
| Repository | [`meshapplied/qwen3.8-flash-next-ternary-latest`](https://huggingface.co/meshapplied/qwen3.8-flash-next-ternary-latest) |
| Kind | Ternary expert pack (Hadamard-rotated `{-1,0,+1}` routed experts) |
| Runtime profile | `flash_next` |

### Engine

[SGLang](https://github.com/sgl-project/sglang) with Goethe `ternary_runtime` (`pip install -e ./ternary-kernels`). Expert layout: `gate_up_down`, specialized K 2560 and 640. Pack contract: [pack.md](../pack.md).

### Hardware

NVIDIA GPU with CUDA matching your PyTorch build. Tensor parallel must respect Hadamard block sizes in the pack (see [pack.md](../pack.md)). The n-gram table is large; host RAM matters for long context (see vendor and community serve notes for Flash-Next).

### Install and serve

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe && pip install -e ./ternary-kernels
```

```python
import ternary_runtime
m = ternary_runtime.fetch("meshapplied/qwen3.8-flash-next-ternary-latest")
ternary_runtime.bootstrap()
```

```bash
python -m sglang.launch_server \
  --model-path "$TERNARY_MODEL_DIR" \
  --tp-size 1 \
  --host 127.0.0.1 --port 30000 \
  --trust-remote-code
```

Example template: [`examples/ternary-flash-next.toml`](../../examples/ternary-flash-next.toml).

### Measurement (ternary, lab)

Internal stack: ternary experts with BF16 dense layers, SGLang, native MTP, context up to about 8.6k tokens on one workstation GPU.

| Setting | Decode tok/s (C=1) | Notes |
|---------|-------------------|--------|
| Ternary v3, BF16 dense | 191 | Measured on lab hardware; not checked in to `nvfp4-kernels/`-style benchmark here |

No agent-replay benchmark is checked in to this repository for the ternary pack. Report engine version, batch size, context length and sampling with any number.

---

## Variant: NVFP4 experts

### Hub

| Field | Value |
|-------|-------|
| meshapplied | No NVFP4 pack or recipe repo in Goethe yet |
| Typical weights | NVIDIA ModelOpt NVFP4 builds and community forks (search Hugging Face for `Qwen3.8-Flash-Next` + NVFP4); official quant cards ship FP8 attention and NVFP4 MLP or expert paths per publisher |

Use the publisher revision you pin; do not assume byte compatibility with the ternary pack.

### Engine

[vLLM](https://github.com/vllm-project/vllm) 0.29+ or SGLang with Flash-Next support (BF16, FP8 and NVFP4 paths per engine release notes). Native MTP is the usual speculative path on NVFP4 stacks. DFlash2 for Qwen3.8-27B does not load on Flash-Next layer ids without a Flash-Next-specific drafter.

### Hardware

Often one Blackwell or datacenter GPU with enough VRAM for NVFP4 experts plus dense and MTP, or multi-GPU pipeline when the full n-gram table stays on host. Community PRO 6000 reports use 100 GB+ host RAM for the embedding table.

### Install and serve

Not automated in Goethe. Install your engine, download the NVFP4 checkpoint you trust, and follow that engine's Flash-Next cookbook. Pin revision, KV dtype and MTP settings in your own runbook.

### Measurement (NVFP4, field and lab anchors)

Numbers below are **not** from the scripts in this repo; harnesses differ (prompt length, power cap, MTP acceptance).

| Source | Decode tok/s (C=1) | Notes |
|--------|-------------------|--------|
| Community, PRO 6000, NVFP4 + MTP | 171 to 231 | Forum and GitHub reports; acceptance 2.58 typical for strict MTP |
| Pennyroyal / online FP8 dense on NVFP4 stack | about 207 | Short context; FP8 dense path on experts |
| Lab ternary (above) | 191 | BF16 dense, not NVFP4 |

At equal context on agentic workloads, 27B NVFP4 + DFlash2 has beaten Flash-Next + MTP on aggregate C=4 in field comparisons because draft acceptance differs; treat model choice as workload-dependent, not a single tok/s ranking.

---

## Repository paths

| Path | Role |
|------|------|
| [`ternary-kernels/`](../../ternary-kernels/) | Ternary loader, Hub fetch, CUDA kernels, SGLang hooks |
| [`hub/qwen38-flash-next-ternary-latest/README.md`](../../hub/qwen38-flash-next-ternary-latest/README.md) | Hugging Face card for the ternary pack |
| [`docs/architecture.md`](../architecture.md) | Ternary runtime design |

## License

Apache-2.0 for Goethe ternary pack and code. Base checkpoint and any third-party NVFP4 weights keep their licenses on Hugging Face.
