# ternary-kernels

The library `goethe serve` calls for the two ternary models. `pip install -e .`, then `import ternary_runtime`.

| Profile | Pack |
|---------|------|
| `flash_next` | [`meshapplied/Qwen3.8-Flash-Next-Ternary-Latest`](https://huggingface.co/meshapplied/Qwen3.8-Flash-Next-Ternary-Latest) |
| `deepseek_v41` | [`meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest`](https://huggingface.co/meshapplied/DeepSeek-V4.1-Flash-Ternary-Latest) |

`fetch` downloads the pack and the public base. `bootstrap()` patches SGLang in that process. Flags: [`../examples/`](../examples/). Pack layout: [`../docs/pack.md`](../docs/pack.md).
