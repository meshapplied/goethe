---
license: apache-2.0
tags:
  - ternary
  - moe
  - qwen
  - flash-next
base_model: Qwen/Qwen3.8-Flash-Next
---

# Qwen3.8-Flash-Next ternary pack

Ternary expert overlay for [`Qwen/Qwen3.8-Flash-Next`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next). Routed experts are Hadamard-rotated `{-1,0,+1}`; attention, router, shared experts and MTP remain in the base checkpoint.

## Run

Engine: SGLang with Goethe `ternary_runtime`. Install, fetch and serve:

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe && pip install -e ./ternary-kernels
```

```python
import ternary_runtime
ternary_runtime.fetch("meshapplied/qwen3.8-flash-next-ternary-latest")
ternary_runtime.bootstrap()
```

Reference (hardware, serve flags, license): [docs/models/qwen38-flash-next.md](https://github.com/meshapplied/Goethe/blob/main/docs/models/qwen38-flash-next.md).

## License

Apache-2.0 for this pack. The Qwen base checkpoint license applies when you download and run the base model.
