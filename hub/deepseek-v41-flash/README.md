---
license: apache-2.0
tags:
  - ternary
  - moe
  - deepseek
base_model: deepseek-ai/DeepSeek-V4.1-Flash
---

# DeepSeek-V4.1-Flash ternary pack

Ternary expert overlay for [`deepseek-ai/DeepSeek-V4.1-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash). This repository is a ternary expert pack, not an NVFP4 weight dump. Routed experts are Hadamard-rotated `{-1,0,+1}`; attention, router, shared experts and MTP remain in the base checkpoint.

## Run

Engine: SGLang with Goethe `ternary_runtime`. Install, fetch and serve:

```bash
git clone https://github.com/meshapplied/Goethe.git
cd Goethe && pip install -e ./ternary-kernels
```

```python
import ternary_runtime
ternary_runtime.fetch("meshapplied/DeepSeek-V4.1-Flash-NVFP4")
ternary_runtime.bootstrap()
```

Reference (hardware, serve flags, license): [docs/models/deepseek-v41-flash.md](https://github.com/meshapplied/Goethe/blob/main/docs/models/deepseek-v41-flash.md).

## License

Apache-2.0 for this pack. The DeepSeek base checkpoint license applies when you download and run the base model.
