# Quickstart

| Model | Doc |
|-------|-----|
| Qwen3.8-Flash-Next (ternary and NVFP4) | [models/qwen38-flash-next.md](models/qwen38-flash-next.md) |
| DeepSeek-V4.1-Flash | [models/deepseek-v41-flash.md](models/deepseek-v41-flash.md) |
| Qwen3.8-27B NVFP4 | [models/qwen38-27b-nvfp4.md](models/qwen38-27b-nvfp4.md) |

Ternary fetch:

```python
import ternary_runtime
m = ternary_runtime.fetch("meshapplied/qwen3.8-flash-next-ternary-latest")
ternary_runtime.bootstrap()
```

Qwen3.8-27B:

```bash
cd nvfp4-kernels && ./install.sh && ./fetch_model.sh && ./serve.sh
```
