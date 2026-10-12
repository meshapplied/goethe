# Quickstart

Pick a model from the [catalog](../README.md#catalog) and open its page. The two ternary models share one Python package; Qwen3.8-27B uses the `nvfp4-kernels` shell scripts.

| Model | Doc |
|-------|-----|
| Qwen3.8-Flash-Next | [models/qwen38-flash-next.md](models/qwen38-flash-next.md) |
| DeepSeek-V4.1-Flash | [models/deepseek-v41-flash.md](models/deepseek-v41-flash.md) |
| Qwen3.8-27B artifact | [models/qwen38-27b-artifact.md](models/qwen38-27b-artifact.md) |
| Qwen3.8-27B recipe | [models/qwen38-27b-recipe.md](models/qwen38-27b-recipe.md) |

Ternary fetch example:

```python
import ternary_runtime
m = ternary_runtime.fetch("meshapplied/qwen3.8-flash-next-ternary-latest")
ternary_runtime.bootstrap()
print(m.profile, m.pack_dir, m.base_dir)
```

NInfer recipe:

```bash
cd nvfp4-kernels && ./install.sh && ./fetch_model.sh && ./serve.sh
```
