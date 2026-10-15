# Qwen3.8-27B NVFP4

23.7 GB file, Blackwell sm_120a, 32 GiB. `goethe serve` runs these scripts. Card: [`meshapplied/Qwen3.8-27B-NVFP4-NInfer`](https://huggingface.co/meshapplied/Qwen3.8-27B-NVFP4-NInfer). Weights: [`neroued/Qwen3.8-27B-nvfp4-NInfer`](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer), file `qwen3_8_27b_nvfp4.ninfer`. Pins in [NINFER_PIN](NINFER_PIN).

Engine: [NInfer](https://github.com/Neroued/ninfer) at the commit in `NINFER_PIN`, plus the patches below. CUDA 13.x, `nvcc`, `cmake`, `ninja`, C++20. RTX 5090 is the machine we measured. An RTX PRO 6000 is the same compute capability. We have not run it.

```bash
./run.sh
```

That builds NInfer, downloads and patches the model, and starts the server (the first run takes a while; later runs only start the server). The same in three steps: `./install.sh && ./fetch_model.sh && ./serve.sh`. Needs `git`, `cmake`, `ninja`, `nvcc`, `curl` and `python3` (standard library only).

| profile | drafter | K | lanes | max context |
|---------|---------|---|-------|-------------|
| `default` | DFlash2 | 7 | 4 | 100000 |
| `long` | DFlash2 | 7 | 1 | 131072 |
| `batch` | MTP | 4 | 8 | 131072 |

`PROFILE=<name> ./serve.sh`. Overrides: `SPEC`, `K`, `MAX_CONCURRENCY`, `MAX_CONTEXT`. fp8 KV, `--lm-head-draft`, `--prefill-chunk 1024`. OpenAI Chat Completions, Responses and Anthropic Messages on port 8080 by default.

DFlash2 does not start at 8 lanes and 131072 context on 32 GiB. Use MTP for 8 lanes.

Throughput (one RTX 5090, replayed agent turns, mean context ~28k, thinking on): see [RESULTS.md](RESULTS.md). Reproduce: `pip install httpx` then `bench/run_matrix.sh "mtp:3:8:131072 mtp:4:8:131072 dflash2:7:4:100000" "1 2 4"`.

## Patches

`install.sh` and `fetch_model.sh` apply two changes by default. Speculative decoding stays exact, so the output distribution does not change.

- `patches/ninfer-dn-sched.patch`: a better tile for the MLP down projection at 16 tokens or fewer. Same output bit for bit, about 1.4 % faster.
- `patches/apply_head.py`: the draft-token shortlist behind `--lm-head-draft` never contained `<tool_call>` and `</tool_call>`, so every tool call cost rejected drafts. This swaps 677 rows of it in the model file, about 3.5 % faster on agent traffic. The original rows are kept in `<model>.head-orig`.

Together about 5 % faster decode, see [RESULTS.md](RESULTS.md#patches). The throughput table above was measured without them. `NVFP4_PATCHES=0 ./install.sh` builds NInfer unpatched, `NVFP4_HEAD=0 ./fetch_model.sh` keeps or restores the original head.

Third-party attribution: [NOTICE](NOTICE).

## Flash-Next NVFP4 weight-only (SGLang)

Serve a Flash-Next NVFP4 weight-only (compressed-tensors) tree with SGLang, not NInfer. Stock SGLang does not dispatch that scheme. Apply the patch to the SGLang checkout first, then start from [`examples/nvfp4-weight-only.toml`](../examples/nvfp4-weight-only.toml):

```bash
git -C "$SGLANG_DIR" apply --check "$PWD/patches/sglang-nvfp4-weight-only.patch"
git -C "$SGLANG_DIR" apply "$PWD/patches/sglang-nvfp4-weight-only.patch"
```

The patch touches two files under `python/sglang/srt/layers/quantization/compressed_tensors/`: scheme selection for weight-only NVFP4 experts and the NVFP4 MoE input-scale defaults (1.0 when the checkpoint has no static activation scales). The WNA16 guard only matches INT weights, so FP8 channel weight-only layers reach the W8A16 FP8 scheme.
