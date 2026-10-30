# Registry entry checklist

A model in `engine/cdn/v1/registry.json` is supported only when all of these pass on every GPU generation in its `supported_sm`, for each tensor-parallel size it can split over:

1. **Flags are complete.** `extra_sglang_args` holds every engine flag the model needs, including the state dtype (`--mamba-ssm-dtype`) and speculative-decoding flags. Goethe sizes memory from these flags.
2. **Default loads work.** `goethe load --model <id>` and `goethe load --model <id> --gpu <one GPU>` reach RUNNING with no extra flags. A "reduced to fit" notice is fine; a failure is not.
3. **The plan matches the engine.** In the engine log, `max_total_num_tokens` is at least the planned KV tokens, and the allocated state is within 5% of the planned state. After the first load, Goethe learns less than 0.5 GB of extra weights.
4. **Unload and load again.** `goethe unload` followed at once by `goethe load` succeeds.
5. **First request.** The first request after RUNNING is answered at normal speed (Goethe warms the engine before RUNNING).
6. **Mixed load.** Every scenario of `engine/tests/load/mixed_load.py` finishes with no 5xx and no 429, and prompts longer than the context get a clean 400.
