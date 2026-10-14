# Checks that still need the 5090

Rescheduled while the card is in use. None of these run in CI. A check is closed only when the command below has been run on current main and the result is written here.

## Not run

- **Context above capacity.** `goethe serve --profile long` is not this. Boot the engine at 220032, 240000 and 262144 and keep the FATAL line and the `capacity | KV` line. Fit was measured once at 262144 stepping to 212992. That was not repeated on current main.
- **Four-lane KV pressure.** One replay at 4 lanes and the same replay at 1 lane. Record preemptions and replayed tokens. Not measured.
- **Cold time-to-ready.** Drop the page cache, then time `goethe serve` until `/health` answers. The 6.7 s figure is a warm start.
- **Memory hog.** A process holding 6 GiB, then 12 GiB, while `goethe serve` fits. Record the context it lands on and confirm no `ninfer-serve` is left running. The plan text says fewer lanes or an MTP or int4 fallback. The code never changes lanes and has no int4 fallback, so write what the boot actually does.
- **Chat acceptance.** `--profile chat` on a prompt that fits in 32768 tokens. The earlier chat attempt was rejected because the replay prompts reach 60k.
- **Greedy digest.** `goethe serve` with 1 lane, temperature 0, the held-out replay. The reference digest is `c9da62f164a33a1f`. It was matched before the fit and connect work. It has not been matched on current main.
- **Harness tool call.** One task that writes a file, for claude, codex, aider, opencode, qwen (`-y`), goose, kilo and mini, against a server started from current main. Done once before current main.
- **Live conformance.** `connect` against `/v1/chat/completions`, `/v1/responses` and `/v1/messages` on a running server from current main. Done once before current main.
- **Prebuilt binary, when one exists.** Boot it on the 5090 and compare the greedy digest with the source build. No tarball exists yet.

## Not the GPU

Parallel-range download throughput from Hugging Face is a phase 0 spike and needs bandwidth, not the card. `goethe update --check` was run on the source build already (engine `81c8ce093b2c`, patched model `afcaac9cd6b6`). `ldd` on that binary resolves libcuda, FFmpeg, libcurl and libcudart. `goethe doctor` in a GPU-less Python 3.11 container fails on `nvidia-smi` and names driver 580.
