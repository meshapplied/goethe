# Goethe PR180 greengoblin E2E session

- **When (UTC):** 2026-11-18T21:07:05Z
- **Host:** greengoblin
- **Binary:** `/home/v/proj/petya/goethe-pr180/engine/target/release/goethe` (`goethe 0.3.3`)
- **GOETHE_HOME (isolated):** `/home/v/proj/petya/goethe-e2e-180/home`
- **Prod goethe (restored after):** `/home/v/.local/bin/goethe`
- **Scope:** install/scan local path, fit, load, run (chat + embed), CLI, tailnet expose only — **no public share**
- **GPU:** GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-aaabe7a6-e328-9869-cca8-a8e56a3d4591)

## Method

Isolated `GOETHE_HOME` so the live 0.3.x install is not mutated. Production `goethe` is stopped for the session (port/GPU), then started again. Findings are written as observed — not softened.

## Preflight
disk:
  Filesystem      Size  Used Avail Use% Mounted on
  /dev/nvme0n1p1  3.6T  3.0T  489G  86% /
  /dev/nvme0n1p1  3.6T  3.0T  489G  86% /
gpu mem:
  memory.used [MiB], memory.free [MiB], memory.total [MiB]
  15 MiB, 32095 MiB, 32607 MiB
prod status:
  goethe: running
  API: https://100.90.97.111:8443/v1
  API KEY NAME  KEY ID
  matteo        95f818903035
  
  No models loaded. Pick one and load it:
  
  	→ goethe models
  	→ goethe load <model-id>
  
  Platform: Linked (workspace Personal, heartbeat OK (10 s ago))
share (prod binary may lack subcommand):
  The public gateway is closed. Open it with: goethe api share 5h --key <key name>

> Stopping production Goethe for isolated E2E (will restore).

## 1. Start (fresh home)

### goethe up

```
+ goethe up
```

```
goethe: running
API: https://127.0.0.1:8443/v1
keys: none, create one with: goethe api keys create my-app

No models loaded. Pick one and load it:

	→ goethe models
	→ goethe load <model-id>

Platform: Not linked (goethe login)
```

exit: `0`
**result:** pass

### goethe status

```
+ goethe status
```

```
goethe: running
API: https://127.0.0.1:8443/v1
keys: none, create one with: goethe api keys create my-app

No models loaded. Pick one and load it:

	→ goethe models
	→ goethe load <model-id>

Platform: Not linked (goethe login)
```

exit: `0`
**result:** pass

### goethe doctor

```
+ goethe doctor
```

```
CHECK     STATUS  DETAIL
hardware  OK      driver 580.178.04
                  gpu0 NVIDIA GeForce RTX 5090 (SM 12.0): 31.3 of 31.8 GB free, 0 foreign processes
platform  INFO    not logged in: optional; goethe login links this node and unlocks optimised kernels
cloud     INFO    no cloud provider detected
py-env    INFO    not set up yet: the first goethe install of a model sets it up (about 30 MB, more for engines)
also OK: driver, glibc, nginx, goethe_home, disk, ram, state_db, environment, ports, goethed, gateway, network, tls, huggingface, registry, nvidia_driver, compiler (goethe -v doctor lists them)

GPU architectures goethe supports: sm_80 (Ampere data center), sm_86 (Ampere), sm_89 (Ada), sm_90a (Hopper), sm_100a (Blackwell data center), sm_103a (Blackwell Ultra data center), sm_120a (Blackwell), sm_121a (Blackwell)
Architectures of this machine: sm_120a
```

exit: `0`
**result:** pass

### goethe doctor --hardware

```
+ goethe doctor --hardware
```

```
Goethe hardware profile at 2026-11-18T21:07:13Z
Driver: 580.178.04
GPU   NAME                     VRAM TOTAL  VRAM FREE  PCIE
gpu0  NVIDIA GeForce RTX 5090  31.8 GB     31.3 GB    PCIe 5 x16
      SM 12.0
CPU: Intel(R) Core(TM) Ultra 7 265K (GenuineIntel), 1 socket(s), 20 cores, 20 threads, flags [avx2]
RAM: 125.1 GB total, 119.7 GB available; swap 3.6/8.0 GB free; hugepages 0/0 free (2048 kB)
Disk goethe_home: /home/v/proj/petya/goethe-e2e-180/home on / (ext4) 525 GB free of 3936 GB
Disk models: /home/v/proj/petya/goethe-e2e-180/home/models on / (ext4) 525 GB free of 3936 GB
OS: Ubuntu 24.04.3 LTS, kernel 7.0.0-42-generic, glibc 2.39
NVIDIA driver 580.178.04, CUDA driver 13.0
Environment: no container, systemd is PID 1, cloud none
```

exit: `0`
**result:** pass

## 2. CLI surface / help consistency

### goethe --help

```
+ goethe --help
```

```
Goethe: serve models through OpenAI and Anthropic compatible gateways

Start and stop
  goethe                                                  the menu: quick setup, install models, API keys and gateway, status
  goethe up                                               start Goethe now and at every boot
  goethe down                                             stop Goethe; loaded models come back on the next up
  goethe status                                           running or not, the API address, keys and served models
  goethe ps                                               running models in detail: GPUs, memory, requests, speed

Models
  goethe install MODEL --load                             download from Hugging Face (or add a folder) and serve it
  goethe load MODEL --gpu gpu2 gpu3                       serve an installed model on these GPUs
  goethe load MODEL --ctx 32768 --save                    the same, and keep the settings as its defaults
  goethe load MODEL --dry-run                             show where it would go, load nothing
  goethe unload MODEL                                     stop serving it (--all stops every model)
  goethe remove MODEL                                     unload and delete it
  goethe run MODEL                                        try it by hand: chat, one reply, embedding or transcript
  goethe models                                           installed and catalog models
  goethe models recommend                                 catalog models that fit your GPUs
  goethe models scan FOLDER                               register a folder you put in GOETHE_HOME/models
  goethe models alias set fast MODEL                      a short name clients can use (remove fast drops it)
  goethe logs MODEL -f                                    follow a model's log (also goethed or gateway)

API and access
  goethe api                                              the address, the keys and a curl command to try it
  goethe api keys                                         list the keys (a menu on a terminal)
  goethe api keys create my-app                           a new key (also --expires 30d --rpm 300 --models ID --admin)
  goethe api keys revoke KEY_ID                           revoke a key
  goethe api expose                                       where the API listens and who may connect
  goethe api expose vpn --interface wg0                   serve a VPN or private network (also local or public)
  goethe api allow add 203.0.113.7/32                     let a client address in (remove takes it out)
  goethe api domain check DOMAIN                          check that a domain reaches the API
  goethe api share 5h --key my-app --allow <their-ip>/32  open a public address for a while (share stop closes it)

Learning and traces
  goethe learn on                                         sessions keep what compaction drops (--model M for one model)
  goethe trace on                                         record requests and replies for training (--key K for one key)

Machine
  goethe gpu                                              GPUs live: memory, processes and bandwidth
  goethe gpu deny gpu2                                    keep new models off a GPU (allow gpu2 undoes it)
  goethe doctor                                           check the machine and the setup, with a fix for each problem
  goethe doctor --hardware                                the hardware profile: GPUs and links, CPU, RAM, disk, OS
  goethe update                                           signed update of Goethe, its engines and kernel packs
  goethe uninstall                                        remove Goethe and the models it installed
  goethe login                                            link this node to the Goethe platform (optional)
  goethe logout                                           unlink it and remove its local credentials
  goethe completion install                               tab completion for your shell

MODEL is a Hugging Face id, an installed id, an alias or words from a name, for example:

	→ goethe install nvidia/Qwen2.5-VL-7B-Instruct-NVFP4 --load
```

exit: `0`
**result:** pass

### goethe api share --help

```
+ goethe api share --help
```

```
Open the API on a public address for a while, for chosen API keys; stop closes it.

Usage: goethe api share [OPTIONS] [DURATION]

Arguments:
  [DURATION]  How long to stay open (30m, 5h, 1d; at most 7d), stop to close now, or nothing for the status.

Options:
      --key <NAME_OR_ID>...            API keys to share (not admin keys), each until DURATION; with stop, the keys to stop sharing.
      --allow <CIDR>...                Only these client addresses or ranges may use the keys given, such as 203.0.113.7.
  -v, --verbose                        Show the output of every underlying process (also GOETHE_VERBOSE=1); it is always kept in <home>/logs.
      --dangerously-allow-all-inbound  Allow every address on the public gateway (0.0.0.0/0). Rate limits still apply. Not with --allow.
      --port <PORT>                    The public port (default 8444, or the last one used).
  -h, --help                           Print help

Examples:

	→ goethe api share 5h --key my-app --allow <their-ip>/32
	→ goethe api share 5h --key my-app --dangerously-allow-all-inbound
	→ goethe api share             (status)
	→ goethe api share stop

--allow is who may connect (their public IP: ask them, or https://ifconfig.me on their device). This machine's IP is only the URL goethe prints. Reopening the same keys keeps their last addresses. --dangerously-allow-all-inbound opens to every address (rate limits still apply). Closes when the last key ends.
```

exit: `0`
**result:** pass

### goethe api expose --help

```
+ goethe api expose --help
```

```
Who can reach the API: local, vpn, public, or one address or interface.

Usage: goethe api expose [OPTIONS] [WHERE]

Arguments:
  [WHERE]  local, vpn or public; or an address (127.0.0.1, 0.0.0.0, an IP of this host) or an interface such as wg0 to listen on.

Options:
      --interface <NAME>  With vpn or public: the network interface to serve, e.g. wg0; the gateway follows its address.
      --ip <IP>           With vpn or public: a fixed address of this machine to listen on, instead of following an interface.
  -v, --verbose           Show the output of every underlying process (also GOETHE_VERBOSE=1); it is always kept in <home>/logs.
      --allow <CIDR>...   With vpn or public: client ranges to allow (default for vpn: the ranges the network reaches).
      --yes               Apply without questions (public still refuses 0.0.0.0/0).
      --json              With no WHERE: the status as JSON.
  -h, --help              Print help

Who can reach the API:

	→ goethe api expose local            (this machine only)
	→ goethe api expose vpn --interface wg0
	→ goethe api expose public --allow 203.0.113.7/32

Or listen on one address or interface directly:

	→ goethe api expose 0.0.0.0
	→ goethe api expose wg0

goethe api expose alone shows where the API listens and who may connect.
```

exit: `0`
**result:** pass

### goethe models --help

```
+ goethe models --help
```

```
Installed and catalog models; also recommend, scan and alias.

Usage: goethe models [OPTIONS]
       goethe models <COMMAND>

Commands:
  recommend  Catalog models that fit the free VRAM of the allowed GPUs.
  scan       Register model folders you downloaded yourself and moved into GOETHE_HOME/models.
  alias      Short model names for clients: list them, set one, or remove some.
  help       Print this message or the help of the given subcommand(s)

Options:
  -v, --verbose    Show the output of every underlying process (also GOETHE_VERBOSE=1); it is always kept in <home>/logs.
      --json       Emit JSON.
      --installed  Only installed models.
      --available  Only catalog models.
      --all        Every line; the default stops after 40.
  -h, --help       Print help
```

exit: `0`
**result:** pass

### goethe run --help

```
+ goethe run --help
```

```
Try a model by hand: a chat screen on a terminal without INPUT, else one reply, an embedding summary or an audio file's transcript. Loads an installed model that is not loaded and unloads it again.

Usage: goethe run [OPTIONS] [MODEL]...

Arguments:
  [MODEL]...  [MODEL] [INPUT]: MODEL may be left out when one model is loaded.

Options:
      --keep      Leave a model loaded that this command loaded (it is unloaded when run ends otherwise).
      --plain     No chat screen: the plain prompt, as when input or output is not a terminal.
  -v, --verbose   Show the output of every underlying process (also GOETHE_VERBOSE=1); it is always kept in <home>/logs.
      --thinking  Show the model's reasoning (dimmed) with the answer.
  -h, --help      Print help
```

exit: `0`
**result:** pass

## 3. What fits

### goethe models recommend

```
+ goethe models recommend
```

```
VRAM free: 31.8 GB

meshapplied/Qwen3.8-27B-NVFP4-NInfer  ctx 100000  concurrency 4

Up to 24 GB
MODEL                                VRAM                               DOWNLOAD
nvidia/Qwen2.5-VL-7B-Instruct-NVFP4  22.2 GB/GPU · ctx 128000 · 8 conc  7.2 GB
nvidia/Llama-3.1-8B-Instruct-NVFP4   23.4 GB/GPU · ctx 65536 · 8 conc   6.0 GB

Up to 48 GB
MODEL                                 VRAM                               DOWNLOAD
nvidia/Qwen3.8-27B-NVFP4              26.8 GB/GPU · ctx 32768 · 8 conc   21.9 GB
microsoft/Phi-4-multimodal-instruct   28.2 GB/GPU · ctx 65536 · 8 conc   12.9 GB
meshapplied/Qwen3.8-27B-NVFP4-NInfer  28.4 GB/GPU · ctx 100000 · 4 conc  -
```

exit: `0`
**result:** pass

### goethe models

```
+ goethe models
```

```
VRAM free: 31.8 GB

Up to 24 GB
MODEL                                STATUS     DISK    FIT ON THIS MACHINE
nvidia/Qwen2.5-VL-7B-Instruct-NVFP4  available  7.2 GB  fits: 22.2 GB per GPU (ctx 128000, 8 concurrent)
nvidia/Llama-3.1-8B-Instruct-NVFP4   available  6.0 GB  fits: 23.4 GB per GPU (ctx 65536, 8 concurrent)

Up to 48 GB
MODEL                                                   STATUS     DISK     FIT ON THIS MACHINE
nvidia/Qwen3.8-27B-NVFP4 (optimised kernels available)  available  21.9 GB  fits: 26.8 GB per GPU (ctx 32768, 8 concurrent)
microsoft/Phi-4-multimodal-instruct                     available  12.9 GB  fits: 28.2 GB per GPU (ctx 65536, 8 concurrent)
meshapplied/Qwen3.8-27B-NVFP4-NInfer                    available  28.4 GB  fits: 28.4 GB per GPU (ctx 100000, 4 concurrent)

Too large for the allowed GPUs
MODEL                                               STATUS     DISK       FIT ON THIS MACHINE
nvidia/Gemma-4-31B-IT-NVFP4                         available  32.7 GB    does not fit: not enough GPU memory
mistralai/Mistral-Small-3.1-24B-Instruct-2503       available  48.1 GB    does not fit: not enough GPU memory
google/gemma-4-26B-A4B-it                           available  51.6 GB    does not fit: not enough GPU memory
Qwen/Qwen3.8-27B                                    available  55.6 GB    does not fit: not enough GPU memory
meta-models/Muse-Glimmer-30B                        available  59.6 GB    does not fit: not enough GPU memory
meshapplied/Qwen3.8-Flash-Next-NVFP4-Latest         available  132.7 GB   does not fit: not enough GPU memory (gated: needs your Hugging Face token or access)
nvidia/Qwen3.8-Flash-Next-NVFP4                     available  132.7 GB   does not fit: not enough GPU memory
deepseek-ai/DeepSeek-V4-Flash-0731                  available  166.9 GB   does not fit: not enough GPU memory
XiaomiMiMo/MiMo-V2.6-Flash-RL                       available  177.8 GB   does not fit: not enough GPU memory
orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4      available  183.5 GB   does not fit: not enough GPU memory (gated: needs your Hugging Face token or access)
RedHatAI/GLM-5.3-Flash-NVFP4                        available  197.9 GB   does not fit: not enough GPU memory
nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16  available  248.9 GB   does not fit: not enough GPU memory
nvidia/MiniMax-M3-NVFP4                             available  250.1 GB   does not fit: not enough GPU memory
nvidia/Qwen3.5-397B-A17B-NVFP4                      available  251.2 GB   does not fit: not enough GPU memory
inclusionAI/Ling-3.0-flash                          available  255.0 GB   does not fit: not enough GPU memory
Qwen/Qwen3.8-Flash-Next                             available  360.0 GB   does not fit: not enough GPU memory
meshapplied/Qwen3.8-Flash-Next-Ternary-Latest       available  391.9 GB   does not fit: not enough GPU memory
nvidia/DeepSeek-V3.1-NVFP4                          available  413.3 GB   does not fit: not enough GPU memory
nvidia/DeepSeek-V3.2-NVFP4                          available  415.1 GB   does not fit: not enough GPU memory
nvidia/DeepSeek-V3-0324-NVFP4                       available  423.7 GB   does not fit: not enough GPU memory
nvidia/DeepSeek-R1-NVFP4                            available  423.7 GB   does not fit: not enough GPU memory
LibertAIDAI/DeepSeek-V4.1-Flash-1017-NVFP4          available  429.0 GB   does not fit: not enough GPU memory
nvidia/GLM-5.2-NVFP4                                available  464.9 GB   does not fit: not enough GPU memory
Qwen/Qwen3-VL-235B-A22B-Instruct                    available  471.4 GB   does not fit: not enough GPU memory
deepseek-ai/DeepSeek-V4.1-Flash-1017                available  510.3 GB   does not fit: not enough GPU memory
... 23 more lines; all of them: goethe models --all
```

exit: `0`
**result:** pass

> Check whether models vs recommend show comparable per-GPU + ctx lines for the same id.

## 4. Local path model (scan + install from folder)

### goethe models scan models/ggml-org__embeddinggemma-2-GGUF

```
+ goethe models scan ggml-org/embeddinggemma-2-GGUF
```

```
ggml-org/embeddinggemma-2-GGUF: No such file or directory (os error 2)
```

exit: `1`
**result:** FAIL (expected success)

### goethe models --installed

```
+ goethe models --installed
```

```
VRAM free: 31.8 GB
```

exit: `0`
**result:** pass

## 5. API key + tailnet expose (NO public)

### create key e2e-tailnet

```
+ goethe api keys create e2e-tailnet --models ggml-org/embeddinggemma-2-GGUF
```

```
API key for e2e-tailnet (id d5be91cb0cc5):

	goethe_sk_REDACTED

goethe api shows it again at any time.
```

exit: `0`
**result:** pass

### goethe api (capture)
```
API endpoint: https://127.0.0.1:8443/v1
SSL_CERT_FILE=/home/v/proj/petya/goethe-e2e-180/home/nginx/certs/server.crt
Accepts: OpenAI, Anthropic (/v1/messages) and LiteLLM (routes with or without /v1, x-litellm-api-key)

API keys
NAME         API KEY                                     ADMIN  EXPIRES
e2e-tailnet  goethe_sk_REDACTED  no     never

The allowlist is empty, so every client is refused. Allow one with:

	→ goethe api allow add <client-ip>/32

Query the API:

	→ curl --cacert /home/v/proj/petya/goethe-e2e-180/home/nginx/certs/server.crt https://127.0.0.1:8443/v1/chat/completions -H "Authorization: Bearer <your_bearer_token>" -H "Content-Type: application/json" -d '{"model": "<model_name>", "messages": [{"role": "user", "content": "Hello!"}]}'
```

### keys list
```
NAME         KEY ID        ADMIN  RPM  EXPIRES  MODELS                          TODAY
e2e-tailnet  d5be91cb0cc5  no     120  never    ggml-org/embeddinggemma-2-GGUF  0 req, 0 tok
```

### expose vpn on tailscale0 with CGNAT allow

```
+ goethe api expose vpn --interface tailscale0 --allow 100.64.0.0/10 --yes
```

```
Gateway now: listening on 127.0.0.1 port 8443; allowlist: empty (every client refused)
The gateway will listen on 100.90.97.111 (tailscale0; follows its address) and on 127.0.0.1, and allow 100.64.0.0/10.
gateway listens on 100.90.97.111 (tailscale0; follows its address) port 8443
allowed 127.0.0.1/32 (this machine)
allowed 100.64.0.0/10 (tailscale0 clients)
API keys: 1 exist (goethe api keys; goethe api keys create <client> for each new client)

API: https://100.90.97.111:8443/v1

For the models and keys run:

	→ goethe api
```

exit: `0`
**result:** pass

### goethe api expose (status)

```
+ goethe api expose
```

```
bind: 100.90.97.111 (follows tailscale0) (remote clients reach it when allowlisted)
port: 8443
url: https://100.90.97.111:8443
state: running
allowlist: 2 entries
```

exit: `0`
**result:** pass

### allowlist shows tailnet

```
+ goethe api allow
```

```
ALLOWED CLIENTS
100.64.0.0/10
127.0.0.1/32
```

exit: `0`
**result:** pass

### public expose without --allow should refuse or warn

```
+ goethe api expose public --yes
```

```
Gateway now: listening on 100.90.97.111 (following tailscale0) port 8443; allowlist: 100.64.0.0/10, 127.0.0.1/32
WARNING: exposing the API on a public network is not recommended.
  - Anyone who can reach this machine can attempt to connect; only the allowlist and API keys stop them.
  - Allow only the client IPs you trust (goethe api allow add <ip>/32) and give every client its own key.
  - Open the gateway port in your firewall or router yourself; Goethe does not change them.
  - The certificate is self-signed, so clients must trust it or skip verification.
Your public IP: 81.182.125.247 -> API: https://81.182.125.247:8443/v1
This machine is behind NAT (81.182.125.247 is not on its interfaces): forward TCP port 8443 on your router or cloud firewall to it.
goethe api expose: restarting nginx to move the listener from 100.90.97.111 to 0.0.0.0
goethe api expose: stopping nginx master [3210154]
gateway listens on 0.0.0.0 port 8443 (restarted)
No client addresses were added: only addresses already on the allowlist can connect (goethe api allow add <ip>/32).
API keys: 1 exist (goethe api keys; goethe api keys create <client> for each new client)

API: https://81.182.125.247:8443/v1

For the models and keys run:

	→ goethe api
```

exit: `0`
**result:** FAIL (expected refusal)

### share without --allow / remembered grant

```
+ goethe api share 5m --key e2e-tailnet
```

```
name who may connect with --allow <their-ip>/32 (their phone or laptop's public IP, not this machine's). this machine's public address is 81.182.125.247 (that becomes the URL, not --allow); ask them; to open to every address use --dangerously-allow-all-inbound
```

exit: `1`
**result:** pass (refused as expected)

### share with this-machine IP as --allow (if public IP known)

```
+ bash -c IP=$(curl -4 -fsS --max-time 3 ifconfig.me || true); if [ -n "$IP" ]; then goethe api share 5m --key e2e-tailnet --allow "$IP/32"; else echo "skip: no public ip"; exit 1; fi
```

```
81.182.125.247/32 is this machine's public address (the URL), not who may connect; use their phone or laptop's public IP
```

exit: `1`
**result:** pass (refused as expected)

### share status (should be closed)

```
+ goethe api share
```

```
The public gateway is closed. Open it with: goethe api share 5h --key <name> --allow <their-ip>/32
```

exit: `0`
**result:** pass

> Confirm no :8444 listener

### ss public ports

```
+ bash -c ss -ltn | grep 8444 || echo no_8444
```

```
no_8444
```

exit: `0`

## 6. Load + run embedding model

### load dry-run embedding

```
+ goethe load ggml-org/embeddinggemma-2-GGUF --dry-run
```

```
model ggml-org/embeddinggemma-2-GGUF is not installed (goethe models lists installed models; goethe install ggml-org/embeddinggemma-2-GGUF adds it)
```

exit: `1`
**result:** FAIL (expected success)

### load embedding

```
+ goethe load ggml-org/embeddinggemma-2-GGUF
```

```
model ggml-org/embeddinggemma-2-GGUF is not installed (goethe models lists installed models; goethe install ggml-org/embeddinggemma-2-GGUF adds it)
```

exit: `1`
**result:** FAIL (expected success)

### goethe ps

```
+ goethe ps
```

```
No models loaded. Pick one and load it:

	→ goethe models
	→ goethe load <model-id>
```

exit: `0`
**result:** pass

### goethe status after load

```
+ goethe status
```

```
goethe: running
API: https://127.0.0.1:8443/v1
API KEY NAME  KEY ID
e2e-tailnet   d5be91cb0cc5

No models loaded. Pick one and load it:

	→ goethe models
	→ goethe load <model-id>

Platform: Not linked (goethe login)
```

exit: `0`
**result:** pass

### goethe run embed (prompt)

```
+ goethe run ggml-org/embeddinggemma-2-GGUF -- embed hello greengoblin e2e
```

```
error: unexpected value 'hello greengoblin e2e' for '[MODEL]...' found; no more were expected

Usage: goethe run [OPTIONS] [MODEL]...

For more information, try '--help'.
```

exit: `2`
