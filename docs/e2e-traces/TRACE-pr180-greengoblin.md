# Goethe PR180 greengoblin E2E session

- **When (UTC):** 2026-11-18T21:10:59Z
- **Host:** greengoblin
- **Binary:** `/home/v/proj/petya/goethe-pr180/engine/target/release/goethe` (`goethe 0.3.3`)
- **GOETHE_HOME (isolated):** `/home/v/proj/petya/goethe-e2e-180/home`
- **Prod goethe (restored after):** `/home/v/.local/bin/goethe`
- **Scope:** install/scan local path, fit, load, run (chat + embed), CLI, tailnet expose — **no lasting public bind**
- **GPU:** GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-aaabe7a6-e328-9869-cca8-a8e56a3d4591)

## Method

Isolated `GOETHE_HOME` so the live 0.3.x install is not mutated. Production `goethe` is stopped for the session (port/GPU), then started again. Findings are written as observed — not softened. Script continues after failures (no mid-run abort).

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
  
  Platform: Linked (workspace Personal, heartbeat OK (11 s ago))
prod listeners:
  LISTEN 0      511                                 127.0.0.1:8443       0.0.0.0:*          
  LISTEN 0      511                             100.90.97.111:8443       0.0.0.0:*          

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
Goethe hardware profile at 2026-11-18T21:11:07Z
Driver: 580.178.04
GPU   NAME                     VRAM TOTAL  VRAM FREE  PCIE
gpu0  NVIDIA GeForce RTX 5090  31.8 GB     31.3 GB    PCIe 5 x16
      SM 12.0
CPU: Intel(R) Core(TM) Ultra 7 265K (GenuineIntel), 1 socket(s), 20 cores, 20 threads, flags [avx2]
RAM: 125.1 GB total, 119.8 GB available; swap 3.6/8.0 GB free; hugepages 0/0 free (2048 kB)
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

> goethe run has no -p/--prompt; INPUT is positional. CI should not teach -p.

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

> Compare models vs recommend per-GPU + ctx wording for the same id.

## 4. Local path model (scan + install from folder)

### models scan HF-id form (footgun check)

```
+ goethe models scan ggml-org/embeddinggemma-2-GGUF
```

```
ggml-org/embeddinggemma-2-GGUF: No such file or directory (os error 2)
```

exit: `1`
**result:** pass (HF-id form rejected)

### models scan folder name --yes

```
+ goethe models scan ggml-org__embeddinggemma-2-GGUF --yes
```

```
ggml-org__embeddinggemma-2-GGUF: No such file or directory (os error 2)
```

exit: `1`
**result:** FAIL (expected success)

### models --installed after scan

```
+ goethe models --installed
```

```
VRAM free: 31.8 GB
```

exit: `0`
**result:** pass

### install from absolute path --name

```
+ goethe install /home/v/proj/petya/goethe-e2e-180/home/models/ggml-org__embeddinggemma-2-GGUF --name ggml-org/embeddinggemma-2-GGUF --yes
```

```
engine   installing 0.3.0 (about 5 GB with PyTorch, once)
engine   0.3.0 ready (4.6 GB, 4 s)
ggml-org/embeddinggemma-2-GGUF  goethe-engine  ctx 1024  0.3 GB + 28 MB/req
files stay in /data/proj/gc/goethe-models/ggml-org__embeddinggemma-2-GGUF (goethe remove only unregisters)
run: goethe run ggml-org/embeddinggemma-2-GGUF
```

exit: `0`
**result:** pass

### models --installed after install path

```
+ goethe models --installed
```

```
VRAM free: 31.8 GB

Installed
MODEL                                                         STATUS     DISK    FIT ON THIS MACHINE
ggml-org/embeddinggemma-2-GGUF (optimised kernels available)  installed  0.4 GB  fits: 0.9 GB per GPU (ctx 1024, 8 concurrent)
```

exit: `0`
**result:** pass

## 5. API key + tailnet expose (NO lasting public)

### create key e2e-tailnet

```
+ goethe api keys create e2e-tailnet --models ggml-org/embeddinggemma-2-GGUF
```

```
API key for e2e-tailnet (id a62afdf8be04):

	goethe_sk_REDACTED

goethe api shows it again at any time.
```

exit: `0`
**result:** pass

### captured bearer (redact in shared copies)
```
goethe_sk_REDACTED
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

### ss after vpn expose

```
+ bash -c ss -ltn | grep -E "8443|8444" || true
```

```
LISTEN 0      511                                 127.0.0.1:8443       0.0.0.0:*          
LISTEN 0      511                             100.90.97.111:8443       0.0.0.0:*          
```

exit: `0`

### public expose without new allow

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
goethe api expose: stopping nginx master [3213137]
gateway listens on 0.0.0.0 port 8443 (restarted)
No client addresses were added: only addresses already on the allowlist can connect (goethe api allow add <ip>/32).
API keys: 1 exist (goethe api keys; goethe api keys create <client> for each new client)

API: https://81.182.125.247:8443/v1

For the models and keys run:

	→ goethe api
```

exit: `0`
**result:** FAIL for prod bar — public bind succeeded; allowlist unchanged (0.0.0.0 listen)

> Reverting to vpn/tailnet immediately after public probe.

### revert expose vpn after public probe

```
+ goethe api expose vpn --interface tailscale0 --allow 100.64.0.0/10 --yes
```

```
Gateway now: listening on 0.0.0.0 port 8443; allowlist: 100.64.0.0/10, 127.0.0.1/32
The gateway will listen on 100.90.97.111 (tailscale0; follows its address) and on 127.0.0.1, and allow 100.64.0.0/10.
goethe api expose: restarting nginx to move the listener from 0.0.0.0 to 100.90.97.111
goethe api expose: stopping nginx master [3213913]
gateway listens on 100.90.97.111 (tailscale0; follows its address) port 8443 (restarted)
API keys: 1 exist (goethe api keys; goethe api keys create <client> for each new client)

API: https://100.90.97.111:8443/v1

For the models and keys run:

	→ goethe api
```

exit: `0`

### ss after revert

```
+ bash -c ss -ltn | grep -E "8443|8444" || true
```

```
LISTEN 0      511                                 127.0.0.1:8443       0.0.0.0:*          
LISTEN 0      511                             100.90.97.111:8443       0.0.0.0:*          
```

exit: `0`
**result:** pass (no 0.0.0.0:8443 after vpn revert)

### share without --allow

```
+ goethe api share 5m --key e2e-tailnet
```

```
name who may connect with --allow <their-ip>/32 (their phone or laptop's public IP, not this machine's). this machine's public address is 81.182.125.247 (that becomes the URL, not --allow); ask them; to open to every address use --dangerously-allow-all-inbound
```

exit: `1`
**result:** pass (refused as expected)

### share with this-machine public IP as --allow

```
+ bash -c 
  IP=$(curl -4 -fsS --max-time 3 ifconfig.me || true)
  if [ -z "$IP" ]; then echo "skip: no public ip"; exit 1; fi
  goethe api share 5m --key e2e-tailnet --allow "$IP/32"

```

```
81.182.125.247/32 is this machine's public address (the URL), not who may connect; use their phone or laptop's public IP
```

exit: `1`
**result:** pass (refused as expected)

### share status closed

```
+ goethe api share
```

```
The public gateway is closed. Open it with: goethe api share 5h --key <name> --allow <their-ip>/32
```

exit: `0`
**result:** pass

### ss :8444 must be absent

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
would place ggml-org/embeddinggemma-2-GGUF tp=1 on GPU 0 at ctx 1024: 0.9 GB per GPU (weights 0.3 + KV 0.0 + sliding-window KV 0.1 + overhead 0.5)
  GPU 0: 31.3 GB free, other software is running there
```

exit: `0`
**result:** pass

### load embedding

```
+ goethe load ggml-org/embeddinggemma-2-GGUF
```

```
GPU 0 has unmanaged process PID 2046 (Xorg, user gdm) using 0.0 GB VRAM
Goethe never stops processes it did not start. To stop it yourself: sudo kill 2046
Free VRAM on GPU 0: 31.3 GB; this load needs 0.9 GB.
Re-run with --yes to load alongside it.
not loading next to processes goethe did not start: re-run with --yes (no terminal to ask on)
```

exit: `3`
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
API: https://100.90.97.111:8443/v1
API KEY NAME  KEY ID
e2e-tailnet   a62afdf8be04

No models loaded. Pick one and load it:

	→ goethe models
	→ goethe load <model-id>

Platform: Not linked (goethe login)
```

exit: `0`
**result:** pass

### goethe run embed positional INPUT

```
+ goethe run --plain ggml-org/embeddinggemma-2-GGUF hello greengoblin e2e
```

```
GPU 0 has unmanaged process PID 2046 (Xorg, user gdm) using 0.0 GB VRAM
Goethe never stops processes it did not start. To stop it yourself: sudo kill 2046
Free VRAM on GPU 0: 31.3 GB; this load needs 0.9 GB.
Re-run with --yes to load alongside it.
not loading next to processes goethe did not start: re-run with --yes (no terminal to ask on)
```

exit: `3`
**result:** FAIL

### goethe run with -- separator (footgun)

```
+ goethe run ggml-org/embeddinggemma-2-GGUF -- embed hello
```

```
error: unexpected value 'hello' for '[MODEL]...' found; no more were expected

Usage: goethe run [OPTIONS] [MODEL]...

For more information, try '--help'.
```

exit: `2`
**result:** pass (bad argv rejected)

### curl /v1/models via 127.0.0.1 + cacert

```
+ curl -sS --max-time 20 --cacert /home/v/proj/petya/goethe-e2e-180/home/nginx/certs/server.crt https://127.0.0.1:8443/v1/models -H Authorization: Bearer goethe_sk_REDACTED
```

```
{"data":[],"object":"list"}
```

exit: `0`
**result:** pass

### curl /v1/models via tailnet IP + cacert

```
+ curl -sS --max-time 20 --cacert /home/v/proj/petya/goethe-e2e-180/home/nginx/certs/server.crt https://100.90.97.111:8443/v1/models -H Authorization: Bearer goethe_sk_REDACTED
```

```
curl: (60) SSL: no alternative certificate subject name matches target host name '100.90.97.111'
More details here: https://curl.se/docs/sslcerts.html

curl failed to verify the legitimacy of the server and therefore could not
establish a secure connection to it. To learn more about this situation and
how to fix it, please visit the web page mentioned above.
```

exit: `60`
**result:** FAIL (expected success)

### curl /v1/embeddings via tailnet

```
+ curl -sS --max-time 90 --cacert /home/v/proj/petya/goethe-e2e-180/home/nginx/certs/server.crt https://100.90.97.111:8443/v1/embeddings -H Authorization: Bearer goethe_sk_REDACTED -H Content-Type: application/json -d {"model":"ggml-org/embeddinggemma-2-GGUF","input":"greengoblin e2e"}
```

```
curl: (60) SSL: no alternative certificate subject name matches target host name '100.90.97.111'
More details here: https://curl.se/docs/sslcerts.html

curl failed to verify the legitimacy of the server and therefore could not
establish a secure connection to it. To learn more about this situation and
how to fix it, please visit the web page mentioned above.
```

exit: `60`
**result:** FAIL (expected success)

### curl embeddings without key
HTTP `000`
```
```
**result:** FAIL (expected 401/403, got 000)

## 7. Chat model (if linked)

> Llama NVFP4 tree not linked; skipped chat load.

## 8. More failure / UX probes

### load missing model

```
+ goethe load definitely-missing-model-xyz
```

```
model definitely-missing-model-xyz is not installed (goethe models lists installed models; goethe install definitely-missing-model-xyz adds it)
```

exit: `1`
**result:** pass (refused as expected)

### run missing model

```
+ goethe run --plain definitely-missing-model-xyz hi
```

```
definitely-missing-model-xyz is not installed: goethe install definitely-missing-model-xyz
```

exit: `1`
**result:** pass (refused as expected)

### load --dry-run missing

```
+ goethe load definitely-missing-model-xyz --dry-run
```

```
model definitely-missing-model-xyz is not installed (goethe models lists installed models; goethe install definitely-missing-model-xyz adds it)
```

exit: `1`

### share dangerously without remembered key

```
+ goethe api share 5m --dangerously-allow-all-inbound
```

```
name the API keys to allow with --key <name> (the next time they are remembered); list them with goethe api keys
```

exit: `1`
**result:** pass (refused as expected)

### gpu table

```
+ goethe gpu
```

```

  ▣ GPU  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
  gpu0   0.5 GB ░░░░░░░░░░ 31.8 GB   2046            ALLOWED   ░░░░░░░░░░   0%

  █ goethe models   █ other processes   ░ free   █ memory bandwidth
```

exit: `0`

### logs goethed (tail)

```
+ bash -c goethe logs goethed 2>&1 | tail -50
```

```
log not found: /home/v/proj/petya/goethe-e2e-180/home/logs/goethed.log
```

exit: `0`

## 8b. URL consistency

### goethe status (URLs)

```
+ goethe status
```

```
goethe: running
API: https://100.90.97.111:8443/v1
API KEY NAME  KEY ID
e2e-tailnet   a62afdf8be04

No models loaded. Pick one and load it:

	→ goethe models
	→ goethe load <model-id>

Platform: Not linked (goethe login)
```

exit: `0`

### goethe api (URLs)

```
+ goethe api
```

```
API endpoint: https://100.90.97.111:8443/v1
SSL_CERT_FILE=/home/v/proj/petya/goethe-e2e-180/home/nginx/certs/server.crt
Accepts: OpenAI, Anthropic (/v1/messages) and LiteLLM (routes with or without /v1, x-litellm-api-key)

API keys
NAME         API KEY                                     ADMIN  EXPIRES
e2e-tailnet  goethe_sk_REDACTED  no     never

Query the API:

	→ curl --cacert /home/v/proj/petya/goethe-e2e-180/home/nginx/certs/server.crt https://100.90.97.111:8443/v1/chat/completions -H "Authorization: Bearer <your_bearer_token>" -H "Content-Type: application/json" -d '{"model": "<model_name>", "messages": [{"role": "user", "content": "Hello!"}]}'
```

exit: `0`

### goethe api expose

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

## 9. Cleanup on box

### unload all

```
+ goethe unload --all
```

```
nothing was loaded
```

exit: `0`

### expose local

```
+ goethe api expose local --yes
```

```
Gateway now: listening on 100.90.97.111 (following tailscale0) port 8443; allowlist: 100.64.0.0/10, 127.0.0.1/32
gateway listens on 127.0.0.1 port 8443
API keys: 1 exist (goethe api keys; goethe api keys create <client> for each new client)

API: https://127.0.0.1:8443/v1

For the models and keys run:

	→ goethe api
```

exit: `0`

### share stop

```
+ goethe api share stop
```

```
The public gateway is not open.
```

exit: `0`

### goethe down (e2e)

```
+ goethe down
```

```
goethed stopped
goethed will not start at boot until the next goethe up
```

exit: `0`

### ss after e2e down

```
+ bash -c ss -ltn | grep -E "8443|8444" || echo none
```

```
none
```

exit: `0`

> Restoring production Goethe.

> Prod not on 0.0.0.0:8443

## Scoreboard

- Passes: **33**
- Failures / unexpected: **7**
- Full command log: `/home/v/proj/petya/goethe-e2e-180/session.log`
- Isolated home: `/home/v/proj/petya/goethe-e2e-180/home` — delete when done: `rm -rf /home/v/proj/petya/goethe-e2e-180`

## Findings (observed, not softened)

1. UX footgun: users typing HF id into models scan get os error 2; hint should say use folder name org__name or goethe install <path>
2. FAIL ok: models scan folder name --yes
3. api expose public --yes binds 0.0.0.0 without requiring a public client allow; only warns. Listener is world-reachable if NAT/firewall forwards; nginx allowlist may still drop non-listed clients.
4. FAIL ok: load embedding
5. goethe run --plain embed INPUT failed (exit 3)
6. UX: goethe run does not take -- extra args; clap error is opaque for users copying engine-flag patterns
7. FAIL ok: curl /v1/models via tailnet IP + cacert
8. FAIL ok: curl /v1/embeddings via tailnet
9. embeddings without bearer returned HTTP 000 (expected 401/403)

## CI / follow-up test ideas

1. Assert `goethe models scan org/name` fails with a hint to use `org__name` folder form.
2. Assert `goethe api expose public --yes` either refuses without a non-VPN client allow, or never binds `0.0.0.0` unless `--dangerously-allow-all-inbound` (product decision).
3. After any public probe in E2E, assert `ss` has no `0.0.0.0:8443` / `:8444`.
4. `goethe run --plain <model> <input>` happy path for embed + chat; reject teaching `run -- -p`.
5. Curl `/v1/models` and `/v1/embeddings` with cacert over tailnet IP; 401 without bearer.
6. Share: machine public IP as `--allow` must refuse; no `--allow` must refuse.
7. Script must not `set -e` across soft failures (v1 aborted mid-session).


## Findings (refined after follow-up — zero gaslight)

### P0 / product

1. **`goethe api expose public --yes` binds `0.0.0.0` without requiring a public client allow.** It keeps the existing VPN allowlist and only warns. Nginx may still drop non-allowlisted clients, but the socket is world-reachable if NAT/firewall forwards. Prod bar decision needed: refuse unless a non-VPN client CIDR is supplied, or require `--dangerously-allow-all-inbound` to bind `0.0.0.0`.

2. **Gateway TLS cert SANs omit the tailnet IP.** Observed SANs: `DNS:localhost`, `IP:127.0.0.1`, `DNS:greengoblin`. **No `100.90.97.111`.** `goethe api` tells users to `curl --cacert … https://100.90.97.111:8443/…` which **fails with curl 60** (name mismatch). Localhost with cacert works. Users must `-k` today — docs/UX lie by omission.

### P1 / real user footguns

3. **`goethe models scan <foldername>` resolves the argument from CWD via `canonicalize`, not from `GOETHE_HOME/models`.** Passing `ggml-org__embeddinggemma-2-GGUF` as the help text suggests yields `No such file or directory (os error 2)` unless that name exists in the current working directory. Code: `scan.rs` `candidate_dirs`. Fix: try `models.join(d)` before raw `d`.

4. **`goethe models scan` (no args) refuses a symlink under models/** with “register the folder it points to with `goethe install <realpath>`”. Symlinking an already-installed tree into a fresh `GOETHE_HOME` (isolated / multi-home) does not work via scan. **`goethe install <abs-path> --name …` works** (creates a symlink farm). Document this; optionally teach scan to follow+register like install.

5. **Non-interactive `goethe load` / `goethe run` refuse GPU 0 when Xorg/gdm holds 0.0 GB VRAM** unless `--yes`. Message is clear. Scripts/SSH/CI without a TTY always need `--yes` on boxes with a display manager. E2E must pass `--yes`.

6. **`goethe run` INPUT is positional; there is no `-p` / `--` engine-flag passthrough.** `goethe run model -- embed hi` → opaque clap error. Users copying install/`--` patterns get confused.

### P2 / isolation & ops

7. **Second `GOETHE_HOME` while systemd user unit owns another home:** `goethe up` warns and may pick **18443** (see `ports.rs` 18443–18499) instead of 8443. Heartbeat unit stays on prod home. Isolated E2E must `goethe down` prod **and** account for port shift, or disable the other unit. Follow-up accidentally curled 8443 while e2e listened on 18443.

8. **v1 E2E script aborted mid-run** because `run()` re-enabled `set -e` and a bare failing command exited the script before cleanup/scoreboard. Fixed in v2.

### What worked

- Isolated `GOETHE_HOME` + PR180 binary `0.3.3` build on greengoblin.
- `goethe up` / `status` / `doctor` / `doctor --hardware`.
- `models recommend` + `models` fit labels with per-GPU / ctx / concurrency.
- `goethe install <abs-path> --name ggml-org/embeddinggemma-2-GGUF --yes` registers local tree.
- Tailnet expose: `api expose vpn --interface tailscale0 --allow 100.64.0.0/10 --yes` → listen `100.90.97.111:8443` + `127.0.0.1`, allowlist CGNAT + localhost.
- Share without `--allow` refuses; machine public IP as `--allow` refuses; no `:8444`.
- After public probe, vpn revert cleared `0.0.0.0:8443`.
- **Follow-up:** `load --yes` + `run --plain --keep` embedding → `dims=768 norm=1.0000 […]` success.
- localhost `/v1/models` with bearer + cacert → JSON list (empty when unloaded).
- Missing model load/run refuse cleanly.
- Prod restored: `100.90.97.111:8443` + `127.0.0.1:8443`, not public; heartbeat OK; key `matteo` intact.

### Not covered this session

- Chat/Llama load: no Llama tree under prod `models/` (only embeddinggemma).
- True multi-client curl from another tailnet node (this runner used on-box curls).
- Public share happy path (intentionally out of scope — fail-closed only).
- Kernel-pack / ternary / ninfer paths.

### Box state after session

- Prod goethe **0.3.1** running, tailnet bind only.
- E2E home left at `/home/v/proj/petya/goethe-e2e-180` for inspection (safe to `rm -rf`).
- No e2e `goethed` left running.

## Incident closeout (public 0.0.0.0)

- **Now:** CLOSED. Prod listens only `100.90.97.111:8443` + `127.0.0.1:8443`. No `:8444`. No e2e daemon.
- **Attack evidence:** e2e nginx access log had only `127.0.0.1`. Prod Nov 18 access was only tailnet `100.x` (our follow-up curls, 401). No foreign public IPs.
- **Why not internet-wide open in practice:** (1) NAT: public IP `81.182.125.247` is not on any iface; inbound needs router forward. (2) nginx allowlist stayed `100.64.0.0/10` + `127.0.0.1` (deny all else). (3) API key still required. **Still wrong:** LAN `192.168.1.134:8443` could accept SYNs while bound to `0.0.0.0` (would then hit deny/401).
- **Windows:** v2 ~5s (`22:11:23` to `22:11:28`). v1 aborted without auto-revert until manual `goethe down` (~1 min class).
- **Fix shipped:** `expose public --yes` without `--allow` refuses; raw `expose 0.0.0.0` refuses; public needs confirm + named clients; `--dangerously-allow-all-inbound` hidden for world-open.
