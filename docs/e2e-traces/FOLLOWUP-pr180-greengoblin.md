# E2E follow-up (load --yes, cert SAN, scan detail)
- when: 2026-11-18T21:12:48Z

## Prior scan folder failure excerpt
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

--
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

## scan/install state
ggml-org__embeddinggemma-2-GGUF: No such file or directory (os error 2)
VRAM free: 31.8 GB

Installed
MODEL                                                         STATUS     DISK    FIT ON THIS MACHINE
ggml-org/embeddinggemma-2-GGUF (optimised kernels available)  installed  0.4 GB  fits: 0.9 GB per GPU (ctx 1024, 8 concurrent)
/home/v/proj/petya/goethe-e2e-180/home/models/ggml-org__embeddinggemma-2-GGUF is already inside /home/v/proj/petya/goethe-e2e-180/home/models
KEY_PRESENT=yes
Gateway now: listening on 127.0.0.1 port 18443; allowlist: 100.64.0.0/10, 127.0.0.1/32
The gateway will listen on 100.90.97.111 (tailscale0; follows its address) and on 127.0.0.1, and allow 100.64.0.0/10.
gateway listens on 100.90.97.111 (tailscale0; follows its address) port 18443
API keys: 1 exist (goethe api keys; goethe api keys create <client> for each new client)

API: https://100.90.97.111:18443/v1

For the models and keys run:

	→ goethe api

## cert SANs
subject=CN = goethe-local
X509v3 Subject Alternative Name: 
    DNS:localhost, IP Address:127.0.0.1, IP Address:0:0:0:0:0:0:0:1, DNS:greengoblin
## load --yes
auto: tp=1 on GPUs 0: 0.9 GB per GPU (weights 0.3 + KV 0.0 + sliding-window KV 0.1 + overhead 0.5)
Loading into VRAM...
loading into vram, 5 s
loading into vram, 10 s
warming up kernels with one request (the first time can take a few minutes)
RUNNING: ggml-org/embeddinggemma-2-GGUF instance inst-18e9401211c53dcc on GPU 0 port 30001 (12.1 s)
INSTANCE               MODEL                           RUNTIME        SLOT                                STATUS   GPUS  PORT   VRAM_GB  RESTARTS  IN_FLIGHT  TOK_PER_S  PREFIX
inst-18e9401211c53dcc  ggml-org/embeddinggemma-2-GGUF  goethe-engine  ggml-org/embeddinggemma-2-GGUF|0|0  RUNNING  0     30001  0.88     0         0          0.0        -
## run --plain embed
dims=768 norm=1.0000 [-0.0346, 0.0333, 0.0366, -0.0186, ...]
## curl localhost cacert
curl: (60) SSL certificate problem: self-signed certificate
More details here: https://curl.se/docs/sslcerts.html

curl failed to verify the legitimacy of the server and therefore could not
establish a secure connection to it. To learn more about this situation and
how to fix it, please visit the web page mentioned above.
## curl tailnet cacert (expect SAN fail)
curl: (60) SSL certificate problem: self-signed certificate
More details here: https://curl.se/docs/sslcerts.html

curl failed to verify the legitimacy of the server and therefore could not
establish a secure connection to it. To learn more about this situation and
how to fix it, please visit the web page mentioned above.
## curl tailnet -k (insecure name check bypass)
{"error":{"code":"invalid_api_key","message":"Invalid or missing Goethe API key","param":null,"type":"auth_error"}}## curl embeddings -k
{"error":{"code":"invalid_api_key","message":"Invalid or missing Goethe API key","param":null,"type":"auth_error"}}## curl embeddings no auth -k
HTTP 401
{"error":{"code":"invalid_api_key","message":"Invalid or missing Goethe API key","param":null,"type":"auth_error"}}## llama trees
ggml-org__embeddinggemma-2-GGUF
## restored prod
goethe: running

API: https://100.90.97.111:8443/v1

API KEY NAME  KEY ID
matteo        95f818903035

For more details run:

	→ goethe api

Platform: Linked (workspace Personal)
Heartbeat: OK (3 s ago)
LISTEN 0      511                                 127.0.0.1:8443       0.0.0.0:*          
LISTEN 0      511                             100.90.97.111:8443       0.0.0.0:*          
