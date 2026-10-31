# Traces

Goethe can record every generation request and its reply as a corpus for training later. Recording is off until you turn it on. Nothing is scored or filtered while it records: deciding which turns go into a training run is a later, separate step (a labeler) that reads this folder.

```bash
goethe trace on                      # every API key
goethe trace on --key my-app         # only this key (name or id); repeat for more
goethe trace off --key ci            # every key but this one, when all keys are on
goethe trace off                     # stop recording
goethe trace                         # what is recorded, folder, size, free disk
goethe trace on --dir /data/traces --max-gb 200
goethe trace --max-gb 100            # change the size limit only
```

Changes apply to the next request; no restart or reload. Recorded routes: `/v1/chat/completions`, `/v1/messages` (Anthropic), `/v1/completions` and `/v1/responses`. Embeddings, rerank and token counts are not recorded.

Traces hold whole prompts and replies, including anything secret that clients send. The folder is `0700` and every file `0600`. Request headers and client keys are never stored, only the key id. When the folder reaches the size limit (`--max-gb`, default 20 GB), recording turns off, as if you had run `goethe trace off`. `goethe trace` says why and when, and `history.jsonl` logs it with the scope that was on. `goethe trace on` refuses while the folder is over the limit: raise it in the same command (`goethe trace on --max-gb 200`) or delete traces first. When the disk has less than 10 GB free, recording pauses instead, and resumes by itself once space is freed. A full write queue drops traces instead of slowing replies. Every drop is counted in the manifest. `goethe uninstall` keeps a trace folder that lives in the Goethe home. `goethe remove --model` asks whether to delete that model's traces too (default no; `--delete-traces` or `--keep-traces` answers without asking). Deleting a model's `v1/<model>` folder by hand works too: goethed drops its lines from `sessions.jsonl`, `models.json` and the manifest on its next write.

## Folder

```
traces/
  manifest.json      schema version, totals (turns per model too), drops, pause reason; rewritten every 2 s by goethed
  models.json        every model version that produced a turn, by fingerprint
  sessions.jsonl     one line per session when it starts: id, start, model, key id, file
  history.jsonl      every goethe trace on/off or setting change, and the automatic off at the size limit
  v1/<model>/<YYYY-MM-DD>/<start>_<session_id>.jsonl.gz
```

A session file holds one JSON object per turn, one per line. Each turn is appended as its own gzip member, so `zcat`, Python's `gzip` and any multi-member gzip reader read the whole file. `v1` is the schema version. A future format goes next to it, not over it.

## Sessions

The API is stateless, so Goethe groups turns into sessions itself:

- A request whose messages start with an earlier request plus the reply Goethe recorded for it continues that session. This is how agent loops and chats resend history. Matching compares role, visible text and tool calls (arguments parsed, keys sorted); ids, reasoning and whitespace around text are ignored, as clients often drop or rewrite them.
- A request that repeats an earlier request exactly (a regeneration) joins that session too. Both replies are kept.
- An `X-Session-Id` header (1 to 128 visible ASCII characters) groups requests explicitly, even unrelated ones. Matching still runs inside that session.
- Matching happens only within one API key and one model, and only against turns from the last 24 hours. A conversation that switches model starts a new session in that model's folder. After goethed restarts, a continuing conversation starts a new session that carries its full history.

## Turn record

| Field | Meaning |
|---|---|
| `schema` | `1` |
| `turn_id`, `session_id`, `seq` | Stable ids; `seq` counts turns within the session from 0 |
| `started_at`, `duration_s`, `ttft_s` | Request start (UTC), time to the last byte, time to the first output token (streams only) |
| `api` | `chat.completions`, `messages`, `completions` or `responses` |
| `model`, `model_fingerprint` | Served model id, and the `models.json` entry that produced the turn |
| `instance_id`, `key_id`, `request_id` | Engine instance, API key id, `X-Request-Id` |
| `session_hint` | The client's `X-Session-Id`, when sent |
| `parent` | `{"turn_id", "with_reply"}` or `null`; chat and messages only |
| `messages_from` | How many history messages come from the parent |
| `messages` | Only the messages that are new since the parent, as the client sent them |
| `params` | Every request field except `messages` (tools, system prompt, sampling settings) |
| `params_from_parent` | `true` instead of `params` when they equal the parent's |
| `request_hash`, `reply_hash` | Hashes of the canonical history before and after this turn's reply; equal `request_hash` means alternative replies to the same request (same key and model) |
| `response.status`, `response.stream` | HTTP status, and whether the client streamed |
| `response.complete` | `false` when the stream was cut or the client disconnected |
| `response.truncated` | `true` when the reply passed 32 MB and the rest was not kept |
| `response.body` | The reply as the non-streaming API returns it; streams are rebuilt into that shape: content, `reasoning_content`, tool calls with whole arguments, finish reason, usage. For Anthropic it is the message object with its content blocks |

Only complete replies with status 200 get a `reply_hash`, so only they can be continued by a later turn.

## Rebuilding a turn

The full message list of a turn is: the parent's full list, then the parent's reply if `with_reply` is true, then this turn's `messages`. Its length before the new messages equals `messages_from`. The parent's reply is the model's own output (with its reasoning and tool-call ids), not the client's resent copy; the two match in role, text and tool calls. Request parameters follow `params_from_parent` up to the first turn that has `params`. For `completions` and `responses`, `params` holds the whole request and there is no parent.

`engine/tests/trace/trace_reader.py` is a reference reader: it loads a folder, rebuilds any turn's history and params, and groups alternative replies.

## Model versions

`models.json` maps each fingerprint to what decides a model's output: the model id, sha256 of `config.json`, `generation_config.json`, the tokenizer files, the chat template and the safetensors index, the weight file names and sizes, and the extra engine flags. A changed checkpoint, chat template or engine flag gets a new fingerprint, so a labeler can keep turns from different model versions apart.

## For a labeler

Everything a later labeler needs is stored as raw facts, not as judgements:

- **Format errors:** the tool schemas (`params.tools`) next to the tool calls in `response.body`.
- **Tool outcomes:** what the tool returned, in the next turn's `messages`.
- **Bad replies:** truncation (`finish_reason`, `complete`), and regenerations (equal `request_hash`).
- **Model versions:** `model_fingerprint`.

A labeler should write its decisions to its own files, keyed by `turn_id` and its own version, and leave this folder unchanged.
