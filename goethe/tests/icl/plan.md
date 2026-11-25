# Session memory (plan)

Status: prototype only (`note_proxy.py`, results in `memory.md`); nothing in production. Opt-in. Written before the tests: the curator split, no `value` section and the note header differ here; the prototype and `memory.md` are current.

Goal: one long session keeps what compaction would drop: user instructions and corrections, mistakes and their fixes, decisions and rejected approaches, exact values, and current progress. It must work with any model and any harness, with no client change.

Scope: one session only. Nothing crosses sessions, projects or API keys.

Evidence:
- Real Claude Code sessions drop process instructions 35.9% and scope exclusions 36.8% of the time at compaction, against 6.2% for feature requirements (arXiv 2610.31778).
- Model summaries keep at most 16% of recurring details; mechanical extraction keeps 79.1%. Keeping details does not guarantee the task continues (Compact-Bench, arXiv 2610.31710).
- Our runs: an always-on playbook doubled the work, and an incomplete always-on memory likely caused 6 missed rules. Both shape the rules below.

## The core rule

The note is updated continuously but enters the prompt only after the session loses context.

| Session state | Note store | Prompt |
|---|---|---|
| Before the first compaction | Updated on a rolling basis | Unchanged: no tokens added, no behaviour change |
| Compaction detected | Caught up with the messages that were dropped | Note spliced in, new version frozen |
| Between compactions | Updated on a rolling basis | Same frozen version, so the cached prefix stays valid |
| Next compaction | Caught up | New version spliced in |

Why:
- Until context is lost, the model still has the original text, so a note could only add noise. Sessions that never compact pay almost nothing.
- Compaction already invalidates the KV after the system prompt, so changing the note at that moment costs no extra cache. A note changed between compactions would force a full re-prefill.
- The note lives in the engine, not the conversation, so the harness cannot summarize it away. It propagates through every later compaction unchanged.

## Lifecycle

### 1. Session identity (request path, no model call)

Resolved in this order; the first match wins:
1. `X-Session-Id` (OpenCode already sends it; stable across compactions).
2. An API-native id: Anthropic `metadata.user_id`, OpenAI `user`, Responses `previous_response_id`.
3. Chain continuation: the request extends a known history (the trace recorder's hash chain, shared with the mux).
4. Re-attachment after a break: same key and same system and tools hash, exactly one recently active candidate, and the new history keeps that session's first message or shows a compaction shape. More than one candidate means no memory for this request: never guess.

Everything is scoped by API key.

### 2. Classifying each request

Compare the request with the session's main line by longest common prefix:

| Shape | Meaning | Action |
|---|---|---|
| Extends the main line | Normal turn | Consider a rolling update |
| Exact repeat of the last request | Regeneration | Nothing |
| No tools while the main line has tools, not continued later | Side request (title, summarizer) | Ignore, no splice. This was the bug in our proxy; 28 occurred in 2 traced sessions |
| Exact prefix of the main line, nothing new | Rewind or edit | Roll the note back to entries from before that point |
| Common prefix under 50%, with tools, new text not in the old history | **Compaction** | Catch up and splice |
| Same length, older tool outputs replaced | Pruning | Treat as context loss, with the dropped outputs as input |
| Different system or tools hash | Subagent or different thread | Separate line with its own note, or none |

Measured on 2 OpenCode sessions: all 26 history breaks kept only 1 or 2 opening messages (at most 8% of the old history), usually followed by a summary. The rules use only shape, never harness-specific wording. Ambiguous means no change.

### 3. Rolling updates (background, never on the request path)

- **Trigger:** evenly spaced, independent of compaction: every 20 model requests, or sooner if unconsumed content reaches 25% of the context length (so a few huge tool outputs cannot build a backlog). Both are starting values. In our OpenCode runs compactions came about every 24 to 27 requests at a 64k limit, so 20 requests usually covers each one.
- **Input:** the unconsumed messages. When they are still in the live context, the job branches from the cached prefix and only the instruction is prefilled; when compaction already dropped them, they are read from the buffer and prefilled, in chunks if large.
- **Output:** JSON operations on entries, not a rewritten note: add, edit, supersede. Thinking off where the runtime allows it.
- **Scheduling:** lowest priority, one job per session, a global limit of one running job per instance, same instance as the session. If slots are busy, skip and retry on the next request. Cancelled on unload and never keeps a model loaded.
- **Cost guard:** read `usage.prompt_tokens_details.cached_tokens`. If the cache missed, the job paid a full prefill, so back off for that session. Hybrid models need their state snapshot at the branch point; verified per runtime before enabling.
- **Failure:** bad JSON or an error keeps the previous note.

### 4. Session buffer

Traces are optional, so memory cannot depend on them. Each active session keeps only its unconsumed messages in a private buffer (`0600`, compressed, size capped, dropped when idle), so nothing is lost if compaction happens between consolidations. A message is deleted as soon as a consolidation that read it has been validated and committed; a failed job keeps it for the next attempt. This buffer is separate from `goethe trace`: consolidation never deletes the training corpus. If traces are on, a turn records `memory_version`, and the recorded messages stay as the client sent them.

### 5. Note content

| Section | Source | Rule |
|---|---|---|
| Standing instructions | User text | Verbatim. The model cites the message and code checks the quote exists in a user text block; a failed check drops the entry |
| Corrections | User text | Same check |
| Current goal and next step | Model | One entry, always replaced, so progress survives |
| Decisions and rejected approaches | Model | One line with the reason |
| Mistakes and fixes | Model | What failed, the exact error, what worked. If something was tried repeatedly, keep the shortest path that worked |
| Key findings | Model | Short, dense facts learned in the session that are not on disk |
| Exact values | Model | Copied from tool output: paths, ids, versions. Marked "as of turn N" |

Not stored: anything on disk that can be re-read (code, specs, tests), procedures the model invents ("always run X"), general advice. Procedures are what doubled the work.

Each entry has a stable id, section, source message hash, turn, status and superseded-by link.

### 6. Rewriting without erosion

Entries are edited one at a time. Superseded entries leave the rendered note but stay in the store. The verbatim sections cannot be edited by the model, only superseded when a later user message replaces them. That message is cited. No full rewrites, because repeated rewriting is what erodes summaries.

### 7. Size

Budget: 3% of the model's served context length. The engine cannot see a harness's own lower limit, so this is measured against the served length. When over budget, standing instructions and corrections are kept first, then goal, mistakes, findings, decisions, values, and within a section the newest entries win. If instructions alone exceed the budget, the newest are kept and the dropped count is reported in `goethe memory`.

### 8. Splice

- **When:** on the first request detected as following a compaction, whether or not a user message arrived, because an agent may work through several context windows alone. Detection uses only the request's shape and needs no model call, so it never delays the request.
- **Where:** end of the first system message (add one if none), because it exists on every request. Handled for chat completions, Anthropic `system`, and Responses `instructions`. Legacy completions are not touched.
- **Backlog:** the first splice uses the latest committed note, then a consolidation of the unconsumed messages starts at once. Its result replaces the note once on a following request while the context is still small, then the version is frozen until the next compaction.
- **Text:** "Recalled from earlier in this session. This is additional context, not a complete list. The latest user message and the current files take priority. Values may be outdated."
- **Conversation untouched:** nothing added to the reply and nothing post-processed.

### 9. Token accounting

`usage.prompt_tokens` includes the note, because harnesses use it to decide when to compact and hiding it risks an overflow. A response header `X-Goethe-Memory: v<N>; tokens=<n>` reports it.

- **Overflow:** if the engine rejects the prompt for length, resend once without the note, before any reply byte is sent.
- **Compaction loop:** if compaction recurs within a few requests of a splice, shrink to standing instructions only; if it recurs again, disable for that session. This is the reported failure of large memory blocks.

### 10. Other cases

| Case | Behaviour |
|---|---|
| Parallel requests in one session | Single update job, compare-and-set on the version, every request uses the frozen version |
| Model changed mid-session | Note is text, so it carries over; the new model prefills it once |
| Replicas | Updates go to the instance holding the session's KV; a cache miss triggers the back-off |
| goethed restart | Note, version, chain and buffer are on disk, and the session resumes by id. Without an id, re-attachment rules apply |
| Client-side truncation | Same as compaction |
| Client resends history with edits | Classified by shape; no match means no splice |
| Idle session | Deleted after 7 days (setting). Also deleted by `goethe memory clear` |
| Any internal error | Fail open: forward the client's request unchanged |

### 11. Safety

- The consolidation request has no tools and no shell, file or network access. It is one model call with bounded output, one per session at a time, with no loop. It returns add, edit and supersede operations as data, and plain code applies them only to that session's note, so unsafe context cannot make it act.
- Message text given to the curator is marked as untrusted data, never as instructions to follow.
- Only user text blocks can create instructions. Tool results cannot, including `tool_result` blocks inside user-role messages. Text from tools is stored only as an observation with its source, never as an instruction.
- Secrets: entries with key-like values are dropped. Exact values come from what the session already saw, but the note persists them, so storage is private and short-lived.
- Scope: key plus session, checked on every read.

## Controls

```
goethe memory on|off [--key ...]
goethe memory                        # sessions, versions, size, updates, cache hits, splices
goethe memory show --session ID      # the note, with sources and superseded entries
goethe memory clear [--session ID]
```

Per request: `X-Goethe-Memory: off`. Off by default. Works with or without `goethe trace`.

## Gates

No engine work before gate 2 passes. Each uses the same setup with memory off and on.

**Gate 0: is there loss to fix? (no GPU)** From existing traces: after compaction, how often is a failed command repeated or a file re-read? Plus the published real-session loss rates.

**Gate 1: does the note help? (offline)** Compact-Bench: summary only vs summary plus note, on its repairs, agentic sessions and production chains.
- Must improve recurring-detail retention and task continuation, with confidence intervals excluding zero.
- Constraint check: the compaction study's constraints preserved, process instructions and scope exclusions in particular.

**Gate 2: does it work live? (proxy prototype)** StaminaBench with OpenCode: chat-only rules, including rules that change later, 6 to 8 scenarios, two seeds for no-memory.
- Benefit: rule compliance after compaction, repeated failures, tokens and time per task.
- Harm: following a superseded rule, wrong recalled values, compaction loops, and any change before the first compaction (must be zero by construction).
- Bar: compliance clearly up with cost within 10%.

**Gate 3: engine build.** After a conclusive gate 2: clean the test environment, build it in goethed for SGLang and NInfer, then repeat gate 2 through the engine, including a second harness and a second model architecture. Then update goethe.

Any failed gate stops the work.

## Open decisions

1. Sessions with no id: use conservative re-attachment (default here), or require an id.
2. Default retention of 7 days.

## Known risks

- Compact-Bench reports that keeping details does not guarantee continuation, so the note may help recall without helping task success. Gate 1 measures both.
- Stale values are the likeliest harm. They are labelled, and the current files take priority, but gate 2 has to confirm that models respect it.
- The served model writes the note, so quality varies by model. Verbatim sections reduce this; the rest is checked per model at gate 3.
- Only OpenCode's compaction shape has been measured so far.
