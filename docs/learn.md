# In-session learning

A long agent session loses things when its client compacts the context: rules the user gave, corrections, facts mentioned in chat. With learning on, Goethe keeps a small note per session and puts it into the prompt after each compaction, so the session keeps them. It works with any model and needs no client change beyond sending a session id. It is off until you turn it on.

```bash
goethe api learn on                      # every model
goethe api learn off                     # every model, off
goethe api learn on --model my-model     # just this model (id or alias); repeat for more
goethe api learn off --model my-model    # this model only; with every model on, it becomes an exception
goethe api learn on --model all          # the same as goethe api learn on
goethe api learn                         # the current scope, and the sessions kept
```

Changes apply to the next request; no restart or reload.

Notes belong to one session. Nothing crosses sessions, API keys or restarts of a conversation under a new id.

## Sessions

A session is the client's `X-Session-Id` header (1 to 128 visible ASCII characters) under one API key. OpenCode sends it. A request without the header passes through unchanged, with nothing stored. Two API keys using the same id have separate notes. A session that moves to another model keeps its note, as long as learning is on for that model.

Routes: `/v1/chat/completions`, `/v1/messages` (Anthropic) and `/v1/responses`. Legacy `/v1/completions`, embeddings and the rest are not touched.

## Reading a note

```bash
goethe api learn --print my-session-id
goethe api learn --print my-session-id --json
```

This shows the live entries by section, the superseded ones, what is in the prompt right now and what the note has cost. `goethe api learn` lists the most recent session ids. Changes reach the file within 5 seconds.

## What is kept

| Section | Written from | In the prompt | Rule |
|---|---|---|---|
| `instruction` | User text | Yes | A standing rule, in the user's exact words: the entry is dropped unless the quote is found in a user message |
| `correction` | User text | Yes | The same, and the entry it replaces is superseded. Without a matching quote it is kept as a `finding` |
| `finding` | User text | Yes, the 20 newest | A fact from chat that is not in the project, such as a contact, a ticket or the reason for a decision |
| `goal` | The whole conversation | Yes | The goal and what comes next; one entry, replaced each time |
| `decision` | The whole conversation | No, `--print` only | A decision or rejected approach, with the reason |
| `mistake` | The whole conversation | No, `--print` only | What failed, the exact error and what worked |

Decisions and mistakes stay out of the prompt because they read as procedures: with them in, agents re-read more files and compacted about twice as often in our tests. Code, specs and anything else on disk is not recorded, since the agent can read it again. Tool results can never create an instruction, a correction or a finding: only user text blocks reach that update. An entry that looks like a credential (a private key, a known token prefix, a long random string) is dropped.

## When it runs

**Updates.** Every 20 requests, when waiting messages reach about a quarter of the context, after a compaction, and when a session has been quiet for 2 minutes and its model is idle. An update is a few tool-less model calls with thinking off, in the background: one per user message for the user sections (at most 8), and one for the rest. Each model runs one update at a time. They use a slot of the engine like any request, are not charged to an API key, are not traced, and do not keep a model from unloading. If an update fails, its messages stay and the next attempt waits longer, from 1 minute up to about 30.

**The prompt.** The note enters the prompt only after a compaction, at the end of the system prompt, and stays the same until the next one. Before the first compaction nothing is added. Everything else in the request keeps its bytes, so the prefix cache stays valid between compactions. If an update lands within 5 requests after a compaction, the note is refreshed once.

The note is at most 3% of the context length of the model's loaded instance. Past that, rules and corrections are kept first, then findings and the goal; the newest entries win within a section. `usage.prompt_tokens` includes the note, since clients use it to decide when to compact. If the engine answers 400 to a prompt with a note, Goethe sends the client's own request once more, and if that works, leaves the note out until the next compaction. An entry is dropped if it tries to imitate the note's labels. If compactions come very close together, the goal leaves the note until they space out. The user sections always stay.

## How requests are classified

Goethe looks only at the shape of a request, never at the wording of a particular client:

| Shape | Treated as |
|---|---|
| Starts with everything the session sent last time | A normal turn |
| No tools, in a session that has used tools | A side request such as a title or a summary: no note |
| A different system prompt, sharing nothing | A subagent: no note. After 20 in a row it is treated as the main thread, until the earlier thread comes back |
| Nothing new compared with the last request | A rewind or regeneration |
| Less than half of the earlier history kept, and shorter than before (at least 6 messages before) | A compaction. If the earlier history comes back on a later request, it was a subagent with the same system prompt: the compaction is undone |
| Anything else, such as old tool output replaced | A normal turn |
| Responses with `previous_response_id` | A normal turn; only the new input is read |

The first new message of a rewritten history is taken for the client's own summary: the user sections never read it. What follows it is read as usual.

## Files

```
learn/sessions/<hash of key and session id>.json
```

in the Goethe home. The folder is `0700` and files are `0600`. A file holds the note, its history and the messages still waiting for an update, clipped: up to 4,000 characters of a user message, 800 of the rest, and 1 MB in all, oldest out first. The text comes from your sessions, so treat it like traces. Files are saved every 5 seconds and when goethed stops. A session continues where it left off after a restart. Sessions idle for 30 days are deleted, and past 2,000 sessions the oldest go first. `goethe uninstall` deletes the folder; `rm` on a file forgets that session once it is no longer active.

With [traces](traces.md) on, a turn stores the messages as the client sent them, and `learn_note` holds the note the engine also received.

## Limits

- Learning helps recall; it has not been shown to improve task success. Replaying two real recorded OpenCode sessions through Goethe found all 63 compactions with no failed request. Asked about facts gone from the context, the model answered 18 of 24 times with the note and 0 of 12 without. Every fact was in the note; the misses were the model answering "unknown" anyway.
- It costs something when the context is tight, so `goethe api learn on` warns about any model with a context under 100k tokens; learning still turns on. On one coding task with OpenCode at a 30k context limit, two runs each: compactions went from 8 and 9 to 11 and 14, uncached input from about 285k to 520k tokens, output from about 38k to 48k. The note's own updates were about 1% of that. Contexts of 100k and above have not been measured this way. In those runs the agent still knew the facts at the end in 1 of 2 runs without a note, from OpenCode's own summary, and in 2 of 2 with one.
- The note is written by the served model, so its quality depends on that model.
- A client that sends no session id gets nothing.
- Only OpenCode's compaction has been tested against a real client.
- A subagent that shares the session id and the main system prompt gets the note on its first request, until the main thread returns.
- The goal, decision and mistake sections are written from the whole conversation, including tool output. The prompt labels them as the agent's own notes rather than orders, and the curator is told to treat that text as data, but a hostile tool result could still shape them. Leave learning off for agents that read untrusted content if that matters to you.
