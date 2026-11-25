# In-session memory prototype (local only)

The production version is `goethe learn` (`docs/learn.md`, `crates/core/src/learn.rs`). This folder is the research behind it.

Goal: keep what compaction drops in one long session, for any model and harness. Nothing crosses sessions.

Design (plan.md): a rolling note per session, updated every 20 requests or at 25% of context in unconsumed content, by tool-less curator calls. One call sees only user text (rules, corrections, facts; quotes verified); one sees the transcript (goal, decisions, mistakes). Spliced at the end of the system message on the first request after each compaction, frozen until the next. Max 3% of context. Consumed messages leave the buffer.

## Files
- `note_proxy.py`: the prototype, run between harness and goethed. `run_test.sh`, `chat_rules.json`, `facts.json`, `stamina.patch`: live test on StaminaBench with OpenCode (`score_note.py`, `score_rules.py`).
- `recall_offline.py`: recall on real sessions, needs no agent. `.work/traces` is the corpus (key `fec10900c742`: no memory).
- `configs/`, `ace-appworld.patch`: the ACE reproduction. `plan.md`: full design.

## Tried
| Attempt | Result (Qwen3.8-27B) |
|---|---|
| Lessons from earlier sessions, SWE-bench | About -8% calls (range includes 0) |
| Always-on playbook, StaminaBench | 2x work, no gain |
| Narrow always-on session memory | 6 extra missed rules; OpenCode kept the rules anyway |
| ACE, AppWorld, 120 tasks | 107 / 108 / 109 tasks; no clear gain, 2.4x+ prompt tokens |
| Paper playbooks on `test_challenge`, stopped early | No gain on the 27 tasks scored |
| Note spliced after compaction, 25 turns | Recall 129/149 vs 0/268; calls about -1% (range -28% to +25%); missed corrections in 3/6 |
| Final, with the user-only curator | Correction 15/15 live; recall 44/44 vs 0/44 offline (user-side note built from a whole session) |

## Open
- Task success did not change in any run. Facts were planted; real sessions untested. Two no-memory runs differ by +52% in calls, so cost needs about 12+ scenarios per condition.
- 2/2 live recall misses at short session length are unexplained. Stale recall needs a malicious-tool-output test.
- Agent curator can duplicate user rules as decisions; block it.

## Setup
Clone StaminaBench (amazon-science) into `.work/stamina`, apply the patch, `uv sync --extra dev`, build the OpenCode image. Create keys, `.work/stamina-<off|mem>.env` with `GOETHE_API_KEY` and `GOETHE_BASE_URL=http://172.16.0.1:<port>/v1`, and start one `note_proxy.py` per condition on those ports.

## Production tests (2026-11-07)
`replay.py` replays a real session through goethe. `.work/agent/agent.py` runs live OpenCode in a container (needs an image with opencode), `.work/agent/results.txt` holds the runs. With decisions and mistakes in the prompt, compactions doubled (16, 16 vs 8, 9) and reads rose, so they stay out now. Final: 11, 14 compactions; input still about 1.8x at a 30k limit. Untested at larger contexts.
