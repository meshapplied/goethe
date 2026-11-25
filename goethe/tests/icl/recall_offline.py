"""Offline recall test: builds each note from a traced session's user messages, then asks about lost facts."""

from __future__ import annotations

import argparse
import importlib
import json
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "trace"))
np = importlib.import_module("note_proxy")
tr = importlib.import_module("trace_reader")


def main() -> None:
    """Replay one key's sessions from a trace folder, with and without the note."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", default=str(HERE / ".work" / "traces"))
    p.add_argument("--key-id", required=True)
    p.add_argument("--curate-key-file", required=True)
    p.add_argument("--model", default="meshapplied/Qwen3.8-27B-NVFP4-NInfer")
    p.add_argument("--upstream", default="http://127.0.0.1:30100")
    p.add_argument("--min-turns", type=int, default=100)
    args = p.parse_args()
    turns = tr.load_turns(Path(args.traces))
    facts = json.loads((HERE / "facts.json").read_text())
    tmp = tempfile.mkdtemp()
    a = argparse.Namespace(memory=True, upstream=args.upstream, model=args.model, facts="", ctx=100000, every=20,
                           probe_every=25, jobs=1, notes=tmp + "/n", log=tmp + "/l",
                           curate_key=Path(args.curate_key_file).read_text().strip())
    st = np.State(a)
    orig = st.chat
    res: Counter[str] = Counter()
    random.seed(0)
    sessions: dict[str, list[dict]] = {}
    for x in turns.values():
        if x.get("session_hint") and x["key_id"] == args.key_id and tr.params(turns, x).get("tools"):
            sessions.setdefault(x["session_hint"], []).append(x)

    def user_only(m: list[dict], t: int, *rest: object, **kw: object) -> str:
        return orig(m, t, *rest, **kw) if m[0]["content"].startswith("You maintain the part") else "NONE"

    for sid, turns_of in sessions.items():
        if len(turns_of) < args.min_turns:
            continue
        ordered = sorted(turns_of, key=lambda x: x["started_at"])
        msgs, seen = [], set()
        for x in ordered:
            for m in tr.history(turns, x)[1:]:
                if m.get("role") == "user" and np.digest(m) not in seen:
                    seen.add(np.digest(m))
                    msgs.append(m)
        s = np.Session()
        st.chat = user_only
        for i in range(0, len(msgs), 2):
            s.buffer, s.busy = msgs[i:i + 2], True
            st.consolidate("t", s)
        st.chat = orig
        block = st.render(s)
        late = ordered[len(ordered) // 2:]
        random.shuffle(late)
        asked = 0
        for x in late:
            h = tr.history(turns, x)
            current = "\n".join(np.text_of(m) for m in h)
            lost = [f for f in facts if all(v["marker"] not in current for v in f["versions"])]
            if not lost:
                continue
            for f in lost:
                ask = f"Quick check, answer in a few words, without any tool: {f['question']} Say unknown if not known."
                for label, b in (("off", ""), ("note", block)):
                    reply = st.chat([*np.splice({"messages": h}, b)["messages"], {"role": "user", "content": ask}], 60,
                                    tr.params(turns, x).get("tools"), "probe")
                    res[label + "_n"] += 1
                    res[label + "_ok"] += f["versions"][-1]["answer"].lower() in reply.lower()
            asked += 1
            if asked >= 2:
                break
        print(sid, "note chars", len(block), dict(res), flush=True)
    print("with note", res["note_ok"], "/", res["note_n"], " without", res["off_ok"], "/", res["off_n"])


if __name__ == "__main__":
    main()
