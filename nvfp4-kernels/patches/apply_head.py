#!/usr/bin/env python3
"""Swap 677 rows of the draft-token shortlist (`--lm-head-draft`) in the model file, or put the original back with --revert.

The original shortlist never proposes <tool_call> and </tool_call>, so every tool call costs rejected drafts. head_swap.json lists the
rows to drop and the tokens to add, head_rows.dat holds the new rows. Nothing is written unless the result matches the hashes in
NINFER_PIN; the original bytes are kept in <model>.head-orig. Standard library only.

usage: apply_head.py MODEL [--revert]
"""
import array, hashlib, json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
model, revert = sys.argv[1], "--revert" in sys.argv[2:]
pin = dict(l.strip().split("=", 1) for l in open(os.path.join(here, "..", "NINFER_PIN")) if "=" in l and l[0] != "#")
HEAD_AT, HEAD_N, IDS_AT, IDS_N = (int(pin[k]) for k in ("HEAD_OFFSET", "HEAD_BYTES", "IDS_OFFSET", "IDS_BYTES"))
ROWS, BASE, SCALE, SCALE_AT = 131072, 2560, 160, 335544320   # q4_g64 rows: 2560 code bytes and 160 scale bytes per row
sha = lambda b: hashlib.sha256(b).hexdigest()
orig = (pin["HEAD_ORIG_SHA256"], pin["IDS_ORIG_SHA256"])
new = (pin["HEAD_PATCHED_SHA256"], pin["IDS_PATCHED_SHA256"])
backup = model + ".head-orig"

f = open(model, "r+b")
def read(at, n):
    f.seek(at)
    return f.read(n)
def write(head, ids):
    f.seek(HEAD_AT); f.write(head)
    f.seek(IDS_AT); f.write(ids)
    f.flush(); os.fsync(f.fileno())

head, ids = read(HEAD_AT, HEAD_N), read(IDS_AT, IDS_N)
now = (sha(head), sha(ids))
if revert:
    if now == orig:
        sys.exit(print("head is original"))
    raw = open(backup, "rb").read()
    write(raw[:HEAD_N], raw[HEAD_N:])
    assert (sha(read(HEAD_AT, HEAD_N)), sha(read(IDS_AT, IDS_N))) == orig
    sys.exit(print("original head restored"))
if now == new:
    sys.exit(print("head already patched"))
if now != orig:
    sys.exit("error: this is not the expected model file, not touching it")

swap = json.load(open(os.path.join(here, "head_swap.json")))
drop, add = set(swap["drop"]), swap["add"]
old_ids = array.array("i"); old_ids.frombytes(ids)
keep = [r for r in range(ROWS) if old_ids[r] not in drop]
fresh = open(os.path.join(here, "head_rows.dat"), "rb").read()
n = len(add)
assert len(keep) + n == ROWS and len(fresh) == n * (BASE + SCALE)
codes = b"".join(head[r * BASE:(r + 1) * BASE] for r in keep) + fresh[:n * BASE]
scales = b"".join(head[SCALE_AT + r * SCALE:SCALE_AT + (r + 1) * SCALE] for r in keep) + fresh[n * BASE:]
new_ids = array.array("i", [old_ids[r] for r in keep] + add)
new_head, new_ids = codes + scales, new_ids.tobytes()
if (sha(new_head), sha(new_ids)) != new:
    sys.exit("error: rebuilt head does not match the expected hashes, nothing written")

if not os.path.exists(backup):
    open(backup, "wb").write(head + ids)
write(new_head, new_ids)
assert (sha(read(HEAD_AT, HEAD_N)), sha(read(IDS_AT, IDS_N))) == new
print(f"head patched, original saved to {backup}")
