#!/usr/bin/env python3
"""Swap 677 rows of the draft-token shortlist (`--lm-head-draft`) in the model file, or put the original back with --revert.

The original shortlist never proposes <tool_call> and </tool_call>, so every tool call costs rejected drafts. head_swap.json
lists the rows to drop and the tokens to add. Kept rows are untouched; added rows are quantized from the model's own FP8
output head with NInfer's converter code. Sizes do not change. The result is checked against the hashes in NINFER_PIN before
anything is written; the original bytes go to <model>.head-orig.

usage: apply_head.py MODEL [--revert] [--ninfer DIR]     (DIR: NInfer checkout, default $NVFP4_HOME/ninfer)
"""
import argparse, hashlib, json, os, sys

here = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("--revert", action="store_true")
ap.add_argument("--ninfer", default=os.path.join(os.environ.get("NVFP4_HOME", os.path.expanduser("~/.cache/goethe/nvfp4")), "ninfer"))
a = ap.parse_args()

pin = dict(l.strip().split("=", 1) for l in open(os.path.join(here, "..", "NINFER_PIN")) if "=" in l and l[0] != "#")
sys.path.insert(0, a.ninfer)
try:
    import numpy as np
    import torch
    from tools.artifact.reader import Artifact
    from tools.artifact.codecs.fp8_row import decode_fp8_row_scaled_words
    from tools.artifact.codecs.row_split import RowPlanes, assemble_row_planes, encode_row_split, gather_row_planes, split_row_planes
    from tools.artifact.layouts import row_split_geometry
    from tools.convert.quantization.groupwise import quantize_matrix
except ImportError as e:
    sys.exit(f"error: {e}\nneeds python3 with torch and numpy, and the NInfer checkout at {a.ninfer} (run install.sh). NVFP4_HEAD=0 skips this step.")

HEAD, IDS, SRC, FMT, K, ROWS = "weight/001070", "weight/001071", "weight/000243", "q4_g64_fp16", 5120, 131072
sha = lambda b: hashlib.sha256(b).hexdigest()
art = Artifact(a.model)
obj = {"head": art.object(HEAD), "ids": art.object(IDS)}
backup = a.model + ".head-orig"
orig = {"head": pin["HEAD_ORIG_SHA256"], "ids": pin["IDS_ORIG_SHA256"]}
new = {"head": pin["HEAD_PATCHED_SHA256"], "ids": pin["IDS_PATCHED_SHA256"]}
state = lambda: {k: sha(art.read_object(o.id)) for k, o in obj.items()}


def write(data):
    fd = os.open(a.model, os.O_RDWR)
    for k, o in obj.items():
        assert len(data[k]) == o.bytes
        os.pwrite(fd, data[k], art.payload_offset + o.offset)
    os.fsync(fd)
    os.close(fd)


now = state()
if a.revert:
    if now == orig:
        sys.exit(print("head is original"))
    raw = open(backup, "rb").read()
    write({"head": raw[:obj["head"].bytes], "ids": raw[obj["head"].bytes:]})
    assert state() == orig
    sys.exit(print("original head restored"))
if now == new:
    sys.exit(print("head already patched"))
if now != orig:
    sys.exit("error: head objects are neither the original nor the patched ones, not touching the file")

old_head, old_ids = art.read_object(HEAD), np.frombuffer(art.read_object(IDS), dtype="<i4").astype(np.int64)
swap = json.load(open(os.path.join(here, "head_swap.json")))
drop, add = set(swap["drop"]), swap["add"]
keep = [r for r in range(ROWS) if int(old_ids[r]) not in drop]
assert len(keep) + len(add) == ROWS

codes, scales = decode_fp8_row_scaled_words(art.read_object(SRC), (248320, K))
i = torch.tensor(add)
q = quantize_matrix(codes[i].view(torch.float8_e4m3fn).float() * scales[i].float().unsqueeze(1), FMT, device="cpu")
fresh = split_row_planes(encode_row_split(q.codes, q.scales, FMT, (len(add), K)), row_split_geometry(FMT, (len(add), K)))
kept = gather_row_planes(old_head, row_split_geometry(FMT, (ROWS, K)), keep)
cat = lambda x, y: bytes(x) + bytes(y)
head = assemble_row_planes(RowPlanes(cat(kept.base, fresh.base), cat(kept.high, fresh.high), cat(kept.scale, fresh.scale), ROWS), FMT, K)
ids = np.concatenate([old_ids[keep], np.array(add)]).astype("<i4").tobytes()
if {"head": sha(head), "ids": sha(ids)} != new:
    sys.exit("error: rebuilt head does not match the expected hashes, nothing written")

if not os.path.exists(backup):
    open(backup, "wb").write(old_head + art.read_object(IDS))
write({"head": head, "ids": ids})
assert state() == new
print(f"head patched, original saved to {backup}")
