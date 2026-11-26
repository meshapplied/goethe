"""A stand-in for `python -m goethe_engine`: the pack commands goethe calls, and `serve`."""

import hashlib
import json
import os
import sys
from pathlib import Path

from . import openai_server


def flag(args, name, default=None):
    return args[args.index(name) + 1] if name in args else default


def say(obj, code=0):
    print(obj if isinstance(obj, str) else json.dumps(obj), file=sys.stderr if code not in (0, 3) else sys.stdout)
    sys.exit(code)


def index_entries(args):
    """The entries of a verified index: the signature file must hold the sha256 of the index bytes."""
    raw = Path(flag(args, "--index")).read_bytes()
    if Path(flag(args, "--sig")).read_text().strip() != hashlib.sha256(raw).hexdigest():
        say("index signature does not verify", 2)
    index = json.loads(raw)
    if index["serial"] < int(flag(args, "--last-serial", "0")):
        say("index serial went back", 2)
    return index


def container(data):
    """(header, manifest bytes) of a pack or its prefix."""
    n = int.from_bytes(data[8:16], "little")
    header = json.loads(data[16 : 16 + n])
    off, length = header["manifest"]
    return header, data[16 + n + off : 16 + n + off + length]


def pack(args):
    cmd = args[0]
    if cmd == "match":
        index = index_entries(args)
        if "--model-dir" not in args:
            say({"ok": True})
        have = {p.name for p in Path(flag(args, "--model-dir")).iterdir()}
        for e in index["entries"]:
            if e["arch"] == flag(args, "--arch") and set(e["e2e_files"]) <= have:
                say({"file": e["file"], "variant": e["e2e_variant"], "model": e["model"]})
        say("no pack has this structure", 3)
    if cmd == "need":
        header, manifest = container(Path(args[1]).read_bytes())
        say({"manifest_sha256": hashlib.sha256(manifest).hexdigest(), "ranges": header["blocks"]})
    if cmd == "verify":
        data = Path(args[1]).read_bytes()
        header, manifest = container(data)
        for b in header["blocks"]:
            if hashlib.sha256(data[b["off"] : b["off"] + b["len"]]).hexdigest() != b["sha256"]:
                say("a block does not hash to its name", 2)
        meta = json.loads(manifest)
        weights = [w["file"] for w in meta["model"]["weights"]]
        missing = [w for w in weights if not (Path(flag(args, "--model-dir", ".")) / w).is_file()]
        if missing:
            say(f"missing weights {missing}", 2)
        say(
            {
                "ok": True,
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "arch": flag(args, "--arch"),
                "variant": next(iter(meta["variants"])),
                "weights": weights,
            }
        )
    if cmd == "runs":
        if os.environ.get("E2E_PACK_RUNS") == "no":
            say(
                "device NVIDIA GeForce RTX 5090 runs no variant of this pack: the variants are built for another SKU", 3
            )
        say("ok")
    if cmd == "plan":
        _, manifest = container(Path(args[1]).read_bytes())
        meta = json.loads(manifest)
        names = [w["file"] for w in meta["model"]["weights"]]
        refused = os.environ.get("E2E_PACK_RUNS") == "no"
        say(
            {
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "variants": [
                    {
                        "id": vid,
                        "role": "target",
                        "quant": v["desc"]["quant"],
                        "source": None,
                        "files": [{"name": names[i]} for i in v["weights"]],
                        "max_ctx": 8192,
                        "runs": not refused,
                        "why": "no SKU for this device" if refused else None,
                    }
                    for vid, v in meta["variants"].items()
                ],
            }
        )
    if cmd == "show":
        _, manifest = container(Path(args[1]).read_bytes())
        for name in json.loads(manifest)["variants"]:
            print(f"bucket {name} maxctx 8192")
        sys.exit(0)
    say(f"unknown pack command {cmd}", 2)


def main():
    args = sys.argv[1:]
    if args[:2] == ["index", "verify"]:
        say({"serial": index_entries(args)["serial"]})
    if args[:1] == ["pack"]:
        pack(args[1:])
    if args[:1] == ["serve"]:
        openai_server.serve(args[1:])
    say(f"unknown command {args}", 2)


main()
