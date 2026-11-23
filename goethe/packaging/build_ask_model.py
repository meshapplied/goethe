#!/usr/bin/env python3
"""Build ask-model-v1.bin (GASK v1, per-row int8) from the pinned minishlab/potion-base-8M."""

import argparse
import hashlib
import json
import math
import struct
import sys
import urllib.request
from pathlib import Path

MODEL_REPO = "minishlab/potion-base-8M"
MODEL_REV = "bf8b056651a2c21b8d2565580b8569da283cab23"
PINNED_SHA256 = {
    "model.safetensors": "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2",
    "tokenizer.json": "e67e803f624fb4d67dea1c730d06e1067e1b14d830e2c2202569e3ef0f70bb50",
}
MAGIC = b"GASK"
VERSION = 1
DOWNLOAD_TIMEOUT_S = 120


def fetch(name: str, cache: Path) -> Path:
    """The pinned file from the cache, or downloaded beside it and kept only once its SHA-256 matches."""
    dest = cache / name
    want = PINNED_SHA256[name]
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == want:
        return dest
    cache.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    url = f"https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REV}/{name}"
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as resp:
        part.write_bytes(resp.read())
    got = hashlib.sha256(part.read_bytes()).hexdigest()
    if got != want:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"sha256 mismatch for {name}: got {got}, want {want}")
    part.replace(dest)
    return dest


def load_safetensors_tensor(path: Path) -> tuple[list[int], bytes, str]:
    """The first 2D tensor (the embeddings) of a safetensors file, read without torch."""
    raw = path.read_bytes()
    header_len = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_len].decode("utf-8"))
    body = raw[8 + header_len :]
    for name, meta in header.items():
        if name != "__metadata__" and len(meta["shape"]) == 2:
            lo, hi = meta["data_offsets"]
            return meta["shape"], body[lo:hi], meta["dtype"]
    raise RuntimeError(f"no 2D embedding tensor found in {path}")


def extract_vocab(tokenizer_path: Path, vocab_size: int) -> tuple[list[str], int]:
    """The WordPiece vocabulary in id order, and the id of the unknown token."""
    tok = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model_vocab = tok.get("model", {}).get("vocab", {})
    if isinstance(model_vocab, dict):
        tokens = [k for k, _ in sorted(model_vocab.items(), key=lambda kv: kv[1])]
    else:
        tokens = [entry[0] if isinstance(entry, list) else str(entry) for entry in model_vocab]
    tokens = (tokens + [""] * vocab_size)[:vocab_size]
    unk = tok.get("model", {}).get("unk_token", "[UNK]")
    return tokens, tokens.index(unk) if unk in tokens else 0


def quantize_per_row_int8(rows: bytes, shape: list[int], dtype: str) -> tuple[bytes, bytes, float]:
    """One f32 scale and int8 values per row; refused if a sample of rows loses cosine similarity."""
    if dtype not in ("F32", "F16"):
        raise RuntimeError(f"unsupported safetensors dtype {dtype}")
    vocab_size, dim = shape
    fmt = "<" + ("e" if dtype == "F16" else "f") * dim
    scales_buf = bytearray()
    q_buf = bytearray()
    step = max(1, vocab_size // 1024)
    cos_sum = 0.0
    cos_n = 0
    for idx, vec in enumerate(struct.iter_unpack(fmt, rows)):
        mx = max(abs(x) for x in vec)
        scale = mx / 127.0 if mx > 0.0 else 1.0
        scales_buf.extend(struct.pack("<f", scale))
        qs = [max(-127, min(127, round(x / scale))) for x in vec]
        q_buf.extend(struct.pack(f"<{dim}b", *qs))
        if idx % step == 0 and mx > 0.0:
            deq = [q * scale for q in qs]
            norms = math.sqrt(sum(a * a for a in vec)) * math.sqrt(sum(b * b for b in deq))
            if norms > 1e-9:
                cos_sum += sum(a * b for a, b in zip(vec, deq, strict=True)) / norms
                cos_n += 1
    cos = cos_sum / max(1, cos_n)
    if cos < 0.9995:
        raise RuntimeError(f"per-row int8 cosine parity too low: {cos:.6f}")
    return bytes(scales_buf), bytes(q_buf), cos


def write_gask(out_path: Path, tokens: list[str], scales: bytes, q: bytes, dim: int, unk_id: int) -> str:
    """Write the GASK v1 file (no adapter) and return its SHA-256."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<4sIIIII", MAGIC, VERSION, len(tokens), dim, unk_id, 0))
        for t in tokens:
            tb = t.encode("utf-8")
            f.write(struct.pack("<H", len(tb)))
            f.write(tb)
        f.write(scales)
        f.write(q)
    return hashlib.sha256(out_path.read_bytes()).hexdigest()


def main() -> int:
    """Fetch the pinned model, quantize it and write ask-model-v1.bin."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=Path(".cache/ask-model"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    shape, raw_bytes, dtype = load_safetensors_tensor(fetch("model.safetensors", args.cache))
    tokens, unk_id = extract_vocab(fetch("tokenizer.json", args.cache), shape[0])
    scales, q, cos = quantize_per_row_int8(raw_bytes, shape, dtype)
    sha = write_gask(args.out, tokens, scales, q, shape[1], unk_id)
    mb = args.out.stat().st_size / (1024 * 1024)
    print(f"wrote {args.out} ({mb:.2f} MiB, vocab={shape[0]}, dim={shape[1]}, cos={cos:.6f}, sha256={sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
