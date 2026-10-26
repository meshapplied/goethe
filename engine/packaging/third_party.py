#!/usr/bin/env python3
"""Write the generated part of NOTICE (every package, crate and license text) from the locks, a py-env and cargo."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent
REPO = ENGINE.parent
NOTICE = REPO / "NOTICE"
MARKER = "=" * 31 + " GENERATED " + "=" * 36
LOCK = ENGINE / "pyenv" / "requirements.lock"
TARGETS = ("x86_64-unknown-linux-gnu", "aarch64-unknown-linux-gnu")

NVIDIA = "NVIDIA proprietary"
ALIASES = {
    "mit license": "MIT",
    "apache software license": "Apache-2.0",
    "apache 2.0": "Apache-2.0",
    "apache license 2.0": "Apache-2.0",
    "apache 2.0 license": "Apache-2.0",
    "bsd license": "BSD",
    "3-clause bsd license": "BSD-3-Clause",
    "isc license (iscl)": "ISC",
    "python software foundation license": "PSF-2.0",
    "mozilla public license 2.0 (mpl 2.0)": "MPL-2.0",
    "gnu lesser general public license v3 (lgplv3)": "LGPL-3.0",
    "other/proprietary license": NVIDIA,
    "licenseref-nvidia-proprietary": NVIDIA,
    "nvidia proprietary software": NVIDIA,
    "public domain": "Public Domain",
}
UNDECLARED = {
    "cuda-toolkit": NVIDIA,
    "nvshmem4py-cu13": NVIDIA,
    "kernels-data": "Apache-2.0",
    "humming-kernels": "Apache-2.0",
    "openai-harmony": "Apache-2.0",
    "quack-kernels": "Apache-2.0",
    "sgl-deep-ep": "Apache-2.0",
    "torch-memory-saver": "MIT",
    "torchcodec": "BSD-3-Clause",
}

def run(cmd: list[str]) -> str:
    """Run a command in the engine workspace and return its stdout."""
    return subprocess.run(cmd, cwd=ENGINE, check=True, capture_output=True, text=True).stdout


def lock_packages() -> dict[str, str]:
    """Name to version for every package pinned in requirements.lock."""
    out = {}
    for line in LOCK.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)(?:==([^\s;\\]+)| @ \S+?-(\d[^-]*?)-cp)", line)
        if m:
            out[m.group(1).lower().replace("_", "-")] = (m.group(2) or m.group(3)).replace("%2B", "+")
    return out


def normalize(raw: str) -> str:
    """One license name per package from a License-Expression, classifiers or a License field."""
    parts = [ALIASES.get(p.strip().lower(), p.strip()) for p in raw.split(";") if p.strip()]
    return " OR ".join(dict.fromkeys(parts)) if parts else "UNKNOWN"


def python_packages(python: str) -> list[tuple[str, str, str]]:
    """(name, version, license) for every distribution installed in the py-env."""
    script = (
        "import importlib.metadata as md, json\n"
        "rows = []\n"
        "for d in md.distributions():\n"
        "    m = d.metadata\n"
        "    lic = (m.get('License-Expression') or '').strip()\n"
        "    if not lic:\n"
        "        lic = '; '.join(c.split('::')[-1].strip() for c in (m.get_all('Classifier') or [])"
        " if c.startswith('License ::'))\n"
        "    if not lic:\n"
        "        lic = ((m.get('License') or '').strip().splitlines() or [''])[0][:60]\n"
        "    rows.append(((m.get('Name') or '?').lower().replace('_', '-'), d.version, lic))\n"
        "print(json.dumps(rows))\n"
    )
    proc = subprocess.run([python, "-I", "-c", script], check=True, capture_output=True, text=True)
    rows = json.loads(proc.stdout)
    out = {}
    for name, version, lic in rows:
        version = re.sub(r"\.d\d{8}$", "", version)
        out[name] = (name, version, UNDECLARED.get(name) or normalize(lic))
    return sorted(out.values())


def rust_crates() -> list[tuple[str, str, str, Path]]:
    """(name, version, license, source dir) of every crate compiled into goethe and goethed on either target."""
    meta = json.loads(run(["cargo", "metadata", "--format-version", "1", "--locked"]))
    by_id = {(p["name"], p["version"]): p for p in meta["packages"] if p["source"]}
    used = set()
    for target in TARGETS:
        tree = run(["cargo", "tree", "--locked", "-e", "normal", "--prefix", "none", "-f", "{p}", "--target", target])
        for line in tree.splitlines():
            m = re.match(r"^(\S+) v(\S+)", line)
            if m and (m.group(1), m.group(2)) in by_id:
                used.add((m.group(1), m.group(2)))
    out = []
    for key in sorted(used):
        p = by_id[key]
        lic = p.get("license") or "see its license file"
        out.append((p["name"], p["version"], lic, Path(p["manifest_path"]).parent))
    return out


def license_texts(crates: list[tuple[str, str, str, Path]]) -> str:
    """Every crate's license files; repeated texts point back to their first appearance."""
    seen: dict[str, str] = {}
    blocks = []
    for name, version, lic, src in crates:
        names = ("LICEN", "COPYING", "UNLICENSE")
        files = sorted(f for f in src.iterdir() if f.is_file() and f.name.upper().startswith(names))
        head = f"\n{'-' * 78}\n{name} {version} ({lic})\n{'-' * 78}\n"
        if not files:
            blocks.append(head + f"The crate ships no license file; it is under {lic}, whose standard text applies.\n")
            continue
        for f in files:
            text = f.read_text(encoding="utf-8", errors="replace").strip()
            key = " ".join(text.split())
            if key in seen:
                blocks.append(head + f"{f.name}: the same text as {seen[key]} above.\n")
            else:
                seen[key] = f"{name} {version} ({f.name})"
                blocks.append(head + f"{f.name}:\n\n{text}\n")
    return "".join(blocks)


def section(title: str) -> list[str]:
    return ["", title, "-" * len(title), ""]


def generated(py: list[tuple[str, str, str]], crates: list[tuple[str, str, str, Path]]) -> str:
    """The generated part of NOTICE: every Python package and Rust crate, then the crates' license texts."""
    lines = [MARKER]
    lines += section(f"Python packages in Goethe's Python environment ({len(py)})")
    lines += [f"  {n:<40} {v:<28} {lic}" for n, v, lic in py]
    lines += section(f"Rust crates compiled into goethe and goethed ({len(crates)})")
    lines += [f"  {n:<40} {v:<28} {lic}" for n, v, lic, _ in crates]
    lines += section("License texts of the Rust crates")
    return "\n".join(lines) + "\n" + license_texts(crates)


def split_notice() -> tuple[str, str]:
    """(hand-written part ending with the marker line, generated part after it)."""
    text = NOTICE.read_text(encoding="utf-8")
    if MARKER not in text:
        sys.exit(f"{NOTICE} has no generated part; it must contain the line {MARKER!r}")
    head, tail = text.split(MARKER, 1)
    return head, tail


def listed_rows(tail: str, title: str) -> set[tuple[str, str]]:
    """(name, version) rows of one generated list."""
    body = tail.split(title, 1)[1].split("\n\n", 2)[1]
    return set(re.findall(r"^  (\S+) +(\S+) +\S", body, re.M))


def check() -> int:
    """Fail when NOTICE's lists no longer match Cargo.lock or requirements.lock."""
    _, tail = split_notice()
    listed_py = dict(listed_rows(tail, "Python packages in"))
    listed_rs = listed_rows(tail, "Rust crates compiled")
    problems = []
    for name, version in lock_packages().items():
        if listed_py.get(name) != version:
            problems.append(f"python {name} {version}: listed as {listed_py.get(name)}")
    want_rs = {(n, v) for n, v, _, _ in rust_crates()}
    if want_rs != listed_rs:
        problems.append(f"rust crates differ: {sorted(want_rs ^ listed_rs)[:10]}")
    for p in problems:
        print(p, file=sys.stderr)
    if problems:
        print("regenerate: python3 engine/packaging/third_party.py --python <py-env>/bin/python", file=sys.stderr)
        return 1
    print("NOTICE matches the locks")
    return 0


def main() -> int:
    """Regenerate NOTICE's generated part from a py-env, or check it against the locks."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", help="python of a py-env built from requirements.lock")
    ap.add_argument("--check", action="store_true", help="only check NOTICE against the locks")
    args = ap.parse_args()
    if args.check:
        return check()
    if not args.python:
        ap.error("--python is required to generate")
    head, _ = split_notice()
    NOTICE.write_text(head + generated(python_packages(args.python), rust_crates()), encoding="utf-8")
    print(f"wrote {NOTICE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
