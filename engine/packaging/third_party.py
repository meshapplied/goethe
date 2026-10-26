#!/usr/bin/env python3
"""Write THIRD_PARTY_NOTICES.md and THIRD_PARTY_LICENSES.txt from the locks, a py-env and the cargo registry."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent
REPO = ENGINE.parent
NOTICES = REPO / "THIRD_PARTY_NOTICES.md"
LICENSES = REPO / "THIRD_PARTY_LICENSES.txt"
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

MAIN = [
    ("SGLang", "sglang", "The serving engine. Goethe ships a modified build (see below).", "Apache-2.0",
     "https://github.com/sgl-project/sglang"),
    ("PyTorch", "torch", "Tensor library the engine runs on.", "BSD-3-Clause", "https://github.com/pytorch/pytorch"),
    ("FlashInfer", "flashinfer-python", "Attention and sampling kernels.", "Apache-2.0",
     "https://github.com/flashinfer-ai/flashinfer"),
    ("Triton", "triton", "GPU kernel compiler.", "MIT", "https://github.com/triton-lang/triton"),
    ("Transformers", "transformers", "Model configurations and tokenizers.", "Apache-2.0",
     "https://github.com/huggingface/transformers"),
    ("Hugging Face Hub", "huggingface-hub", "Model downloads.", "Apache-2.0",
     "https://github.com/huggingface/huggingface_hub"),
    ("XGrammar", "xgrammar", "Structured output.", "Apache-2.0", "https://github.com/mlc-ai/xgrammar"),
    ("NVIDIA CUDA libraries", "nvidia-cuda-runtime",
     "CUDA runtime, nvcc, cuBLAS, cuDNN, NCCL, NVSHMEM, CUTLASS DSL and related packages from PyPI.", NVIDIA,
     "https://docs.nvidia.com/cuda/eula/"),
    ("uv", "uv", "Builds the Python environment (shipped in bin/uv).", "Apache-2.0 OR MIT",
     "https://github.com/astral-sh/uv"),
    ("CPython", "cpython", "The Python interpreter uv installs (python-build-standalone).", "PSF-2.0",
     "https://github.com/astral-sh/python-build-standalone"),
    ("nginx", None, "The HTTPS gateway. Installed from your distribution's packages, not shipped by Goethe.",
     "BSD-2-Clause", "https://nginx.org/LICENSE"),
    ("SQLite", None, "State database, compiled into goethe and goethed.", "Public Domain",
     "https://sqlite.org/copyright.html"),
]


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


def python_version(python: str) -> str:
    return subprocess.run([python, "-I", "-c", "import platform; print(platform.python_version())"], check=True,
                          capture_output=True, text=True).stdout.strip()


def uv_version() -> str:
    m = re.search(r'^UV_VERSION="([^"]+)"', (ENGINE / "packaging" / "build_cdn.sh").read_text(encoding="utf-8"), re.M)
    return m.group(1) if m else ""


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
    blocks = [
        "Licenses of the Rust crates compiled into goethe and goethed.\n"
        "Generated by engine/packaging/third_party.py from Cargo.lock; see THIRD_PARTY_NOTICES.md for the rest.\n"
    ]
    for name, version, lic, src in crates:
        names = ("LICEN", "COPYING", "UNLICENSE")
        files = sorted(f for f in src.iterdir() if f.is_file() and f.name.upper().startswith(names))
        head = f"\n{'=' * 78}\n{name} {version} ({lic})\n{'=' * 78}\n"
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


def notices(py: list[tuple[str, str, str]], crates: list[tuple[str, str, str, Path]], cpython: str) -> str:
    """The human-readable notice: main components, special cases, then every package and crate."""
    versions = {n: v for n, v, _ in py} | {"uv": uv_version(), "cpython": cpython}
    lines = [
        "# Third-party software",
        "",
        "Goethe is licensed under the [Apache License 2.0](LICENSE). It is built on open-source and NVIDIA "
        "software, "
        "each under its own license, listed below. The installer places this file and "
        "[THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt) in `$GOETHE_HOME`.",
        "",
        "Generated by `engine/packaging/third_party.py`; regenerate it when `Cargo.lock` or `requirements.lock` "
        "changes.",
        "",
        "## Main components",
        "",
        "| Component | Version | What Goethe uses it for | License |",
        "|---|---|---|---|",
    ]
    for title, pkg, use, lic, url in MAIN:
        version = versions.get(pkg, "") if pkg else ""
        lines.append(f"| [{title}]({url}) | {version} | {use} | {lic} |")
    lines += [
        "",
        "## Notes",
        "",
        "- **SGLang is modified.** Goethe ships a build of SGLang with its own patches (NVFP4 and ternary model "
        "support, "
        "serving fixes). The changes are Goethe's; SGLang's copyright and Apache-2.0 license apply to the rest, and "
        "SGLang's license file ships in the wheel's `.dist-info` directory.",
        "- **NVIDIA software** (the `nvidia-*`, `cuda-*` and NVSHMEM packages) is proprietary. Goethe installs the "
        "packages NVIDIA publishes on PyPI, unmodified, under NVIDIA's terms: the "
        "[CUDA EULA](https://docs.nvidia.com/cuda/eula/) and the license files in each package's `.dist-info` "
        "directory.",
        "- **CUDA forward compatibility:** on NVIDIA drivers 535 to 579, data-center GPUs use NVIDIA's CUDA forward "
        "compatibility libraries, unmodified, under NVIDIA's license; they come from NVIDIA and are not redistributed "
        "by Goethe.",
        "- **LGPL:** `pycountry` (LGPL-2.1) and `easydict` (LGPL-3.0) are used unmodified as separate Python packages; "
        "you can replace them in the py-env, and their source is on PyPI.",
        "- **MPL-2.0:** `certifi`, `tqdm` and `orjson` are used unmodified; their source is on PyPI.",
        "- **Python packages** keep their own license files in `$GOETHE_HOME/py-env/lib/python3.11/site-packages/"
        "<package>.dist-info/`.",
        "- **Rust crates** compiled into `goethe` and `goethed` have their license texts in "
        "[THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt).",
        "",
        f"## Python packages in the py-env ({len(py)})",
        "",
        "| Package | Version | License |",
        "|---|---|---|",
    ]
    lines += [f"| {n} | {v} | {lic} |" for n, v, lic in py]
    lines += [
        "",
        f"## Rust crates in goethe and goethed ({len(crates)})",
        "",
        "| Crate | Version | License |",
        "|---|---|---|",
    ]
    lines += [f"| {n} | {v} | {lic} |" for n, v, lic, _ in crates]
    return "\n".join(lines) + "\n"


def table_rows(text: str, title: str) -> set[tuple[str, str]]:
    """(name, version) rows of the table under a heading, without its header row."""
    body = text.split(title, 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| (\S+) \| (\S+) \| .+ \|$", body, re.M)
    return {r for r in rows if r[1] != "Version" and not r[1].startswith("---")}


def check() -> int:
    """Fail when the committed lists no longer match Cargo.lock or requirements.lock."""
    text = NOTICES.read_text(encoding="utf-8")
    listed_py = dict(table_rows(text, "## Python packages"))
    listed_rs = table_rows(text, "## Rust crates")
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
    print("third-party notices match the locks")
    return 0


def main() -> int:
    """Generate both files from a py-env, or check the committed ones against the locks."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", help="python of a py-env built from requirements.lock")
    ap.add_argument("--check", action="store_true", help="only check the committed files against the locks")
    args = ap.parse_args()
    if args.check:
        return check()
    if not args.python:
        ap.error("--python is required to generate")
    crates = rust_crates()
    NOTICES.write_text(notices(python_packages(args.python), crates, python_version(args.python)), encoding="utf-8")
    LICENSES.write_text(license_texts(crates), encoding="utf-8")
    print(f"wrote {NOTICES} and {LICENSES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
