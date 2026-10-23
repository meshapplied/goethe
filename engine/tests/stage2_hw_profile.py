#!/usr/bin/env python3
"""Stage 2 hardware profile gate vs nvidia-smi ground truth."""

from __future__ import annotations

import json
import re
import subprocess

from lib.goethe_test import main_exit, require_cli, run_goethe, temp_home

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _strip_ansi(text: str) -> str:
    """Remove ANSI color and cursor control sequences."""
    return _ANSI.sub("", text)


def _gpu_index_token(token: str) -> int | None:
    """Parse GPU followed by digits only (ignore GPU NUMA and bare GPU)."""
    m = re.fullmatch(r"GPU(\d+)", _strip_ansi(token).strip())
    return int(m.group(1)) if m else None


def _smi_gpu_rows() -> list[dict[str, str | int]]:
    """Query nvidia-smi for per-GPU fields."""
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,compute_cap,memory.total,memory.used,memory.free,"
            "pcie.link.gen.current,pcie.link.width.current",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    rows: list[dict[str, str | int]] = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        rows.append(
            {
                "index": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "compute_cap": parts[3],
                "vram_total_mb": int(float(parts[4])),
                "vram_used_mb": int(float(parts[5])),
                "vram_free_mb": int(float(parts[6])),
                "pcie_gen": int(parts[7]),
                "pcie_width": int(parts[8]),
            }
        )
    return rows


def _smi_pcie_snapshot() -> dict[int, dict[str, int]]:
    """Sample current and max PCIe link fields per GPU index."""
    proc = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    out: dict[int, dict[str, int]] = {}
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        out[int(parts[0])] = {
            "pcie_gen": int(parts[1]),
            "pcie_width": int(parts[2]),
            "pcie_gen_max": int(parts[3]),
            "pcie_width_max": int(parts[4]),
        }
    return out


def _check_pcie_link(idx: int, goethe_gen: int, goethe_width: int, samples: list[dict[int, dict[str, int]]]) -> None:
    """Exact match when the link was stable across all samples; within capability when power states moved it."""
    gens = {s[idx]["pcie_gen"] for s in samples}
    widths = {s[idx]["pcie_width"] for s in samples}
    gen_max = max(s[idx]["pcie_gen_max"] for s in samples)
    width_max = max(s[idx]["pcie_width_max"] for s in samples)
    if len(gens) == 1 and goethe_gen not in gens:
        raise RuntimeError(f"gpu {idx} pcie_gen {goethe_gen} != stable smi value {gens.pop()}")
    if len(widths) == 1 and goethe_width not in widths:
        raise RuntimeError(f"gpu {idx} pcie_width {goethe_width} != stable smi value {widths.pop()}")
    if not 1 <= goethe_gen <= gen_max or not 1 <= goethe_width <= width_max:
        raise RuntimeError(f"gpu {idx} pcie {goethe_gen}x{goethe_width} outside 1..{gen_max} x 1..{width_max}")


def _topo_map() -> dict[tuple[int, int], str]:
    """Parse nvidia-smi topo -m into link types."""
    proc = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30)
    if proc.returncode != 0:
        return {}
    lines = [_strip_ansi(ln) for ln in proc.stdout.splitlines()]
    if len(lines) < 2:
        return {}
    header = lines[0].split()
    idxs = [gi for h in header if (gi := _gpu_index_token(h)) is not None]
    mapping: dict[tuple[int, int], str] = {}
    for line in lines[1:]:
        cols = line.split()
        if not cols:
            continue
        a = _gpu_index_token(cols[0])
        if a is None:
            continue
        for j, cell in enumerate(cols[1 : 1 + len(idxs)]):
            b = idxs[j]
            if a >= b:
                continue
            code = cell.strip()
            if code.startswith("NV"):
                mapping[(a, b)] = "NVLINK"
            elif code in {"PIX", "PXB", "PHB", "NODE", "SYS"}:
                mapping[(a, b)] = code
    return mapping


def main() -> None:
    """Run goethe hw profile into a temp home and compare hardware_state.json with nvidia-smi."""
    goethe = require_cli("goethe")
    with temp_home(stub_python=True, tag="hwprof") as home:
        samples = [_smi_pcie_snapshot() for _ in range(3)]
        run_goethe([str(goethe), "hw", "profile"], env={"GOETHE_HOME": str(home)})
        samples += [_smi_pcie_snapshot() for _ in range(3)]
        path = home / "hardware_state.json"
        if not path.is_file():
            raise RuntimeError("hardware_state.json missing")
        state = json.loads(path.read_text(encoding="utf-8"))
    smi = {int(r["index"]): r for r in _smi_gpu_rows()}
    reported = {int(g["index"]) for g in state.get("gpus", [])}
    if not smi or reported != set(smi):
        raise RuntimeError(f"profile reports GPUs {sorted(reported)}, nvidia-smi sees {sorted(smi)}")
    for gpu in state.get("gpus", []):
        idx = int(gpu["index"])
        ref = smi.get(idx)
        if ref is None:
            raise RuntimeError(f"smi missing gpu {idx}")
        if gpu["uuid"] != ref["uuid"]:
            raise RuntimeError("uuid mismatch")
        if gpu["name"] != ref["name"]:
            raise RuntimeError("name mismatch")
        if gpu["compute_capability"] != ref["compute_cap"]:
            raise RuntimeError("compute_capability mismatch")
        total = int(gpu["vram_total_mb"])
        ref_total = int(ref["vram_total_mb"])
        if abs(total - ref_total) / max(ref_total, 1) > 0.01:
            raise RuntimeError("vram_total_mb out of 1 percent")
        if any(idx not in snap for snap in samples):
            raise RuntimeError(f"pcie snapshot missing gpu {idx}")
        _check_pcie_link(idx, int(gpu["pcie_gen"]), int(gpu["pcie_width"]), samples)
    topo = _topo_map()
    for link in state.get("links", []):
        key = (int(link["a"]), int(link["b"]))
        if key in topo and link["type"] != topo[key]:
            raise RuntimeError(f"link {key} type {link['type']} != smi {topo[key]}")
    print("stage2_hw_profile ok")


if __name__ == "__main__":
    main_exit(main)
