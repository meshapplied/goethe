"""NVIDIA GPU probe through nvidia-smi. No NVML bindings, standard library only."""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass

Runner = Callable[[list[str]], tuple[int, str]]

QUERY = ("index,name,memory.total,memory.free,driver_version,compute_cap,power.limit,"
         "pcie.link.width.current,pcie.link.width.max")


def run(cmd: list[str], timeout: float = 15) -> tuple[int, str]:
    """(returncode, stdout); 127 when the program is missing, 124 on timeout."""
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, ""
    return done.returncode, done.stdout


@dataclass(frozen=True)
class Gpu:
    index: int
    name: str
    total_mib: int
    free_mib: int
    driver: str
    compute_cap: str
    power_limit_w: float | None
    pcie_width: int | None
    pcie_width_max: int | None

    @property
    def total_gib(self) -> float:
        return self.total_mib / 1024


def _num(text: str, cast: Callable[[str], float]) -> float | None:
    try:
        return cast(text)
    except ValueError:
        return None  # "[N/A]" and friends


def parse(csv: str) -> list[Gpu]:
    gpus = []
    for line in csv.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) != 9:
            raise ValueError(f"unexpected nvidia-smi line: {line!r}")
        gpus.append(Gpu(
            index=int(f[0]), name=f[1], total_mib=int(f[2]), free_mib=int(f[3]), driver=f[4],
            compute_cap=f[5], power_limit_w=_num(f[6], float),
            pcie_width=_num(f[7], int), pcie_width_max=_num(f[8], int)))
    return gpus


def probe(runner: Runner = run) -> tuple[list[Gpu], str | None]:
    """All GPUs, or ([], reason) when none can be read."""
    code, out = runner(["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"])
    if code == 127:
        return [], "nvidia-smi not found"
    if code != 0:
        return [], f"nvidia-smi failed (exit {code})"
    try:
        gpus = parse(out)
    except ValueError as e:
        return [], str(e)
    return (gpus, None) if gpus else ([], "nvidia-smi reports no GPU")


def compute_apps(runner: Runner = run) -> list[tuple[int, str, int]]:
    """(pid, process name, used MiB) of compute processes currently on any GPU."""
    code, out = runner(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                        "--format=csv,noheader,nounits"])
    apps = []
    for line in out.strip().splitlines() if code == 0 else []:
        pid, rest = line.split(",", 1)
        name, used = rest.rsplit(",", 1)  # process names may contain commas
        apps.append((int(pid), name.strip(), _num(used.strip(), int) or 0))
    return apps
