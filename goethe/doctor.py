"""Read-only diagnosis. Every check says what it saw and, when it is not fine, how to fix it."""
from __future__ import annotations

import os
import platform
import re
import shutil
import socket
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from goethe import hw
from goethe.engine import Install
from goethe.registry import Model
from goethe.resolve import tier_for

MIN_DRIVER = 580  # CUDA 13 runtime
MIN_NVCC = (13, 1)
MIN_CMAKE = (3, 28)
MIN_FREE_RAM_GIB = 12  # the server pins 9.5 GiB of host context
BUILD_TREE_GB = 2
FFMPEG_LIBS = ("libavformat", "libavcodec", "libavutil", "libswscale")


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # ok, warn, fail, info
    detail: str
    fix: str = ""


class System:
    """Everything doctor reads from the machine; tests substitute a fake."""

    def run(self, cmd: list[str]) -> tuple[int, str]:
        return hw.run(cmd)

    def which(self, name: str) -> str | None:
        return shutil.which(name)

    def read(self, path: Path) -> str | None:
        try:
            return path.read_text()
        except OSError:
            return None

    def free_bytes(self, path: Path) -> int:
        while not path.exists():
            path = path.parent
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize

    def mem_available_kib(self) -> int | None:
        text = self.read(Path("/proc/meminfo")) or ""
        m = re.search(r"^MemAvailable:\s+(\d+)", text, re.M)
        return int(m[1]) if m else None

    def port_free(self, port: int) -> bool:
        with socket.socket() as s:
            return s.connect_ex(("127.0.0.1", port)) != 0

    def network_error(self, url: str) -> str | None:
        """None when the server answers (any HTTP status counts), else why it does not."""
        try:
            urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=10).close()
        except urllib.error.HTTPError:
            return None
        except OSError as e:
            return str(getattr(e, "reason", e))
        return None

    def direct_io(self, path: Path) -> bool:
        """Whether files under `path` open with O_DIRECT, which the engine uses to load weights."""
        while not path.exists():
            path = path.parent
        try:
            with tempfile.NamedTemporaryFile(dir=path) as f:
                os.close(os.open(f.name, os.O_RDONLY | getattr(os, "O_DIRECT", 0)))
        except OSError:
            return False
        return hasattr(os, "O_DIRECT")


def _version(text: str) -> tuple[int, ...] | None:
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text)
    return tuple(int(x) for x in m.groups() if x) if m else None


def _fmt(v: tuple[int, ...]) -> str:
    return ".".join(map(str, v))


def check_python() -> Check:
    v = sys.version_info[:3]
    if v >= (3, 11):
        return Check("python", "ok", _fmt(v))
    return Check("python", "fail", f"{_fmt(v)}, goethe needs 3.11 or newer", "install python 3.11 or newer")


def check_platform() -> Check:
    libc = " ".join(platform.libc_ver())
    if platform.system() == "Linux" and platform.machine() == "x86_64":
        return Check("platform", "ok", f"linux x86_64, {libc}")
    return Check("platform", "fail", f"{platform.system()} {platform.machine()}, NInfer runs on Linux x86_64 only")


def check_gpus(sys_: System, model: Model) -> tuple[list[Check], list[hw.Gpu]]:
    gpus, err = hw.probe(sys_.run)
    if not gpus:
        return [Check("nvidia-smi", "fail", err or "no GPU",
                      "install the NVIDIA driver (580 or newer) and check that nvidia-smi lists the card")], []
    out = []
    for g in gpus:
        t = tier_for(model, g)
        out.append(Check(f"gpu {g.index}", {"tested": "ok", "compatible": "warn", "unsupported": "fail"}[t.name],
                         f"{g.name}, {g.total_gib:.1f} GiB, sm_{g.compute_cap.replace('.', '')}; "
                         f"{model.title}: {t.name}, {t.reason}"))
    driver = _version(gpus[0].driver)
    if driver and driver[0] >= MIN_DRIVER:
        out.append(Check("driver", "ok", gpus[0].driver))
    else:
        out.append(Check("driver", "fail", f"{gpus[0].driver}, CUDA 13 needs {MIN_DRIVER} or newer",
                         "update the NVIDIA driver"))
    for g in gpus:
        if g.pcie_width and g.pcie_width_max and g.pcie_width < g.pcie_width_max:
            out.append(Check(f"pcie {g.index}", "warn", f"link width x{g.pcie_width} of x{g.pcie_width_max}",
                             "reseat the card or check the slot; loading and prefill suffer"))
    return out, gpus


def check_free_vram(sys_: System, gpus: list[hw.Gpu], model: Model) -> list[Check]:
    if not gpus:
        return []
    g = gpus[0]
    detail = f"{g.free_mib / 1024:.1f} GiB free of {g.total_gib:.1f}"
    if g.free_mib >= model.min_vram_gib * 1024 * 0.95:
        return [Check("free vram", "ok", detail)]
    who = ", ".join(f"{n} (pid {p}, {m} MiB)" for p, n, m in hw.compute_apps(sys_.run)) or "unknown processes"
    return [Check("free vram", "warn", f"{detail}, in use by {who}", "stop the other GPU programs before serving")]


def check_resources(sys_: System, inst: Install) -> list[Check]:
    out = []
    kib = sys_.mem_available_kib()
    if kib is not None:
        gib = kib / 1024 / 1024
        enough = gib >= MIN_FREE_RAM_GIB
        out.append(Check("free ram", "ok" if enough else "warn", f"{gib:.1f} GiB available",
                         "" if enough else f"the server pins about 9.5 GiB, keep {MIN_FREE_RAM_GIB} GiB free"))
    need_gb = (0 if inst.has_model else inst.model_bytes / 1e9 + 0.4) + (0 if inst.has_binary else BUILD_TREE_GB)
    free_gb = sys_.free_bytes(inst.home) / 1e9
    detail = f"{free_gb:.1f} GB free under {inst.home}" + (f", {need_gb:.1f} GB needed" if need_gb else "")
    enough = free_gb >= need_gb
    out.append(Check("disk", "ok" if enough else "fail", detail,
                     "" if enough else "free space or point NVFP4_HOME at a bigger disk"))
    models = inst.home / "models"
    if sys_.direct_io(models):
        out.append(Check("direct io", "ok", f"O_DIRECT works under {models}"))
    else:
        out.append(Check("direct io", "warn",
                         f"O_DIRECT fails under {models}, loading the weights will be slow or fail",
                         "keep the model on a local disk, not a network or FUSE mount"))
    return out


def check_nvcc(sys_: System) -> Check:
    nvcc = sys_.which("nvcc")
    if not nvcc:
        return Check("nvcc", "fail", "not found", "install CUDA 13.1 or newer and put nvcc on PATH (or set CUDA_HOME)")
    code, text = sys_.run([nvcc, "--version"])
    m = re.search(r"release (\d+)\.(\d+)", text)
    if code != 0 or not m:
        return Check("nvcc", "fail", "version unreadable")
    v = (int(m[1]), int(m[2]))
    if v < MIN_NVCC:
        return Check("nvcc", "fail", f"{_fmt(v)}, need {_fmt(MIN_NVCC)} or newer", "install a newer CUDA toolkit")
    header = sys_.read(Path(nvcc).resolve().parent.parent / "include" / "cuda_runtime_api.h") or ""
    h = re.search(r"#define CUDART_VERSION\s+(\d+)", header)
    if not h:
        return Check("nvcc", "warn", f"{_fmt(v)}, CUDA headers not found next to it")
    headers = (int(h[1]) // 1000, int(h[1]) % 1000 // 10)
    if headers != v:
        return Check("nvcc", "fail",
                     f"compiler {_fmt(v)} but CUDA headers {_fmt(headers)}; the build stops with a CCCL "
                     "'compiler and toolkit headers are incompatible' error",
                     "use a toolkit whose compiler and headers are the same version")
    return Check("nvcc", "ok", f"{_fmt(v)}, headers match")


def check_toolchain(sys_: System, inst: Install) -> list[Check]:
    """Needed only to build the engine from source, so informational once it is built."""
    if inst.has_binary:
        return [Check("build tools", "info", "not needed, the engine is already built")]
    out = []
    for tool, minimum in (("git", None), ("cmake", MIN_CMAKE), ("ninja", None), ("pkg-config", None),
                          ("curl", None), ("c++", None)):
        path = sys_.which(tool)
        if not path:
            out.append(Check(tool, "fail", "not found", f"install {tool}"))
        elif minimum:
            code, text = sys_.run([tool, "--version"])
            v = _version(text) if code == 0 else None
            ok = bool(v) and v[:2] >= minimum
            out.append(Check(tool, "ok" if ok else "fail", _fmt(v) if v else "version unreadable",
                             "" if ok else f"need {_fmt(minimum)} or newer"))
        else:
            out.append(Check(tool, "ok", path))
    out.append(check_nvcc(sys_))
    if sys_.which("pkg-config"):
        for what, libs in (("libcurl dev", ["libcurl"]), ("ffmpeg dev", list(FFMPEG_LIBS))):
            ok = sys_.run(["pkg-config", "--exists", *libs])[0] == 0
            out.append(Check(what, "ok" if ok else "fail", "found" if ok else "missing",
                             "" if ok else f"install the development packages for {', '.join(libs)}"))
    return out


def check_install(sys_: System, inst: Install, port: int) -> list[Check]:
    return [
        Check("engine", "ok" if inst.has_binary else "info",
              str(inst.binary) if inst.has_binary else "not built yet, goethe serve builds it"),
        Check("model file", "ok" if inst.has_model else "info",
              str(inst.model_file) if inst.has_model else "not downloaded yet, goethe serve fetches it"),
        Check("port", "ok", f"{port} is free") if sys_.port_free(port)
        else Check("port", "warn", f"{port} is already in use", "stop the other server or pass --port")]


def check_network(sys_: System, inst: Install, repo: str) -> Check:
    error = sys_.network_error(f"https://huggingface.co/{repo}")
    if error is None:
        return Check("network", "ok", "huggingface.co reachable")
    if inst.has_model:
        return Check("network", "info", f"huggingface.co not reachable ({error}), not needed, the model is present")
    return Check("network", "fail", f"huggingface.co not reachable ({error})", "check the connection or proxy settings")


def run_checks(model: Model, inst: Install, pin: dict[str, str], port: int = 8080,
               sys_: System | None = None) -> list[Check]:
    sys_ = sys_ or System()
    gpu_checks, gpus = check_gpus(sys_, model)
    return [check_python(), check_platform(), *gpu_checks, *check_free_vram(sys_, gpus, model),
            *check_resources(sys_, inst), *check_toolchain(sys_, inst), *check_install(sys_, inst, port),
            check_network(sys_, inst, pin["MODEL_REPO"])]


def as_dicts(checks: list[Check]) -> list[dict]:
    return [asdict(c) for c in checks]


def exit_code(checks: list[Check]) -> int:
    return 1 if any(c.status == "fail" for c in checks) else 0
