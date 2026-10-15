import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from goethe import doctor, registry
from goethe.engine import Install
from tests.fixtures import PRO_6000, RTX_3090, RTX_5090, gpu

MODEL = registry.load()["qwen3-8-27b"]
PIN = {"MODEL_REPO": "org/model"}
SIZE = 23_719_715_844
TOOLS = ("git", "cmake", "ninja", "pkg-config", "curl", "c++", "nvcc")


def smi(*gpus):
    return "\n".join(f"{g.index}, {g.name}, {g.total_mib}, {g.free_mib}, {g.driver}, {g.compute_cap}, "
                     f"{g.pcie_width}, {g.pcie_width_max}" for g in gpus)


class Fake(doctor.System):
    """A machine described by keyword arguments; defaults are a healthy 5090 box."""

    def __init__(self, gpus=(RTX_5090,), tools=TOOLS, nvcc="release 13.4", header="13040", cmake="cmake version 3.31.1",
                 pkg=True, free_gb=500.0, ram_gib=64.0, direct=True, port_open=True, net=True, apps="", smi_ok=True,
                 ldd="\tlibc.so.6 => /lib/libc.so.6\n"):
        self.gpus, self.tools, self.nvcc, self.header, self.cmake, self.pkg = gpus, tools, nvcc, header, cmake, pkg
        self.free_gb, self.ram_gib, self.direct, self.port_open = free_gb, ram_gib, direct, port_open
        self.net, self.apps, self.smi_ok, self.ldd = net, apps, smi_ok, ldd

    def run(self, cmd):
        if cmd[0] == "nvidia-smi":
            if not self.smi_ok:
                return 127, ""
            return 0, self.apps if "--query-compute-apps" in cmd[1] else smi(*self.gpus)
        if cmd[0].endswith("nvcc"):
            return 0, f"Cuda compilation tools, {self.nvcc}, V13.4.92"
        if cmd[0] == "cmake":
            return 0, self.cmake
        if cmd[0] == "pkg-config":
            return (0 if self.pkg else 1), ""
        if cmd[0] == "ldd":
            return 0, self.ldd
        return 1, ""

    def which(self, name):
        return f"/usr/bin/{name}" if name in self.tools else None

    def read(self, path):
        return f"#define CUDART_VERSION  {self.header}\n" if path.name == "cuda_runtime_api.h" and self.header else None

    def free_bytes(self, path):
        return int(self.free_gb * 1e9)

    def mem_available_kib(self):
        return int(self.ram_gib * 1024 * 1024)

    def port_free(self, port):
        return self.port_open

    def network_error(self, url):
        return None if self.net else "timed out"

    def direct_io(self, path):
        return self.direct


class DoctorTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)

    def install(self, binary=False, model=False):
        inst = Install(Path("/recipe"), self.home, self.home / "ninfer-serve", self.home / "model", SIZE)
        if binary:
            inst.binary.write_text("")
            inst.binary.chmod(inst.binary.stat().st_mode | stat.S_IXUSR)
        if model:
            with open(inst.model_file, "wb") as f:
                f.truncate(SIZE)
        return inst

    def run_checks(self, machine, **installed):
        return {c.name: c for c in doctor.run_checks(MODEL, self.install(**installed), PIN, 8080, machine)}


class SystemWhich(unittest.TestCase):
    def test_cuda_home_bin_is_searched_first(self):
        with tempfile.TemporaryDirectory() as cuda:
            nvcc = Path(cuda) / "bin" / "nvcc"
            nvcc.parent.mkdir()
            nvcc.write_text("")
            nvcc.chmod(0o755)
            with mock.patch.dict(os.environ, {"CUDA_HOME": cuda, "PATH": "/nonexistent"}):
                self.assertEqual(doctor.System().which("nvcc"), str(nvcc))
            with mock.patch.dict(os.environ, {"PATH": "/nonexistent"}, clear=True):
                self.assertIsNone(doctor.System().which("nvcc"))


class Machines(DoctorTest):
    def test_healthy_5090_with_nothing_installed_has_no_failures(self):
        checks = self.run_checks(Fake())
        self.assertEqual([n for n, c in checks.items() if c.status == "fail"], [])
        self.assertEqual(checks["gpu 0"].status, "ok")
        self.assertIn("tested", checks["gpu 0"].detail)
        self.assertEqual(doctor.exit_code(list(checks.values())), 0)

    def test_machine_without_a_gpu(self):
        checks = self.run_checks(Fake(smi_ok=False, tools=()))
        self.assertEqual(checks["nvidia-smi"].status, "fail")
        self.assertIn("580", checks["nvidia-smi"].fix)
        self.assertNotIn("driver", checks)
        self.assertNotIn("free vram", checks)
        for tool in ("git", "cmake", "ninja", "nvcc"):
            self.assertEqual(checks[tool].status, "fail", tool)
        self.assertEqual(doctor.exit_code(list(checks.values())), 1)

    def test_unsupported_and_untested_gpus(self):
        c = self.run_checks(Fake(gpus=(RTX_3090,)))["gpu 0"]
        self.assertEqual(c.status, "fail")
        self.assertIn("sm_120a", c.detail)
        self.assertEqual(self.run_checks(Fake(gpus=(PRO_6000,)))["gpu 0"].status, "warn")

    def test_old_driver(self):
        c = self.run_checks(Fake(gpus=(gpu(driver="570.10"),)))["driver"]
        self.assertEqual(c.status, "fail")
        self.assertIn("580", c.detail)

    def test_pcie_width_warning(self):
        self.assertEqual(self.run_checks(Fake(gpus=(gpu(width=4, width_max=16),)))["pcie 0"].status, "warn")
        self.assertNotIn("pcie 0", self.run_checks(Fake()))

    def test_busy_gpu_names_the_process(self):
        c = self.run_checks(Fake(gpus=(gpu(free_gib=6),), apps="999, python3, 25000"))["free vram"]
        self.assertEqual(c.status, "warn")
        self.assertIn("python3 (pid 999, 25000 MiB)", c.detail)
        self.assertEqual(self.run_checks(Fake())["free vram"].status, "ok")

    def test_disk_ram_and_direct_io(self):
        checks = self.run_checks(Fake(free_gb=3, ram_gib=4, direct=False))
        self.assertEqual((checks["disk"].status, checks["free ram"].status, checks["direct io"].status),
                         ("fail", "warn", "warn"))

    def test_disk_needs_nothing_once_installed(self):
        self.assertIn("needed", self.run_checks(Fake())["disk"].detail)
        c = self.run_checks(Fake(free_gb=0.5), binary=True, model=True)["disk"]
        self.assertEqual(c.status, "ok")
        self.assertNotIn("needed", c.detail)

    def test_toolchain_is_informational_once_built(self):
        checks = self.run_checks(Fake(tools=()), binary=True, model=True)
        self.assertEqual(checks["build tools"].status, "info")
        self.assertNotIn("nvcc", checks)
        self.assertEqual(checks["engine"].status, "ok")
        self.assertEqual(checks["libraries"].status, "ok")

    def test_missing_runtime_libraries_name_the_apt_line(self):
        ldd = "\tlibavcodec.so.60 => not found\n\tlibcurl.so.4 => not found\n\tlibcuda.so.1 => not found\n"
        c = self.run_checks(Fake(ldd=ldd), binary=True)["libraries"]
        self.assertEqual(c.status, "fail")
        self.assertEqual(c.fix, "apt install ffmpeg libcurl4; install the NVIDIA driver (580 or newer)")

    def test_nvcc_version_and_header_match(self):
        self.assertEqual(self.run_checks(Fake(nvcc="release 13.4"))["nvcc"].status, "ok")
        self.assertEqual(self.run_checks(Fake(nvcc="release 12.8"))["nvcc"].status, "fail")
        mismatch = self.run_checks(Fake(nvcc="release 13.4", header="13000"))["nvcc"]
        self.assertEqual(mismatch.status, "fail")
        self.assertIn("CCCL", mismatch.detail)
        self.assertEqual(self.run_checks(Fake(header=""))["nvcc"].status, "warn")

    def test_old_cmake_and_missing_dev_packages(self):
        self.assertEqual(self.run_checks(Fake(cmake="cmake version 3.22.1"))["cmake"].status, "fail")
        checks = self.run_checks(Fake(pkg=False))
        self.assertEqual((checks["libcurl dev"].status, checks["ffmpeg dev"].status), ("fail", "fail"))

    def test_port_and_network(self):
        self.assertEqual(self.run_checks(Fake(port_open=False))["port"].status, "warn")
        down = self.run_checks(Fake(net=False))["network"]
        self.assertEqual(down.status, "fail")
        self.assertIn("timed out", down.detail)
        self.assertEqual(self.run_checks(Fake(net=False), binary=True, model=True)["network"].status, "info")

    def test_json_shape(self):
        rows = doctor.as_dicts(doctor.run_checks(MODEL, self.install(), PIN, 8080, Fake()))
        self.assertTrue(all(set(r) == {"name", "status", "detail", "fix"} for r in rows))
        self.assertTrue(all(r["status"] in {"ok", "warn", "fail", "info"} for r in rows))


if __name__ == "__main__":
    unittest.main()
