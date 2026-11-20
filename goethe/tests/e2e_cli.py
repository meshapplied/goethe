#!/usr/bin/env python3
"""The real goethe and goethed binaries as a pristine user with no GPU; the world around them is faked."""

from __future__ import annotations

import re
import stat
import subprocess
import time
import unittest
from contextlib import contextmanager

from e2e.world import Cell, Result, World

BAD = re.compile(r"Traceback|panicked at|No such file or directory|os error")
INSTALL_LINES = 12
SILENCE = 10.0


def clean(test: unittest.TestCase, r: Result, ok: bool = True) -> None:
    """The exit code asked for, nothing a user should never see, and no step silent for longer than SILENCE seconds."""
    test.assertEqual(r.code == 0, ok, f"exit {r.code}: {r.out}{r.err}")
    test.assertIsNone(BAD.search(r.out + r.err), f"{r.out}{r.err}")
    test.assertLess(r.gap, SILENCE, f"silent for {r.gap:.1f} s: {r.out}{r.err}")


class FreshUser(unittest.TestCase):
    world: World

    @classmethod
    def setUpClass(cls) -> None:
        cls.world = World().start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.world.stop()

    @contextmanager
    def pristine(self, **kw):
        """An empty home with no engine layer, no key file, no login and a fake RTX 5090."""
        c = Cell(self.world, warm=False, up=True, **kw)
        try:
            yield c
        finally:
            c.close()

    def downloaded(self, repo: str) -> bool:
        return any(f"/{repo}/resolve/" in r for r in self.world.requests)

    def journey(self, c: Cell, install: list[str], model: str) -> None:
        """up (the cell) -> login -> install --load -> ps shows goethe-engine -> run -> remove, each in few lines."""
        r = c.run("login")
        clean(self, r)
        r = c.run("install", *install, "--yes", "--load")
        clean(self, r)
        self.assertLessEqual(r.lines, INSTALL_LINES, r.out + r.err)
        self.assertRegex(r.out, r"engine +installing \d+\.\d+\.\d+", "the engine layer was created by this install")
        ps = c.run("ps")
        clean(self, ps)
        self.assertRegex(next(line for line in ps.out.splitlines() if model in line), r"\bgoethe-engine\b")
        r = c.run("run", model, "say hi")
        clean(self, r)
        self.assertIn(f"hello from {model}", r.out)
        r = c.run("remove", model, "--yes")
        clean(self, r)
        self.assertNotIn(model, c.run("ps").out)

    def test_a_pack_model_by_hub_id_from_nothing_to_run_and_remove(self) -> None:
        with self.pristine() as c:
            self.journey(c, ["e2e/packed"], "e2e/packed")

    def test_a_local_folder_by_the_same_path(self) -> None:
        with self.pristine() as c:
            folder = c.home / "mine"
            folder.mkdir()
            for name, data in self.world.repos["e2e/packed"].files.items():
                (folder / name).write_bytes(data)
            self.journey(c, [str(folder), "--name", "local/mine"], "local/mine")

    def test_no_engine_layer_and_no_release_key_file_installs_anyway(self) -> None:
        """No key file anywhere: the release key is compiled in, and the fake host is signed with the test key."""
        with self.pristine() as c:
            clean(self, c.run("login"))
            r = c.run("install", "e2e/packed", "--yes")
            clean(self, r)
            self.assertRegex(r.out, r"engine +installing \d+\.\d+\.\d+")

    def test_a_pack_only_model_on_a_card_with_no_pack_variant_is_refused_before_any_download(self) -> None:
        with self.pristine() as c:
            clean(self, c.run("login"))
            r = c.run("install", "nvidia/Qwen3.8-27B-NVFP4", "--yes")
            clean(self, r, ok=False)
            self.assertIn("nothing was downloaded", r.out + r.err)
            self.assertIn("built for sm_100a", r.out + r.err)
            self.assertLessEqual(r.lines, 2, r.out + r.err)
        self.assertFalse(self.downloaded("nvidia/Qwen3.8-27B-NVFP4"))

    def test_the_engine_layer_created_after_the_pack_runner_was_built_is_used(self) -> None:
        with self.pristine() as c:
            clean(self, c.run("login"))
            r = c.run("install", "e2e/packed", "--yes")
            clean(self, r)
            self.assertIn("kernels  sm_120a verified", r.out)


class Matrix(unittest.TestCase):
    """The other user paths on a warm home: cards, failures, Ctrl-C, fallback, updates, pack-store faults."""

    world: World

    @classmethod
    def setUpClass(cls) -> None:
        cls.world = World().start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.world.stop()

    def setUp(self) -> None:
        self.world.serial, self.world.bad_sig, self.world.pack_401 = 7, False, False
        self.mark = len(self.world.requests)

    def fetched(self, repo: str) -> bool:
        """Whether the weights of a repo were downloaded since the test began (its small config files may be read)."""
        return any(
            f"/{repo}/resolve/" in r and r.endswith((".safetensors", ".gguf")) for r in self.world.requests[self.mark :]
        )

    @contextmanager
    def cell(self, **kw):
        c = Cell(self.world, **kw)
        try:
            yield c
        finally:
            c.close()

    def runtime_of(self, c: Cell, model: str) -> str:
        ps = c.run("ps")
        return next(line for line in ps.out.splitlines() if model in line).split()[2]

    def lifecycle(self, c: Cell, model: str, runtime: str) -> None:
        """install --load, ps, a one-shot run, unload, remove."""
        r = c.run("install", model, "--yes", "--load")
        clean(self, r)
        self.assertLessEqual(r.lines, INSTALL_LINES, r.out + r.err)
        self.assertEqual(self.runtime_of(c, model), runtime)
        r = c.run("run", model, "say hi")
        clean(self, r)
        self.assertIn(f"hello from {model}", r.out)
        for args, want in ((("unload", model), "unloaded"), (("remove", model, "--yes"), "removed")):
            r = c.run(*args)
            clean(self, r)
            self.assertIn(want, r.out)

    def test_each_gpu_model_is_read_through_the_driver_library(self) -> None:
        for gpu, name in (("pro6000", "RTX PRO 6000"), ("5090", "RTX 5090"), ("gtx1080", "GTX 1080")):
            with self.cell(gpu=gpu) as c:
                r = c.run("doctor", "--hardware")
                clean(self, r)
                self.assertIn(name, r.out)
        with self.cell(gpu="none") as c:
            clean(self, c.run("install", "e2e/plain-gguf", "--yes", "--load"), ok=False)

    def test_a_plain_model_and_a_pack_model_run_end_to_end_on_each_card(self) -> None:
        with self.cell(gpu="5090") as c:
            self.lifecycle(c, "e2e/plain-gguf", "llama.cpp")
            log = c.logs("install")[0]
            self.assertEqual(stat.S_IMODE(log.stat().st_mode), 0o600)
            self.assertIn("Fetching", log.read_text())
        for gpu in ("5090", "pro6000"):
            with self.cell(gpu=gpu, login=True) as c:
                self.lifecycle(c, "e2e/packed", "goethe-engine")

    def test_not_logged_in_and_an_unsupported_card_are_refused_before_any_download(self) -> None:
        with self.cell() as c:
            r = c.run("install", "e2e/packed-only", "--yes")
            clean(self, r, ok=False)
            self.assertIn("run goethe login first", r.out + r.err)
            self.assertLessEqual(r.lines, 3)
        with self.cell(gpu="gtx1080", login=True) as c:
            r = c.run("install", "e2e/packed-only", "--yes")
            clean(self, r, ok=False)
            self.assertIn("no GPU here can run this model", r.out + r.err)
        self.assertFalse(self.fetched("e2e/packed-only"))

    def test_unknown_gated_and_offline_each_end_in_one_line_and_a_next_step(self) -> None:
        with self.cell() as c:
            r = c.run("install", "e2e/does-not-exist", "--yes")
            clean(self, r, ok=False)
            self.assertEqual(r.lines, 1, r.err)
            r = c.run("install", "e2e/gated", "--yes")
            clean(self, r, ok=False)
            self.assertEqual(r.lines, 1, r.err)
            self.assertIn("--hf-token", r.err)
            dead = {"HTTPS_PROXY": "http://127.0.0.1:9", "HTTP_PROXY": "http://127.0.0.1:9", "NO_PROXY": ""}
            r = c.run("install", "e2e/plain-gguf", "--yes", env=dead)
            clean(self, r, ok=False)
            self.assertIn("full log: ", r.out + r.err)
            self.assertLessEqual(r.lines, 4, r.out + r.err)

    def test_ctrl_c_stops_the_downloader_and_the_rerun_finishes(self) -> None:
        with self.cell() as c:
            r = c.run("install", "e2e/slow-gguf", "--yes", interrupt_after=1.5)
            self.assertEqual(r.code, 130, r.out + r.err)
            self.assertIn("run the same command to resume", r.err)
            time.sleep(1.0)
            pattern = f"download .*{c.home}"
            left = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout.split()
            self.assertEqual(left, [], "no downloader is left running")
            r = c.run("install", "e2e/slow-gguf", "--yes")
            clean(self, r)
            self.assertIn("e2e/slow-gguf  llama.cpp", r.out, "the rerun finished the install")

    def test_a_pack_that_runs_on_no_variant_serves_plain_unless_fallback_is_turned_off(self) -> None:
        with self.cell(login=True, extra={"E2E_PACK_RUNS": "no", "GOETHE_ALLOW_FALLBACK": "0"}) as c:
            r = c.run("install", "e2e/packed-gguf", "--yes")
            clean(self, r, ok=False)
            self.assertIn("--fallback", r.out + r.err)
        with self.cell(login=True, extra={"E2E_PACK_RUNS": "no"}) as c:
            r = c.run("install", "e2e/packed-gguf", "--yes", "--load")
            clean(self, r)
            self.assertIn("using its plain runtime", r.out + r.err)
            self.assertEqual(self.runtime_of(c, "e2e/packed-gguf"), "llama.cpp")

    def test_a_pack_index_with_a_bad_signature_is_refused_and_nothing_is_downloaded(self) -> None:
        with self.cell(login=True) as c:
            self.world.bad_sig = True
            r = c.run("install", "e2e/packed", "--yes")
            clean(self, r, ok=False)
            self.assertIn("signature", r.out + r.err)
        self.assertFalse(self.fetched("e2e/packed"))

    def test_an_index_serial_that_went_back_is_refused(self) -> None:
        with self.cell(login=True) as c:
            clean(self, c.run("install", "e2e/packed", "--yes"))
            self.world.serial = 3
            r = c.run("install", "e2e/packed", "--yes", "--force")
            clean(self, r, ok=False)
            self.assertIn("serial", r.out + r.err)

    def test_a_pack_too_big_for_the_card_is_refused_before_the_download(self) -> None:
        with self.cell(gpu="4060", login=True) as c:
            r = c.run("install", "e2e/toobig", "--yes")
            clean(self, r, ok=False)
            self.assertIn("does not fit", r.out + r.err)
            self.assertIn("nothing was downloaded", r.out + r.err)
        self.assertFalse(self.fetched("e2e/toobig"))

    def test_a_pack_that_needs_a_newer_engine_is_named_and_the_plain_runtime_serves(self) -> None:
        with self.cell(login=True) as c:
            r = c.run("install", "e2e/newer-engine", "--yes", "--load")
            clean(self, r)
            self.assertIn("engine_too_old", r.out + r.err)
            self.assertIn("goethe update", r.out + r.err)
            self.assertEqual(self.runtime_of(c, "e2e/newer-engine"), "llama.cpp")

    def test_a_pack_download_the_platform_refuses_with_401_says_to_log_in_again(self) -> None:
        with self.cell(login=True) as c:
            self.world.pack_401 = True
            r = c.run("install", "e2e/packed", "--yes")
            clean(self, r, ok=False)
            self.assertIn("goethe login", r.out + r.err)
            self.assertLessEqual(r.lines, 4, r.out + r.err)

    def test_update_on_a_build_newer_than_the_published_one_keeps_it_without_an_error(self) -> None:
        old = World(pkg_version="0.0.1").start()
        try:
            c = Cell(old)
            try:
                for args in (["update", "--check", "--yes"], ["update", "--yes"], ["update", "engine", "--yes"]):
                    r = c.run(*args)
                    clean(self, r)
                    self.assertLessEqual(r.lines, 4, r.out + r.err)
                    if "--check" in args:
                        self.assertIn("everything is up to date", r.out, "--check ends with its summary")
            finally:
                c.close()
        finally:
            old.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
