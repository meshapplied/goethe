#!/usr/bin/env python3
"""The real goethe and goethed binaries as a pristine user with no GPU; the world around them is faked."""

from __future__ import annotations

import re
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
        self.assertRegex(r.out, r"engine +\d+\.\d+\.\d+ ready", "the engine layer was created by this install")
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
            self.assertRegex(r.out, r"engine +\d+\.\d+\.\d+ ready")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
