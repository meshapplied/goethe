"""engine/tests/main_in_dev.sh fails exactly when main has commits dev lacks."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.fixtures import REPO

SCRIPT = REPO / "engine" / "tests" / "main_in_dev.sh"
WHO = {"NAME": "t", "EMAIL": "t@example.com"}
GIT_ENV = {**os.environ, **{f"GIT_{who}_{k}": v for who in ("AUTHOR", "COMMITTER") for k, v in WHO.items()}}


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, check=True, capture_output=True, text=True).stdout


class MainInDev(unittest.TestCase):
    def check(self, cwd):
        return subprocess.run(["bash", str(SCRIPT), "main", "dev"], cwd=cwd, capture_output=True, text=True)

    def test_a_commit_only_on_main_fails_with_its_name_and_a_merge_makes_it_pass(self):
        with tempfile.TemporaryDirectory() as d:
            git(d, "init", "-q", "-b", "main")
            Path(d, "a").write_text("1")
            git(d, "add", "a")
            git(d, "commit", "-q", "-m", "first")
            git(d, "branch", "dev")
            self.assertEqual(self.check(d).returncode, 0, "equal branches pass")
            Path(d, "a").write_text("2")
            git(d, "commit", "-q", "-am", "a hotfix only on main")
            out = self.check(d)
            self.assertEqual(out.returncode, 1)
            self.assertIn("merge main into dev first", out.stderr)
            self.assertIn("a hotfix only on main", out.stderr)
            git(d, "checkout", "-q", "dev")
            git(d, "merge", "-q", "--no-edit", "main")
            self.assertEqual(self.check(d).returncode, 0, "dev holding main's commits passes")
            Path(d, "b").write_text("dev work")
            git(d, "add", "b")
            git(d, "commit", "-q", "-m", "dev ahead of main is fine")
            self.assertEqual(self.check(d).returncode, 0)


if __name__ == "__main__":
    unittest.main()
