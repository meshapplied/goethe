import hashlib
import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from goethe.engine import Install
from goethe.update import check

ENV = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
       "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}


def git(cwd, *args, out=False):
    done = subprocess.run(["git", "-C", str(cwd), *args], check=not out, env=ENV, capture_output=True, text=True)
    return done.stdout.strip() if out else None


def install(root: Path, body: bytes, applied: bool, rev: str | None = None,
            patched: bytes | None = None) -> tuple[Install, dict]:
    src = root / "home" / "ninfer"
    src.mkdir(parents=True)
    (src / "f").write_text("a\n")
    git(src, "init", "-q")
    git(src, "add", "f")
    git(src, "commit", "-q", "-m", "x")
    pinned = git(src, "rev-parse", "HEAD", out=True)
    (src / "f").write_text("b\n")
    patch = subprocess.check_output(["git", "-C", str(src), "diff", "--", "f"])
    git(src, "checkout", "-q", "--", "f")
    recipe = root / "recipe"
    dest = recipe / "patches"
    dest.mkdir(parents=True)
    (dest / "ninfer-dn-sched.patch").write_bytes(patch)
    if applied:
        git(src, "apply", str(dest / "ninfer-dn-sched.patch"))
    if rev == "other":
        (src / "f").write_text("c\n")
        git(src, "add", "f")
        git(src, "commit", "-q", "-m", "y")
    binary = src / "build" / "apps" / "ninfer-serve"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n")
    binary.chmod(binary.stat().st_mode | stat.S_IEXEC)
    model = root / "home" / "models" / "m"
    model.parent.mkdir()
    model.write_bytes(body)
    pin = {"NINFER_REV": pinned, "MODEL_BYTES": str(len(body)),
           "MODEL_PATCHED_SHA256": hashlib.sha256(body if patched is None else patched).hexdigest(),
           "MODEL_SHA256": hashlib.sha256(body).hexdigest()}
    return Install(recipe, root / "home", binary, model, len(body)), pin


class Check(unittest.TestCase):
    def test_matching_install(self):
        with tempfile.TemporaryDirectory() as d:
            inst, pin = install(Path(d), b"hello", applied=True)
            rows = {c.name: c for c in check(inst, pin)}
        self.assertTrue(all(c.status == "ok" for c in rows.values()), rows)
        self.assertEqual(rows["model"].detail, hashlib.sha256(b"hello").hexdigest()[:12] + ", patched")

    def test_missing_patch_wrong_rev_and_unpatched_model(self):
        with tempfile.TemporaryDirectory() as d:
            inst, pin = install(Path(d), b"plain", applied=False, rev="other", patched=b"hello")
            rows = {c.name: c for c in check(inst, pin)}
        self.assertEqual(rows["engine"].status, "fail")
        self.assertEqual(rows["patch"].status, "fail")
        self.assertEqual(rows["model"].status, "fail")
        self.assertIn("head patch", rows["model"].detail)

    def test_size_mismatch_is_not_hashed_as_a_match(self):
        with tempfile.TemporaryDirectory() as d:
            inst, pin = install(Path(d), b"hello", applied=True)
            pin["MODEL_BYTES"] = "1"
            rows = {c.name: c for c in check(inst, pin)}
        self.assertEqual(rows["model"].status, "fail")
        self.assertIn("bytes", rows["model"].detail)

    def test_nothing_installed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inst = Install(root, root / "home", root / "home" / "ninfer" / "build" / "apps" / "ninfer-serve",
                           root / "home" / "models" / "m", 1)
            rows = {c.name: c for c in check(inst, {"NINFER_REV": "abc", "MODEL_BYTES": "1",
                                                    "MODEL_PATCHED_SHA256": "aa", "MODEL_SHA256": "bb"})}
        self.assertTrue(all(c.status == "fail" for c in rows.values()))


if __name__ == "__main__":
    unittest.main()
