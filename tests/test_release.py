import os
import tempfile
import unittest
from pathlib import Path

from goethe.release import ReleaseError, rollback, switch


def engine(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    return path


class Switch(unittest.TestCase):
    def test_refusal_leaves_current_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            switch(root, engine(root, "a"), lambda _: True)
            with self.assertRaises(ReleaseError):
                switch(root, engine(root, "b"), lambda _: False)
            self.assertEqual(str((root / "current").readlink()), "a")
            self.assertEqual((root / "history").read_text(), "")
            self.assertTrue((root / "b").is_dir())

    def test_keeps_the_previous_two_and_rollback_flips(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for name in "abcd":
                switch(root, engine(root, name), lambda _: True)
            self.assertEqual(str((root / "current").readlink()), "d")
            self.assertEqual((root / "history").read_text().split(), ["c", "b"])
            self.assertFalse((root / "a").exists())
            self.assertEqual(rollback(root), "c")
            self.assertEqual(str((root / "current").readlink()), "c")
            self.assertEqual(rollback(root), "d")
            self.assertEqual(str((root / "current").readlink()), "d")

    def test_rollback_without_history_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ReleaseError):
                rollback(Path(d))

    def test_history_cannot_point_outside_the_engines_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "history").write_text("../outside\n")
            with self.assertRaises(ReleaseError):
                rollback(root)
            self.assertFalse((root / "current").exists())

    def test_a_link_that_leaves_the_directory_is_not_followed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            switch(root, engine(root, "a"), lambda _: True)
            (root / "current").unlink()
            (root / "current").symlink_to("../outside")
            with self.assertRaises(ReleaseError):
                switch(root, engine(root, "b"), lambda _: True)
            self.assertEqual(os.readlink(root / "current"), "../outside")
            self.assertTrue((root / "a").is_dir())
            self.assertTrue((root / "b").is_dir())


if __name__ == "__main__":
    unittest.main()
