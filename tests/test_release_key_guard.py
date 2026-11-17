"""The lint that keeps the test release key out of shipped binaries."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.fixtures import REPO

LINT = REPO / "engine" / "tests" / "lint_style.py"


def lint(tree: Path) -> str:
    out = subprocess.run([sys.executable, str(LINT), str(tree / "engine")], capture_output=True, text=True)
    return out.stdout + out.stderr


class ReleaseKeyGuard(unittest.TestCase):
    def tree(self, root: Path, toml: str = "[features]\ntest-release-key = []\n", release: str = "") -> None:
        (root / "engine" / "packaging").mkdir(parents=True, exist_ok=True)
        (root / "engine" / "Cargo.toml").write_text(toml)
        (root / ".github" / "workflows").mkdir(parents=True, exist_ok=True)
        (root / ".github" / "workflows" / "release.yml").write_text(release)
        (root / ".github" / "workflows" / "ci.yml").write_text(
            "jobs:\n  engine-test:\n    run: cargo build --features goethe/test-release-key\n"
        )

    def test_the_feature_can_exist_but_not_ship(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.tree(root)
            self.assertNotIn("test-release-key", lint(root))
            self.tree(root, toml='[features]\ndefault = ["x", "test-release-key"]\n')
            self.assertIn("is a default feature", lint(root))
            self.tree(root, release="run: cargo build --release --features goethe/test-release-key\n")
            self.assertIn("release.yml: passes test-release-key", lint(root))
            (root / ".github" / "workflows" / "ci.yml").write_text(
                "jobs:\n  a:\n    run: x\n\n  package-smoke:\n    run: cargo build --features goethe/test-release-key\n"
            )
            self.assertIn("package-smoke job passes", lint(root))


if __name__ == "__main__":
    unittest.main()
