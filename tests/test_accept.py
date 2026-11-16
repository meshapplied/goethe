"""engine/tests/accept.sh against a fake goethe: the table, the JSON, the pack-runtime rule."""

import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from tests.fixtures import REPO

SCRIPT = REPO / "engine" / "tests" / "accept.sh"
FAKE = r"""#!/usr/bin/env bash
home="$GOETHE_HOME/models"
case "$1" in
  models) echo '[{"id":"org/packed","status":"available","gb":1.0},{"id":"org/plain","status":"available","gb":1.0},'\
    '{"id":"org/lost","status":"available","gb":1.0}]' ;;
  install)
    id="${2%%:*}"; d="$home/${id//\//__}"; mkdir -p "$d"
    case "$id" in
      org/packed) echo "installing the optimised kernels for sm_120a ..."
        echo 'runtime = "goethe-engine"' > "$d/goethe_card.toml"
        echo '{"ok": true, "manifest_sha256": "0123456789abcdef0123456789abcdef"}' > "$d/pack.verify.json" ;;
      org/plain) echo "(no kernel pack: none for this model on this GPU)"
        echo 'runtime = "llama.cpp"' > "$d/goethe_card.toml" ;;
      org/lost) echo "installing the optimised kernels for sm_120a ..."
        echo 'runtime = "llama.cpp"' > "$d/goethe_card.toml" ;;
    esac ;;
  run) echo "hello there my friend" ;;
  unload | remove) : ;;
esac
"""


class Accept(unittest.TestCase):
    def run_accept(self, *args):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d, "goethe")
            fake.write_text(FAKE)
            fake.chmod(0o755)
            Path(d, "models").mkdir()
            env = {**os.environ, "GOETHE_HOME": d, "GOETHE_BIN": str(fake), "ACCEPT_RESERVE_GB": "0"}
            cmd = ["bash", str(SCRIPT), *args, "org/packed", "org/plain", "org/lost"]
            return subprocess.run(cmd, env=env, capture_output=True, text=True)

    def test_json_names_runtime_pack_and_verdict_and_a_pack_model_must_run_on_its_pack(self):
        out = self.run_accept("--json")
        doc = json.loads(out.stdout)
        rows = {m["model"]: m for m in doc["models"]}
        packed = rows["org/packed"]
        got = (packed["runtime"], packed["pack"], packed["result"])
        self.assertEqual(got, ("goethe-engine", "0123456789abcdef", "PASS"))
        plain = rows["org/plain"]
        self.assertEqual((plain["runtime"], plain["result"]), ("llama.cpp", "PASS"), "no pack: plain is right")
        self.assertEqual(rows["org/lost"]["result"], "FAIL")
        self.assertIn("kernel pack", rows["org/lost"]["reason"])
        self.assertEqual((doc["pass"], doc["fail"], doc["skip"]), (2, 1, 0))
        self.assertEqual(out.returncode, 1, "a failing model fails the run")

    def test_text_mode_ends_with_one_aligned_table(self):
        out = self.run_accept()
        self.assertIn("MODEL", out.stdout)
        table = out.stdout.split("\n\n", 1)[1]
        self.assertEqual(len([ln for ln in table.splitlines() if ln.startswith("org/")]), 3)
        self.assertIn("RUNTIME", table)
        self.assertIn("accept: 2 pass, 1 fail, 0 skip", table)


if __name__ == "__main__":
    unittest.main()
