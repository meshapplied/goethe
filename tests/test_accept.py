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
    '{"id":"org/lost","status":"available","gb":1.0},{"id":"org/noisy","status":"available","gb":0.0},'\
    '{"id":"org/slow","status":"available","gb":0.0},{"id":"org/chatty","status":"available","gb":0.0},'\
    '{"id":"org/big","status":"available","gb":5.0},{"id":"org/biggood","status":"available","gb":5.0}]' ;;
  up) echo "$GOETHE_HOME" >> "$FAKE_LOG" ;;
  down) : ;;
  install)
    case "$2" in
      org/does-not-exist-xyz-123) [[ -n "$FAKE_TRACE" ]] && echo "Traceback (most recent call last):" >&2
        echo "$2: not found on Hugging Face (check the id; add --hf-token if private)" >&2; exit 1 ;;
      orcarouter/OrcaCyber-Nano-1.0) echo "$2 is gated: goethe install $2 --hf-token" >&2; exit 1 ;;
      org/offline-probe-xyz) [[ -z "$HTTPS_PROXY" ]] || { echo "cannot reach the hub; check the net" >&2; exit 1; } ;;
      nvidia/Qwen3.8-27B-NVFP4) [[ -d "$GOETHE_HOME/platform" ]] || { echo "$2: goethe login first" >&2; exit 1; } ;;
      ggml-org/embeddinggemma-2-GGUF:Q8_0) f="$GOETHE_HOME/.slow_once"; [[ ! -f "$f" ]] || { rm -f "$f"; sleep 30; } ;;
      org/noisy) echo "Traceback (most recent call last):" >&2 ;;
      org/chatty) for i in $(seq 12); do echo "note $i" >&2; done ;;
      org/slow) sleep 3 ;;
      org/biggood) echo "org/biggood (gguf, 5.0 GB) with llama.cpp" ;;
    esac
    id="${2%%:*}"; [[ "$2" == /* ]] && id="$4"
    d="$home/${id//\//__}"; mkdir -p "$d"
    case "$id" in
      org/packed) echo "installing the optimised kernels for sm_120a (1.0 GB) ..."
        echo 'runtime = "goethe-engine"' > "$d/goethe_card.toml"
        echo '{"ok": true, "manifest_sha256": "0123456789abcdef0123456789abcdef"}' > "$d/pack.verify.json" ;;
      org/plain) echo "(no kernel pack: none for this model on this GPU)"
        echo 'runtime = "llama.cpp"' > "$d/goethe_card.toml" ;;
      org/lost) echo "installing the optimised kernels for sm_120a ..."
        echo 'runtime = "llama.cpp"' > "$d/goethe_card.toml" ;;
      org/noisy | org/slow | org/chatty | org/big | org/biggood) echo 'runtime = "llama.cpp"' > "$d/goethe_card.toml" ;;
      local/accept) echo "installing the optimised kernels for sm_120a ..."
        echo "runtime = \"${FAKE_LOCAL_RUNTIME:-goethe-engine}\"" > "$d/goethe_card.toml" ;;
    esac ;;
  run) echo "hello there my friend" ;;
  unload | remove) : ;;
esac
"""


class Accept(unittest.TestCase):
    def run_accept(self, *args, extra=None, models=("org/packed", "org/plain", "org/lost"), slow_once=False):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d, "goethe")
            fake.write_text(FAKE)
            fake.chmod(0o755)
            Path(d, "models").mkdir()
            if slow_once:
                Path(d, ".slow_once").write_text("")
            log = Path(d, "up.log")
            env = {**os.environ, "GOETHE_HOME": d, "GOETHE_BIN": str(fake), "ACCEPT_RESERVE_GB": "0"}
            env["FAKE_LOG"] = str(log)
            env.update(extra or {})
            out = subprocess.run(["bash", str(SCRIPT), *args, *models], env=env, capture_output=True, text=True)
            out.warm = d
            out.started = log.read_text().split() if log.exists() else []
            out.warm_models = sorted(p.name for p in Path(d, "models").iterdir())
            return out

    def test_json_names_runtime_pack_and_verdict_and_a_pack_model_must_run_on_its_pack(self):
        out = self.run_accept("--json")
        doc = json.loads(out.stdout)
        rows = {m["model"]: m for m in doc["models"]}
        packed = rows["org/packed"]
        got = (packed["runtime"], packed["pack"], packed["result"])
        self.assertEqual(got, ("goethe-engine", "0123456789abcdef", "PASS"))
        plain = rows["org/plain"]
        self.assertEqual((plain["runtime"], plain["result"]), ("llama.cpp", "PASS"), "no pack: plain is right")
        self.assertEqual((packed["home"], plain["home"]), ("warm", "warm"))
        self.assertEqual(rows["org/lost"]["result"], "FAIL")
        self.assertIn("kernel pack", rows["org/lost"]["reason"])
        self.assertEqual((doc["pass"], doc["fail"], doc["skip"]), (2, 1, 0))
        self.assertEqual(out.returncode, 1, "a failing model fails the run")

    def test_text_mode_ends_with_one_aligned_table(self):
        out = self.run_accept()
        self.assertIn("MODEL", out.stdout)
        self.assertIn("HOME", out.stdout)
        table = out.stdout.split("\n\n", 1)[1]
        self.assertEqual(len([ln for ln in table.splitlines() if ln.startswith("org/")]), 3)
        self.assertIn("RUNTIME", table)
        self.assertIn("accept: 2 pass, 1 fail, 0 skip", table)

    def test_a_fresh_home_without_a_login_to_clone_is_skipped_with_the_reason(self):
        out = self.run_accept("--json", "--fresh")
        doc = json.loads(out.stdout)
        self.assertEqual((doc["pass"], doc["fail"], doc["skip"]), (0, 0, 3))
        self.assertTrue(all(m["home"] == "fresh" and "ACCEPT_LOGIN_HOME" in m["reason"] for m in doc["models"]))
        self.assertEqual(out.started, [], "goethe was not started anywhere")
        self.assertEqual(out.returncode, 0)

    def test_a_fresh_cell_runs_in_a_throwaway_home_with_the_login_cloned_and_leaves_nothing(self):
        with tempfile.TemporaryDirectory() as linked, tempfile.TemporaryDirectory() as root:
            Path(linked, "platform").mkdir()
            Path(linked, "platform", "credentials.json").write_text("{}")
            out = self.run_accept("--json", "--fresh", extra={"ACCEPT_LOGIN_HOME": linked, "ACCEPT_FRESH_ROOT": root},
                                  models=("org/packed", "org/plain"))
            doc = json.loads(out.stdout)
            self.assertEqual([m["home"] for m in doc["models"]], ["fresh", "fresh"])
            self.assertEqual((doc["pass"], doc["fail"]), (2, 0))
            self.assertEqual(len(out.started), 2, "goethe started once per cell")
            self.assertTrue(all(h.startswith(root) and h != out.warm for h in out.started), out.started)
            self.assertEqual(list(Path(root).iterdir()), [], "the throwaway homes are gone")
            self.assertEqual(out.warm_models, [], "the warm home was not touched")

    def test_a_local_folder_cell_must_end_on_the_pack_runtime_like_a_hub_cell(self):
        with tempfile.TemporaryDirectory() as folder:
            ok = self.run_accept("--json", extra={"ACCEPT_LOCAL_DIR": folder}, models=("org/packed",))
            rows = {m["model"]: m for m in json.loads(ok.stdout)["models"]}
            local = rows["local/accept"]
            self.assertEqual((local["runtime"], local["result"]), ("goethe-engine", "PASS"))
            plain = {"ACCEPT_LOCAL_DIR": folder, "FAKE_LOCAL_RUNTIME": "llama.cpp"}
            bad = self.run_accept("--json", extra=plain, models=())
            row = json.loads(bad.stdout)["models"][-1]
            self.assertEqual((row["model"], row["result"]), ("local/accept", "FAIL"))
            self.assertIn("kernel pack", row["reason"], "a plain runtime on a local folder is a failure")

    def test_ux_mode_times_each_cell_enforces_the_limits_and_runs_the_failure_cells(self):
        models = ("org/packed", "org/noisy", "org/slow", "org/chatty", "org/big", "org/biggood")
        out = self.run_accept("--json", "--ux", extra={"ACCEPT_UX_GAP": "1"}, models=models, slow_once=True)
        rows = {m["model"]: m for m in json.loads(out.stdout)["models"]}
        self.assertEqual(rows["org/packed"]["result"], "PASS")
        self.assertRegex(rows["org/packed"]["ux"], r"^ttfa [0-9.]+s gap [0-9.]+s err 0$")
        reasons = {k: rows[k]["reason"] for k in ("org/noisy", "org/slow", "org/chatty", "org/big")}
        self.assertTrue(all(rows[k]["result"] == "FAIL" for k in reasons), reasons)
        self.assertRegex(reasons["org/noisy"], "[Tt]raceback")
        self.assertIn("silent for", reasons["org/slow"])
        self.assertIn("12 stderr lines", reasons["org/chatty"])
        self.assertIn("not announced", reasons["org/big"])
        self.assertEqual(rows["org/biggood"]["result"], "PASS", "a download announced with its size passes")
        cells = {k: v["result"] for k, v in rows.items() if k.startswith("ux/")}
        wanted = ("ux/unknown-id", "ux/gated-no-token", "ux/offline", "ux/interrupt-rerun", "ux/not-logged-in")
        self.assertEqual(cells, {k: "PASS" for k in wanted}, rows)

    def test_a_failure_cell_that_prints_a_traceback_fails(self):
        out = self.run_accept("--json", "--ux", extra={"FAKE_TRACE": "1"}, models=("org/plain",), slow_once=True)
        rows = {m["model"]: m for m in json.loads(out.stdout)["models"]}
        self.assertEqual(rows["ux/unknown-id"]["result"], "FAIL")
        self.assertRegex(rows["ux/unknown-id"]["reason"], "[Tt]raceback")


if __name__ == "__main__":
    unittest.main()
