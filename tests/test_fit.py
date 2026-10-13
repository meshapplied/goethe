import json
import tempfile
import unittest
from pathlib import Path

from goethe import fit, registry, resolve
from tests.fixtures import RTX_5090

MODEL = registry.load()["qwen3-8-27b"]
FATAL = ("FATAL server failed during startup | minimum Engine runtime reservation requires {need} bytes "
         "in addition to {head} bytes of automatic headroom, but only {have} bytes are available")


def plan(profile="long"):
    return resolve.resolve(MODEL, [RTX_5090], profile, env={})


def boots(*results):
    """A boot() stand-in that returns scripted log text and records the argv it was given."""
    seen = []

    def boot(argv, log):
        seen.append(argv)
        text = results[len(seen) - 1]
        return text() if callable(text) else text

    boot.seen = seen
    return boot


def started(tokens):
    return f"2026 INFO capacity | KV {tokens:,} tokens, fp8, auto | pages 1/1 | runtime 1 GiB | free 1 GiB"


class FitLoop(unittest.TestCase):
    def test_first_boot_success_is_one_attempt(self):
        boot = boots(started(131_072))
        with tempfile.TemporaryDirectory() as d:
            got = fit.fit(plan(), ["serve", "--port", "8080", "--max-context", "1", "--max-concurrency", "1"],
                          Path(d) / "tune.log", boot=boot)
        self.assertEqual((got.context, got.lanes, got.kv_tokens, got.attempts), (131072, 1, 131072, 1))

    def test_real_boot_creates_the_log_and_kills_the_engine(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "new" / "tune.log"
            fake = Path(d) / "engine.sh"
            fake.write_text('#!/bin/sh\nprintf "FATAL minimum Engine runtime reservation requires 9 bytes in addition '
                            'to 1 bytes of automatic headroom, but only 5 bytes are available\\n"; sleep 30\n')
            fake.chmod(0o755)
            before = set(self._engines())
            with self.assertRaises(fit.ProbeError):
                fit.fit(plan("long"), [str(fake)], log, seconds=3)
            self.assertTrue(log.exists())
            self.assertEqual(set(self._engines()) - before, set())

    def _engines(self):
        import os
        found = []
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\x00", b" ").decode()
            except OSError:
                continue
            if "engine.sh" in cmd:
                found.append(pid)
        return found

    def test_probe_uses_its_own_port(self):
        boot = boots(started(1000))
        with tempfile.TemporaryDirectory() as d:
            fit.fit(plan(), ["b", "--port", "8080"], Path(d) / "t.log", boot=boot)
        self.assertEqual(boot.seen[0][boot.seen[0].index("--port") + 1], str(fit.PROBE_PORT))

    def test_failure_steps_the_context_down(self):
        boot = boots(FATAL.format(need=19, head=1, have=10), started(65_536))
        with tempfile.TemporaryDirectory() as d:
            got = fit.fit(plan("long"), [], Path(d) / "t.log", boot=boot)
        self.assertEqual(got.attempts, 2)
        sizes = [int(a[a.index("--max-context") + 1]) for a in boot.seen]
        self.assertEqual(sizes, [131072, 131072 - fit.STEP])

    def test_lanes_never_change(self):
        # fewer lanes reserve more KV, not less, so shrinking them cannot help
        fail = FATAL.format(need=100, head=0, have=50)
        boot = boots(fail, started(4096))
        with tempfile.TemporaryDirectory() as d:
            got = fit.fit(plan("agent"), [], Path(d) / "t.log", boot=boot)
        lanes = [int(a[a.index("--max-concurrency") + 1]) for a in boot.seen]
        self.assertEqual(lanes, [4, 4])
        self.assertEqual(got.lanes, 4)
        self.assertLess(got.context, 100000)

    def test_four_failures_raise(self):
        boot = boots(*[FATAL.format(need=100, head=0, have=50)] * 4)
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(fit.ProbeError, "still does not fit"):
            fit.fit(plan(), [], Path(d) / "t.log", boot=boot)

    def test_unrecognized_log_raises_instead_of_guessing(self):
        boot = boots("ERROR something unexpected\n")
        with tempfile.TemporaryDirectory() as d, self.assertRaisesRegex(fit.ProbeError, "neither started nor gave"):
            fit.fit(plan(), [], Path(d) / "t.log", boot=boot)

    def test_argv_keeps_engine_flags_and_replaces_context(self):
        argv = ["bin", "model", "--host", "127.0.0.1", "--max-context", "999", "--max-concurrency", "9",
                "--request-log-jsonl", "/tmp/r"]
        out = fit.engine_argv(argv, 4096, 1, port=18099)
        self.assertEqual(out[:4], ["bin", "model", "--host", "127.0.0.1"])
        self.assertNotIn("999", out)
        self.assertEqual(out[out.index("--request-log-jsonl") + 1], "/tmp/r")
        self.assertEqual(out[-6:], ["--max-context", "4096", "--max-concurrency", "1", "--port", "18099"])


class Cache(unittest.TestCase):
    def key(self, path):
        path.write_bytes(b"x" * 100)
        return fit.cache_key(plan(), path)

    def test_round_trip_and_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            model, path = Path(d) / "m", Path(d) / "tune.json"
            key = self.key(model)
            fit.save(path, key, fit.Fit(65536, 1, 65000, 2), free_mib=20000)
            self.assertEqual(fit.load(path, key, 20000)["context"], 65536)
            self.assertIsNone(fit.load(path, key, 19000))  # less free memory than the fit saw
            self.assertIsNone(fit.load(path, {**key, "driver": "other"}, 30000))
            self.assertIsNone(fit.load(Path(d) / "missing.json", key, 30000))
            path.write_text("{")
            self.assertIsNone(fit.load(path, key, 30000))

    def test_saved_file_is_plain_json(self):
        with tempfile.TemporaryDirectory() as d:
            model, path = Path(d) / "m", Path(d) / "tune.json"
            fit.save(path, self.key(model), fit.Fit(1, 1, 1, 1), 5)
            self.assertEqual(set(json.loads(path.read_text())), {"key", "context", "lanes", "kv_tokens", "free_mib"})


if __name__ == "__main__":
    unittest.main()
