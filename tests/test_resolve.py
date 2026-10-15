import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from goethe import registry, resolve
from tests.fixtures import PRO_4000, PRO_6000, REPO, RTX_3090, RTX_5090, gpu

MODEL = registry.load()["qwen3-8-27b"]
BIN, FILE = Path("/b/ninfer-serve"), Path("/m/model.ninfer")


def plan(gpus=(RTX_5090,), profile="agent", **kw):
    kw.setdefault("env", {})
    return resolve.resolve(MODEL, list(gpus), profile, **kw)


class Profiles(unittest.TestCase):
    def test_agent_on_a_5090(self):
        p = plan()
        self.assertEqual((p.tier.name, p.gpu.index), ("tested", 0))
        self.assertEqual(resolve.server_argv(p, BIN, FILE), [
            "/b/ninfer-serve", "/m/model.ninfer", "--host", "127.0.0.1", "--port", "8080",
            "--model-id", "qwen3.8-27b", "--max-context", "100000", "--kv-capacity", "auto",
            "--max-concurrency", "4", "--kv-dtype", "fp8", "--spec", "dflash2", "--draft-tokens", "7",
            "--lm-head-draft", "--prefill-chunk", "1024"])

    def test_chat_and_long(self):
        c, long = plan(profile="chat"), plan(profile="long")
        self.assertEqual((c.value("lanes"), c.value("context")), (1, 32768))
        self.assertEqual((long.value("lanes"), long.value("context")), (1, 131072))
        self.assertTrue(all(p.value("spec") == "dflash2" and p.value("draft_tokens") == 7 for p in (c, long)))

    def test_profile_settings_carry_their_reason(self):
        self.assertIn("RESULTS.md", plan().settings["lanes"].why)

    def test_goethe_server_flags_equal_serve_sh(self):
        """cli and nvfp4-kernels/serve.sh must start the engine the same way."""
        with tempfile.TemporaryDirectory() as d:
            stub, model = Path(d) / "stub", Path(d) / "model.ninfer"
            stub.write_text('#!/bin/sh\nfor a in "$@"; do echo "$a"; done\n')
            stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
            model.write_bytes(b"")
            for profile, shell_env in (("agent", {"PROFILE": "default"}), ("long", {"PROFILE": "long"}),
                                      ("chat", {"PROFILE": "default", "MAX_CONCURRENCY": "1", "MAX_CONTEXT": "32768"})):
                env = {**os.environ, "NINFER_SERVE": str(stub), "MODEL": str(model), "HOME": d, **shell_env}
                out = subprocess.run(["bash", str(REPO / "nvfp4-kernels" / "serve.sh")], env=env, check=True,
                                     capture_output=True, text=True).stdout.split("\n")[:-1]
                self.assertEqual(resolve.server_argv(plan(profile=profile), stub, model)[1:], out, profile)


class Layers(unittest.TestCase):
    def test_precedence_is_profile_config_env_flag(self):
        config = {"serve": {"port": 9000}, "profiles.agent": {"lanes": 2}}
        p = plan(config=config)
        self.assertEqual((p.settings["port"].source, p.settings["lanes"].source), ("config", "config"))
        p = plan(config=config, env={"GOETHE_PORT": "9100", "GOETHE_LANES": "3"})
        self.assertEqual((p.value("port"), p.settings["port"].source, p.value("lanes")), (9100, "env", 3))
        p = plan(config=config, env={"GOETHE_PORT": "9100"}, flags={"port": "9200"})
        self.assertEqual((p.value("port"), p.settings["port"].source), (9200, "flag"))
        self.assertEqual(p.settings["context"].source, "profile")

    def test_config_for_another_profile_is_not_applied(self):
        self.assertEqual(plan(config={"profiles.chat": {"lanes": 2}}).value("lanes"), 4)

    def test_extra_engine_args_go_last(self):
        argv = resolve.server_argv(plan(extra_args=["--request-log-jsonl", "/x"]), BIN, FILE)
        self.assertEqual(argv[-2:], ["--request-log-jsonl", "/x"])

    def test_invalid_values(self):
        cases = [({"lanes": "9"}, "lanes from flag"), ({"lanes": "0"}, "lanes"), ({"port": "0"}, "port"),
                 ({"port": "http"}, "not an integer"), ({"spec": "eagle"}, "spec"), ({"kv_dtype": "q2"}, "kv_dtype"),
                 ({"context": "100"}, "context"), ({"context": "999999"}, "above the model"),
                 ({"draft_tokens": "16"}, "draft_tokens"), ({"nope": "1"}, "unknown setting"), ({"host": ""}, "host")]
        for flags, message in cases:
            with self.subTest(flags=flags), self.assertRaisesRegex(resolve.ResolveError, message):
                plan(flags=flags)

    def test_draft_tokens_bound_depends_on_spec(self):
        self.assertEqual(plan(flags={"spec": "mtp", "draft_tokens": "3"}).value("draft_tokens"), 3)
        with self.assertRaisesRegex(resolve.ResolveError, "outside 1..5 for spec mtp"):
            plan(flags={"spec": "mtp"})

    def test_bad_env_value_names_its_source(self):
        with self.assertRaisesRegex(resolve.ResolveError, "lanes from env"):
            plan(env={"GOETHE_LANES": "many"})

    def test_non_fp8_kv_warns(self):
        self.assertEqual(plan().warnings, [])
        self.assertIn("kv_dtype nvfp4", plan(flags={"kv_dtype": "nvfp4"}).warnings[0])


class ApiKey(unittest.TestCase):
    def test_loopback_has_no_key(self):
        self.assertNotIn("api_key", plan().settings)
        self.assertNotIn("--api-key", resolve.server_argv(plan(), BIN, FILE))

    def test_public_host_generates_a_private_stable_key(self):
        with tempfile.TemporaryDirectory() as d:
            env = {"XDG_STATE_HOME": d}
            first = plan(flags={"host": "0.0.0.0"}, env=env)
            second = plan(flags={"host": "0.0.0.0"}, env=env)
            key_file = Path(d) / "goethe" / "api_key"
            self.assertEqual(first.settings["api_key"].source, "generated")
            self.assertEqual(first.value("api_key"), second.value("api_key"))
            self.assertEqual(stat.S_IMODE(key_file.stat().st_mode), 0o600)
            argv = resolve.server_argv(first, BIN, FILE)
            self.assertEqual(argv[argv.index("--api-key") + 1], first.value("api_key"))
            redacted = resolve.server_argv(first, BIN, FILE, redact=True)
            self.assertEqual(redacted[argv.index("--api-key") + 1], "<hidden>")

    def test_an_empty_key_file_is_replaced(self):
        with tempfile.TemporaryDirectory() as d:
            key_file = Path(d) / "goethe" / "api_key"
            key_file.parent.mkdir()
            key_file.write_text("\n")
            p = plan(flags={"host": "0.0.0.0"}, env={"XDG_STATE_HOME": d})
            self.assertTrue(p.value("api_key").startswith("gk-"))
            self.assertEqual(key_file.read_text().strip(), p.value("api_key"))

    def test_configured_key_wins_over_generation(self):
        with tempfile.TemporaryDirectory() as d:
            p = plan(flags={"host": "0.0.0.0"}, config={"serve": {"api_key": "mine"}}, env={"XDG_STATE_HOME": d})
            self.assertEqual((p.value("api_key"), p.settings["api_key"].source), ("mine", "config"))
            self.assertFalse((Path(d) / "goethe").exists())


class Hardware(unittest.TestCase):
    def test_tiers(self):
        tiers = {g.name: resolve.tier_for(MODEL, g) for g in (RTX_5090, PRO_6000, PRO_4000, RTX_3090)}
        self.assertEqual([t.name for t in tiers.values()], ["tested", "compatible", "unsupported", "unsupported"])
        self.assertIn("sm_120a", tiers[RTX_3090.name].reason)
        self.assertIn("at least 30 GiB", tiers[PRO_4000.name].reason)

    def test_unsupported_gpu_is_a_clear_error(self):
        with self.assertRaisesRegex(resolve.ResolveError, "sm_120a"):
            plan(gpus=[RTX_3090])
        with self.assertRaisesRegex(resolve.ResolveError, "no NVIDIA GPU"):
            plan(gpus=[])

    def test_first_usable_gpu_wins_and_device_is_passed(self):
        second = gpu(index=1)
        p = plan(gpus=[RTX_3090, second])
        self.assertEqual(p.gpu.index, 1)
        argv = resolve.server_argv(p, BIN, FILE)
        self.assertEqual(argv[-2:], ["--device", "1"])


class Config(unittest.TestCase):
    def load(self, text):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "config.toml"
            path.write_text(text)
            return resolve.load_config(path)

    def test_missing_file_is_empty(self):
        self.assertEqual(resolve.load_config(Path("/nonexistent/config.toml")), {})

    def test_valid(self):
        cfg = self.load('[serve]\nport = 9000\n[profiles.chat]\nlanes = 2\ncontext = 8192\n')
        self.assertEqual(cfg, {"serve": {"port": 9000}, "profiles.chat": {"lanes": 2, "context": 8192}})

    def test_unknowns_are_errors(self):
        cases = (("[other]\na = 1\n", "unknown top-level"), ("[profiles.fast]\nlanes = 1\n", "unknown profile"),
                 ("[serve]\nlanes = 1\n", "unknown keys"), ("[profiles.agent]\nport = 1\n", "unknown keys"))
        for text, message in cases:
            with self.subTest(text=text), self.assertRaisesRegex(resolve.ResolveError, message):
                self.load(text)


if __name__ == "__main__":
    unittest.main()
