import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from goethe import cli, engine, hw
from tests.fixtures import PRO_6000, RTX_3090, RTX_5090


def run(*argv, gpus=(RTX_5090,), env=None, config=None):
    """(exit code, stdout, stderr) of goethe with a fake GPU list and an empty home."""
    out, err = io.StringIO(), io.StringIO()
    with tempfile.TemporaryDirectory() as home:
        environ = {"HOME": home, "NVFP4_HOME": f"{home}/nvfp4", **(env or {})}
        if config:
            (Path(home) / ".config" / "goethe").mkdir(parents=True)
            (Path(home) / ".config" / "goethe" / "config.toml").write_text(config)
        probe = (list(gpus), None) if gpus else ([], "nvidia-smi not found")
        with mock.patch.dict(os.environ, environ, clear=True), mock.patch.object(hw, "probe", lambda *a: probe), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
    return code, out.getvalue(), err.getvalue()


class Surface(unittest.TestCase):
    def test_help_lists_exactly_the_implemented_commands(self):
        code, out, _ = run()
        self.assertEqual(code, 0)
        commands = out.split("positional arguments:")[1].split("options:")[0].splitlines()[2:]
        listed = [line.split()[0] for line in commands if line.strip()]
        self.assertEqual(listed, ["serve", "list", "doctor"])

    def test_unknown_command_exits_2(self):
        with self.assertRaises(SystemExit) as e, contextlib.redirect_stderr(io.StringIO()):
            cli.main(["frobnicate"])
        self.assertEqual(e.exception.code, 2)


class Serve(unittest.TestCase):
    def test_dry_run_prints_plan_commands_and_runs_nothing(self):
        with mock.patch("subprocess.run") as sub, mock.patch("os.execv") as ex:
            code, out, _ = run("serve", "--dry-run")
        self.assertEqual(code, 0)
        sub.assert_not_called()
        ex.assert_not_called()
        self.assertIn("RTX 5090", out)
        self.assertIn("tested", out)
        self.assertRegex(out, r"run\s+\S+/install\.sh")
        self.assertRegex(out, r"run\s+\S+/fetch_model\.sh")
        self.assertRegex(out, r"exec\s+\S+ninfer-serve \S+ --host 127\.0\.0\.1 --port 8080 --model-id qwen3\.8-27b")
        self.assertIn("--max-concurrency 4", out)

    def test_explain_says_where_each_value_came_from(self):
        _, out, _ = run("serve", "--explain", "--for", "chat", "--port", "9001", env={"GOETHE_LANES": "2"},
                        config='[serve]\nport = 7\n[profiles.chat]\ncontext = 8192\n')
        self.assertRegex(out, r"port\s+9001\s+flag")
        self.assertRegex(out, r"lanes\s+2\s+env")
        self.assertRegex(out, r"context\s+8192\s+config")
        self.assertRegex(out, r"kv_dtype\s+fp8\s+default")
        self.assertNotIn("exec", out)

    def test_passthrough_goes_to_the_engine(self):
        _, out, _ = run("serve", "--dry-run", "--", "--request-log-jsonl", "/tmp/r.jsonl")
        self.assertTrue(out.splitlines()[-1].endswith("--request-log-jsonl /tmp/r.jsonl"), out)

    def test_set_flag(self):
        _, out, _ = run("serve", "--explain", "--set", "lanes=2", "--set", "kv_dtype=nvfp4")
        self.assertRegex(out, r"lanes\s+2\s+flag")
        self.assertIn("warning  kv_dtype nvfp4", out)

    def test_public_host_hides_the_key(self):
        _, out, _ = run("serve", "--dry-run", "--host", "0.0.0.0")
        self.assertIn("--api-key '<hidden>'", out)
        self.assertRegex(out, r"api_key\s+<hidden>\s+generated")

    def test_errors_exit_1_with_a_message(self):
        for argv, gpus, message in ((["serve", "--dry-run", "--set", "lanes=abc"], (RTX_5090,), "not an integer"),
                                    (["serve", "--dry-run", "--set", "lanes"], (RTX_5090,), "KEY=VALUE"),
                                    (["serve", "--dry-run"], (RTX_3090,), "sm_120a"),
                                    (["serve", "--dry-run"], (), "nvidia-smi not found"),
                                    (["serve", "nope", "--dry-run"], (RTX_5090,), "unknown model")):
            with self.subTest(argv=argv):
                code, _, err = run(*argv, gpus=gpus)
                self.assertEqual(code, 1)
                self.assertIn(message, err)

    def test_bad_config_file_is_reported(self):
        code, _, err = run("serve", "--explain", config="[serve]\nnope = 1\n")
        self.assertEqual(code, 1)
        self.assertIn("unknown keys", err)

    def test_already_running_server_is_not_started_twice(self):
        with mock.patch.object(engine, "healthy", return_value=True), mock.patch("os.execv") as ex:
            code, out, _ = run("serve")
        self.assertEqual(code, 0)
        self.assertIn("already serving on http://127.0.0.1:8080", out)
        ex.assert_not_called()

    def test_serve_installs_then_replaces_itself_with_the_engine(self):
        with mock.patch.object(engine, "healthy", return_value=False), mock.patch("subprocess.run") as sub, \
                mock.patch("os.execv") as ex:
            code, out, _ = run("serve", "--for", "long")
        self.assertFalse(code)  # os.execv never returns for real
        self.assertEqual(sub.call_count, 2)  # install.sh, fetch_model.sh
        self.assertTrue(all(c.kwargs["env"]["NVFP4_HOME"] for c in sub.call_args_list))
        self.assertIn("serving qwen3.8-27b on http://127.0.0.1:8080 (1 lanes, 131072 context, long)", out)
        argv = ex.call_args.args[1]
        self.assertEqual(ex.call_args.args[0], argv[0])
        self.assertIn("131072", argv)

    def test_generated_api_key_location_is_printed_on_start(self):
        with mock.patch.object(engine, "healthy", return_value=False), mock.patch("subprocess.run"), \
                mock.patch("os.execv") as ex:
            _, out, _ = run("serve", "--host", "0.0.0.0")
        self.assertRegex(out, r"API key: \S+/goethe/api_key")
        self.assertIn("--api-key", ex.call_args.args[1])

    def test_engine_args_only_apply_to_serve(self):
        with self.assertRaises(SystemExit) as e, contextlib.redirect_stderr(io.StringIO()):
            cli.main(["list", "--", "--x"])
        self.assertEqual(e.exception.code, 2)

    def test_failed_install_step_exits_1(self):
        import subprocess
        failing = subprocess.CalledProcessError(2, ["/x/install.sh"])
        with mock.patch.object(engine, "healthy", return_value=False), \
                mock.patch("subprocess.run", side_effect=failing), mock.patch("os.execv") as ex:
            code, _, err = run("serve")
        self.assertEqual(code, 1)
        self.assertIn("/x/install.sh exited with 2", err)
        ex.assert_not_called()


class ListAndDoctor(unittest.TestCase):
    def test_list_text_and_json(self):
        _, out, _ = run("list", gpus=(PRO_6000,))
        self.assertRegex(out, r"qwen3-8-27b\s+Qwen3\.8 27B NVFP4\s+compatible\s+not installed")
        rows = json.loads(run("list", "--json", gpus=(RTX_3090,))[1])
        self.assertEqual((rows[0]["id"], rows[0]["status"], rows[0]["installed"]),
                         ("qwen3-8-27b", "unsupported", False))
        self.assertEqual(json.loads(run("list", "--json", gpus=())[1])[0]["status"], "no gpu")

    def test_doctor_exit_code_follows_failures(self):
        code, out, _ = run("doctor", gpus=())
        self.assertEqual(code, 1)
        self.assertRegex(out, r"fail\s+nvidia-smi\s+nvidia-smi not found")
        self.assertIn("fix:", out)

    def test_doctor_json(self):
        rows = json.loads(run("doctor", "--json", gpus=())[1])
        self.assertIn("nvidia-smi", [r["name"] for r in rows])


if __name__ == "__main__":
    unittest.main()
