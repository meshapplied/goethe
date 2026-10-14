import contextlib
import io
import os
import unittest
from unittest import mock

from goethe import cli, connect

PORT = 8080
URL = f"http://127.0.0.1:{PORT}/v1"


class Commands(unittest.TestCase):
    def test_each_launch_harness_points_at_the_server(self):
        env, argv = connect.command("claude", "qwen3.8-27b", PORT, context=212992)
        self.assertEqual(env["ANTHROPIC_BASE_URL"], f"http://127.0.0.1:{PORT}")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "")
        for tier in ("OPUS", "SONNET", "HAIKU"):
            self.assertEqual(env[f"ANTHROPIC_DEFAULT_{tier}_MODEL"], "qwen3.8-27b")
        self.assertEqual(env["CLAUDE_CODE_SUBAGENT_MODEL"], "qwen3.8-27b")
        self.assertEqual(env["CLAUDE_CODE_ATTRIBUTION_HEADER"], "0")
        self.assertEqual(env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"], "212992")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_THINKING"], "1")
        self.assertEqual(argv, ["claude", "--effort", "medium"])

        env, argv = connect.command("codex", "qwen3.8-27b", PORT)
        self.assertIn(f'model_providers.goethe.base_url="{URL}"', argv)
        self.assertIn('model_providers.goethe.wire_api="responses"', argv)
        self.assertIn('web_search="disabled"', argv)
        self.assertEqual(env, {})

        env, argv = connect.command("aider", "qwen3.8-27b", PORT)
        self.assertEqual(env["OPENAI_API_BASE"], URL)
        self.assertIn("openai/qwen3.8-27b", argv)

        env, argv = connect.command("opencode", "qwen3.8-27b", PORT)
        self.assertIn(URL, env["OPENCODE_CONFIG_CONTENT"])
        self.assertIn("goethe/qwen3.8-27b", env["OPENCODE_CONFIG_CONTENT"])

        for harness, base in (("qwen", "OPENAI_BASE_URL"), ("goose", "OPENAI_HOST")):
            env, _ = connect.command(harness, "qwen3.8-27b", PORT)
            self.assertTrue(env[base].startswith("http://127.0.0.1"), harness)

        self.assertIn(URL, connect.kilo_config(URL, "qwen3.8-27b"))
        env, argv = connect.command("mini-swe-agent", "qwen3.8-27b", PORT)
        self.assertEqual(env["OPENAI_API_BASE"], URL)
        self.assertEqual(env["MSWEA_CONFIGURED"], "true")
        self.assertEqual(argv, ["mini"])

    def test_file_harness_is_refused(self):
        for harness in ("cursor", "continue", "zed", "cline"):
            with self.subTest(harness=harness), self.assertRaises(SystemExit):
                connect.command(harness, "qwen3.8-27b", PORT)

    def test_unknown_harness_is_refused(self):
        with self.assertRaises(SystemExit):
            connect.command("emacs", "qwen3.8-27b", PORT)


class Print(unittest.TestCase):
    def test_print_runs_nothing_and_shows_the_command(self):
        out = io.StringIO()
        with mock.patch("os.execve") as ex, contextlib.redirect_stdout(out):
            code = cli.main(["connect", "aider", "--print", "--", "--message", "hi"])
        self.assertEqual(code, 0)
        ex.assert_not_called()
        self.assertIn("OPENAI_API_BASE=http://127.0.0.1:8080/v1", out.getvalue())
        self.assertIn("aider --model openai/qwen3.8-27b --message hi", out.getvalue())

    def test_refuses_without_a_server(self):
        with mock.patch("goethe.engine.healthy", return_value=False), mock.patch("os.execve") as ex:
            code = cli.main(["connect", "claude"])
        self.assertEqual(code, 1)
        ex.assert_not_called()

    def test_missing_harness_is_an_error(self):
        with mock.patch("goethe.engine.healthy", return_value=True), \
                mock.patch("goethe.connect.conforms", return_value=[]), \
                mock.patch("shutil.which", return_value=None), mock.patch("os.execve") as ex:
            code = cli.main(["connect", "claude"])
        self.assertEqual(code, 1)
        ex.assert_not_called()

    def test_execs_with_the_harness_environment(self):
        with mock.patch("goethe.engine.healthy", return_value=True), \
                mock.patch("goethe.connect.conforms", return_value=[]), \
                mock.patch("shutil.which", return_value="/usr/bin/claude"), \
                mock.patch.dict(os.environ, {"HOME": "/h"}, clear=True), mock.patch("os.execve") as ex:
            cli.main(["connect", "claude"])
        argv, env = ex.call_args.args[1], ex.call_args.args[2]
        self.assertEqual(argv, ["claude", "--effort", "medium"])
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:8080")
        self.assertEqual(env["HOME"], "/h")


if __name__ == "__main__":
    unittest.main()
