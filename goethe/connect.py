"""Point a coding agent at the running server for one command, by environment variables only.

Nothing here edits a config file. Harnesses that can only be configured through a file are named in FILES, so the
command can say so instead of pretending. The variables come from each project's own docs; the checks against a live
server are in tests/test_connect.py.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

# These need a file or a GUI, which goethe connect does not touch.
FILES = {
    "continue": "~/.continue/config.yaml", "zed": "~/.config/zed/settings.json",
    "cursor": "the Cursor settings UI, and it needs a public URL", "jan": "the Jan UI",
    "roo": "the Roo Code settings import", "cline": "the Cline settings UI",
}

BASE = "http://127.0.0.1:{port}/v1"


def conforms(port: int, timeout: float = 10) -> list[str]:
    """Endpoints the harnesses need. Empty list means the server speaks all of them."""
    url = BASE.format(port=port)
    missing = []
    chat = json.dumps({"model": "qwen3.8-27b", "messages": [{"role": "user", "content": "hi"}],
                       "max_tokens": 1}).encode()
    responses = json.dumps({"model": "qwen3.8-27b", "input": "hi", "max_output_tokens": 1}).encode()

    def post(path: str, body: bytes) -> int:
        try:
            urllib.request.urlopen(urllib.request.Request(
                url + path, data=body, headers={"Content-Type": "application/json"}), timeout=timeout)
        except urllib.error.HTTPError as e:
            return e.code
        except OSError:
            return 0
        return 200

    if post("/chat/completions", chat) != 200:
        missing.append("/v1/chat/completions")
    if post("/responses", responses) != 200:
        missing.append("/v1/responses")
    if post("/messages", chat) != 200:
        missing.append("/v1/messages")
    return missing


def _anthropic(url: str, model: str, context: int) -> dict[str, str]:
    # background tasks use the Haiku alias, so every tier must name the served model or they hit Anthropic
    tiers = {f"ANTHROPIC_DEFAULT_{t}_MODEL": model for t in ("OPUS", "SONNET", "HAIKU")}
    return {"ANTHROPIC_BASE_URL": url.removesuffix("/v1"), "ANTHROPIC_AUTH_TOKEN": "goethe",
            "ANTHROPIC_API_KEY": "", "CLAUDE_CODE_SUBAGENT_MODEL": model,
            "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",  # without this the prefix cache is invalidated every turn
            "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context),
            "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(min(context // 4, 8192)),
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", **tiers}


def kilo_config(url: str, model: str) -> str:
    return json.dumps({"model": f"goethe/{model}",
                       "provider": {"goethe": {"npm": "@ai-sdk/openai-compatible",
                                               "options": {"baseURL": url, "apiKey": "goethe"},
                                               "models": {model: {}}}}})


def command(harness: str, model: str, port: int, context: int = 100000) -> tuple[dict[str, str], list[str]]:
    """(environment, argv) for `harness`. Kilo is the exception: the caller writes kilo_config()."""
    url, m = BASE.format(port=port), model
    if harness == "claude":
        # NInfer rejects Claude Code's defaults: thinking.display and reasoning effort "high"
        env = {**_anthropic(url, m, context), "CLAUDE_CODE_DISABLE_THINKING": "1"}
        return env, ["claude", "--effort", "medium"]
    if harness == "codex":
        # codex speaks the Responses API only, and takes its provider as -c flags, so nothing is written
        # web_search is on by default and NInfer rejects that tool type, so the request never starts
        return {}, ["codex", "-c", "model_provider=goethe",
                    "-c", 'model_providers.goethe.name="goethe"',
                    "-c", f'model_providers.goethe.base_url="{url}"',
                    "-c", 'model_providers.goethe.wire_api="responses"',
                    "-c", 'web_search="disabled"', "-m", m]
    if harness == "aider":
        return {"OPENAI_API_BASE": url, "OPENAI_API_KEY": "goethe"}, ["aider", "--model", f"openai/{m}"]
    if harness == "opencode":
        config = json.dumps({"provider": {"goethe": {"npm": "@ai-sdk/openai-compatible",
                                                     "options": {"baseURL": url}, "models": {m: {}}}},
                             "model": f"goethe/{m}"})
        return {"OPENCODE_CONFIG_CONTENT": config}, ["opencode"]
    if harness == "qwen":
        return {"OPENAI_BASE_URL": url, "OPENAI_API_KEY": "goethe", "OPENAI_MODEL": m}, \
            ["qwen", "--auth-type", "openai"]
    if harness == "goose":
        return {"GOOSE_PROVIDER": "openai", "GOOSE_MODEL": m, "OPENAI_HOST": url.removesuffix("/v1"),
                "OPENAI_BASE_PATH": "v1/chat/completions", "OPENAI_API_KEY": "goethe"}, ["goose"]
    if harness == "kilo":
        # current kilo has no config env var; the caller writes kilo_config() and points XDG_CONFIG_HOME at it
        return {}, ["kilo"]
    if harness == "mini-swe-agent":
        # MSWEA_CONFIGURED skips the first-run prompt, which otherwise blocks a headless run
        return {"MSWEA_MODEL_NAME": f"openai/{m}", "MSWEA_CONFIGURED": "true",
                "OPENAI_API_BASE": url, "OPENAI_API_KEY": "goethe",
                "MSWEA_COST_TRACKING": "ignore_errors"}, ["mini"]
    if harness in FILES:
        raise SystemExit(f"{harness} is configured through {FILES[harness]}, which goethe connect does not change")
    known = "claude, codex, aider, opencode, qwen, goose, kilo, mini-swe-agent"
    raise SystemExit(f"unknown harness {harness!r}; known: {known}")
