"""Mechanical style linter for the goethe package tree."""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/ternary-flash/goethe-release/engine")
SKIP_DIRS = {"target", "dist", "build", "__pycache__", ".git", "node_modules", "experimental"}
ARCHIVED_TESTS = {
    "tests/stage1_archs.sh",
    "tests/stage1_hook.sh",
    "tests/stage1_ops.py",
}
SRC = {".py", ".rs", ".cu", ".cuh", ".cpp", ".h", ".sh", ".toml", ".conf", ".sql", ".tmpl", ".service"}
BAD_CHARS = re.compile("[\u2013\u2014\u2015\u2212]|[\U0001F000-\U0001FFFF]|[\u2600-\u27BF]|[\u2B00-\u2BFF]")
CTRL = re.compile(r"^\s*(if|for|while|switch|else|do|catch|try|namespace|struct|class|enum|union|match|loop|impl|trait|mod|unsafe)\b")
problems: list[str] = []


def flag(path: Path, line: int, msg: str) -> None:
    """Record a lint problem."""
    problems.append(f"{path.relative_to(ROOT)}:{line}: {msg}")


def lint_python(path: Path, text: str) -> None:
    """Lint a Python source file."""
    tree = ast.parse(text)
    allowed = set()
    body = tree.body
    if not (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(getattr(body[0], "value", None), ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        flag(path, 1, "missing one-line module docstring")
    else:
        ds = body[0]
        allowed.add(ds.lineno)
        if ds.end_lineno != ds.lineno:
            flag(path, ds.lineno, "module docstring must be one line")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if (
                b
                and isinstance(b[0], ast.Expr)
                and isinstance(getattr(b[0], "value", None), ast.Constant)
                and isinstance(b[0].value.value, str)
            ):
                if b[0].end_lineno != b[0].lineno:
                    flag(path, b[0].lineno, "docstring must be one line")
                allowed.add(b[0].lineno)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
            and node.lineno not in allowed
        ):
            flag(path, node.lineno, "stray string statement used as comment")
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            if tok.start[0] == 1 and tok.string.startswith("#!"):
                continue
            flag(path, tok.start[0], f"comment: {tok.string[:50]}")


def strip_strings(line: str) -> str:
    """Remove string literals for comment scanning."""
    out, i, q = [], 0, None
    while i < len(line):
        c = line[i]
        if q:
            if c == "\\":
                i += 2
                continue
            if c == q:
                q = None
            i += 1
            continue
        if c in "\"'":
            q = c
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def lint_clike(path: Path, lines: list[str], marker: str) -> None:
    """Lint C-like or shell sources."""
    n = len(lines)
    header_ok = False
    start = 0
    if lines and lines[0].startswith("#!"):
        start = 1
    if start < n and lines[start].strip().startswith(marker):
        header_ok = True
        if start + 1 < n and lines[start + 1].strip().startswith(marker):
            flag(path, start + 2, "header must be one line")
    if not header_ok:
        flag(path, start + 1, "missing one-line header comment")
    fn_open = None
    pending_sig = None
    for i, raw in enumerate(lines):
        s = raw.strip()
        code = strip_strings(raw)
        is_comment_line = (i == 0 and s.startswith("#!")) is False and (
            s.startswith(marker) or s.startswith("/*") or (marker == "//" and s.startswith("*"))
        )
        if marker == "//" and "/*" in code:
            flag(path, i + 1, "block comment")
        if is_comment_line:
            if i == start and header_ok:
                continue
            if fn_open is not None and i == fn_open + 1:
                continue
            flag(path, i + 1, f"comment: {s[:50]}")
            continue
        idx = code.find(marker)
        if idx > 0 and marker == "//" and not code[:idx].rstrip().endswith(":") and "http" not in code[:idx + 8]:
            flag(path, i + 1, f"inline comment: {raw.strip()[-50:]}")
        if marker == "#" and re.search(r"\s#\s", code):
            flag(path, i + 1, f"inline comment: {raw.strip()[-50:]}")
        text = code.strip()
        if not text:
            continue
        if marker == "//":
            if (
                pending_sig is None
                and "(" in text
                and not CTRL.match(text)
                and not text.startswith(("return", "}", ")", ".", "="))
                and "=" not in text.split("(")[0]
            ):
                pending_sig = i
            if pending_sig is not None and text.endswith(";"):
                pending_sig = None
            if text.endswith("{") and pending_sig is not None:
                fn_open = i
                pending_sig = None
            elif text.endswith("{"):
                fn_open = None
            if text in ("}", "};"):
                pending_sig = None


def lint_shell_like(path: Path, lines: list[str]) -> None:
    """Lint shell scripts."""
    lint_clike(path, lines, "#")


def main() -> int:
    """Walk goethe/ and print style violations."""
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if not path.is_file() or any(p in SKIP_DIRS for p in rel.parts):
            continue
        if str(rel).replace("\\", "/") in ARCHIVED_TESTS:
            continue
        if path.suffix not in SRC and path.name not in {"Dockerfile"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if BAD_CHARS.search(line):
                flag(path, i, "emoji or dash character")
        lines = text.splitlines()
        if path.suffix == ".py":
            lint_python(path, text)
        elif path.suffix in {".rs", ".cu", ".cuh", ".cpp", ".h"}:
            lint_clike(path, lines, "//")
        elif path.suffix in {".sh", ".toml", ".conf", ".service", ".tmpl"} or path.name == "Dockerfile":
            lint_shell_like(path, lines)
        elif path.suffix == ".sql":
            lint_clike(path, lines, "--")
    for other in sorted(ROOT.rglob("*")):
        if other.is_file() and other.suffix in {".md", ".patch", ".json", ".lock", ".txt"} and not any(
            p in SKIP_DIRS for p in other.relative_to(ROOT).parts
        ):
            for i, line in enumerate(other.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if other.suffix == ".patch" and not (line.startswith("+") and not line.startswith("+++")):
                    continue
                if BAD_CHARS.search(line):
                    flag(other, i, "emoji or dash character")
    print("\n".join(problems) if problems else "lint clean")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
