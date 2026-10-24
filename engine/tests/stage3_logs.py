#!/usr/bin/env python3
"""goethe logs index and tail."""

from __future__ import annotations

import json

from lib.goethe_test import main_exit, require_cli, run, temp_home


def main() -> None:
    goethe = require_cli("goethe")
    with temp_home(stub_python=True, tag="logs") as home:
        logs = home / "logs"
        logs.mkdir(exist_ok=True)
        inst = "inst-test1"
        logf = logs / f"{inst}.log"
        logf.write_text("line1\nline2\nline3\n", encoding="utf-8")
        (logs / "index.jsonl").write_text(
            json.dumps({"instance_id": inst, "model_id": "m/a", "log": str(logf), "ts": "t"}) + "\n",
            encoding="utf-8",
        )
        rr = run(
            [str(goethe), "logs", "--model", "m/a", "-n", "2"],
            env={"GOETHE_HOME": str(home)},
        )
        if "line2" not in rr.stdout or "line3" not in rr.stdout:
            raise RuntimeError(rr.stdout)
        legacy = logs / "local__x.log"
        legacy.write_text("legacy\n", encoding="utf-8")
        rr2 = run([str(goethe), "logs", "--model", "local/x", "-n", "5"], env={"GOETHE_HOME": str(home)})
        if "legacy" not in rr2.stdout:
            raise RuntimeError("legacy log path failed")
    print("stage3_logs ok")


if __name__ == "__main__":
    main_exit(main)
