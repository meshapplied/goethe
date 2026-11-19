"""Run a command, time its output, print one line of shell assignments: exit code, seconds, lines, longest silence."""

import argparse
import os
import selectors
import signal
import subprocess
import sys
import threading
import time


def pump(proc, sinks, stamps, trace=None, start=0.0):
    """Copy each stream to its file as it arrives, counting lines (newlines) and noting when any bytes arrived."""
    sel = selectors.DefaultSelector()
    counts = {}
    for name, stream in (("out", proc.stdout), ("err", proc.stderr)):
        sel.register(stream, selectors.EVENT_READ, name)
        counts[name] = 0
    while sel.get_map():
        for key, _ in sel.select(timeout=0.2):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                sel.unregister(key.fileobj)
                continue
            sinks[key.data].write(chunk)
            sinks[key.data].flush()
            counts[key.data] += chunk.count(b"\n")
            stamps.append(time.monotonic())
            for line in chunk.decode(errors="replace").replace("\r", "\n").splitlines():
                if trace and line.strip():
                    trace.write(f"{stamps[-1] - start:8.1f} {key.data} {line.rstrip()}\n")
    return counts


def main(argv):
    """Parse the options, run the command to its end (or its limit) and print the record."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=0)
    ap.add_argument("--interrupt", type=float, default=0, help="send SIGINT to the command after this many seconds")
    ap.add_argument("--trace", help="write every output line with its time since the start to this file")
    ap.add_argument("out")
    ap.add_argument("err")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    start = time.monotonic()
    pipes = {"stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    proc = subprocess.Popen(cmd, start_new_session=True, **pipes)
    for after, sig in ((a.interrupt, signal.SIGINT), (a.timeout, signal.SIGKILL)):
        if after > 0:
            timer = threading.Timer(after, lambda s=sig: os.killpg(proc.pid, s) if proc.poll() is None else None)
            timer.daemon = True
            timer.start()
    stamps = [start]
    trace = open(a.trace, "w") if a.trace else None
    with open(a.out, "wb") as out, open(a.err, "wb") as err:
        counts = pump(proc, {"out": out, "err": err}, stamps, trace, start)
    if trace:
        trace.close()
    rc = proc.wait()
    end = time.monotonic()
    stamps.append(end)
    gap = max(b - a_ for a_, b in zip(stamps, stamps[1:], strict=False))
    print(f"U_RC={rc} U_WALL={end - start:.1f} U_OUT={counts['out']} U_ERR={counts['err']} U_GAP={gap:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
