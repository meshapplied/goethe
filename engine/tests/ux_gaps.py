"""Print the silences of a ux_run.py --trace file: each gap over a limit, with the line before and the line after."""

import sys
from pathlib import Path


def gaps(lines, limit):
    """(seconds, line before, line after) for every pause of at least `limit` seconds, the end of the run included."""
    rows = [(float(x.split(None, 1)[0]), x.split(None, 1)[1].rstrip()) for x in lines if x.strip()]
    out = []
    for (t0, before), (t1, after) in zip(rows, rows[1:], strict=False):
        if t1 - t0 >= limit:
            out.append((t1 - t0, before, after))
    return out


def main(argv):
    """ux_gaps.py TRACE [LIMIT_SECONDS]: the table, longest first, and the line count."""
    lines = Path(argv[0]).read_text().splitlines()
    limit = float(argv[1]) if len(argv) > 1 else 3.0
    print(f"{len(lines)} lines, {len([x for x in lines if ' out ' in x[:20] or ' err ' in x[:20]])} counted")
    for secs, before, after in sorted(gaps(lines, limit), reverse=True):
        print(f"{secs:6.1f} s  after: {before[:90]}")
        print(f"{'':9}before: {after[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
