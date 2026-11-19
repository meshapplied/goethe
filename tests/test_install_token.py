"""install.sh sends the private-package token to curl through a config pipe, never on its command line."""

import http.server
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from tests.fixtures import REPO

INSTALL = REPO / "goethe" / "cdn" / "install.sh"
TOKEN = 'tok"en\\with-quotes'


def functions():
    text = INSTALL.read_text()
    names = ("auth_conf", "get")
    found = [re.search(rf"^{name}\(\) \{{.*?^\}}", text, re.S | re.M) for name in names]
    return "\n".join(m.group(0) for m in found if m)


@unittest.skipUnless(shutil.which("curl") and Path("/proc/self/cmdline").exists(), "needs curl and /proc")
class PackageToken(unittest.TestCase):
    def test_the_token_reaches_the_server_and_never_the_process_list(self):
        seen, hold, arrived = {}, threading.Event(), threading.Event()

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                seen["auth"] = self.headers.get("Authorization")
                arrived.set()
                hold.wait(10)
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        with tempfile.TemporaryDirectory() as d:
            script = functions() + '\nget SHA256SUMS "$OUT"\n'
            base = f"http://127.0.0.1:{server.server_port}"
            env = {**os.environ, "GOETHE_PKG_TOKEN": TOKEN, "BASE": base, "OUT": f"{d}/out"}
            proc = subprocess.Popen(["bash", "-c", script], env=env)
            self.assertTrue(arrived.wait(10), "curl reached the server")
            files = [Path("/proc") / p / "cmdline" for p in os.listdir("/proc") if p.isdigit()]
            cmdlines = [f.read_bytes() for f in files if f.exists()]
            curls = [c for c in cmdlines if b"curl" in c and f":{server.server_port}".encode() in c]
            hold.set()
            proc.wait(10)
            server.shutdown()
        self.assertTrue(curls, "the running curl was found")
        self.assertFalse([c for c in curls if b"tok" in c or b"Authorization" in c], "no token on curl's command line")
        self.assertEqual(seen["auth"], f"Bearer {TOKEN}")


if __name__ == "__main__":
    unittest.main()
