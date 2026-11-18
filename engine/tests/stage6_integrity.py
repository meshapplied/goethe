#!/usr/bin/env python3
"""Stage 6 CDN signature and checksum verification gate (packaging keygen/sign scripts and install.sh)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from lib.goethe_test import (
    GOETHE_PKG,
    Proc,
    free_port,
    http_get,
    main_exit,
    run_temp_prefix,
    spawn,
    temp_home,
    wait_until,
)

ARCH = "aarch64" if platform.machine() in ("aarch64", "arm64") else platform.machine()
PACKAGING = GOETHE_PKG / "packaging"
STUB = b"""#!/bin/sh
[ "$1" = --version ] && { echo "goethe 0.0.0"; exit 0; }
echo "$*" >"$GOETHE_HOME/setup.ran"
echo SETUP_OK
"""


def _sh(*cmd: str | Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=60)


def _mini_cdn(cdn: Path, pub: Path) -> None:
    """A package with a stub goethe that records how install.sh calls goethe setup; install.sh embeds pub."""
    cdn.mkdir(parents=True)
    bindir = cdn / "bin" / f"linux-{ARCH}"
    bindir.mkdir(parents=True)
    for name in ("goethe", "goethed", "nginx"):
        (bindir / name).write_bytes(STUB)
        (bindir / name).chmod(0o755)
    rr = _sh("bash", PACKAGING / "subst_install_pubkey.sh", GOETHE_PKG / "cdn" / "install.sh", pub, cdn / "install.sh")
    if rr.returncode != 0:
        raise RuntimeError(f"subst_install_pubkey failed: {rr.stderr}")


def _install(install_sh: Path, pkg_url: str, home: Path, extra: dict[str, str] | None = None) -> str:
    """Run install.sh against pkg_url into home like curl | bash (stdin closed); rc, output and the setup call."""
    env = {
        "GOETHE_HOME": str(home / "g"), "GOETHE_PKG_URL": pkg_url,
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), **(extra or {}),
    }
    rr = subprocess.run(["bash", str(install_sh)], env=env, capture_output=True, text=True, timeout=120,
                        stdin=subprocess.DEVNULL)
    ran = home / "g" / "setup.ran"
    setup = ran.read_text().strip() if ran.exists() else "not run"
    return f"rc={rr.returncode}\n{rr.stdout}{rr.stderr}\nsetup: {setup}"


def _variant(root: Path, name: str, cdn: Path) -> Path:
    """Copy of the signed CDN under root/name for a negative case."""
    out = root / name
    shutil.copytree(cdn, out)
    return out


def _answers(url: str) -> bool:
    """True once the local HTTP server serves url."""
    try:
        return http_get(url, timeout=2)[0] == 200
    except OSError:
        return False


def main() -> None:
    """Exercise keygen, sign_cdn.sh and the signed-CDN verification paths in install.sh."""
    root = Path(f"/tmp/{run_temp_prefix('integrity')}root")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    srv: Proc | None = None
    try:
        keys, other = root / "keys", root / "other-keys"
        for d in (keys, other):
            if _sh("bash", PACKAGING / "keygen.sh", d).returncode != 0:
                raise RuntimeError("keygen.sh failed")
        if _sh("bash", PACKAGING / "keygen.sh", keys).returncode == 0:
            raise RuntimeError("keygen.sh overwrote an existing key")
        if (keys / "release.key").stat().st_mode & 0o077:
            raise RuntimeError("release.key is readable by group or others")
        cdn = root / "cdn"
        _mini_cdn(cdn, keys / "release_pub.pem")
        wrong = _variant(root, "cdn-wrongkey", cdn)
        rr = _sh("bash", PACKAGING / "sign_cdn.sh", wrong, other / "release.key")
        if rr.returncode == 0 or "does not match" not in rr.stderr:
            raise RuntimeError(f"sign_cdn.sh accepted a key that differs from install.sh's: {rr.stderr}")
        rr = _sh("bash", PACKAGING / "sign_cdn.sh", cdn, keys / "release.key")
        if rr.returncode != 0:
            raise RuntimeError(f"sign_cdn.sh failed: {rr.stderr}")
        tampered = _variant(root, "cdn-tamper", cdn)
        with open(tampered / "bin" / f"linux-{ARCH}" / "goethe", "ab") as f:
            f.write(b"# changed\n")
        badsums = _variant(root, "cdn-badsums", cdn)
        text = (badsums / "SHA256SUMS").read_text()
        (badsums / "SHA256SUMS").write_text(text.replace("a", "b", 1))
        nokey = _variant(root, "cdn-nokey", cdn)
        shutil.copy2(GOETHE_PKG / "cdn" / "install.sh", nokey / "install.sh")
        ver = (cdn / "VERSION").read_text().strip()
        versioned = _variant(root, "cdn-versioned", cdn)
        shutil.copytree(cdn, versioned / "v" / ver)
        older = root / "cdn-older"
        shutil.copytree(cdn, older)
        old_bin = older / "bin" / f"linux-{ARCH}" / "goethe"
        old_bin.write_bytes(b'#!/bin/sh\n[ "$1" = --version ] && { echo "goethe 0.3.2"; exit 0; }\nexit 2\n')
        template = root / "old-install.sh"
        template.write_text("#!/usr/bin/env bash\nGOETHE_RELEASE_PUB_PEM='__GOETHE_RELEASE_PUB_PEM__'\n"
                            "echo OLD_INSTALLER \"$GOETHE_PKG_URL\"\n")
        _sh("bash", PACKAGING / "subst_install_pubkey.sh", template, keys / "release_pub.pem", older / "install.sh")
        rr = _sh("bash", PACKAGING / "sign_cdn.sh", older, keys / "release.key")
        shutil.copy2(cdn / "install.sh", older / "new-install.sh")
        if rr.returncode != 0:
            raise RuntimeError(f"sign_cdn.sh failed for the older release: {rr.stderr}")
        port = free_port()
        srv = spawn(
            [os.environ.get("GOETHE_PYTHON", "python3"), "-m", "http.server", str(port), "--bind", "127.0.0.1"],
            cwd=root,
            label="integrity-cdn",
        )
        base = f"http://127.0.0.1:{port}"
        if not wait_until(lambda: _answers(f"{base}/"), timeout=30):
            raise RuntimeError(f"local CDN server on port {port} did not come up")
        cases = [
            ("valid", cdn, {}, ["rc=0", "SETUP_OK", f"setup: setup --pkg-url {base}/cdn --from {base}/cdn"]),
            ("tampered", tampered, {}, ["does not match the signed package list", "setup: not run"]),
            ("bad sums", badsums, {}, ["not signed by the Goethe release key", "setup: not run"]),
            ("no key", nokey, {}, ["has no release key", "setup: not run"]),
            ("newest release folder", versioned, {}, ["rc=0", f"--from {base}/cdn-versioned/v/{ver}"]),
            ("pinned", versioned, {"GOETHE_VERSION": f"v{ver}"},
             ["rc=0", f"--pkg-url {base}/cdn-versioned --from {base}/cdn-versioned/v/{ver}"]),
            ("unknown version", versioned, {"GOETHE_VERSION": "1.2.3"},
             ["is 1.2.3 a published version?", "setup: not run"]),
            ("bad version", cdn, {"GOETHE_VERSION": "latest; rm"}, ["is not a version like", "setup: not run"]),
            ("release before goethe setup", older, {},
             ["rc=0", "predates goethe setup", f"OLD_INSTALLER {base}/cdn-older"]),
        ]
        with temp_home(stub_python=False, tag="integrity", pyenv=False) as home:
            for name, tree, extra, wants in cases:
                out = ""
                for _ in range(20):
                    script = tree / "new-install.sh" if (tree / "new-install.sh").exists() else tree / "install.sh"
                    out = _install(script, f"{base}/{tree.name}", home / name.replace(" ", "-"), extra)
                    if "Connection refused" not in out and "Couldn't connect" not in out:
                        break
                missing = [w for w in wants if w not in out]
                if missing or ("rc=0" not in wants and out.startswith("rc=0")):
                    raise RuntimeError(f"{name}: expected {missing}: {out[-600:]}")
    finally:
        if srv is not None:
            srv.stop()
        shutil.rmtree(root, ignore_errors=True)
    print("stage6_integrity ok")


if __name__ == "__main__":
    main_exit(main)
