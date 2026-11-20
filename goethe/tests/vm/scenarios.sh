#!/usr/bin/env bash
# Inside one fresh VM as a sudo user: curl | bash installs (user and root), reinstall, upgrade with goethed running, a tampered package, a pinned version, the first model install and uninstall.

set -uo pipefail

DISTRO="$1"
VERSION="$2"
CDN="http://10.0.2.2:8999"
export GOETHE_RELEASE_PUB=/tmp/test_pub.pem GOETHE_LOGIN=0 GOETHE_ALLOW_HTTP=1
G="${HOME}/.goethe/bin/goethe"
fails=0

pass() { echo "PASS $*"; }
fail() { echo "FAIL $*"; fails=$((fails + 1)); }
check() { local name="$1"; shift; if "$@"; then pass "${name}"; else fail "${name}"; fi; }
has() { grep -q -- "$1" "$2"; }
sum_of() { curl -fsSL "${CDN}/$1/SHA256SUMS" | awk -v p="bin/linux-x86_64/$2" '$2 == p {print $1}'; }
sha() { sha256sum "$1" | awk '{print $1}'; }
health() { local c; c="$(curl -sk -o /dev/null -w '%{http_code}' https://127.0.0.1:8443/health)"; [[ -n "${c}" && "${c}" != 000 ]]; }

install_as_user() {
  # $1 package, the rest extra env; output in /tmp/out.log, rc returned.
  local pkg="$1"
  shift
  curl -fsSL "${CDN}/${pkg}/install.sh" | env GOETHE_PKG_URL="${CDN}/${pkg}" "$@" bash >/tmp/out.log 2>&1
}

install_as_root() {
  local pkg="$1"
  shift
  curl -fsSL "${CDN}/${pkg}/install.sh" | sudo env GOETHE_PKG_URL="${CDN}/${pkg}" GOETHE_RELEASE_PUB="${GOETHE_RELEASE_PUB}" GOETHE_LOGIN=0 GOETHE_ALLOW_HTTP=1 "$@" bash >/tmp/out.log 2>&1
}

show() { sed 's/^/    | /' /tmp/out.log | tail -n "${1:-12}"; }

echo "--- $(. /etc/os-release && echo "${PRETTY_NAME}"), glibc $(getconf GNU_LIBC_VERSION | awk '{print $2}')$(command -v getenforce >/dev/null && echo ", SELinux $(getenforce)")"

if [[ "${DISTRO}" == debian ]]; then
  echo "== no openssl: a user is told how to get it, root gets it installed"
  sudo env DEBIAN_FRONTEND=noninteractive apt-get remove -y -qq openssl >/dev/null 2>&1
  if command -v openssl >/dev/null; then
    fail "openssl could not be removed for the test"
  else
    install_as_user a
    check "a user without openssl is told what to install" has "missing openssl" /tmp/out.log
    check "and nothing was installed" test ! -e "${HOME}/.goethe/bin/goethe"
  fi
fi

echo "== root install (systemd)"
if install_as_root a; then pass "root install"; else fail "root install"; show; fi
check "root's home is /var/lib/goethe" sudo test -x /var/lib/goethe/bin/goethe
check "goethe and goethed are linked in /usr/local/bin" test -L /usr/local/bin/goethe -a -L /usr/local/bin/goethed
command -v openssl >/dev/null && pass "openssl is there for root" || fail "openssl is there for root"
check "goethe nginx, not the distribution's" sudo test -x /var/lib/goethe/nginx-root/usr/sbin/nginx
check "no system nginx was installed" bash -c '! command -v nginx >/dev/null && ! test -x /usr/sbin/nginx'
check "no Python environment yet" sudo test ! -e /var/lib/goethe/py-env
sudo /usr/local/bin/goethe up >/tmp/up.log 2>&1 || { fail "root goethe up"; sed 's/^/    | /' /tmp/up.log | tail -8; }
check "systemd runs goethed" bash -c '[[ "$(systemctl is-active goethed 2>/dev/null)" == active ]]'
[[ "$(systemctl is-active goethed 2>/dev/null)" == active ]] || {
  systemctl status goethed --no-pager 2>&1 | tail -n 6 | sed 's/^/    | /'
  command -v ausearch >/dev/null && sudo ausearch -m avc -ts recent 2>/dev/null | grep -o 'denied.*' | sort -u | head -n 4 | sed 's/^/    | avc /'
  ls -Z /var/lib/goethe/bin/goethed 2>/dev/null | sed 's/^/    | /'
}
for _ in $(seq 1 30); do health && break; sleep 1; done
check "root's gateway answers" health
health || {
  sudo tail -n 5 /var/lib/goethe/logs/gateway.log /var/lib/goethe/logs/goethed.log 2>&1 | sed 's/^/    | /'
  command -v ausearch >/dev/null && sudo ausearch -m avc -ts recent 2>/dev/null | grep -o 'denied.*' | sort -u | head -n 5 | sed 's/^/    | avc /'
  ps -eZ 2>/dev/null | grep -E "goethed|nginx" | head -n 3 | sed 's/^/    | /'
}
sudo /usr/local/bin/goethe uninstall --yes >/tmp/un.log 2>&1 || { fail "root uninstall"; sed 's/^/    | /' /tmp/un.log | tail -5; }
check "root uninstall removed the links" test ! -e /usr/local/bin/goethe -a ! -e /usr/local/bin/goethed
check "root uninstall removed the unit" bash -c '! systemctl is-enabled goethed >/dev/null 2>&1'

echo "== user install, no terminal"
if install_as_user a; then pass "user install"; else fail "user install"; show; fi
check "it ends with Goethe install complete" has "Goethe install complete" /tmp/out.log
check "goethe runs" "${G}" --version
check "goethe nginx in the home" test -x "${HOME}/.goethe/nginx-root/usr/sbin/nginx"
check "the gateway certificate exists" test -s "${HOME}/.goethe/nginx/certs/server.crt"
check "no Python environment and no uv yet" test ! -e "${HOME}/.goethe/py-env" -a ! -e "${HOME}/.goethe/bin/uv"
check "goethe and goethed are linked in ~/.local/bin" test -L "${HOME}/.local/bin/goethe" -a -L "${HOME}/.local/bin/goethed"
check "without a terminal the PATH line is shown, not written" bash -c 'grep -q "To run goethe by name, add this line" /tmp/out.log && ! grep -q "added by goethe setup" ~/.bashrc'
check "goethe.env keeps the package host" grep -qx "GOETHE_PKG_URL=${CDN}/a" "${HOME}/.goethe/goethe.env"
"${G}" doctor --fix >/tmp/fix.log 2>&1
check "doctor --fix finds nothing to fix without a GPU" has "nothing to fix" /tmp/fix.log

echo "== goethe up and the gateway"
"${G}" up --no-boot >/tmp/up.log 2>&1 || { fail "goethe up"; sed 's/^/    | /' /tmp/up.log | tail -8; }
for _ in $(seq 1 30); do health && break; sleep 1; done
check "the gateway answers" health || true
health || { "${G}" status 2>&1 | sed 's/^/    | /' | tail -6; tail -n 8 ~/.goethe/logs/goethed.log ~/.goethe/logs/gateway.log 2>/dev/null | sed 's/^/    | /'; }
pid="$(pgrep -u "$(id -u)" -f "${HOME}/.goethe/bin/goethed" | head -n1)"

echo "== the same install again"
if install_as_user a; then pass "reinstall"; else fail "reinstall"; show; fi
check "it replaces nothing" has "already installed; nothing to replace" /tmp/out.log
check "goethed kept running" test "$(pgrep -u "$(id -u)" -f "${HOME}/.goethe/bin/goethed" | head -n1)" = "${pid}"

echo "== upgrade to another build while goethed runs"
if install_as_user b; then pass "upgrade"; else fail "upgrade"; show 20; fi
new="$(pgrep -u "$(id -u)" -f "${HOME}/.goethe/bin/goethed" | head -n1)"
check "a new goethed runs" test -n "${new}" -a "${new}" != "${pid}"
check "it is the new build" test "$(sha "/proc/${new}/exe" 2>/dev/null)" = "$(sum_of b goethed)"
check "goethe is the new build" test "$(sha "${HOME}/.goethe/bin/goethe")" = "$(sum_of b goethe)"
for _ in $(seq 1 30); do health && break; sleep 1; done
check "the gateway answers after the upgrade" health

echo "== a tampered package"
if install_as_user t; then fail "a tampered goethe is refused"; show; else pass "a tampered goethe is refused"; fi
check "it says why" has "does not match the signed package list" /tmp/out.log
check "the install is untouched" test "$(sha "${HOME}/.goethe/bin/goethe")" = "$(sum_of b goethe)"

echo "== plain http to a host that is not this machine"
if install_as_user a GOETHE_ALLOW_HTTP=0; then fail "plain http is refused"; show; else pass "plain http is refused"; fi
check "it says why" has "is plain http" /tmp/out.log

echo "== a pinned version"
if install_as_user a GOETHE_VERSION="${VERSION}"; then pass "pinned install"; else fail "pinned install"; show; fi
check "it installs that release's build" test "$(sha "${HOME}/.goethe/bin/goethed")" = "$(sum_of a goethed)"
check "goethe.env still names the package, not the release folder" grep -qx "GOETHE_PKG_URL=${CDN}/a" "${HOME}/.goethe/goethe.env"

if [[ "${DISTRO}" == ubuntu ]]; then
  echo "== PATH line when asked for"
  install_as_user a GOETHE_MODIFY_PATH=1
  install_as_user a GOETHE_MODIFY_PATH=1
  check "~/.bashrc gets the PATH line once" test "$(grep -c "added by goethe setup" ~/.bashrc)" = 1
  check "a new shell finds goethe" bash -ic 'command -v goethe' >/dev/null 2>&1

  echo "== the first model install sets up the Python environment"
  timeout 900 "${G}" install bartowski/SmolLM2-135M-Instruct-GGUF --yes >/tmp/model.log 2>&1
  echo "    goethe install exited $? (no GPU here; only the download is checked)"
  tail -n 4 /tmp/model.log | sed 's/^/    | /'
  check "uv came from the package" test -x "${HOME}/.goethe/bin/uv"
  check "the base Python environment exists" test -x "${HOME}/.goethe/py-env/bin/python"
  check "the GGUF was downloaded" bash -c 'ls ~/.goethe/models/*/*.gguf >/dev/null 2>&1'
fi

echo "== uninstall"
"${G}" down >/dev/null 2>&1
"${G}" uninstall --yes >/tmp/un.log 2>&1 || { fail "uninstall"; sed 's/^/    | /' /tmp/un.log | tail -5; }
check "the links are gone" test ! -e "${HOME}/.local/bin/goethe" -a ! -e "${HOME}/.local/bin/goethed"
check "the home is gone" test ! -e "${HOME}/.goethe/bin/goethe"

echo
if [[ ${fails} -gt 0 ]]; then
  echo "${fails} FAILED"
  exit 1
fi
echo "ALL PASS"
