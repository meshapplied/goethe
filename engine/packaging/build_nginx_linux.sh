#!/usr/bin/env bash
# goethe's own nginx for Linux (x86_64 or aarch64, the host's): static OpenSSL and PCRE2, built on Ubuntu 20.04 so glibc 2.31 is enough.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARCH="$(uname -m)"
[[ "${ARCH}" == arm64 ]] && ARCH=aarch64
OUT="${ROOT}/dist/cdn/bin/linux-${ARCH}"
CACHE="${GOETHE_BUILD_CACHE:-${HOME}/.cache/goethe-build}/nginx"
mkdir -p "${OUT}" "${CACHE}"

docker run --rm --name "goethe-build-nginx-${ARCH}" \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "${ROOT}/packaging:/packaging:ro" -v "${CACHE}:/cache" -v "${OUT}:/out" \
  ubuntu:20.04 bash -c '
    set -euo pipefail
    . /packaging/nginx_versions.sh
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq build-essential curl ca-certificates perl file binutils >/dev/null
    cd /cache
    fetch() { local f; f="$(basename "$1")"; [[ -f "$f" ]] || curl -fsSL -o "$f" "$1"; echo "$2  $f" | sha256sum -c --quiet; tar -xzf "$f" -C /tmp; }
    fetch "$NGINX_URL" "$NGINX_SHA256"
    fetch "$OPENSSL_URL" "$OPENSSL_SHA256"
    fetch "$PCRE2_URL" "$PCRE2_SHA256"
    cd "/tmp/nginx-$NGINX_VERSION"
    ./configure \
      --prefix=/usr/local/goethe-nginx \
      --error-log-path=stderr \
      --with-http_ssl_module \
      --with-http_v2_module \
      --with-http_realip_module \
      --without-http_gzip_module \
      --without-http_auth_basic_module \
      --with-openssl="/tmp/openssl-$OPENSSL_VERSION" \
      --with-openssl-opt="no-shared no-tests no-docs" \
      --with-pcre="/tmp/pcre2-$PCRE2_VERSION" \
      --with-pcre-jit \
      --with-cc-opt="-O2 -fstack-protector-strong -D_FORTIFY_SOURCE=2" \
      --with-ld-opt="-Wl,-z,relro,-z,now" >/tmp/configure.log
    make -j"$(nproc)" >/tmp/make.log
    install -m 0755 objs/nginx /out/nginx
    strip /out/nginx
    chown "$HOST_UID:$HOST_GID" /out/nginx
    libs="$(objdump -p /out/nginx | awk "/NEEDED/ {print \$2}" | sort | tr "\n" " ")"
    for l in $libs; do
      case "$l" in libc.so.6 | libpthread.so.0 | libdl.so.2 | libcrypt.so.1 | libm.so.6 | ld-linux-aarch64.so.1 | ld-linux-x86-64.so.2) ;; *) echo "nginx needs $l, which is not part of glibc" >&2; exit 1 ;; esac
    done
    glibc="$(objdump -T /out/nginx | sed -n "s/.*GLIBC_\([0-9.]*\).*/\1/p" | sort -V | tail -n1)"
    echo "nginx needs: $libs(glibc $glibc)"
    /out/nginx -V 2>&1 | head -1
  '
echo "built ${OUT}/nginx"
