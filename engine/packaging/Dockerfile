# Goethe in a container: the signed package, no systemd, goethed as PID 1.
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl zstd openssl nginx-core \
    && rm -rf /var/lib/apt/lists/*

ARG GOETHE_VERSION=0.1.7
ENV GOETHE_HOME=/opt/goethe \
    GOETHE_NO_AUTOSTART=1 \
    UV_PYTHON_INSTALL_DIR=/opt/goethe/uv-python
RUN curl -fsSL https://meshapplied.com/install.sh \
      | GOETHE_HOME=/opt/goethe GOETHE_NO_AUTOSTART=1 GOETHE_REQUIRE_GPU=0 bash \
    && useradd -u 1000 -m goethe \
    && chown -R goethe:goethe /opt/goethe \
    && rm -rf /root/.cache /root/.local /opt/goethe/nginx/debs

ENV PATH=/opt/goethe/bin:$PATH

RUN mkdir -p /models && ln -s /models /opt/goethe/models && chown goethe:goethe /opt/goethe/models

USER goethe
VOLUME /models
EXPOSE 8443

HEALTHCHECK --interval=15s --timeout=3s --start-period=360s --retries=3 \
  CMD curl -fsS --cacert /opt/goethe/nginx/certs/server.crt https://127.0.0.1:8443/health || exit 1

ENTRYPOINT ["/bin/sh","-c","goethe gateway --bind \"${GOETHE_GATEWAY_BIND:-0.0.0.0}\" >/dev/null 2>&1; goethe allowlist --add 127.0.0.1/32 ${GOETHE_ALLOW} >/dev/null 2>&1; exec goethed"]
