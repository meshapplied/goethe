# Goethe container image from a verified release tree: goethe, goethed, nginx, uv and the Ask model; engines, kernel packs and models go to the /workspace volume when first used.
ARG TARGETARCH

FROM scratch AS bin-amd64
COPY bin/linux-x86_64/goethe bin/linux-x86_64/goethed bin/linux-x86_64/uv bin/linux-x86_64/nginx /

FROM scratch AS bin-arm64
COPY bin/linux-aarch64/goethe bin/linux-aarch64/goethed bin/linux-aarch64/uv bin/linux-aarch64/nginx /

FROM bin-${TARGETARCH} AS bin

FROM ubuntu:24.04@sha256:9c7f200c07e3d4a03e7249d2c16bf66a51120fb876aa2242cf31451412543289
ARG VERSION
ARG REVISION=unknown
RUN test -n "${VERSION}" \
 && apt-get update \
 && apt-get install -y --no-install-recommends bash ca-certificates curl g++ libavdevice60 openssl tar tini \
 && rm -rf /var/lib/apt/lists/*
COPY --from=bin / /opt/goethe/bin/
COPY ask/ask-model-v1.bin /opt/goethe/ask/
COPY LICENSE NOTICE VERSION /opt/goethe/
ENV GOETHE_IMAGE=${VERSION} \
    GOETHE_HOME=/workspace/.goethe \
    HOME=/workspace \
    PATH=/opt/goethe/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    GOETHE_NO_AUTOSTART=1 \
    GOETHE_NO_LINK=1 \
    GOETHE_NO_COMPLETION=1
LABEL org.opencontainers.image.title="goethe" \
      org.opencontainers.image.description="Serve models through OpenAI and Anthropic compatible gateways" \
      org.opencontainers.image.source="https://github.com/meshapplied/goethe" \
      org.opencontainers.image.url="https://goethe.sh" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}" \
      org.opencontainers.image.licenses="Apache-2.0"
VOLUME /workspace
WORKDIR /workspace
EXPOSE 8443 8444
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD ["goethe", "container", "--check"]
ENTRYPOINT ["tini", "-g", "--", "goethe", "container"]
