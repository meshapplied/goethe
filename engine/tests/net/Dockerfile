# Test image for vpn_interface_e2e.py: the tools goethe needs for its gateway, without GPUs.
FROM ubuntu:24.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      nginx-light openssl iproute2 curl python3 ca-certificates procps \
    && rm -rf /var/lib/apt/lists/*
