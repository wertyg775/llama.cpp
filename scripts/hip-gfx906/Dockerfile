FROM mixa3607/rocm-gfx906@sha256:2dfc7673bc16a68ad1b3b72d4e9ec9304f335f8a41c037143e58f1541538b3ae
USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends cmake ninja-build g++ \
    && rm -rf /var/lib/apt/lists/*
