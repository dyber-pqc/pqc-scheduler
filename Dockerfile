# PQC Scheduler - Docker Container
# Copyright (c) 2025 Dyber, Inc. All Rights Reserved.
#
# Build: docker build -t pqc-scheduler .
# Run:   docker run -it --rm pqc-scheduler --help

# Build stage
FROM rust:1.75-bookworm AS builder

RUN apt-get update && apt-get install -y \
    cmake ninja-build pkg-config libssl-dev libclang-dev git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone --depth 1 --branch 0.10.0 https://github.com/open-quantum-safe/liboqs.git && \
    cd liboqs && mkdir build && cd build && \
    cmake -GNinja -DOQS_USE_AVX2=ON -DOQS_BUILD_ONLY_LIB=ON -DCMAKE_BUILD_TYPE=Release .. && \
    ninja && ninja install && ldconfig

WORKDIR /app
COPY . .
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y libssl3 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/liboqs* /usr/local/lib/
RUN ldconfig
COPY --from=builder /app/target/release/pqc-scheduler /usr/local/bin/
COPY config/ /etc/pqc-scheduler/config/
COPY workloads/ /etc/pqc-scheduler/workloads/
RUN mkdir -p /var/lib/pqc-scheduler/results && \
    useradd -r -s /bin/false pqc && chown -R pqc:pqc /var/lib/pqc-scheduler
USER pqc
WORKDIR /var/lib/pqc-scheduler
ENV RUST_LOG=info
EXPOSE 9090
ENTRYPOINT ["pqc-scheduler"]
CMD ["--config", "/etc/pqc-scheduler/config/default.yaml", "--workload", "/etc/pqc-scheduler/workloads/constant_rate.json"]
