# Hybrid Classical-Quantum Cryptographic Operation Scheduler

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.75+-orange.svg)](https://www.rust-lang.org)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()

A cryptographic scheduling framework with formal guarantees for the post-quantum transition, implementing dynamic algorithm selection, multi-objective resource optimization, graceful degradation, and verified migration management.

## Overview

This framework addresses the critical challenge of managing hybrid classical-quantum cryptographic operations during the transition from classical to post-quantum cryptography. It provides:

1. **Dynamic Algorithm Selection Engine**: MDP-based optimal algorithm selection with proven convergence and bounded regret under adversarial threat evolution
2. **Multi-Objective Resource Optimization**: Pareto-optimal resource allocation with game-theoretic multi-tenant support
3. **Graceful Degradation Framework**: Formal availability guarantees with sub-millisecond failover
4. **Verified Migration Path Management**: Risk-quantified transition strategies with automatic rollback

## Key Results

- **23.7% throughput improvement** over static PQC scheduling (p < 0.001, Cohen's d = 2.14)
- **99.97% SLA compliance** under varying threat conditions
- **847μs mean algorithm switch time** with warm standby (27.6× improvement over cold start)
- **67% migration risk reduction** compared to big-bang approaches

## Supported Algorithms

### Classical
- RSA-2048, RSA-4096
- ECDSA-P256, ECDSA-P384
- X25519

### Post-Quantum
- ML-KEM-512, ML-KEM-768, ML-KEM-1024 (FIPS 203)
- ML-DSA-44, ML-DSA-65, ML-DSA-87 (FIPS 204)
- SLH-DSA-128f, SLH-DSA-128s, SLH-DSA-192f (FIPS 205)

### Hybrid
- X25519+ML-KEM-768
- P256+ML-KEM-768
- P384+ML-KEM-1024
- Ed25519+ML-DSA-65
- P256+ML-DSA-65
- P384+ML-DSA-87

## Requirements

### Software
- Ubuntu 22.04 LTS (kernel 5.15+)
- Rust 1.75.0+
- OpenSSL 3.2.0+
- liboqs 0.10.0
- oqs-provider 0.5.3
- Redis 7.2.0+

### Hardware (for FPGA acceleration)
- Intel Xeon Gold 6342 or equivalent
- AMD EPYC 7763 (optional, for PQC software)
- Xilinx Alveo U280 FPGA (optional, for hardware acceleration)
- 512GB DDR4-3200 ECC (recommended)

## Installation

### Quick Start

```bash
# Clone repository
git clone https://github.com/dyber-pqc/pqc-scheduler.git
cd pqc-scheduler

# Install dependencies
./scripts/install_deps.sh

# Build
cargo build --release

# Run tests
cargo test --release

# Run benchmarks
cargo bench
```

### Building liboqs

```bash
git clone https://github.com/open-quantum-safe/liboqs.git
cd liboqs && git checkout a5e4814
mkdir build && cd build
cmake -DOQS_USE_AVX2=ON -DOQS_USE_AVX512=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local ..
make -j$(nproc) && sudo make install
```

## Usage

### Basic Usage

```bash
# Run scheduler with default configuration
cargo run --release -- \
  --config config/default.yaml \
  --workload workloads/cloudflare_mmpp.json \
  --duration 3600 \
  --seed 42

# Run with custom threat level
cargo run --release -- \
  --config config/high_threat.yaml \
  --threat-level 0.7

# Run benchmark suite
./scripts/run_benchmarks.sh
```

### Configuration

See `config/default.yaml` for configuration options:

```yaml
scheduler:
  mdp:
    gamma: 0.99          # Discount factor
    epsilon: 1.0e-6      # Convergence threshold
    recompute_interval_sec: 60
  weights:
    security: 1.0        # Security weight
    latency: 0.5         # Latency weight  
    cost: 0.3            # Cost weight
```

### Programmatic API

```rust
use pqc_scheduler::{Scheduler, Config, Algorithm};

// Create scheduler
let config = Config::from_file("config/default.yaml")?;
let mut scheduler = Scheduler::new(config).await?;

// Submit cryptographic operation
let result = scheduler.submit(Operation::KeyExchange {
    algorithm: Algorithm::MlKem768,
    priority: Priority::High,
}).await?;

// Get performance metrics
let metrics = scheduler.metrics();
println!("Throughput: {} ops/s", metrics.throughput);
println!("P99 Latency: {} μs", metrics.latency_p99);
```

## Benchmarks

### Scheduling Performance (n=10 runs, 1 hour each)

| Strategy | Throughput (ops/s) | Latency (μs) | SLA % | Security |
|----------|-------------------|--------------|-------|----------|
| Static-Classical | 147,230±4,521 | 124±9 | 99.92±0.02 | 0.41 |
| Static-PQC | 89,450±3,124 | 287±18 | 96.34±0.41 | 1.00 |
| Round-Robin | 98,720±3,847 | 231±15 | 94.21±0.67 | 0.71 |
| **MDP-Optimal** | **110,640±2,847** | **198±12** | **99.97±0.01** | **0.94** |

### Graceful Degradation (n=100 trials)

| Scenario | Mean (μs) | p99 (μs) | Continuity |
|----------|-----------|----------|------------|
| Algorithm Vulnerability | 847±124 | 1,823 | 100% |
| Hardware Failure | 1,234±189 | 2,847 | 100% |
| Resource Exhaustion | 423±67 | 1,102 | 100% |
| Cold Start (baseline) | 23,400±2,140 | 47,200 | 98.7% |

## Directory Structure

```
pqc-scheduler/
├── Cargo.toml              # Rust project configuration
├── README.md               # This file
├── LICENSE                 # Apache 2.0 license
├── config/                 # Configuration files
│   ├── default.yaml        # Default scheduler config
│   ├── high_threat.yaml    # High threat level config
│   └── production.yaml     # Production deployment config
├── workloads/              # Workload definitions
│   ├── cloudflare_mmpp.json
│   ├── constant_rate.json
│   └── burst_pattern.json
├── src/                    # Source code
│   ├── main.rs             # Entry point
│   ├── lib.rs              # Library root
│   ├── scheduler/          # Scheduler implementation
│   ├── mdp/                # MDP algorithm selection
│   ├── optimizer/          # Multi-objective optimization
│   ├── degradation/        # Graceful degradation
│   ├── migration/          # Migration management
│   ├── crypto/             # Cryptographic backends
│   └── metrics/            # Performance metrics
├── scripts/                # Utility scripts
│   ├── install_deps.sh     # Dependency installation
│   ├── run_benchmarks.sh   # Benchmark suite
│   ├── synth_ntt.tcl       # FPGA synthesis
│   └── program_fpga.tcl    # FPGA programming
├── notebooks/              # Jupyter analysis notebooks
│   ├── statistical_analysis.ipynb
│   └── figure_generation.ipynb
├── benches/                # Benchmark code
│   └── scheduler_bench.rs
└── tests/                  # Integration tests
    └── integration_tests.rs
```

## FPGA Acceleration

For FPGA-accelerated deployment with Xilinx Alveo U280:

```bash
# Synthesize NTT accelerator
vivado -mode batch -source scripts/synth_ntt.tcl \
  -tclargs xcu280-fsvh2892-2L-e 250MHz

# Program FPGA
vivado -mode batch -source scripts/program_fpga.tcl \
  -tclargs bitstreams/mlkem768_ntt_250mhz.bit

# Run with FPGA acceleration enabled
cargo run --release --features fpga -- \
  --config config/fpga_accelerated.yaml
```

## Citation

If you use this software in your research, please cite:

```bibtex
@inproceedings{kleckner2025hybrid,
  title={Hybrid Classical-Quantum Cryptographic Operation Scheduling with 
         Dynamic Algorithm Selection and Multi-Objective Resource Optimization},
  author={Kleckner, Zachary},
  booktitle={Proceedings of ICCFN 2025: The International Conference on 
             Cybertechnology and Future Networks},
  year={2025},
  organization={Dyber, Inc.}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Open Quantum Safe project for liboqs
- NIST Post-Quantum Cryptography Standardization team
- Cloudflare Radar for workload characterization data

## Contact

- **Author**: Zachary Kleckner
- **Email**: zkleckner@dyber.org
- **Organization**: [Dyber, Inc.](https://dyber.org)

---
© 2025 Dyber, Inc. All Rights Reserved.
