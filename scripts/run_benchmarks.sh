#!/bin/bash
# PQC Scheduler - Benchmark Suite Runner
# Copyright (c) 2025 Dyber, Inc. All Rights Reserved.

set -e

echo "=============================================="
echo "PQC Scheduler Benchmark Suite"
echo "=============================================="
echo ""

# Configuration
DURATION=${DURATION:-3600}
WARMUP=${WARMUP:-300}
RUNS=${RUNS:-10}
OUTPUT_DIR=${OUTPUT_DIR:-"results"}
CONFIG=${CONFIG:-"config/default.yaml"}
WORKLOAD=${WORKLOAD:-"workloads/cloudflare_mmpp.json"}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)
            DURATION="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --runs)
            RUNS="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --workload)
            WORKLOAD="$2"
            shift 2
            ;;
        --quick)
            DURATION=60
            WARMUP=10
            RUNS=3
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --duration SECS    Duration per run (default: 3600)"
            echo "  --warmup SECS      Warmup period (default: 300)"
            echo "  --runs N           Number of runs (default: 10)"
            echo "  --output DIR       Output directory (default: results)"
            echo "  --config FILE      Config file (default: config/default.yaml)"
            echo "  --workload FILE    Workload file (default: workloads/cloudflare_mmpp.json)"
            echo "  --quick            Quick test mode (60s duration, 3 runs)"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/benchmark_$TIMESTAMP"
mkdir -p "$RUN_DIR"

echo "Configuration:"
echo "  Duration:  $DURATION seconds"
echo "  Warmup:    $WARMUP seconds"
echo "  Runs:      $RUNS"
echo "  Config:    $CONFIG"
echo "  Workload:  $WORKLOAD"
echo "  Output:    $RUN_DIR"
echo ""

# Build release binary
echo -e "${BLUE}Building release binary...${NC}"
cargo build --release
echo -e "${GREEN}Build complete${NC}"
echo ""

# System information
echo "System Information:" | tee "$RUN_DIR/system_info.txt"
echo "===================" | tee -a "$RUN_DIR/system_info.txt"
uname -a | tee -a "$RUN_DIR/system_info.txt"
echo "" | tee -a "$RUN_DIR/system_info.txt"

if [ -f /proc/cpuinfo ]; then
    echo "CPU:" | tee -a "$RUN_DIR/system_info.txt"
    grep "model name" /proc/cpuinfo | head -1 | tee -a "$RUN_DIR/system_info.txt"
    echo "Cores: $(nproc)" | tee -a "$RUN_DIR/system_info.txt"
fi

if [ -f /proc/meminfo ]; then
    echo "" | tee -a "$RUN_DIR/system_info.txt"
    echo "Memory:" | tee -a "$RUN_DIR/system_info.txt"
    grep "MemTotal" /proc/meminfo | tee -a "$RUN_DIR/system_info.txt"
fi
echo ""

# Copy configuration files
cp "$CONFIG" "$RUN_DIR/config.yaml"
cp "$WORKLOAD" "$RUN_DIR/workload.json"

# Run benchmarks
echo "=============================================="
echo "Running $RUNS benchmark iterations"
echo "=============================================="
echo ""

SUCCESSFUL=0
FAILED=0

for seed in $(seq 0 $((RUNS - 1))); do
    RUN_NUM=$((seed + 1))
    echo -e "${BLUE}[$RUN_NUM/$RUNS] Running with seed $seed...${NC}"
    
    OUTPUT_FILE="$RUN_DIR/run_${seed}.json"
    LOG_FILE="$RUN_DIR/run_${seed}.log"
    
    START_TIME=$(date +%s)
    
    if ./target/release/pqc-scheduler \
        --config "$CONFIG" \
        --workload "$WORKLOAD" \
        --duration "$DURATION" \
        --warmup "$WARMUP" \
        --seed "$seed" \
        --output "$OUTPUT_FILE" \
        2>&1 | tee "$LOG_FILE"; then
        
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        
        echo -e "${GREEN}[$RUN_NUM/$RUNS] Complete in ${ELAPSED}s${NC}"
        SUCCESSFUL=$((SUCCESSFUL + 1))
    else
        echo -e "${YELLOW}[$RUN_NUM/$RUNS] Failed${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
done

# Aggregate results
echo "=============================================="
echo "Aggregating Results"
echo "=============================================="
echo ""

# Create Python script for aggregation
cat > "$RUN_DIR/aggregate.py" << 'PYTHON_SCRIPT'
#!/usr/bin/env python3
import json
import sys
import os
import numpy as np
from pathlib import Path

def load_results(run_dir):
    results = []
    for f in sorted(Path(run_dir).glob("run_*.json")):
        with open(f) as fp:
            results.append(json.load(fp))
    return results

def aggregate(results):
    if not results:
        return None
    
    metrics = {}
    for key in results[0].keys():
        values = [r[key] for r in results if key in r]
        if isinstance(values[0], (int, float)):
            metrics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
            }
    return metrics

def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    results = load_results(run_dir)
    
    if not results:
        print("No results found")
        return
    
    agg = aggregate(results)
    
    print(f"\nAggregated Results ({len(results)} runs)")
    print("=" * 50)
    
    key_metrics = [
        ('throughput_mean', 'Throughput', 'ops/s'),
        ('latency_mean_us', 'Latency (mean)', 'μs'),
        ('latency_p99_us', 'Latency (p99)', 'μs'),
        ('sla_compliance', 'SLA Compliance', '%'),
        ('security_score', 'Security Score', ''),
    ]
    
    for key, name, unit in key_metrics:
        if key in agg:
            m = agg[key]
            if key == 'sla_compliance':
                print(f"{name}: {m['mean']*100:.2f}±{m['std']*100:.2f} {unit}")
            else:
                print(f"{name}: {m['mean']:.2f}±{m['std']:.2f} {unit}")
    
    # Save aggregated results
    with open(os.path.join(run_dir, 'aggregated.json'), 'w') as f:
        json.dump(agg, f, indent=2)
    
    print(f"\nResults saved to {run_dir}/aggregated.json")

if __name__ == '__main__':
    main()
PYTHON_SCRIPT

python3 "$RUN_DIR/aggregate.py" "$RUN_DIR"

# Summary
echo ""
echo "=============================================="
echo "Benchmark Summary"
echo "=============================================="
echo ""
echo "Successful runs: $SUCCESSFUL/$RUNS"
echo "Failed runs:     $FAILED/$RUNS"
echo ""
echo "Results saved to: $RUN_DIR"
echo ""
echo "Files:"
ls -la "$RUN_DIR"
echo ""

# Run Rust benchmarks
echo ""
echo "=============================================="
echo "Running Criterion Benchmarks"
echo "=============================================="
echo ""

if cargo bench -- --noplot 2>&1 | tee "$RUN_DIR/criterion_bench.log"; then
    echo -e "${GREEN}Criterion benchmarks complete${NC}"
else
    echo -e "${YELLOW}Criterion benchmarks skipped or failed${NC}"
fi

echo ""
echo -e "${GREEN}Benchmark suite complete!${NC}"
echo "Results directory: $RUN_DIR"
