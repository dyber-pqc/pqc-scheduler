#!/bin/bash
# PQC Scheduler - Quick Test Script
# Copyright (c) 2025 Dyber, Inc. All Rights Reserved.
#
# Runs a quick smoke test to verify the scheduler is working

set -e

echo "=============================================="
echo "PQC Scheduler Quick Test"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
DURATION=${DURATION:-30}
WARMUP=${WARMUP:-5}

# Build if needed
if [ ! -f target/release/pqc-scheduler ] && [ ! -f target/debug/pqc-scheduler ]; then
    echo "Building pqc-scheduler..."
    cargo build --release
fi

BINARY="target/release/pqc-scheduler"
if [ ! -f "$BINARY" ]; then
    BINARY="target/debug/pqc-scheduler"
fi

echo "Using binary: $BINARY"
echo "Duration: ${DURATION}s"
echo "Warmup: ${WARMUP}s"
echo ""

# Create temp output
OUTPUT_DIR=$(mktemp -d)
trap "rm -rf $OUTPUT_DIR" EXIT

echo "Running quick test..."
echo ""

if $BINARY \
    --config config/default.yaml \
    --workload workloads/constant_rate.json \
    --duration "$DURATION" \
    --warmup "$WARMUP" \
    --seed 42 \
    --output "$OUTPUT_DIR/results.json" \
    2>&1; then
    
    echo ""
    echo -e "${GREEN}✓ Test completed successfully${NC}"
    echo ""
    
    if [ -f "$OUTPUT_DIR/results.json" ]; then
        echo "Results:"
        echo "--------"
        cat "$OUTPUT_DIR/results.json" | python3 -m json.tool 2>/dev/null || cat "$OUTPUT_DIR/results.json"
    fi
else
    echo ""
    echo -e "${RED}✗ Test failed${NC}"
    exit 1
fi

echo ""
echo "=============================================="
echo "Quick Test Complete"
echo "=============================================="
