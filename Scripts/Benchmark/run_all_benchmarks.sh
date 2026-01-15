#!/bin/bash
# Run all benchmarks (Swift and Python) and compare results

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "=============================================="
echo "Full Benchmark Suite - Swift vs Python MLX"
echo "=============================================="
echo "Project: $PROJECT_DIR"
echo "Date: $(date)"
echo ""

# Ensure results directory exists
mkdir -p "$SCRIPT_DIR/results"

# Note: Swift CLI build is done in benchmark_swift.sh with xcodebuild

# Run Swift benchmark
echo ""
echo "=============================================="
echo "Running Swift MLX Benchmark..."
echo "=============================================="
bash "$SCRIPT_DIR/benchmark_swift.sh"

# Run Python benchmark
echo ""
echo "=============================================="
echo "Running Python MLX Benchmark..."
echo "=============================================="

# Use venv Python directly
"$PROJECT_DIR/.venv/bin/python3" "$SCRIPT_DIR/benchmark_python.py"

echo ""
echo "=============================================="
echo "All benchmarks complete!"
echo "Results in: $SCRIPT_DIR/results/"
echo "=============================================="
