#!/usr/bin/env bash
set -euo pipefail

echo "========================================"
echo "GEMM Accelerator Simulation Build"
echo "========================================"

# Configuration - compute project root relative to this script
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/sim/build"
RESULTS_DIR="$PROJECT_ROOT/sim/results"

echo "PROJECT_ROOT = $PROJECT_ROOT"
echo "BUILD_DIR    = $BUILD_DIR"
echo "RESULTS_DIR  = $RESULTS_DIR"

# Check prerequisites
echo "Checking prerequisites..."
if ! command -v verilator &> /dev/null; then
    echo "Error: Verilator not found. Please install verilator."
    exit 1
fi

if ! command -v gtkwave &> /dev/null; then
    echo "Warning: GTKWave not found. Waveform viewing will not be available."
fi

echo "Prerequisites check complete."

# Create necessary directories
echo "Creating directories..."
mkdir -p "$BUILD_DIR"
mkdir -p "$RESULTS_DIR"/{performance,traces,synthesis}

# Change to build directory
cd "$BUILD_DIR"

# Clean previous builds (ignore errors)
echo "Cleaning previous builds..."
make clean || true

# Build simulation
echo "Building simulation..."
make all

echo "Build finished. Running a basic test now..."

# Run basic test
set +e
make run
RC=$?
set -e

if [ $RC -ne 0 ]; then
    echo "✗ Basic test failed (exit code $RC)"
    exit $RC
fi

echo "✓ Basic test passed"

# Performance test
echo "Running performance tests..."
make performance_test || true

# Run full suite if requested
if [ "${1-}" = "--full" ]; then
    echo "Running comprehensive test suite..."
    make size_sweep || true
    make trace || true
    echo "✓ Full test suite completed"
fi

echo ""
echo "========================================"
echo "Build and Test Summary"
echo "========================================"
echo "Build: SUCCESS"
echo "Basic Test: SUCCESS"
echo "Performance Test: done (see results)"
if [ "${1-}" = "--full" ]; then
    echo "Extra: size sweep + trace (if available)"
fi
echo ""
echo "Results available in: $RESULTS_DIR"
echo "To view waveforms: make view_trace"
echo "To run more tests: make help"
echo ""
echo "Build process complete!"
