#!/bin/bash
# build_nvfp4_cutlass.sh - Build NVFP4 GEMM benchmark for Jetson AGX Thor
#
# CRITICAL FLAGS (learned the hard way):
#   -arch=sm_110a      NOT sm_110 - the "a" suffix enables architecture-accelerated
#                      features (TMA, tcgen05.mma.blockscaled, etc.)
#   --expt-relaxed-constexpr  REQUIRED for CUTLASS 3.x (sm100_blockscaled_mma_warpspecialized.hpp)
#   --expt-extended-lambda    Required for some CUTLASS kernel lambdas
#
# Source: modified CUTLASS 4.5.2 example 72a with sm_110 runtime guard bypassed

set -e

NVCC=${NVCC:-/usr/local/cuda-13.3/bin/nvcc}
CUTLASS_DIR=${CUTLASS_DIR:-/home/rm01/opencodeWorkspace/cutlass}
SRC_DIR=${SRC_DIR:-/home/rm01/opencodeWorkspace/thor-gemm-bench}
OUTPUT=${OUTPUT:-/home/rm01/opencodeWorkspace/thor-gemm-bench/bench_nvfp4_cutlass}

echo "Building NVFP4 GEMM benchmark for Thor sm_110a..."
echo "  NVCC:        $NVCC"
echo "  CUTLASS:     $CUTLASS_DIR"
echo "  Source:      $SRC_DIR/bench_nvfp4_cutlass.cu"
echo "  Output:      $OUTPUT"
echo

$NVCC -std=c++17 -O3 -arch=sm_110a \
  -I $CUTLASS_DIR/include \
  -I $CUTLASS_DIR/tools/util/include \
  -I $CUTLASS_DIR/examples/common \
  --expt-relaxed-constexpr --expt-extended-lambda \
  $SRC_DIR/bench_nvfp4_cutlass.cu \
  -o $OUTPUT \
  -lcudart

echo
echo "Build SUCCESS: $OUTPUT"
echo
echo "Run with: $OUTPUT --m=1024 --n=1024 --k=1024 --iterations=20"
