#!/bin/bash
# Build script for Linux/macOS.
# Override SM_ARCH for non-Ampere GPUs:
#   native = the GPU in this machine (default)
#   sm_75 = RTX 20xx, GTX 16xx
#   sm_86 = RTX 30xx, A100/A40
#   sm_89 = RTX 40xx, L40
#   sm_90 = H100, GH200
#   sm_120 = RTX 50xx (CUDA 12.8+)
set -euo pipefail

SM_ARCH="${SM_ARCH:-native}"
NVCC="${NVCC:-nvcc}"

"$NVCC" -O3 -arch="$SM_ARCH" -Xcompiler -pthread -o vanity_gpu vanity.cu
echo "Build OK: vanity_gpu (arch=$SM_ARCH)"
