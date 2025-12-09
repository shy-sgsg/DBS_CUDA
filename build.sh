#!/bin/bash

# DBS cuFFT 编译脚本

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== DBS Core (CUDA/cuFFT) Build Script ==="
echo "Source dir: ${SCRIPT_DIR}"
echo "Build dir: ${BUILD_DIR}"

# 创建构建目录
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# 配置（根据需要调整 GPU 架构）
echo ""
echo "Running CMake configuration..."
cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_OPENMP=ON \
  "${SCRIPT_DIR}"

# 编译
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executable: ${BUILD_DIR}/dbs_core"
