#!/bin/bash
export CUDA_HOME=/opt/ohpc/pub/apps/nvidia/nvhpc/24.11/Linux_x86_64/24.11/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
export CFLAGS="-I${CUDA_HOME}/include ${CFLAGS}"
export CXXFLAGS="-I${CUDA_HOME}/include ${CXXFLAGS}"
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export FORCE_CUDA=1
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
