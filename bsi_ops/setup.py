from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME
import torch
import os
import sys

# CPU-only sources (no CUDA wrapper)
cpu_sources = [
    'bsiFunctions.cpp',
]

# CUDA sources include the CUDA wrapper
cuda_sources = [
    'bsiFunctions.cpp',
    'csrc/cuda/bsi_vector_cuda.cpp',
    'csrc/cuda/bsi_cuda.cpp',
    'csrc/cuda/bsi_cuda_kernels.cu',
]

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
common_cxx_flags = ['-O3', '-std=c++20']
common_link_args = [f'-Wl,-rpath,{torch_lib_dir}']

# BSI word size (bits per bitmap word). Default 64. Supported: 8, 16, 32, 64, 128, 256.
bsi_word_bits = int(os.getenv('BSI_WORD_BITS', '64'))
assert bsi_word_bits in (8, 16, 32, 64, 128, 256), \
    f"BSI_WORD_BITS must be 8, 16, 32, 64, 128, or 256, got {bsi_word_bits}"

# Add bsiCPP include path
bsicpp_include = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'bsiCPP'))

# Prefer toolchain-based detection; allow FORCE_CUDA=1 to override
with_cuda = (os.getenv('FORCE_CUDA', '0') == '1') or (CUDA_HOME is not None and getattr(torch.version, 'cuda', None))

if with_cuda:
    ext = CUDAExtension(
        name='bsi_ops',
        sources=cuda_sources,
        include_dirs=[bsicpp_include],
        define_macros=[('BSI_WITH_CUDA', '1'), ('BSI_WORD_BITS', str(bsi_word_bits))],
        extra_compile_args={
            'cxx': common_cxx_flags,
            'nvcc': ['-O3', '-std=c++20', f'-DBSI_WORD_BITS={bsi_word_bits}']
        },
        extra_link_args=common_link_args,
    )
else:
    ext = CppExtension(
        name='bsi_ops',
        sources=cpu_sources,
        include_dirs=[bsicpp_include],
        define_macros=[('BSI_WORD_BITS', str(bsi_word_bits))],
        extra_compile_args=common_cxx_flags,
        extra_link_args=common_link_args,
    )

setup(
    name='bsi_ops',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
