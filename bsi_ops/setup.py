from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
import os


def has_cuda():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


cpu_sources = [
    'bsiFunctions.cpp',
]

cuda_sources = cpu_sources + [
    'csrc/cuda/bsi_cuda.cpp',
    'csrc/cuda/bsi_cuda_kernels.cu',
]

common_cxx_flags = ['-std=c++20', '-O3']

if has_cuda():
    ext = CUDAExtension(
        name='bsi_ops',
        sources=cuda_sources,
        extra_compile_args={
            'cxx': common_cxx_flags + ['-DBSI_WITH_CUDA=1'],
            'nvcc': ['-O3']
        },
    )
else:
    ext = CppExtension(
        name='bsi_ops',
        sources=cpu_sources,
        extra_compile_args=common_cxx_flags,
    )

setup(
    name='bsi_ops',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
