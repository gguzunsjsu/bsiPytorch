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
    'csrc/cuda/bsi_cuda.cpp',
]

cuda_sources = cpu_sources + [
    'csrc/cuda/bsi_cuda_kernels.cu',
]

torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), 'lib')
common_cxx_flags = ['-std=c++20', '-O3']
common_link_args = [f'-Wl,-rpath,{torch_lib_dir}']

if has_cuda():
    ext = CUDAExtension(
        name='bsi_ops',
        sources=cuda_sources,
        extra_compile_args={
            'cxx': common_cxx_flags + ['-DBSI_WITH_CUDA=1'],
            'nvcc': ['-O3', '-std=c++20']
        },
        extra_link_args=common_link_args,
    )
else:
    ext = CppExtension(
        name='bsi_ops',
        sources=cpu_sources,
        extra_compile_args=common_cxx_flags,
        extra_link_args=common_link_args,
    )

setup(
    name='bsi_ops',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExtension},
)
