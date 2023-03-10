from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='bsi_ops',
    ext_modules=[
        CppExtension('bsi_ops', [
            'bsiFunctions.cpp',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)