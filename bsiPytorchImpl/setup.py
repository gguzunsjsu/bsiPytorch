from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='bsi_ops',
    ext_modules=[
        CUDAExtension('bsi_ops', [
            'bsiFunctions.cpp',
            '../bsiCopy/BsiAttribute.cpp',
            '../bsiCopy/BsiSigned.cpp',
            '../bsiCopy/BsiUnsigned.cpp',
            '../bsiCopy/hybridBitmap/hybridbitmap.cpp',
            '../bsiCopy/hybridBitmap/boolarray.cpp',
            '../bsiCopy/hybridBitmap/ewahutil.cpp',
            '../bsiCopy/hybridBitmap/hybridutil.cpp',
            '../bsiCopy/hybridBitmap/runninglengthword.cpp',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)