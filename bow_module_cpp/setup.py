import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

os.environ['CFLAGS'] = '-fopenmp'
os.environ['LDFLAGS'] = '-fopenmp'

setup(
    name='bow_module',
    ext_modules=[
        CppExtension(
            name='bow_module',
            sources=['bow_module.cpp'],
            extra_compile_args=['-std=c++20', '-O3'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
