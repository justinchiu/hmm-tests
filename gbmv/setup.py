from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='gbmv_cpp',
    ext_modules=[cpp_extension.CppExtension('gbmv_cpp', ['gbmv.cpp'])],
    cmdclass={'build_ext': cpp_extension.BuildExtension})

Extension(
   name='gbmv_cpp',
   sources=['gbmv.cpp'],
   include_dirs=cpp_extension.include_paths(),
   language='c++')
