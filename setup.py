from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import os
import sys

__version__ = "0.0.3"

ext_modules = [
    Pybind11Extension("librapid",
        ["librapid/pybind_ndarray.cpp"],
        define_macros = [('LIBRAPID_VERSION', __version__)],
        include_dirs=[os.getcwd()]
        )
]

setup(
    name="librapid",
    version=__version__,
    author="Toby Davis",
    author_email="pencilcaseman@gmail.com",
    url="https://github.com/Pencilcaseman/librapid",
    description="A fast math and neural network library for Python and C++",
    long_description="",
    ext_modules=ext_modules,
    extras_require={"test": "pytest"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False
)
