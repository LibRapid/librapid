from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

import os
import sys

# Load the version number from VERSION.hpp
version_file = open("librapid/VERSION.hpp", "r")
__version__ = version_file.readlines()[1].split()[2].replace("\"", "")
version_file.close()

ext_modules = [
    Pybind11Extension("librapid",
        ["librapid/pybind_librapid.cpp"],
        define_macros = [('LIBRAPID_BUILD', 1)],
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
