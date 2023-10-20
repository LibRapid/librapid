# -*- coding: utf-8 -*-

import os
import platform
import shutil
import sys
import site
import pathlib
from setuptools import find_packages
from skbuild import setup
from skbuild.cmaker import get_cmake_version
from skbuild.exceptions import SKBuildError
import re
from icecream import ic


class Version:
    def __init__(self, versionStr):
        self.versionStr = versionStr
        self.version = tuple(map(int, versionStr.split(".")))

    def major(self):
        return self.version[0]

    def minor(self):
        return self.version[1]

    def patch(self):
        return self.version[2]

    def __str__(self):
        return self.versionStr

    def __lt__(self, other):
        for a, b in zip(self.version, other.version):
            if a < b:
                return True
            elif a > b:
                return False


# Python implementation (CPython, PyPy, Jython, IronPython)
PYTHON_IMPLEMENTATION = ic(platform.python_implementation())

# Root directory
ROOT_DIR = ic(os.path.dirname(os.path.abspath(__file__)))

setup_requires = []
install_requires = []

try:
    if ic(Version(get_cmake_version())) < Version("3.10"):
        setup_requires.append('cmake')
        install_requires.append("cmake")
except SKBuildError:
    setup_requires.append('cmake')
    install_requires.append("cmake")

if ic(platform.system()) == "Windows" and PYTHON_IMPLEMENTATION == "CPython":
    setup_requires.append('pywin32')
    install_requires.append("pywin32")

# The full version, including alpha/beta/rc tags
currentMajorVersion = None
currentMinorVersion = None
currentPatchVersion = None

try:
    with open("./version.txt") as versionFile:
        text = versionFile.read()
        currentMajorVersion = ic(re.search("MAJOR [0-9]+", text).group().split()[1])
        currentMinorVersion = ic(re.search("MINOR [0-9]+", text).group().split()[1])
        currentPatchVersion = ic(re.search("PATCH [0-9]+", text).group().split()[1])
except Exception as e:
    print("[ ERROR ] Failed to read version.txt")
    print(e)
    sys.exit(1)

release = ic(f"{currentMajorVersion}.{currentMinorVersion}.{currentPatchVersion}")

# Locate and read the contents of README.md
with open(os.path.join(ROOT_DIR, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

cmakeArgs = ["-DLIBRAPID_USE_MULTIPREC=ON"]
if ic(os.environ.get("LIBRAPID_NATIVE_ARCH")):  # Only defined on GitHub Actions
    cmakeArgs.append(f"-DLIBRAPID_NATIVE_ARCH={os.environ.get('LIBRAPID_NATIVE_ARCH')}")

if ic(os.environ.get("LIBRAPID_CUDA_WHEEL")):
    moduleName = "librapid_cuda_" + os.environ["LIBRAPID_CUDA_WHEEL"]
else:
    moduleName = "librapid"

# Use multiple cores if possible
cmakeArgs.append("-DCMAKE_BUILD_PARALLEL_LEVEL=0")

setup(
    name=moduleName,
    version=release,
    author="Toby Davis",
    author_email="pencilcaseman@gmail.com",
    url="https://github.com/LibRapid/librapid",
    description="A highly optimised C++ library for high-performance computing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=ic(["librapid." + mod for mod in find_packages("librapid")]),
    package_dir={"": "librapid"},
    cmake_args=cmakeArgs,
    cmake_install_dir="librapid",
    license="MIT License",
    keywords=["librapid",
              "high-performance computing",
              "c++",
              "mathematics",
              "array",
              "matrix",
              "vector",
              "tensor",
              "gpu",
              "cuda",
              "openmp",
              "multithreading",
              "multicore"
              "parallel"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    extras_require={"test": "pytest"},
    install_requires=install_requires,
    setup_requires=setup_requires,
    include_package_data=False,
    zip_safe=True
)
