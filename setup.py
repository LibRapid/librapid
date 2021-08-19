# -*- coding: utf-8 -*-
import os
from skbuild import setup
import distutils
from setuptools import find_packages
import platform
import shutil
from packaging.version import LegacyVersion
from skbuild.exceptions import SKBuildError
from skbuild.cmaker import get_cmake_version

if not os.path.exists(os.path.join("src", "librapid", "pybind11")):
	shutil.copytree("pybind11", os.path.join("src", "librapid", "pybind11"))

if not os.path.exists(os.path.join("src", "librapid", "jitify")):
	shutil.copytree("jitify", os.path.join("src", "librapid", "jitify"))

# Add CMake as a build requirement if cmake is not installed or is too low a version
setup_requires = []
install_requires = []

try:
	if LegacyVersion(get_cmake_version()) < LegacyVersion("3.10"):
		setup_requires.append('cmake')
		install_requires.append("cmake")
except SKBuildError:
	setup_requires.append('cmake')
	install_requires.append("cmake")

if platform.system() == "Windows":
	setup_requires.append('pypiwin32')
	install_requires.append("pypiwin32")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the version number from VERSION.hpp
version_file = open(os.path.join("src", "librapid", "VERSION.hpp"), "r")
__version__ = version_file.readlines()[1].split()[2].replace("\"", "")
version_file.close()

# Locate and read the contents of README.md
with open(os.path.join(ROOT_DIR, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

cmake_args = []

setup(
	name="librapid",
	version=__version__,
	author="Toby Davis",
	author_email="pencilcaseman@gmail.com",
	url="https://github.com/Pencilcaseman/librapid",
	description="A fast math and neural network library for Python and C++",
	long_description=long_description,
	long_description_content_type="text/markdown",
	packages=find_packages("src"),
	package_dir={"" : "src"},
	cmake_install_dir="src/librapid",
	cmake_args=cmake_args,
	license="Boost Software License",
	keywords=["math", "neural network", "ndarray", "array", "matrix",
			"high-performance computing"],
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
	],
	extras_require={"test": "pytest"},
	install_requires=install_requires,
	setup_requires=setup_requires,
	include_package_data=True,
	zip_safe=False
)
