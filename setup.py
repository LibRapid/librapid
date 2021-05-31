from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

from pathlib import Path
import platform
import os
import sys

try:
	from setuptools import setup, Extension, find_packages
except ImportError:
	from distutils.core import setup, Extension

	def find_packages(where='.'):
		return [folder.replace("/", ".").lstrip(".")
				for (folder, _, files) in os.walk(where)
        		if "__init__.py" in files]

def get_compiler_name():
	import re
	import distutils.ccompiler
	comp = distutils.ccompiler.get_default_compiler()
	getnext = False

	for a in sys.argv[2:]:
		if getnext:
			comp = a
			getnext = False
			continue
		# Separated by space
		if a == '--compiler'  or  re.search('^-[a-z]*c$', a):
			getnext = True
			continue
		# Without space
		m = re.search('^--compiler=(.+)', a)
		if m == None:
			m = re.search('^-[a-z]*c(.+)', a)
		if m:
			comp = m.group(1)

	return comp

# Load the version number from VERSION.hpp
version_file = open("librapid/VERSION.hpp", "r")
__version__ = version_file.readlines()[1].split()[2].replace("\"", "")
version_file.close()

# Locate and read the contents of README.md
long_description = Path("./README.md").read_text(encoding="utf-8")

# Set the C++ version to use
def std_version():
	c = get_compiler_name()
	if c == "msvc": return ["/std:c++latest"]
	elif c in ("gcc", "g++"): return ["-std=c++17"]
	elif c == "clang": return ["-std=c++17"]
	elif c == "unix": return ["-std=c++17"]
	return []

def compile_with_omp():
	if platform.system() == "Darwin":
		return []

	c = get_compiler_name()
	if c == "msvc": return ["/openmp"]
	elif c in ("gcc", "g++"): return ["-fopenmp"]
	elif c == "clang": return ["-fopenmp"]
	elif c == "unix": return ["-fopenmp"]
	return []

def link_omp():
	if platform.system() == "Darwin":
		return []
	
	c = get_compiler_name()
	if c == "msvc": return []
	elif c in ("gcc", "g++"): return ["-lgomp"]
	elif c == "unix": return ["-lgomp"]
	elif c == "clang": return ["-lgomp"]
	return []

def enable_optimizations():	
	c = get_compiler_name()
	if c == "msvc":
		res = ["/O2", "/Ot", "/Ob1"]
		p = platform.processor().split()[0]
		if p == "AMD64":
			res += ["/favor:AMD64"]
		elif p == "INTEL64":
			res += ["/favor:INTEL64"]
		elif p == "ATOM":
			res += ["/favor:ATOM"]
		return res
	elif c in ("gcc", "g++"): return ["-O3", "-mavx"]
	elif c == "clang": return ["-O3", "-mavx"]
	elif c == "unix": return ["-O3", "-mavx"]
	return []

compiler_flags = std_version() + compile_with_omp() + enable_optimizations()
linker_flags = link_omp()

ext_modules = [
	Pybind11Extension("librapid",
		["librapid/pybind_librapid.cpp"],
		extra_compile_args=compiler_flags,
		extra_link_args=linker_flags,
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
	long_description=long_description,
	long_description_content_type="text/markdown",
	ext_modules=ext_modules,
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
	cmdclass={"build_ext": build_ext},
	zip_safe=False
)
