from inspect import FullArgSpec
from setuptools import setup

# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir

from cpufeature import CPUFeature as features

from pathlib import Path
import platform
import os
from os import listdir
from os.path import isfile, join
import sys

try:
	from setuptools import setup, Extension, find_packages
except ImportError:
	from distutils.core import setup, Extension

	def find_packages(where='.'):
		return [folder.replace("/", ".").lstrip(".")
				for (folder, _, files) in os.walk(where)
				if "__init__.py" in files]


"""
Find an OpenBLAS directory somewhere to link against for
faster matrix operations. This file is mainly for use
when building the wheels on GitHub, though may also be
applicable to users building from source with OpenBLAS
installed in a suitable directory.

If you have a different CBlas compatible BLAS library
installed on your system, please specify the location
of the relevant files to setup.py when compiling.
"""

import platform
import os
from os import listdir
from os.path import isfile, join

def find_blas(args):
	# Attempt to find a BLAS library

	if "--no-blas" in args:
		print("Not searching for a BLAS library")
		return None

	if len(args) == 0:
		# Search through default directories for OpenBLAS
		search_dirs = []

		vcdir = None

		if platform.system() == "Windows":
			if platform.architecture()[0] == "64bit":
				vcdir = "openblas_x64-windows"
			else:
				vcdir = "openblas_x86-windows"
		elif platform.system() == "Linux":
			if platform.architecture()[0] == "64bit":
				vcdir = "openblas_x64-linux"
			else:
				print("OpenBLAS cannot be installed on 32bit Linux")
				return None
		elif platform.system() == "Darwin":
			vcdir = "openblas_x64-osx"

		if platform.system() == "Windows":
			search_dirs.append("C:/opt/openblas")
			search_dirs.append("C:/opt/vcpkg/packages/" + vcdir)

			search_dirs.append("C:/dev/openblas")
			search_dirs.append("C:/dev/vcpkg/packages/" + vcdir)
		else:
			search_dirs.append("/opt/openblas")
			search_dirs.append("/opt/vcpkg/packages/" + vcdir)

			search_dirs.append("/dev/openblas")
			search_dirs.append("/dev/vcpkg/packages/" + vcdir)

		# Search a few relative directories for vcpkg packages
		search_dirs.append(os.getcwd() + "/vcpkg/packages/" + vcdir)
		search_dirs.append(os.getcwd() + "/../vcpkg/packages/" + vcdir)
		search_dirs.append(os.getcwd() + "/../../vcpkg/packages/" + vcdir)
		search_dirs.append(os.getcwd() + "/../../../vcpkg/packages/" + vcdir)

		for dir in search_dirs:
			print("Searching directory", dir, "for BLAS libraries")
			cmd = {"--blas-dir": dir}
			blas = find_blas(cmd)
			if blas is not None:
				return blas
		return None

	blas_dir = {}
	blas_bin = None
	blas_include = None
	blas_lib = None
	found_blas = False

	if "--blas-dir" in args:
		# If there is a specified directory, search through it

		search_dir = args["--blas-dir"]
		
		try:
			dirs = listdir(search_dir)
			bin_dir = ""
			include_dir = ""
			lib_dir = ""

			print("Found directory ", search_dir)

			for dir in dirs:
				if dir.lower() == "bin":
					bin_dir = search_dir + "/" + dir
				elif dir.lower() == "include":
					# For vcpkg OpenBLAS, first check if there
					# is another directory

					if "openblas" in listdir(search_dir + "/" + dir):
						print("Found vcpkg OpenBLAS, so shifting into 'openblas' directory for cblas.h")
						include_dir = search_dir + "/" + dir + "/openblas"
					else:
						include_dir = search_dir + "/" + dir
				elif dir.lower() == "lib":
					lib_dir = search_dir + "/" + dir

			cmd = {}

			if bin_dir is not None:
				cmd["--blas-bin"] = bin_dir
			
			if include_dir is not None:
				cmd["--blas-include"] = include_dir

			if lib_dir is not None:
				cmd["--blas-lib"] = lib_dir

			return find_blas(cmd)
		except FileNotFoundError:
			print("Directory", search_dir, "was not found")
			return None
	else:
		# Attempt to use specified folders for each section

		has_required = False
		if "--blas-include" in args and "--blas-lib" in args:
			if platform.system() == "Windows":
				if "--blas-bin" in args:
					has_required = True
					blas_bin = args["--blas-bin"]
				else:
					has_required = False
			else:
				has_required = True

		if not has_required:
			print("Required inputs not provided")
			found_blas = False
		else:
			print("Required inputs were found. Searching for BLAS library")

			blas_include = args["--blas-include"]
			blas_lib = args["--blas-lib"]

			# Find the corresponding file in each directory
			found = [platform.system() != "Windows", False, False]

			if blas_bin is not None and not found[0]:
				filename = find_bin(blas_bin)

				if filename is not None:
					# A valid file was found
					print("Found a valid .dll at", blas_bin)
					found[0] = True
					blas_dir["bin"] = [blas_bin, filename]

			if blas_include is not None:
				filename = find_include(blas_include)
				
				if filename is not None:
					# A valid file was found
					print("Found a valid cblas file at", blas_include)
					found[1] = True
					blas_dir["include"] = [blas_include, filename]

			if blas_lib is not None:
				filename = find_lib(blas_lib)

				if filename is not None:
					# A valid library was found
					print("Found a valid library at", blas_lib)
					found[2] = True
					blas_dir["lib"] = [blas_lib, filename]

			# Only set this if the correct files are found
			found_blas = all(found)

	if found_blas:
		return blas_dir
	return None

def find_bin(base_path):
	"""
	Search for a Windows DLL file in the
	provided directory. If the file is found,
	return its name, otherwise return None
	"""
	try:
		files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

		for file in files:
			if file.endswith(".dll"):
				return file
		return None
	except FileNotFoundError:
		print("Suitable binary file not found")
		return None

def find_include(base_path):
	"""
	Search for cblas.h in the provided directory.
	If it exists, return its name, otherwise
	return none.
	"""

	known_files = ["cblas.h", "mkl_cblas.h"]

	try:
		files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

		for file in files:
			if file in known_files:
				return file
		return None
	except FileNotFoundError:
		print("Suitable header file not found")
		return None

def find_lib(base_path):
	"""
	Search for a valid BLAS compiled library.
	If it is found, return its name, otherwise
	return None
	"""

	# We know about .a .so .lib
	# but they only work on specifc
	# operating systems

	known_endings = []

	if platform.system() == "Windows":
		known_endings.append(".lib")
	if platform.system() == "Linux":
		known_endings.append(".a")
		known_endings.append(".so")
	if platform.system() == "Darwin":
		known_endings.append(".a")

	try:
		files = [f for f in listdir(base_path) if isfile(join(base_path, f))]

		for file in files:
			for ending in known_endings:
				if file.endswith(ending):
					# Remove the filetype from the filename
					new_name = file[:-len(ending)]

					if platform.system() in ["Linux", "Darwin"]:
						# If the name starts with "lib", remove it
						return new_name[3:]
					return new_name

		return None
	except FileNotFoundError:
		print("Suitable library file not found")
		return None

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
	"""
	Enable a load of compiler optimizations. The specific
	optimizations enabled will differ based on the compiler
	and the host architecture. Aim to produce the fastest
	code possible.
	"""

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

		if features["OS_AVX512"]:
			res += ["/arch:AVX512"]
		elif features["AVX2"]:
			res += ["/arch:AVX2"]
		elif features["OS_AVX"]:
			res += ["/arch:AVX"]

		return res
	elif c in ("gcc", "g++"):
		return ["-O3", "-mavx"] + ["-m64"] if platform.architecture()[0] == "64bit" else []
	elif c == "clang":
		return ["-O3", "-mavx"] + ["-m64"] if platform.architecture()[0] == "64bit" else []
	elif c == "unix":
		return ["-O3", "-mavx"] + ["-m64"] if platform.architecture()[0] == "64bit" else []
	return []

# The following command line options are available
# They should be used as follows:
# e.g. --blas-dir=c:/opt/openblas
#
# Options:
# --no-blas			<<  Do not attempt to link against any BLAS library
#						Use only the pre-installed routines (which are slower)
# --blas-dir		<<  Set the directory where LibRapid can find a CBlas
#						compatible library. LibRapid will expect the directory
#						to contain a file structure like this (Windows example):
#
#						blas-dir
#						├── bin
#						|	└── libopenblas.dll
#						├── include
#						|	└── cblas.h
#						└── lib
#							└── libopenblas.lib
# --blas-include	<<  Set the BLAS include directory. LibRapid will expect
#						cblas.h to be in this directory
# --blas-lib		<<  Set the BLAS library directory. LibRapid will expect
#						a library file to be here, such as libopenblas.lib
#						or openblas.a
# --blas-bin		<<	 Set the directory of the BLAS binaries on Windows.
#						 LibRapid will search for a DLL file

# Format the inputs
valid = ["--no-blas", "--blas-dir", "--blas-include", "--blas-lib", "--blas-bin"]
args = {}
index = 0
while index < len(sys.argv):
	passed = sys.argv[index]

	is_valid = False
	for seg in valid:
		if seg in passed:
			is_valid = True
			break

	if is_valid:
		if "=" in passed:
			# Add a dictionary containing the name and value
			pos = passed.index("=")
			args[passed[:pos]] = passed[pos + 1:]
		else:
			args[passed] = None
		
		sys.argv.pop(index)
	else:
		index += 1

print("The current working directory is", os.getcwd())
blas_dir = find_blas(args)

if blas_dir is not None:
	print("Found BLAS")
	print("Binaries:", blas_dir["bin"] if "bin" in blas_dir else None)
	print("Includes:", blas_dir["include"])
	print("Libraries:", blas_dir["lib"])

	# If on Windows and BLAS was found, add binaries to PATH
	if platform.system() == "Windows":
		if blas_dir["bin"][0] not in os.environ["PATH"]:
			os.environ["PATH"] += os.pathsep + blas_dir["bin"][0]

compiler_flags = std_version() + compile_with_omp() + enable_optimizations()
linker_flags = link_omp()
define_macros = [("LIBRAPID_BUILD", 1)]
include_dirs = [os.getcwd()]
library_dirs = []
libraries = []

if blas_dir is not None:
	# BLAS library was found, so link against it
	define_macros.append(("LIBRAPID_CBLAS", 1))
	include_dirs.append(blas_dir["include"][0])
	library_dirs.append(blas_dir["lib"][0])
	libraries.append(blas_dir["lib"][1])

ext_modules = [
	Pybind11Extension("librapid",
		["librapid/pybind_librapid.cpp"],
		extra_compile_args=compiler_flags,
		extra_link_args=linker_flags,
		define_macros=define_macros,
		include_dirs=include_dirs,
		library_dirs=library_dirs,
		libraries=libraries
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
	packages=["librapid"] + ["librapid." + mod for mod in find_packages("librapid")],
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
