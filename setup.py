# -*- coding: utf-8 -*-
import os
import platform
import shutil
import sys
import site
from packaging.version import LegacyVersion
from skbuild import setup
from skbuild.cmaker import get_cmake_version
from skbuild.exceptions import SKBuildError

LR_PYTHON_IMPL = platform.python_implementation()

# Copy OpenBLAS build if present in the root directory
if os.path.exists("openblas_install") and not os.path.exists(os.path.join("src", "librapid", "openblas_install")):
    shutil.copytree("openblas_install", os.path.join("src", "librapid", "openblas_install"))

# Remove _skbuild directory if it already exists. It can lead to issues
if os.path.exists("_skbuild"):
    shutil.rmtree("_skbuild")

# Remove the _librapid_python_cmake directory if it's present. This can cause more issues...
if os.path.exists("_librapid_python_cmake"):
    shutil.rmtree("_librapid_python_cmake")

# If the directory "src/librapid/blas" is empty and "src/librapid/openblas_install" is empty,
# run CMake to automatically detect BLAS before installing the Python library
if not os.path.exists(os.path.join("src", "librapid", "blas")) and not os.path.exists(os.path.join("src", "librapid", "openblas_install")):
    out = os.system("mkdir _librapid_python_cmake && cd _librapid_python_cmake && cmake ..")
    if out != 0:
        print("\nCMake failed to run correctly, so it is likely that BLAS will not be installed with LibRapid")

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

# ======= Uncomment this to install win32api as well =======
if platform.system() == "Windows" and LR_PYTHON_IMPL == "CPython":
    setup_requires.append('pywin32')
    install_requires.append("pywin32")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

data_files = []
if platform.system() == "Windows":
    try:
        root, lib = site.getsitepackages()
        libdir = lib.replace(root, "")
        libdir = libdir.lstrip("\\")
        libdir = libdir.lstrip("/")
    except:
        from distutils.sysconfig import get_python_lib
        libdir = get_python_lib()
    
    dll_files = ["openblas.dll", "libopenblas.dll", "flang.dll", "flangrti.dll", "pgmath.dll", "libomp.dll"]
    
    for filename in dll_files:
        try:
            print("Attempting to open 'src/librapid/blas/{}".format(filename))
            with open(os.path.join("src", "librapid", "blas", filename), "r") as _:
                data_files.append(os.path.join("src", "librapid", "blas", filename))
        except:
            print("Failed to open 'src/librapid/blas/bin/{}'".format(filename))
            pass

    if data_files == []:
        for filename in dll_files:
            try:
                print("Attempting to open 'src/librapid/openblas_install/bin/{}'".format(filename))
                with open(os.path.join("src", "librapid", "openblas_install", "bin", filename), "r") as _:
                    data_files.append(os.path.join("src", "librapid", "openblas_install", "bin", filename))
            except:
                print("Failed to open 'src/librapid/openblas_install/bin/{}'".format(filename))
                pass

	# Log some information
    if data_files == []:
        print("Was not able to load datafiles")
        print("File information:")
        if os.path.exists("src"):
            print("./src")
            print(os.listdir("./src"))

            if (os.path.exists("src/librapid")):
                print("./src/librapid")
                print(os.listdir("./src/librapid"))

                if (os.path.exists("src/librapid/openblas_install")):
                    print("./src/librapid/openblas_install")
                    print(os.listdir("./src/librapid/openblas_install"))

                    if (os.path.exists("src/librapid/openblas_install/bin")):
                        print("./src/librapid/openblas_install/bin")
                        print(os.listdir("./src/librapid/openblas_install/bin"))

# Adjust the datafiles object
if data_files != []:
	data_files = [(os.path.join(".", libdir, "librapid"), data_files[:])]

print("Operating System: {}".format(platform.system()))
print("Additional files being included: {}".format(data_files))
print("\n\n\n\n")

# Load the version number from VERSION.hpp
version_file = open(os.path.join("src", "librapid", "VERSION.hpp"), "r")
__version__ = version_file.readlines()[1].split()[2].replace("\"", "")
version_file.close()

# Locate and read the contents of README.md
with open(os.path.join(ROOT_DIR, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

if sys.maxsize > 2 ** 32 and platform.system() == "Windows":
    # print("Using 64bit Python")
    cmake_args = ["-DCMAKE_GENERATOR_PLATFORM=x64"]
else:
    # print("Using 32bit Python (or an OS other than Windows)")
    cmake_args = []

if os.environ.get("LIBRAPID_NO_ARCH"):
    cmake_args.append("-DLIBRAPID_NO_ARCH=yes")

# cmake_args.append("-DCMAKE_BUILD_TYPE=DEBUG")

setup(
    name="librapid",
    version=__version__,
    author="Toby Davis",
    author_email="pencilcaseman@gmail.com",
    url="https://github.com/Pencilcaseman/librapid",
    description="A fast math and neural network library for Python and C++",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=["librapid"],
    package_dir={"": "src"},
    cmake_args=cmake_args,
    cmake_install_dir="src/librapid",
    license="MIT License",
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
		"Programming Language :: Python :: 3.10",
    ],
    extras_require={"test": "pytest"},
    install_requires=install_requires,
    setup_requires=setup_requires,
    data_files=data_files,
    include_package_data=True,
    zip_safe=False
)
