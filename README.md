<p align="center">
<img src="https://github.com/Pencilcaseman/librapid/blob/master/branding/logo_transparent_trimmed.png" width="800">
</p>

[![Wheels (master)](https://github.com/LibRapid/librapid/actions/workflows/wheels.yaml/badge.svg)](https://github.com/LibRapid/librapid/actions/workflows/wheels_master.yaml) [![Documentation Status](https://readthedocs.org/projects/librapid/badge/?version=latest)](https://librapid.readthedocs.io/en/latest/?badge=latest) [![PyPI version fury.io](https://badge.fury.io/py/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![PyPI license](https://img.shields.io/pypi/l/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![Discord](https://img.shields.io/discord/848914274105557043)](https://discord.gg/cGxTFTgCAC) [![Downloads](https://static.pepy.tech/personalized-badge/librapid?period=total&units=international_system&left_color=grey&right_color=purple&left_text=Downloads%20(total))](https://pepy.tech/project/librapid) [![Downloads](https://static.pepy.tech/personalized-badge/librapid?period=month&units=international_system&left_color=grey&right_color=orange&left_text=Downloads%20(month))](https://pepy.tech/project/librapid)

<!--
[![Downloads per month](https://img.shields.io/pypi/dm/librapid.svg)](https://pypi.python.org/pypi/librapid/)
-->

## Credits

Thanks to @NervousNullPtr for his help on the library. He and I collaborate to provide the best user experience possible, as
well as to produce innovative new ideas and faster, more efficient code.

## Why Use Librapid?

In short, LibRapid aims to allow for faster mathematical calculations, including multidimensional arrays, machine
learning and arbitrary-precision arithmetic. LibRapid also includes some helper functions to reduce dependencies, such
as basic colour manipulation and access to console properties. The entire library is highly optimised and supports GPU
calculations where applicable.

## How it Works

LibRapid is a highly-optimized C++ (and CUDA) library which can be found at ```./src/librapid```. The C++ library is
interfaced with Python using [PyBind11](https://github.com/pybind/pybind11), meaning very little performance is lost
between the C++ backend and Python frontend of the library. LibRapid also makes use of some
of [Agner Fog's](https://agner.org) libraries and optimisations to accelerate low-level functions and increase
performance.

LibRapid also aims to provide a consistent interface between the C++ and Python libraries, allowing you to use the
library comprehensively in both languages without having to trawl through two sets of documentation.

Please note that the Python interface has been adjusted slightly from the C++ interface to provide a more "pythonic"
feel without reducing the overall functionality.

## Installing LibRapid

### Python

To install LibRapid as a Python library, simply run ```pip install librapid``` in the command line. Hopefully, there will be precompiled wheels available for your operating system and python version, meaning you will not need a C++ compiler to install it (if this is not the case, a modern C++ compiler will be required)

#### Note

Occasionally, the `pip install` version of LibRapid will cause a `segfault` error. This is a known bug we are trying to fix, but in the meantime, you can run `pip install librapid --no-binary librapid`

#### Building from Source

To enable CUDA support, to use your own BLAS library or to get a (potentially) more optimised install of the library,
you will have to build LibRapid from source:

``` bash
git clone https://github.com/LibRapid/librapid.git --recursive
cd librapid
pip install . -vvv
```

### C++

To use the library in C++, you have a few options. The first option is to download the zip file for librapid, copy +
paste all of the source and header files and add them all to your project.

You can also use librapid in your `CMake` projects by either using the `FetchContent` feature, or by adding the librapid
subdirectory:

Using `FetchContent`

```cmake
add_executable(MyApp myapp.cpp)

include(FetchContent)
FetchContent_Declare(librapid GIT_REPOSITORY https://github.com/librapid/librapid.git)
FetchContent_MakeAvailable(librapid)

target_link_libraries(MyApp librapid)
```

Using `add_subdirectory`

```cmd
# Clone the library into your project
git clone https://github.com/librapid/librapid.git --recursive
```

```cmake
add_executable(MyApp myapp.cpp)

add_subdirectory(librapid)

target_link_libraries(MyApp librapid)
```

## Documentation

### Viewing Online

Documentation can be found online here: https://librapid.readthedocs.io/en/latest/

### Building from Source

If you would like to build it yourself, you will need to instal the required software, which can be found below:

Some python packages are required to build the docs. These can be installed by
running ```pip install -r docs/requirements.txt```

You will also need to install a recent version of Doxygen, which you can
find [here](https://www.doxygen.nl/download.html)

To build the documentation, open a command line window in the ```docs``` directory and run ```make html```. You can then
open the file ```docs/build/html/index.html``` in a web-browser to view the documentation.

## Performance

LibRapid has highly optimized functions and routines, meaning your code will be faster as a result. Nearly all functions
exceed the performance of [NumPy](https://github.com/numpy/numpy) and equivlent libraries, and functions are being
optimised all the time. Anything slower than `NumPy` is concidered a bug, and LibRapid developers will attempt to
optimise it until they are satisfied with the performance.

Both the C++ and Python libraries are designed to work with any CBLAS compatible library, such
as [ATLAS](https://github.com/math-atlas/math-atlas) or [OpenBLAS](https://github.com/xianyi/OpenBLAS), though will be
fully functional without one, using built-in, but slower, routines.

### Parallel Code

LibRapid is designed to use multiple theads to run code faster. The number of threads used for these operations will
default to the number of threads available on the system, however this may end up leading to slower code when the number
of threads exceeds 12-16. To set the number of threads, use the following functions:

``` cpp
// C++
librapid::setNumThreads(<num>); // Set threads to <num>. <num> must be  apositive integer.
librapid::getNumThreads();      // Get the number of threads. If OpenMP was not found, it returns 1.
```

``` python
# Python
librapid.setNumThreads(<num>); # Set threads to <num>. <num> must be a positive integer.
librapid.getNumThreads();      # Get the number of threads. If OpenMP was not found, it returns 1.
```

The `CMakeLists.txt` file will attempt to find a Blas installation automatically, though it may fail if the files are
not strucutred correctly (see below).

If you build LibRapid from source, it will automatically search some specific directories for a BLAS install, though if
one is not found, BLAS will not be linked and internal routines will be used instead. Please note that the BLAS install
must have the following file structure:

``` None
blas-dir
├── include
|   └── cblas.h
├── lib
|   └── your-blas-lib.lib
└── bin (Only on Windows)
    └── your-blas-dll.dll
```

### Recommended Setup for Optimal Performance

The recommended Blas library to use is `OpenBLAS`, though it is very slow and tedious to build (especially on Windows).
For this reason, pre-built binaries are provided which are optimised for most processors and architectures. To download
these, go to https://github.com/LibRapid/librapid/actions and select the most recent build (don't worry if it failed!).
Scroll down to the bottom of the page and download the zip file for your operating system:

``` None
OpenBLAS on macos-latest    :   MacOS
OpenBLAS on ubuntu-latest   :   Linux (not just Ubuntu)
OpenBLAS on windows-latest  :   Windows
```

Unzip this and put the contents in `C:/opt/OpenBLAS` on Windows or `/opt/OpenBLAS` on Linux and MacOS. The folder should
look similar to this (example is for Windows):

``` None
<root directory> # C:/, /, etc.
└── opt
    ├── bin
    │   └── openblas.dll
    ├── include
    │   └── openblas
    │       ├── cblas.h
    │       ├── f77blas.h
    │       ├── lapack.h
    │       ├── lapacke.h
    │       ├── lapacke_config.h
    │       ├── lapacke_example_aux.h
    │       ├── lapacke_mangling.h
    │       ├── lapacke_utils.h
    │       ├── openblas
    │       │   └── lapacke_mangling.h
    │       └── openblas_config.h
    ├── lib
    │   ├── openblas.lib
    │   └── pkgconfig
    │       └── openblas.pc
    └── share
        └── cmake
            └── OpenBLAS
                ├── OpenBLASConfig.cmake
                ├── OpenBLASConfigVersion.cmake
                ├── OpenBLASTargets-release.cmake
                └── OpenBLASTargets.cmake
```

With OpenBLAS set up in this way, LibRapid will automatically find and use it, whether you're in C++ or building from
source for Python. This will (most likely) also give the best performance for the library.
