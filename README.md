<p align="center">
<img src="https://github.com/Pencilcaseman/librapid/blob/master/branding/logo_transparent_trimmed.png" width="800">
</p>

[![Wheels (master)](https://github.com/LibRapid/librapid/actions/workflows/wheels.yaml/badge.svg)](https://github.com/LibRapid/librapid/actions/workflows/wheels_master.yaml) [![Documentation Status](https://readthedocs.org/projects/librapid/badge/?version=latest)](https://librapid.readthedocs.io/en/latest/?badge=latest) [![PyPI version fury.io](https://badge.fury.io/py/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![PyPI license](https://img.shields.io/pypi/l/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![Downloads per month](https://img.shields.io/pypi/dm/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![Discord](https://img.shields.io/discord/848914274105557043)](https://discord.gg/cGxTFTgCAC)

## Credits

Thanks to @TheWindoof for his help on the library

## How it Works

LibRapid is a highly-optimized C++ (and CUDA) library which can be found at ```./src/librapid```. The C++ library is interfaced with Python using [PyBind11](https://github.com/pybind/pybind11), meaning very little performance is lost between the C++ backend and Python frontend of the library.

LibRapid also aims to provide a consistent interface between the C++ and Python libraries, allowing you to use the library comprehensively in both languages without having to trawl through two sets of documentation.

Please note that the Python interface has been adjusted slightly from the C++ interface to provide a more "pythonic" feel without reducing the overall functionality.

## Installing LibRapid

### Python

To install LibRapid as a Python library, simply run ```pip install librapid``` in the command line. Hopefully, there will be precompiled wheels available for your operating system and python version, meaning you will not need a C++ compiler to install it (if this is not the case, a modern C++ compiler will be required)

#### Building from Source

To enable CUDA support, to use your own BLAS library or to get a (potentially) more optimised install of the library, you will have to build LibRapid from source:

``` bash
git clone https://github.com/LibRapid/librapid.git --recursive
cd librapid
pip install . -vvv
```

### C++

To use the library for C++ use, a modern C++ compiler will definitely be required. You will need to add all sources to your project and include the main header file ```librapid/librapid.hpp```.

***This method is very tedious. In the future, CMake support will be added to enable easier building and linking***

## Documentation

### Viewing Online

Documentation can be found online here: https://librapid.readthedocs.io/en/latest/

### Building from Source

If you would like to build it yourself, you will need to instal the required software, which can be found below:

```bash
pip install sphinx
pip install breathe
pip install exhale
pip install furo
```

You will also need to install a recent version of Doxygen, which you can find [here](https://www.doxygen.nl/download.html)

To build the documentation, open a command line window in the ```docs``` directory and run ```make html```. You can then open the file ```docs/build/html/index.html``` in a web-browser to view the documentation.

## Performance

LibRapid has highly optimized functions and routines, meaning your code will be faster as a result. Many of the functions can match or even exceed the performance of [NumPy](https://github.com/numpy/numpy)

Both the C++ and Python libraries are designed to work with any CBLAS compatible library, such as [ATLAS](https://github.com/math-atlas/math-atlas) or [OpenBLAS](https://github.com/xianyi/OpenBLAS), though will be fully functional without one, using built-in, but slower, routines.

To use BLAS in C++, simply allow LibRapid to find the ```cblas.h``` file by adding its directory to the additional include paths, and link the required library, such as ```libopenblas.lib``` on Windows. Finally, before you ```#include<librapid/librapid.hpp>```, you'll have to ```#define LIBRAPID_CBLAS``` to let LibRapid know it should use the provided BLAS library.

For the Python API to LibRapid, things are much simpler. If you install the library via ```pip```, it should come precompiled with OpenBLAS, so you don't have to do anything yourself.

If you build LibRapid from source, it will automatically search some specific directories for a BLAS install, though if one is not found, BLAS will not be linked and internal routines will be used instead. Please note that the BLAS install must have the following file structure:

``` None
blas-dir
├── include
|   └── cblas.h
├── lib
|   └── your-blas-lib.lib
└── bin (Only on Windows)
    └── your-blass-dll.dll
```
