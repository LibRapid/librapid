<p align="center">
<img src="https://github.com/Pencilcaseman/librapid/blob/master/branding/logo_transparent_trimmed.png" width="800">
</p>

[![Build Status](https://github.com/pencilcaseman/librapid/actions/workflows/wheels.yaml/badge.svg)](https://github.com/Pencilcaseman/librapid/actions/workflows/wheels.yaml) [![Documentation Status](https://readthedocs.org/projects/librapid/badge/?version=latest)](https://librapid.readthedocs.io/en/latest/?badge=latest) [![PyPI version fury.io](https://badge.fury.io/py/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![PyPI license](https://img.shields.io/pypi/l/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![Downloads per month](https://img.shields.io/pypi/dm/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![Discord](https://img.shields.io/discord/848914274105557043)](https://discord.gg/cGxTFTgCAC)

## Credits

Thanks to @Windoof-Colourblindoof for his help on the library

## How it Works

LibRapid is a fully templated header-only C++ library, which can be found at ```./librapid/```. The C++ library is then interfaced with Python using the amazing [PyBind11](https://github.com/pybind/pybind11) library, meaning very little performance is lost between the C++ and Python versions of the library.

LibRapid also aims to provide a consistent interface between the C++ and Python libraries, allowing you to use the library comprehensively in both language.

Additionally, the Python interface has been adjusted slightly from the C++ interface, providing a more "pythonic" feel, while not reducing the functionality in the slightest.

## Installing LibRapid

To install LibRapid as a Python library, simply run ```pip install librapid``` in the command line. Hopefully, there will be precompiled wheels available for your operating system and python version, meaning you will not need a C++ compiler to install it (if this is not the case, a modern C++ compiler will be required)

To use the library for C++ use, a modern C++ compiler will definitely be required. To use the library simply add the directory of this file to your extra include paths, and include the library using ```#include<librapid/librapid.hpp>```

## Documentation

Unfortunately, the documentation cannot currently be built by [ReadTheDocs](https://readthedocs.org/) due to a bug in the version of [Doxygen](https://www.doxygen.nl/index.html) they are using. It would seem that the systems will be getting an upgrade in the near future, hopefully moving to a newer version of Doxygen, though this is still not certain.

If you would like to view the documentation, you will have to build it yourself. This is very simple once you have the correct software installed, most of which can be installed via ```pip```.

```bash
pip install sphinx
pip install breathe
pip install exhale
pip install furo
```

You will also need to install a recent version of Doxygen, which you can find [here](https://www.doxygen.nl/download.html)

To build the documentation, open a command line window in the ```docs``` directory and run ```make html```. You can then open the file ```docs/build/html/index.html``` in a web-browser to view the documentation.

Hopefully, the documentation will be available online in the near future, so you will not have to build it yourself...

## Performance

LibRapid has highly optimized functions and routines, meaning your code will be faster as a result. Many of the functions can match or even exceed the performance of [NumPy](https://github.com/numpy/numpy)

Both the C++ and Python libraries are designed to work with any CBLAS compatible library, such as [ATLAS](https://github.com/math-atlas/math-atlas) or [OpenBLAS](https://github.com/xianyi/OpenBLAS), though will be fully functional without one, using built-in, but slower, routines.

To use BLAS in C++, simply allow LibRapid to find the ```cblas.h``` file by adding its directory to the additional include paths, and link the required library, such as ```libopenblas.lib``` on Windows. Finally, before you ```#include<librapid/librapid.hpp>```, you'll have to ```#define LIBRAPID_CBLAS``` to let LibRapid know it should use the provided BLAS library.

For the Python API to LibRapid, things are much simpler. If you install the library via ```pip```, it should come precompiled with OpenBLAS, so you don't have to do anything yourself.

Unfortunately, when building OpenBLAS for the Python wheels, they are designed to be optimal on the build system, which is often use very different hardware to normal users. For this reason, the BLAS installs are often not optimal, and may lead to strange and inconsistent speeds while running. To mitigate this slightly, we recommend you install a BLAS library on your machine and then build LibRapid from source if you want to get the most out of it.

LibRapid will automatically search some specific directories for OpenBLAS, though if you have it installed in a differnet place, or you have a different library installed alltogether, you can specify where LibRapid should search for the files when you build it from the command line.

``` None
The following command line options are available
They should be used as follows:
e.g. --blas-dir=c:/opt/openblas

Options:
--no-blas	   << Do not attempt to link against any BLAS library
				   Use only the pre-installed routines (which are slower)
--blas-dir	  << Set the directory where LibRapid can find a CBlas
				   compatible library. LibRapid will expect the directory
				   to contain a file structure like this (Windows example):

				   blas-dir
                   ├── bin
                   |   └── libopenblas.dll
                   ├── include
                   |   └── cblas.h
                   └── lib
                       └── libopenblas.lib
--blas-include  << Set the BLAS include directory. LibRapid will expect
				   cblas.h to be in this directory
--blas-lib	  << Set the BLAS library directory. LibRapid will expect
				   a library file to be here, such as libopenblas.lib
				   or openblas.a
--blas-bin	  << Set the directory of the BLAS binaries on Windows.
				   LibRapid will search for a DLL file
```

To build LibRapid from source and specify commands, use the following commands:

``` bash
git clone https://github.com/Pencilcaseman/librapid.git
cd librapid
python setup.py sdist --blas-dir=C:/opt/openblas
#					 ^~~~~~~~~~~~~~~~~~~~~~~~~~
#					 Change this to your arguments
pip install .
```
