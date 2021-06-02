<p align="center">
<img src="https://github.com/Pencilcaseman/librapid/blob/master/branding/color.png" width="800"> 
</p>

[![Build Status](https://github.com/pencilcaseman/librapid/actions/workflows/wheels.yaml/badge.svg)](https://github.com/Pencilcaseman/librapid/actions/workflows/wheels.yaml) [![Documentation Status](https://readthedocs.org/projects/librapid/badge/?version=latest)](https://librapid.readthedocs.io/en/latest/?badge=latest) [![PyPI version fury.io](https://badge.fury.io/py/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![PyPI license](https://img.shields.io/pypi/l/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![PyPI pyversions](https://img.shields.io/pypi/pyversions/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![Downloads per month](https://img.shields.io/pypi/dm/librapid.svg)](https://pypi.python.org/pypi/librapid/) [![Discord](https://img.shields.io/discord/848914274105557043)](https://discord.gg/cGxTFTgCAC)

LibRapid -- a fast, general purpose math and neural network library for C++ and Python

## How it Works

LibRapid is a fully templated header-only C++ library, which can be found at ```./librapid/```. The C++ library is then interfaced with Python using the amazing [PyBind11](https://github.com/pybind/pybind11) library, meaning very little performance is lost between the C++ and Python versions of the library.

LibRapid also aims to provide a consistent interface between the C++ and Python libraries, allowing you to use the library comprehensively in both language.

Additionally, the Python interface has been adjusted slightly from the C++ interface, providing a more "pythonic" feel, while not reducing the functionality in the slightest.

## Installing LibRapid

To install LibRapid as a Python library, simply run ```pip install librapid``` in the command line. Hopefully, there will be precompiled wheels available for your operating system and python version, meaning you will not need a C++ compiler to install it (if this is not the case, a modern C++ compiler will be required)

To use the library for C++ use, a modern C++ compiler will definitely be required. To use the library simply add the directory of this file to your extra include paths, and include the library using ```#include<librapid/librapid.hpp>```

## Things to Note

1. Parallel code -- If using LibRapid mulit-dimensional array with OpenMP, ensure that any parallel blocks define the necessary arrays as ```shared```, otherwise the application will ```segfault```, due to uninitialized arrays being used.

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

Hopefully in the near future, the documentation will be available online, so you will not have to build it yourself...
