<p align="center">
<img src="https://github.com/LibRapid/librapid_bin/blob/master/branding/LibRapid_light.png#gh-light-mode-only" width="800">
</p>

<p align="center">
<img src="https://github.com/LibRapid/librapid_bin/blob/master/branding/LibRapid_dark.png#gh-dark-mode-only" width="800">
</p>

![PyPI](https://img.shields.io/pypi/v/librapid?color=green&label=Release&logo=python&logoColor=green) ![PyPI - License](https://img.shields.io/pypi/l/librapid?color=gray&label=Licensed%20under) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/librapid?color=blue&label=Version&logo=python&logoColor=green) [![Discord](https://img.shields.io/discord/848914274105557043?color=blue&label=Discord&logo=Discord)](https://discord.gg/cGxTFTgCAC) ![PyPI - Downloads](https://img.shields.io/pypi/dm/librapid?color=blue&label=Downloads&logo=python&logoColor=green) ![C++ Version](https://img.shields.io/badge/Language-C%2B%2B%2017-orange)


---

[![Build (Windows)](https://github.com/LibRapid/librapid/actions/workflows/build-windows.yaml/badge.svg)](https://github.com/LibRapid/librapid/actions/workflows/build-windows.yaml)

[![Build (Linux)](https://github.com/LibRapid/librapid/actions/workflows/build-linux.yaml/badge.svg)](https://github.com/LibRapid/librapid/actions/workflows/build-linux.yaml)

[![Build (MacOS)](https://github.com/LibRapid/librapid/actions/workflows/build-macos.yaml/badge.svg)](https://github.com/LibRapid/librapid/actions/workflows/build-macos.yaml)

[![Wheels](https://github.com/LibRapid/librapid/actions/workflows/wheels.yaml/badge.svg)](https://github.com/LibRapid/librapid/actions/workflows/wheels.yaml)

[![Documentation Status](https://readthedocs.org/projects/librapid/badge/?version=latest)](https://librapid.readthedocs.io/en/latest/?badge=latest)

---

# What is LibRapid?

LibRapid is a high performance Array library, supporting a wide range of optimised calculations which can be performed on the CPU or GPU (via CUDA). All calculations are vectorised with SIMD instructions and are run on multiple threads (if necessary) to make them as fast as possible on any given machine.

There are also a wide range of helper functions and classes to aid the development of your own project.

LibRapid is highly templated, meaning it can conform to exactly your needs with minimal compile-times and even support for custom datatypes.

# Current Development Stage

At the current point in time, LibRapid C++ is under rapid development by me ([pencilcaseman](https://github.com/Pencilcaseman)).

I am currently doing my A-Levels and do not have time to work on the library as much as I would like, so if you or someone you know might be willing to support the development of the library, feel free to create a pull request or chat to us on [Discord](https://discord.com/invite/cGxTFTgCAC). Any help is greatly appreciated!

# Future Plans

My goal for LibRapid is to develop the C++ interface further, at least initially. At some point I want to add Python and Javascript interfaces (in that order) to increase the range of people who can benefit from the library, but the most important thing is the performance of the underlying C++ code.