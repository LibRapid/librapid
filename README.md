<!-- <p align="center">
<img src="https://raw.githubusercontent.com/LibRapid/librapid_extras/master/branding/LibRapid_light.png#gh-light-mode-only" width="800">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/LibRapid/librapid_extras/master/branding/LibRapid_dark.png#gh-dark-mode-only" width="800">
</p> -->

<picture>
  <source 
    srcset="https://raw.githubusercontent.com/LibRapid/librapid_extras/master/branding/LibRapid_dark.png" 
    media="(prefers-color-scheme: dark)">
  <img src="https://raw.githubusercontent.com/LibRapid/librapid_extras/master/branding/LibRapid_light.png">
</picture>

![C++ Version](https://img.shields.io/badge/C++-20/23-purple.svg?style=flat&logo=c%2B%2B) ![License](https://img.shields.io/badge/License-MIT-orange.svg?style=flat) [![Discord](https://img.shields.io/discord/848914274105557043?color=blue&label=Discord&logo=Discord)](https://discord.gg/cGxTFTgCAC)

---

[![Continuous Integration](https://github.com/LibRapid/librapid/actions/workflows/continuous-integration.yaml/badge.svg)](https://github.com/LibRapid/librapid/actions/workflows/continuous-integration.yaml)
[![Documentation Status](https://readthedocs.org/projects/librapid/badge/?version=latest)](https://librapid.readthedocs.io/en/latest/?badge=latest)

---

[![Documentation](https://img.shields.io/badge/LibRapid%20Documentation-F01F7A?style=for-the-badge&logo=ReadTheDocs&logoColor=white)](https://librapid.readthedocs.io/en/latest/index.html)

---

![Simple Demo](https://raw.githubusercontent.com/LibRapid/librapid_extras/master/images/librapidSimpleDemo.png)

# What is LibRapid?

LibRapid is an extremely fast, highly-optimised and easy-to-use C++ library for mathematics, linear algebra and more,
with an extremely powerful multidimensional array class at it's core. Every part of LibRapid is designed to provide the
best possible performance without making the sacrifices that other libraries often do.

Everything in LibRapid is templated, meaning it'll just work with almost any datatype you throw at it. In addition,
LibRapid is engineered with compute-power in mind, meaning it's easy to make the most out of the hardware you have.
All array operations are vectorised with SIMD instructions, parallelised via OpenMP and can even be run on external
devices via CUDA and OpenCL. LibRapid also supports a range of BLAS libraries to make linear algebra operations even
faster.

![GPU Array](https://raw.githubusercontent.com/LibRapid/librapid_extras/master/images/simpleGpuArray.png)

What's more, LibRapid provides lazy evaluation of expressions, allowing us to perform optimisations at compile-time to
further improve performance. For example, `dot(3 * a, 2 * transpose(b))` will be compiled into a single `GEMM` call,
with `alpha=6`, `beta=0`, `transA=false` and `transB=true`.

## Why use LibRapid?

If you need the best possible performance and an intuitive interface that doesn't sacrifice functionality, LibRapid is
for you. You can fine-tune LibRapid's performance via the CMake configuration and change the device used for a
computation by changing a single template parameter (e.g. `librapid::backend::CUDA` for CUDA compute).

Additionally, LibRapid provides highly-optimised vectors, complex numbers, multiprecision arithmetic (via custom forks
of MPIR and MPFR) and a huge range of mathematical functions that operate on all of these types. LibRapid also provides
a range of linear algebra functions, machine learning activation functions, and more.

### When to use LibRapid

- When you need the best possible performance
- When you want to write one program that can run on multiple devices
- When you want to use a single library for all of your mathematical needs
- When you want a simple interface to develop with

### When not to use LibRapid

- When you need a rigorously tested and documented library
  - LibRapid is still in early development, so it's not yet ready for production use. That said, we still have a wide
    range of tests which are run on every push to the repository, and we're working on improving the documentation.
- When you need a well-established library.
  - LibRapid hasn't been around for long, and we've got a very small community.
- When you need a wider range of functionality.
  - While LibRapid implements a lot of functions, there are some features which are not yet present in the library. If
    you need these features, you may want to look elsewhere. If you would still like to use LibRapid, feel free to
    [open an issue](https://github.com/LibRapid/librapid/issues/new/choose) and I'll do my best to implement it.

# Documentation

<a href="https://librapid.readthedocs.io/en/latest/" target="_blank"><b>Latest
Documentation</b></a> \
<a href="https://librapid.readthedocs.io/en/develop/" target="_blank"><b>Develop Branch Docs</b></a>

LibRapid uses [Doxygen](https://doxygen.nl/) to parse the source code and extract documentation information. We then use
a combination
of [Breathe](https://breathe.readthedocs.io/en/latest/), [Exhale](https://exhale.readthedocs.io/en/latest/)
and [Sphinx](https://www.sphinx-doc.org/en/master/) to generate a website from this data. The final website is hosted on
[Read the Docs](https://readthedocs.org/).

The documentation is rebuilt every time a change is made to the source code, meaning it is always up-to-date.

# Current Development Stage

At the current point in time, LibRapid C++ is being developed solely by
me ([pencilcaseman](https://github.com/Pencilcaseman)).

I'm currently a student in my first year of university, so time and money are both tight. I'm working on LibRapid in my
spare time, and I'm not able to spend as much time on it as I'd like to.

If you like the library and would like to support its development, feel free to create issues or pull requests, or reach
out to me via [Discord](https://discord.com/invite/cGxTFTgCAC) and we can chat about new features. Any support is massively appreciated.

## [Roadmap](https://github.com/orgs/LibRapid/projects/5/views/1)

The [roadmap](https://github.com/orgs/LibRapid/projects/5/views/1) is a rough outline of what I want to get implemented
in the library and by what point, but **please don't count on features being implemented quickly** -- I can't promise
I'll have the time to implement everything as soon as I'd like... (I'll try my best though!)

If you have any feature requests or suggestions, feel free to create an issue describing it. I'll try to get it working
as soon as possible. If you really need something implemented quickly, a small donation would be appreciated, and would
allow me to bump it to the top of my to-do list.

# Dependencies

LibRapid has a few dependencies to improve functionality and performance. Some of these are optional, and can
be configured with a CMake option. The following is a list of the external dependencies and their purpose (these are all
submodules of the library -- you don't need to install anything manually):

###### Submodules

- [fmt](https://github.com/fmtlib/fmt) - Advanced string formatting
- [doxygen-awesome-css](https://github.com/jothepro/doxygen-awesome-css) - A theme for the Doxygen docs
- [CLBlast](https://github.com/CNugteren/CLBlast) - An OpenCL BLAS library
- [Vc](https://github.com/VcDevel/Vc) - SIMD primitives for C++
- [Jitify](https://github.com/Pencilcaseman/jitify.git) - A CUDA JIT compiler
- [pocketfft](https://github.com/mreineck/pocketfft) - A fast, lightweight FFT library
- [scnlib](https://github.com/eliaskosunen/scnlib.git) - Advanced string parsing

###### External

- [OpenMP](https://www.openmp.org/) - Multi-threading library
- [CUDA](https://developer.nvidia.com/cuda-zone) - GPU computing library
- [OpenCL](https://www.khronos.org/opencl/) - Multi-device computing library
- [OpenBLAS](https://www.openblas.net/) - Highly optimised BLAS library
- [MPIR](https://github.com/wbhart/mpir) - Arbitrary precision integer arithmetic
- [MPFR](https://www.mpfr.org/) - Arbitrary precision real arithmetic
- [FFTW](http://www.fftw.org/) - Fast(est) Fourier Transform library

# Star History

<div align="center">
  <a href="https://star-history.com/#LibRapid/librapid&Timeline">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=LibRapid/librapid&type=Timeline&theme=dark" />
      <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=LibRapid/librapid&type=Timeline" />
      <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=LibRapid/librapid&type=Timeline" />
    </picture>
  </a>
</div>

# Contributors

<div align="center">
  <a href="https://github.com/LibRapid/librapid/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=LibRapid/librapid&max=300&columns=20" alt="Contributors">
  </a>
</div>

# Support

Thanks to JetBrains for providing LibRapid with free licenses for their amazing tools!

<p align="center">
  <a href="https://www.jetbrains.com">
    <img src="https://devclass.com/wp-content/uploads/2018/12/jetbrains-variant-4.png" alt="JetBrains" width="200"/>
  </a>
</p>
