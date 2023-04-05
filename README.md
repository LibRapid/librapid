<p align="center">
<img src="https://raw.githubusercontent.com/LibRapid/librapid_extras/master/branding/LibRapid_light.png#gh-light-mode-only" width="800">
</p>

<p align="center">
<img src="https://raw.githubusercontent.com/LibRapid/librapid_extras/master/branding/LibRapid_dark.png#gh-dark-mode-only" width="800">
</p>

![C++ Version](https://img.shields.io/badge/C++-17/22-purple.svg?style=flat&logo=c%2B%2B) ![License](https://img.shields.io/badge/License-MIT-orange.svg?style=flat) [![Discord](https://img.shields.io/discord/848914274105557043?color=blue&label=Discord&logo=Discord)](https://discord.gg/cGxTFTgCAC)

---

[![Continuous Integration](https://github.com/LibRapid/librapid/actions/workflows/continuous-integration.yaml/badge.svg)](https://github.com/LibRapid/librapid/actions/workflows/continuous-integration.yaml)
[![Documentation Status](https://readthedocs.org/projects/librapid/badge/?version=latest)](https://librapid.readthedocs.io/en/latest/?badge=latest)

---

# An Important Note

I am currently preparing for my A-Level exams and, as such, my ability to work on LibRapid is severely limited. I will try to work on it as much as
reasonably possible, but I cannot guarantee that development will be as fast as it has been in the past. Development will continue in short bursts
when I have time, and I'll be available to answer questions on [Discord](https://discord.gg/cGxTFTgCAC).

Any external contributions are greatly appreciated, and I will most likely
review them within a day or two. If you have any questions, feel free to ask them on the Discord server.

Thank you for your understanding.

# What is LibRapid?

LibRapid is a high performance Array library, supporting a wide range of optimised calculations which can be performed
on the CPU or GPU (via CUDA). All calculations are vectorised with SIMD instructions and are run on multiple threads (if
necessary) to make them as fast as possible on any given machine.

## Why use LibRapid?

LibRapid aims to provide a cohesive ecosystem of functions that interoperate with each other, allowing for faster
development ***and*** faster code execution.

For example, LibRapid implements a wide range of mathematical functions which can operate on primitive types,
multi-precision types, vectors, and arrays. Due to the way these functions are implemented, a single function call can
be used to operate on all of these types, reducing code duplication.

### A small example

To prove the point made above, let's take a look at a simple example. Here, we have a function that maps a value from
one range to another:

```cpp
// Standard "double" implementation
double map(double val, double start1, double stop1, double start2, double stop2) {
    return start2 + (stop2 - start2) * ((val - start1) / (stop1 - start1));
}

// map(0.5, 0, 1, 0, 10) = 5
// map(10, 0, 100, 0, 1) = 0.1
// map(5, 0, 10, 0, 100) = 50
```

This function will accept integers, floats and doubles, but nothing else can be used, limiting its functionality.

Of course, this could be templated to accept other types, but if you passed a `std::vector<double>` to this function,
for example, you'd have to create an edge case to support it. **This is where LibRapid comes in.**

Look at the function below:

```cpp
// An extremely versatile mapping function (used within LibRapid!)
template<typename V, typename B1, typename E1, typename B2, typename E2>
V map(V val, B1 start1, E1 stop1, B2 start2, E2 stop2) {
    return start2 + (stop2 - start2) * ((val - start1) / (stop1 - start1));
}
```

This may look excessively complicated with that many template parameters, but you don't actually need all of those! This
just gives the greatest flexibility. This function can be called with ***almost any LibRapid type!***.

```cpp
map(0.5, 0, 1, 0, 100); //  . . . . . . . . . . . . . . . | 50
map(lrc::Vec2d(0.2, 0.8), 0, 1, 0, 100); // . . . . . . . | (20, 80)
map(0.5, 0, 1, 0, lrc::Vec2d(100, 200)); // . . . . . . . | (50, 100)
map(lrc::Vec2d(-1, -2), 1, 0, lrc::Vec2d(100, 300)); // . | (75, 250)

// ---------------------------------------------------------------------

using namespace lrc::literals; // To use "_f" suffix (also requires multiprecision to be enabled)
// "0.5"_f in this case creates a multiprecision float :)
map("0.5"_f, "0"_f, "1"_f, "0"_f, "100"_f); //  . . . . . | 50.00000000000000

// ---------------------------------------------------------------------

auto val    = lrc::fromData<float>({{1, 2}, {3, 4}});
auto start1 = lrc::fromData<float>({{0, 0}, {0, 0}});
auto end1   = lrc::fromData<float>({{10, 10}, {10, 10}});
auto start2 = lrc::fromData<float>({{0, 0}, {0, 0}});
auto end2   = lrc::fromData<float>({{100, 100}, {100, 100}});

fmt::print("{}\n", lrc::map(val, start1, end1, start2, end2));
// [[10 20]
//  [30 40]]

fmt::print("{}\n", lrc::map(val, start1, end1, start2, end2));
// [[10 20]
//  [30 40]]
```

Note: LibRapid's built-in `map` function has even more functionality! See
the [documentation](https://librapid.readthedocs.io/en/develop/api/function_namespacelibrapid_1a4d7a85c238d46f4f45a0a76b5159a652.html) for
details.

This is just one example of how LibRapid's functions can be used to make your code more concise and more efficient, and
hopefully it's clear to see how powerful this could be when working with more complex functions and types.

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

At the current point in time, LibRapid C++ is under rapid development by
me ([pencilcaseman](https://github.com/Pencilcaseman)).

I am currently doing my A-Levels and do not have time to work on the library as much as I would like, so if you or
someone you know might be willing to support the development of the library, feel free to create a pull request or chat
to us on [Discord](https://discord.com/invite/cGxTFTgCAC). Any help is greatly appreciated!

## [Roadmap](https://github.com/orgs/LibRapid/projects/5/views/1)

The [roadmap](https://github.com/orgs/LibRapid/projects/5/views/1) is a rough outline of what I want to get implemented
in the library and by what point, but **please don't count on features being implemented quickly** -- I can't promise
I'll have the time to implement everything as soon as I'd like... (I'll try my best though!)

If you have any feature requests or suggestions, feel free to create an issue describing it. I'll try to get it working
as soon as possible. If you really need something implemented quickly, a small donation would be appreciated, and would
allow me to bump it to the top of my list of features.

# Future Plans

My goal for LibRapid is to make it faster and easier to use than existing libraries, such as Eigen and XTensor. I plan
to develop an extensive testing and benchmarking suite alongside the code base, to ensure that everything is running as
fast as possible.

My main goal for the future is to implement as many features as possible, while maintaining the high performance
LibRapid requires.

# External Dependencies

LibRapid has a few external dependencies to improve functionality and performance. Some of these are optional, and can
be included with a CMake option. The following is a list of the external dependencies and their purpose (these are all
submodules of the library. You don't need to do anything different):

- Required
    - [fmt](https://github.com/fmtlib/fmt) - Advanced string formatting
    - [scnlib](https://github.com/eliaskosunen/scnlib) - Advanced string parsing
    - [Vc](https://github.com/VcDevel/Vc) - SIMD library
- Optional
    - [OpenMP](https://www.openmp.org/) - Multi-threading library
    - [CUDA](https://developer.nvidia.com/cuda-zone) - GPU computing library
    - [mpfr](https://github.com/Pencilcaseman/mpfr) - Arbitrary precision numbers (integer, real, rational)

# Star History

<div align="center">
  <a href="https://star-history.com/#Librapid/librapid/#date">
    <img src="https://api.star-history.com/svg?repos=LibRapid/librapid&type=Date" alt="Star History" width="700">
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
