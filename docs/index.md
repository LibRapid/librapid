# LibRapid Documentation

![LibRapid](https://github.com/LibRapid/librapid_bin/blob/master/branding/LibRapid_dark.png)

---

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

auto val    = lrc::Array<float>(lrc::Shape({2, 2}));
auto start1 = lrc::Array<float>(lrc::Shape({2, 2}));
auto end1   = lrc::Array<float>(lrc::Shape({2, 2}));
auto start2 = lrc::Array<float>(lrc::Shape({2, 2}));
auto end2   = lrc::Array<float>(lrc::Shape({2, 2}));

val    << 1, 2, 3, 4;
start1 << 0, 0, 0, 0;
end1   << 10, 10, 10, 10;
start2 << 0, 0, 0, 0;
end2   << 100, 100, 100, 100;

fmt::print("{}\n", lrc::map(val, start1, end1, start2, end2));
// [[10 20]
//  [30 40]]
```

Note: LibRapid's built-in `map` function has even more functionality! See
the [documentation](https://librapid.github.io/librapid/namespacelibrapid.html#aa77bd94fdda1a889654275f47d258390) for
details.

This is just one example of how LibRapid's functions can be used to make your code more concise and more efficient, and
hopefully it's clear to see how powerful this could be when working with more complex functions and types.

# Current Development Stage

At the current point in time, LibRapid C++ is under rapid development by
me ([pencilcaseman](https://github.com/Pencilcaseman)).

I am currently doing my A-Levels and do not have time to work on the library as much as I would like, so if you or
someone you know might be willing to support the development of the library, feel free to create a pull request or chat
to us on [Discord](https://discord.com/invite/cGxTFTgCAC). Any help is greatly appreciated!

## Roadmap

The [roadmap](https://github.com/orgs/LibRapid/projects/5/views/1) is a rough outline of what I want to get implemented
in the library and by what point, but **please don't count on features being implemented quickly** -- I can't promise
I'll have the time to implement everything as soon as I'd like... (I'll try my best though!)

If you have any feature requests or suggestions, feel free to create an issue describing it. I'll try to get it working
as soon as possible. If you really need something implemented quickly, a small donation would be appreciated, and would
allow me to bump it to the top of my list of features.
