# Getting Started

## Installation

To use LibRapid in your CMake project, first clone the project:

``git clone --recursive https://github.com/LibRapid/libRapid.git``

:::{warning}
Make sure to use the ``--recursive`` flag when cloning the repository. This will ensure that all submodules are
cloned as well!
:::

Make sure you have a structure similar to the following:

```
yourProject/
    CMakeLists.txt
    main.cpp
    librapid/
        CMakeLists.txt
        ...
    ...
```

Next, add the following to your ``CMakeLists.txt``

```cmake
add_subdirectory(librapid)
target_link_libraries(yourTarget PUBLIC librapid)
```

:::{note}
If you are not familiar with CMake, I suggest you follow a quick tutorial on it just to get the hang of the basics.
After that, check out the sample ``CMakeLists.txt`` file in the ``examples`` directory of the repository.

(``examples/templateCMakeLists.txt``)[https://github.com/LibRapid/librapid/blob/master/examples/templateCMakeLists.txt]
:::

That's it! LibRapid will now be compiled and linked with your project!

## Your First Program

```{code-block} cpp
---
linenos: True
---

#include <librapid>
namespace lrc = librapid;

int main() {
    lrc::Array<int> myFirstArray = lrc::fromData({{1, 2, 3, 4},
                                                  {5, 6, 7, 8}});

    lrc::Array<int> mySecondArray = lrc::fromData({{8, 7, 6, 5},
                                                   {4, 3, 2, 1}});

    fmt::print("{}\n\n", myFirstArray);
    fmt::print("{}\n", mySecondArray);

    fmt::print("Sum of two Arrays:\n{}\n", myFirstArray + mySecondArray);
    fmt::print("First row of my Array: {}\n", myFirstArray[0]);
    fmt::print("First row of my Array: {}\n", myFirstArray[0] + mySecondArray[1]);

    return 0;
}
```

## Your First Program: Explained

```{code-block} cpp
---
lineno-start: 1
---

#include <librapid>
namespace lrc = librapid;
```

The first line here allows you to use all of LibRapid's features in your file. The second line isn't required,
but it makes your code shorter and quicker to type.

```{code-block} cpp
---
lineno-start: 5
---

lrc::Array<int> myFirstArray = lrc::fromData({{1, 2, 3, 4},
                                              {5, 6, 7, 8}});

lrc::Array<int> mySecondArray = lrc::fromData({{8, 7, 6, 5},
                                               {4, 3, 2, 1}});
```

These lines create two Array instances from a list of values. Both arrays are 2-dimensional and have 2 rows and 4
columns.

```{code-block} cpp
---
lineno-start: 11
---

fmt::print("{}\n\n", myFirstArray);
fmt::print("{}\n", mySecondArray);
```

Here, we print out the Arrays we just created. Try changing the numbers to see how the formatting changes!

```{code-block} cpp
---
lineno-start: 14
---

fmt::print("Sum of two Arrays:\n{}\n", myFirstArray + mySecondArray);
```

This line performs a simple arithmetic operation on our Arrays and prints the result.

```{code-block} cpp
---
lineno-start: 15
---

fmt::print("First row of my Array: {}\n", myFirstArray[0]);
fmt::print("First row of my Array: {}\n", myFirstArray[0] + mySecondArray[1]);
```

As you can see, Array instances can be indexed with the traditional square bracket notation. This means you can
easily access sub-arrays of higher-dimensional array objects.

Now that you've seen how easy it is to use LibRapid, check out the rest of the documentation to learn more about
the library's features! There are more example programs in the ``examples`` directory of the repository.

---

(``examples/``)[https://github.com/LibRapid/librapid/tree/master/examples]

## Troubleshooting

While I have done my best to make LibRapid compile with as few issues as possible, there are cases where it will not work the first time around. Some issues I have experienced myself or have been told about by other users. Some of these issues and their solutions are shown below:

### Linux with CUDA

If you want to use LibRapid with CUDA on a Linux machine and your code is not compiling, please ensure you have the **development OpenGL** packages installed.

On Ubuntu and similar distros, this can be done with the following:

```
sudo apt-get install libgl1-mesa-dev
```