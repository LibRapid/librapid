Getting Started
###############

Installation
-------------

To use LibRapid in your CMake project, first clone the project: ``git clone --recursive https://github.com/LibRapid/libRapid.git``

Next, add the following to your ``CMakeLists.txt``

.. code-block:: cmake

    add_subdirectory(librapid)
    target_link_libraries(yourTarget PUBLIC librapid)

That's it! LibRapid will now be compiled and linked with your project!

Your First Program
------------------

.. code-block:: cpp
    :linenos:

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

Your First Program: Explained
=============================

.. code-block:: cpp
    :lineno-start: 1

    #include <librapid>
    namespace lrc = librapid;

The first line here allows you to use all of LibRapid's features in your file. The second line isn't required,
but it makes your code shorter and quicker to type.

.. code-block:: cpp
    :lineno-start: 5

    lrc::Array<int> myFirstArray = lrc::fromData({{1, 2, 3, 4},
                                                  {5, 6, 7, 8}});

    lrc::Array<int> mySecondArray = lrc::fromData({{8, 7, 6, 5},
                                                   {4, 3, 2, 1}});

These lines create two Array instances from a list of values. Both arrays are 2-dimensional and have 2 rows and 4
columns.

.. code-block:: cpp
    :lineno-start: 11

    fmt::print("{}\n\n", myFirstArray);
    fmt::print("{}\n", mySecondArray);

Here, we print out the Arrays we just created. Try changing the numbers to see how the formatting changes!

.. code-block:: cpp
    :lineno-start: 14

    fmt::print("Sum of two Arrays:\n{}\n", myFirstArray + mySecondArray);

This line performs a simple arithmetic operation on our Arrays and prints the result.

.. code-block:: cpp
    :lineno-start: 15

    fmt::print("First row of my Array: {}\n", myFirstArray[0]);
    fmt::print("First row of my Array: {}\n", myFirstArray[0] + mySecondArray[1]);

As you can see, Array instances can be indexed with the traditional square bracket notation. This means you can
easily access sub-arrays of higher-dimensional array objects.
