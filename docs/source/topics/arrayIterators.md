# Array Iterators

LibRapid provides many methods to iterate over the elements of an array. Each one has its own advantages and
disadvantages, and the best one to use depends heavily upon the situation.

## Implicit Iteration

This is the **simplest and easiest** way to iterate over an array, but is also the **slowest**. This method should only
be used when performance is not a concern or when the array is known to be relatively small.

```cpp
auto a = lrc::Array<int>(lrc::Shape({4, 5}));

for (auto val : a) {
    for (auto val2 : val) {
        val2 = lrc::randint(1, 10);
    }
}

for (const auto &val : a) {
    for (const auto &val2 : val) {
        fmt::print("{} ", val2);
    }
    fmt::print("\n");
}
```

:::{warning}
Due to the way LibRapid works internally, the iterator type returned by `Array::begin()` and `Array::end()` makes use
of the `ArrayView` class. Since this is ***not a direct C++ reference*** many IDEs will claim that the value is unused
and will suggest removing it. **Do not remove it!** The `ArrayView` is still referencing the original array and your
data will still be updated correctly :)

I am currently looking into ways to fix this issue, but it is proving to be quite difficult...
:::

## Subscript Iteration

This method of iterating over an array is slightly faster than implicit iteration, but is still slow compared to other
methods. This involves using a `for` loop to iterate over each axis of the array and then using the `operator[]` to
access the elements.

```cpp
auto a = lrc::Array<int>(lrc::Shape({4, 5}));

for (auto i = 0; i < a.shape()[0]; i++) {
    for (auto j = 0; j < a.shape()[1]; j++) {
        a[i][j] = lrc::randint(1, 10);
    }
}

for (auto i = 0; i < a.shape()[0]; i++) {
    for (auto j = 0; j < a.shape()[1]; j++) {
        fmt::print("{} ", a[i][j]);
    }
    fmt::print("\n");
}
```

## Direct Iteration

This approach is the fastest safe way to iterate over an array. Again, using a `for` loop to iterate over each axis of
the array, but this time using the `operator()` method to access the elements.

This method is ***much faster*** than using the `operator[]` method because no temporary `ArrayView` objects are
created.

```cpp
auto a = lrc::Array<int>(lrc::Shape({4, 5}));

for (auto i = 0; i < a.shape()[0]; i++) {
    for (auto j = 0; j < a.shape()[1]; j++) {
        a(i, j) = lrc::randint(1, 10);
    }
}

for (auto i = 0; i < a.shape()[0]; i++) {
    for (auto j = 0; j < a.shape()[1]; j++) {
        fmt::print("{} ", a(i, j));
    }
    fmt::print("\n");
}
```

## Direct Storage Access

LibRapid's array types have a `Storage` object which stores the actual data of the array. This object can be accessed
via the `Array::storage()` method. This method is the fastest way to iterate over an array, but it is also the most
dangerous, and you should ***only use it if you know what you are doing***.

:::{danger}
This method only works on `ArrayContainer` instances (Array types which own their own data). If you try to use this
approach on any other datatype, such as an `ArrayView` or `Function`, your code will not compile because these types
do not store their own data and hence do not have a `storage()` method.

Note also that this does not give any information about the shape of the array, so you must be careful to ensure that
you are accessing the correct elements.
:::

```cpp
auto a = lrc::Array<int>(lrc::Shape({4, 5}));

for (auto i = 0; i < a.shape().size(); i++) {
	a.storage()[i] = lrc::randint(1, 10);
}

for (auto i = 0; i < a.shape().size(); i++) {
    fmt::print("{} ", a.storage()[i]);
}
```

:::{warning}
The `Storage` object stores the data in row-major order, so you must be careful that you are accessing the correct
elements.

For example, if you have a 3D array with shape `{2, 3, 4}`, the elements will be accessed in the following order:

```
(0, 0, 0)
(0, 0, 1)
(0, 1, 0)
(0, 1, 1)
(0, 2, 0)
(0, 2, 1)
(1, 0, 0)
(1, 0, 1)
(1, 1, 0)
(1, 1, 1)
(1, 2, 0)
(1, 2, 1)
```
:::

## Benchmarks

These benchmarks were performed on a Ryzen 9 3950x CPU with 64GB of RAM using a 2D array of `float` values with shape
`{25000, 25000}`. The code used is included below.

### MSVC

```
Iterator Timer [     ITERATOR     ] -- Elapsed: 1.25978m | Average: 25.19570s
Iterator Timer [ FOR LOOP INDEXED ] -- Elapsed: 1.06851m | Average: 10.68511s
Iterator Timer [ FOR LOOP DIRECT  ] -- Elapsed: 1.03243m | Average: 2.13607s
Iterator Timer [     STORAGE      ] -- Elapsed: 1.00972m | Average: 712.74672ms
```

### GCC (WSL2)

```
Iterator Timer [     ITERATOR     ] -- Elapsed: 1.25978m | Average: 25.19570s
Iterator Timer [ FOR LOOP INDEXED ] -- Elapsed: 1.06851m | Average: 10.68511s
Iterator Timer [ FOR LOOP DIRECT  ] -- Elapsed: 1.03243m | Average: 2.13607s
Iterator Timer [     STORAGE      ] -- Elapsed: 1.00972m | Average: 712.74672ms
```

### Code

```cpp
lrc::Shape benchShape({25000, 25000});

{
    auto a = lrc::Array<float>(benchShape);
    lrc::Timer iteratorTimer(fmt::format("Iterator Timer [ {:^16} ]", "ITERATOR"));
    iteratorTimer.setTargetTime(10);

    while (iteratorTimer.isRunning()) {
        for (auto val : a) {
            for (auto val2 : val) { val2 = 1; }
        }
    }

    fmt::print("{:.5f}\n", iteratorTimer);
}

{
    auto a = lrc::Array<float>(benchShape);
    lrc::Timer iteratorTimer(fmt::format("Iterator Timer [ {:^16} ]", "FOR LOOP INDEXED"));
    iteratorTimer.setTargetTime(10);

    while (iteratorTimer.isRunning()) {
        for (int64_t i = 0; i < a.shape()[0]; i++) {
            for (int64_t j = 0; j < a.shape()[1]; j++) { a[i][j] = 1; }
        }
    }

    fmt::print("{:.5f}\n", iteratorTimer);
}

{
    auto a = lrc::Array<float>(benchShape);
    lrc::Timer iteratorTimer(fmt::format("Iterator Timer [ {:^16} ]", "FOR LOOP DIRECT"));
    iteratorTimer.setTargetTime(10);

    while (iteratorTimer.isRunning()) {
        for (int64_t i = 0; i < a.shape()[0]; i++) {
            for (int64_t j = 0; j < a.shape()[1]; j++) { a(i, j) = 1; }
        }
    }

    fmt::print("{:.5f}\n", iteratorTimer);
}

{
    auto a = lrc::Array<float>(benchShape);
    lrc::Timer iteratorTimer(fmt::format("Iterator Timer [ {:^16} ]", "STORAGE"));
    iteratorTimer.setTargetTime(10);

    while (iteratorTimer.isRunning()) {
        for (int64_t i = 0; i < a.shape().size(); i++) { a.storage()[i] = 1; }
    }

    fmt::print("{:.5f}\n", iteratorTimer);
}
```
