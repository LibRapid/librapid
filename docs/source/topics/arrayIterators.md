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
of the `GeneralArrayView` class. Since this is ***not a direct C++ reference*** many IDEs will claim that the value is unused
and will suggest removing it. **Do not remove it!** The `GeneralArrayView` is still referencing the original array and your
data will still be updated correctly :)

Keep in mind that this issue only comes up when you're using the non-const iterator, which is when you're assigning to
the iterator.

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

This method is ***much faster*** than using the `operator[]` method because no temporary `GeneralArrayView` objects are
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
approach on any other datatype, such as an `GeneralArrayView` or `Function`, your code will not compile because these types
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

These benchmarks were performed on a Ryzen 9 3950x CPU with 64GB of RAM. The code used is included below.

### $25000 \times 25000$ array of `float`s

#### MSVC

```none
Iterator Timer [     ITERATOR     ] -- Elapsed: 1.25978m | Average: 25.19570s
Iterator Timer [ FOR LOOP INDEXED ] -- Elapsed: 1.06851m | Average: 10.68511s
Iterator Timer [ FOR LOOP DIRECT  ] -- Elapsed: 1.03243m | Average: 2.13607s
Iterator Timer [     STORAGE      ] -- Elapsed: 1.00972m | Average: 712.74672ms
```

#### GCC (WSL2)

```none
Iterator Timer [     ITERATOR     ] -- Elapsed: 1.30497m | Average: 26.09936s
Iterator Timer [ FOR LOOP INDEXED ] -- Elapsed: 1.00171m | Average: 12.02046s
Iterator Timer [ FOR LOOP DIRECT  ] -- Elapsed: 1.00257m | Average: 222.79388ms
Iterator Timer [     STORAGE      ] -- Elapsed: 1.00265m | Average: 268.56730ms
```

### $1000 \times 1000$ array of `float`s

#### MSVC

```none
Iterator Timer [     ITERATOR     ] -- Elapsed: 20.03113s | Average: 60.51699ms
Iterator Timer [ FOR LOOP INDEXED ] -- Elapsed: 20.01374s | Average: 20.56911ms
Iterator Timer [ FOR LOOP DIRECT  ] -- Elapsed: 20.00305s | Average: 3.65019ms
Iterator Timer [     STORAGE      ] -- Elapsed: 20.00049s | Average: 1.45257ms
```

#### GCC (WSL2)

```none
Iterator Timer [     ITERATOR     ] -- Elapsed: 20.03222s | Average: 75.30909ms
Iterator Timer [ FOR LOOP INDEXED ] -- Elapsed: 20.00276s | Average: 23.67190ms
Iterator Timer [ FOR LOOP DIRECT  ] -- Elapsed: 20.00003s | Average: 62.70073us
Iterator Timer [     STORAGE      ] -- Elapsed: 20.00014s | Average: 242.00937us
```

### $100 \times 100$ array of `float`s

#### MSVC

```none
Iterator Timer [     ITERATOR     ] -- Elapsed: 10.00005s | Average: 594.18031us
Iterator Timer [ FOR LOOP INDEXED ] -- Elapsed: 10.00007s | Average: 210.48345us
Iterator Timer [ FOR LOOP DIRECT  ] -- Elapsed: 10.00003s | Average: 14.38816us
Iterator Timer [     STORAGE      ] -- Elapsed: 10.00001s | Average: 14.94997us
```

#### GCC (WSL2)

```none
Iterator Timer [     ITERATOR     ] -- Elapsed: 10.00055s | Average: 621.22918us
Iterator Timer [ FOR LOOP INDEXED ] -- Elapsed: 10.00001s | Average: 235.57702us
Iterator Timer [ FOR LOOP DIRECT  ] -- Elapsed: 10.00000s | Average: 650.03031ns
Iterator Timer [     STORAGE      ] -- Elapsed: 10.00000s | Average: 2.44980us
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
