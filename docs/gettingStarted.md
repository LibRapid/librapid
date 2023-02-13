# Getting Started

## Installation

To use LibRapid in your CMake project, first clone the project:

```
git clone --recursive https://github.com/LibRapid/libRapid.git
```

add the following to your `CMakeLists.txt`:

```cmake
add_subdirectory(librapid)
target_link_libraries(yourTarget PUBLIC librapid)
```

That's it! LibRapid will now be compiled and linked with your project

## Your First Program

```cpp
#include <librapid>
namespace lrc = librapid;

int main() {
    lrc::Array<float> myArray(lrc::Shape({3, 4});
	
    for (int i = 0; i < myArray.shape()[0]; ++i) {
        for (int j = 0; j < myArray.shape()[1]; ++j) {
            myArray[i][j] = j + i * myArray.shape()[1];
        }
    }
    
    fmt::print("{}\n", myArray);
	
    fmt::print("Sum of two Arrays:\n{}\n", myArray + myArray);
    fmt::print("First row of my Array: {}\n", myArray[0]);
    
    return 0;
}
```

### Your First Program: Explained

```cpp
#include <librapid>
namespace lrc = librapid;
```

The first line here allows you to use all of LibRapid's features in your file. The second line isn't required,
but it makes your code shorter and quicker to type.


```cpp
lrc::Array<float> myArray(lrc::Shape({3, 4});
```

This line constructs a 2-dimensional Array of floats with dimensions (3, 4) allocated on the host. For GPU
allocations, see "Using LibRapid with CUDA."

```cpp
for (int i = 0; i < myArray.shape()[0]; ++i) {
    for (int j = 0; j < myArray.shape()[1]; ++j) {
        myArray[i][j] = j + i * myArray.shape()[1];
    }
}
```

This simple 2-dimensional for loop iterates over every element in the Array, setting each to a different value.

```cpp
fmt::print("{}\n", myArray);
```

This line prints the Array we just created. Look how it's formatted -- what happens if you change the
numbers in the Array?

```cpp
fmt::print("Sum of two Arrays:\n{}\n", myArray + myArray);
```

This line prints the result of adding our Array to itself. LibRapid Arrays support a wide range of
operations, arithmetic operations among them.

```cpp
fmt::print("First row of my Array: {}\n", myArray[0]);
```

The last line of the program prints the first row of the Array we constructed. You already saw that Arrays
could be indexed, but printing it out should help to show that they operate almost the same as a complete
Array object.
