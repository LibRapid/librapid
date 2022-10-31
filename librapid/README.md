# LibRapid

High Performance C++ Library Mathematical Programs

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

Now, in your code, add the following where required:

```cpp
#include <librapid>

namespace lrc = librapid; // Optional -- for brevity
```

## Options

While I've done the best to provide optimal default settings, sometimes they won't give you the results you want.
LibRapid provides a range of runtime and compile-time options to customise the behaviour of your code.

### CMake Options

When using LibRapid in your CMake project, the following options are configurable (`name => default`):

- `LIBRAPID_BUILD_EXAMPLES => OFF` (Build examples?)
- `LIBRAPID_BUILD_TESTS => OFF` (Build tests?)
- `LIBRAPID_STRICT => OFF` (Force warnings into errors?)
- `LIBRAPID_QUIET => OFF` (Disable warnings)
- `LIBRAPID_GET_BLAS => OFF` (Clone a prebuilt version of OpenBLAS?)
- `LIBRAPID_USE_CUDA => ON` (Automatically search for CUDA?)
- `LIBRAPID_USE_OMP => ON` (Automatically search for OpenMP?)
- `LIBRAPID_USE_MULTIPREC => OFF` (Include multiprecision library -- more on this elsewhere in documentation)

### Multithreading

By default, LibRapid will automatically run sufficiently large loops in paralle, however, the branch required for this
dynamic selection can cause a very slight performance hit with smaller arrays. For this reason, if you know you'll only
be dealing with relatively small arrays (500x500 or smaller), it might make sense to disable this.

To do so, simply define `LIBRAPID_OPTIMISE_SMALL_ARRAYS` before your `#include <librapid>`

For more multithreading options, check out the `librapid::global` namespace in the documentation.

### Debug Information

When compiling LibRapid in Release mode, you may want to enable the error checking and logging. It's enabled by default
in debug mode.

To enable this, define `LIBRAPID_ENABLE_ASSERT` before including LibRapid.

Again, there are a few more settings for assertions in `librapid::global`
