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
