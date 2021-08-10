# LibRapid CUDA bindings

Due to PyBind11 not fully supporting CUDA compilation in all cases, LibRapid has resorted to creating a library file with the required CUDA bindings which is then linked against the program LibRapid is used in. Unfortunately, this means you may have to build the library yourself.

If you are using LibRapid in Python, running `pip install .` in the home directory should build the library automatically, as well as copy the resulting files to the required location. This means there is (in theory) nothing you need to do to build LibRapid Python with CUDA support.

If you are using the C++ interface to LibRapid and wish to use CUDA, simply run `build.bat` to run the required `nvcc` command to build the library. To use CUDA inside LibRapid, you will need to `#define LIBRAPID_CUDA` before you `#include <librapid/librapid.hpp>`. You will also need to link the generated `librapid_cuda_bindings.lib` file, and the OS must be able to locate `librapid_cuda_bindings.dll`. It is recommended that you copy the generated files into a common directory (such as `C:\opt\librapid\cuda`) and add the path to the system environment variables.

#### Note: CUDA bindings are currently only available for Windows
