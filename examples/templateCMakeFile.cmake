# An example CMake script to use LibRapid in your project.
# Read the comments to better understand the options and limitations.

cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17) # LibRapid should compile with C++17 and beyond

add_executable(MyProject main.cpp)

set(LIBRAPID_BUILD_EXAMPLES ON) # Compile the examples
set(LIBRAPID_BUILD_TESTS ON) # Compile the testing suite

set(LIBRAPID_STRICT OFF) # Enable all warnings and treat them as errors
set(LIBRAPID_QUIET OFF) # Silence all warnings

set(LIBRAPID_USE_BLAS ON) # Attempt to use a BLAS library -- if not found,
                          # LibRapid falls back to less optimized routines
set(LIBRAPID_USE_CUDA ON) # Attempt to enable CUDA support
set(LIBRAPID_USE_OMP ON) # Attempt to enable OpenMP
set(LIBRAPID_USE_MULTIPREC ON) # Enable the multiprecision library (longer
                               # compile times and larger executables)

set(LIBRAPID_OPTIMISE_SMALL_ARRAYS ON) # Optimise performance for smaller arrays
                                       # (~500x500) at the cost of lower performance
                                       # for larger matrices (1000x1000 +). In practice
                                       # this enables multithreaded array operations.
                                       # The multithreading overhead makes array operations
                                       # significantly slower for small arrays, but also
                                       # significantly faster for large arrays.

set(LIBRAPID_GET_BLAS OFF) # Clone a pre-built version of OpenBLAS and use it

add_subdirectory(librapid) # This makes the assumption that LibRapid is located in the same
                           # directory as this file. Change this line to suit your setup.

target_link_libraries(MyProject PUBLIC librapid)
