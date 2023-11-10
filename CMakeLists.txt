cmake_minimum_required(VERSION 3.16)
project(librapid)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
cmake_policy(SET CMP0077 NEW)

if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(STATUS "[ LIBRAPID ] LibRapid is a top-level project. Using C++23")
    set(CMAKE_CXX_STANDARD 23)
    message(STATUS "[ LIBRAPID ] Building ${CMAKE_BUILD_TYPE}")
endif ()

# LibRapid requires C++20 or later
if (CMAKE_CXX_STANDARD LESS 20)
    message(FATAL_ERROR "LibRapid requires C++20 or later")
endif ()

# Extract version information
file(READ "version.txt" ver)

string(REGEX MATCH "MAJOR ([0-9]*)" _ ${ver})
set(LIBRAPID_MAJOR ${CMAKE_MATCH_1})

string(REGEX MATCH "MINOR ([0-9]*)" _ ${ver})
set(LIBRAPID_MINOR ${CMAKE_MATCH_1})

string(REGEX MATCH "PATCH ([0-9]*)" _ ${ver})
set(LIBRAPID_PATCH ${CMAKE_MATCH_1})

message(STATUS "[ LIBRAPID ] LibRapid Version: v${LIBRAPID_MAJOR}.${LIBRAPID_MINOR}.${LIBRAPID_PATCH}")

# Optional LibRapid settings
option(LIBRAPID_OPTIMISE_SMALL_ARRAYS "Optimise small arrays" OFF)

option(LIBRAPID_BUILD_EXAMPLES "Compile LibRapid C++ Examples" OFF)
option(LIBRAPID_BUILD_TESTS "Compile LibRapid C++ Tests" OFF)
option(LIBRAPID_CODE_COV "Compile LibRapid C++ with Coverage" OFF)

option(LIBRAPID_STRICT "Force all warnings into errors (use with caution)" OFF)
option(LIBRAPID_QUIET "Hide warnings generated WITHIN LibRapid's source" OFF)

option(LIBRAPID_USE_PRECOMPILED_HEADER "Use precompiled headers to speed up compilation" OFF)

option(LIBRAPID_GET_FFTW "Clone and use FFTW -- WARNING: SEE DOCUMENTATION" OFF)
option(LIBRAPID_GET_BLAS "Download pre-built OpenBLAS binaries and use them" OFF)
option(LIBRAPID_GET_MULTIPREC "Download generic multiprecision libraries, as opposed to trying to find one on the system" OFF)

option(LIBRAPID_USE_BLAS "Attempt to use a BLAS library" ON)
option(LIBRAPID_USE_OMP "Attempt to use OpenMP to allow multithreading" ON)
option(LIBRAPID_USE_OPENCL "Search for OpenCL and use it if possible" ON)
option(LIBRAPID_USE_CUDA "Attempt to use CUDA" ON)
option(LIBRAPID_USE_MULTIPREC "Include MPIR and MPFR in the LibRapid build" OFF)
option(LIBRAPID_FAST_MATH "Use potentially less accurate operations to increase performance" OFF)
option(LIBRAPID_NATIVE_ARCH "Use the native architecture of the system" ON)

option(LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH "Preferred vector width for vectorised kernels" 2)
option(LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH "Preferred vector width for vectorised kernels" 4)

option(LIBRAPID_NO_WINDOWS_H "Don't include the Windows.h header" OFF)

option(LIBRAPID_MKL_CONFIG_PATH "Path to the 'MKLConfig.cmake' file" "")

MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
    SET(dirlist "")
    FOREACH (child ${children})
        IF (IS_DIRECTORY ${curdir}/${child})
            LIST(APPEND dirlist ${child})
        ENDIF ()
    ENDFOREACH ()
    SET(${result} ${dirlist})
ENDMACRO()

include(FetchContent)

set(LIBRAPID_HAS_OMP false)
set(LIBRAPID_HAS_BLAS false)
set(LIBRAPID_HAS_CUDA false)

file(GLOB_RECURSE LIBRAPID_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/librapid/include/*.hpp" # Header files
        "${CMAKE_CURRENT_SOURCE_DIR}/librapid/src/*.cpp" # Source files
        "${CMAKE_CURRENT_SOURCE_DIR}/librapid/cxxblas/*.h" # Header files
        "${CMAKE_CURRENT_SOURCE_DIR}/librapid/cxxblas/*.tcc" # Template files
        "${CMAKE_CURRENT_SOURCE_DIR}/librapid/cxxblas/*.cxx" # Source files
)

# Extract system information
set(IS_LINUX OFF)
set(IS_MACOS OFF)
set(IS_WINDOWS OFF)
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(IS_LINUX ON)
endif ()
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(IS_MACOS ON)
endif ()
if (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(IS_WINDOWS ON)

    # Disable shared libraries, since they just cause problems on Windows
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
endif ()

if (${SKBUILD})
    message(STATUS "[ LIBRAPID ] Building for Python")
    set(module_name "_librapid")
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries" FORCE)
    set(LIBRAPID_QUIET ON) # Disable warnings for a cleaner output.

    message(STATUS "[ LIBRAPID ] Cloning PyBind11")
    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
    )
    FetchContent_MakeAvailable(pybind11)

    # Set CMAKE_BUILD_PARALLEL_LEVEL=1 to reduce memory usage
    set(CMAKE_BUILD_PARALLEL_LEVEL 1)

    # Set environment variable CMAKE_BUILD_PARALLEL_LEVEL=1 to reduce memory usage
    set(ENV{CMAKE_BUILD_PARALLEL_LEVEL} 1)

    # Run auto-generation script
    # 1. Find the Python executable
    # 2. cd into 'librapid/bindings/generators'
    # 3. Run '${pythonExe} main.py'

    find_package(Python3 COMPONENTS Interpreter)
    if (NOT Python3_FOUND)
        message(FATAL_ERROR "Python3 not found")
    endif ()
    message(STATUS "[ LIBRAPID ] Python3 executable: ${Python3_EXECUTABLE}")
    execute_process(
            COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/librapid/bindings/generators/main.py"
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/librapid/bindings/generators"
    )

    # Since it is not possible to set the CMake options in setup.py (we have to use pyproject.toml), we have to manually
    # read from the environment to set everything correctly:

    # Enable BLAS
    if (NOT ${LIBRAPID_NO_BLAS})
        set(LIBRAPID_USE_BLAS ON)
    endif ()

    # Get BLAS if the environment variable is set
    if ($ENV{LIBRAPID_GET_BLAS})
        set(LIBRAPID_GET_BLAS ON)
    endif ()

    # Use fast math if the environment variable is set
    if ($ENV{LIBRAPID_FAST_MATH})
        set(LIBRAPID_FAST_MATH ON)
    endif ()

    # Get FFTW if the environment variable is set
    if ($ENV{LIBRAPID_GET_FFTW})
        set(LIBRAPID_GET_FFTW ON)
    endif ()

    # Use CUDA/OpenCL only when not inside GitHub Actions
    if (NOT $ENV{GITHUB_ACTIONS})
        set(LIBRAPID_USE_CUDA ON)
        set(LIBRAPID_USE_OPENCL ON)
        set(LIBRAPID_NATIVE_ARCH ON)
    else ()
        set(LIBRAPID_NO_ALWAYS_INLINE ON) # Reduce compile memory
    endif ()

    set(LIBRAPID_USE_OMP ON)
    set(LIBRAPID_OPTIMISE_SMALL_ARRAYS OFF)

    # Disable multiprec
    set(LIBRAPID_USE_MULTIPREC OFF)
    set(LIBRAPID_GET_MULTIPREC OFF)


    file(GLOB_RECURSE PYTHON_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/librapid/bindings/python/*.hpp" # Header files
            "${CMAKE_CURRENT_SOURCE_DIR}/librapid/bindings/python/*.cpp" # Source files
    )

    pybind11_add_module(
            ${module_name} MODULE

            ${LIBRAPID_SOURCES}
            ${PYTHON_SOURCES}
    )

    # If using MSVC with GitHub, set /Zm512 to reduce memory usage
    if (MSVC AND "$ENV{CI}")
        target_compile_options(${module_name} PRIVATE /Zm512)
    endif ()

    target_compile_definitions(${module_name} PUBLIC
            LIBRAPID_PYTHON
            WIN32_LEAN_AND_MEAN
    )
    install(TARGETS ${module_name} DESTINATION .)
else ()
    set(module_name "librapid")
    add_library(${module_name} STATIC ${LIBRAPID_SOURCES})
endif ()

# clang-format off
target_compile_definitions(${module_name} PUBLIC LIBRAPID_MAJOR=${LIBRAPID_MAJOR})
target_compile_definitions(${module_name} PUBLIC LIBRAPID_MINOR=${LIBRAPID_MINOR})
target_compile_definitions(${module_name} PUBLIC LIBRAPID_PATCH=${LIBRAPID_PATCH})

set(LIBRAPID_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/librapid")
target_compile_definitions(${module_name} PUBLIC LIBRAPID_SOURCE="${LIBRAPID_SOURCE_DIR}")
# clang-format on

# Precompiled Headers
if (LIBRAPID_USE_PRECOMPILED_HEADER)
    target_precompile_headers(${module_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/librapid/include/librapid/core/librapidPch.hpp")
endif ()

if (MINGW)
    message(FATAL_ERROR "LibRapid does not compile with MinGW. Please use MSVC or Clang instead.")
endif ()

if (LIBRAPID_STRICT AND LIBRAPID_QUIET)
    message(FATAL_ERROR "LIBRAPID_STRICT and LIBRAPID_QUIET cannot be enabled at the same time")
endif ()

if (LIBRAPID_STRICT)
    # Enable all warnings and treat them as errors
    if (MSVC)
        target_compile_options(${module_name} PRIVATE /W4 /WX)
    else ()
        target_compile_options(${module_name} PRIVATE -Wall -Wextra -pedantic -Werror)
    endif ()
endif ()

if (LIBRAPID_QUIET)
    # Disable warnings because they're really annoying
    target_compile_options(${module_name} PRIVATE
            $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
            -w>
            $<$<CXX_COMPILER_ID:MSVC>:
            /w>)
endif ()

if (MSVC)
    # Disable MSVC warnings about unsafe functions
    target_compile_definitions(${module_name} PRIVATE _CRT_SECURE_NO_WARNINGS)
endif ()

# See if OpenMP should be linked against
if (LIBRAPID_USE_OMP)
    find_package(OpenMP QUIET)

    if (${OpenMP_FOUND})
        message(STATUS "[ LIBRAPID ] Linking against OpenMP")

        # Link the required library
        target_link_libraries(${module_name} PUBLIC OpenMP::OpenMP_CXX)

        # Add the compile definition so LibRapid knows it has OpenMP
        target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_OMP)
        set(LIBRAPID_HAS_OMP true)
    else ()
        message(WARNING "OpenMP support was requested, but OpenMP was not found")
    endif ()
endif ()

# Include any required modules
if (LIBRAPID_USE_BLAS)
    include(blasConfig)
    configure_blas()
endif ()

# Check if CUDA should be used
if (LIBRAPID_USE_CUDA)
    find_package(CUDAToolkit QUIET)

    if (${CUDAToolkit_FOUND})
        message(STATUS "[ LIBRAPID ] Using CUDA ${CUDAToolkit_VERSION}")

        # Ensure jitify is accessible
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/jitify")
            message(STATUS "[ LIBRAPID ] Jitify exists in LibRapid source tree")
        else ()
            message(FATAL_ERROR "Jitify not found. Ensure the full LibRapid GitHub repo is cloned!")
        endif ()

        target_include_directories(${module_name} PUBLIC
                ${CUDAToolkit_INCLUDE_DIRS}
                "${CMAKE_CURRENT_SOURCE_DIR}/librapid/cudahelpers"
        )

        target_link_directories(${module_name} PUBLIC
                ${CUDA_LIBRARIES}
                ${CUDA_CUBLAS_LIBRARIES}
                ${CUDA_CUFFT_LIBRARIES}
                ${CUDA_CUFFTW_LIBRARIES}
                ${CUDA_CURAND_LIBRARIES}
                ${CUDA_CUSOLVER_LIBRARIES}
                ${CUDA_CUSPARSE_LIBRARIES}
        )

        target_link_libraries(${module_name} PUBLIC
                CUDA::cudart
                CUDA::cuda_driver
                CUDA::nvrtc
                CUDA::cublas
                CUDA::cublasLt
                # CUDA::cublasXt
                CUDA::cufft
                CUDA::cufftw
                CUDA::curand
                CUDA::cusolver
                CUDA::cusparse
        )

        # Link with Dbghelp on Windows
        if (IS_WINDOWS)
            target_link_libraries(${module_name} PUBLIC Dbghelp)
        endif ()

        target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_CUDA)
        target_compile_definitions(${module_name} PUBLIC LIBRAPID_CUDA_STREAM)
        message(STATUS "[ LIBRAPID ] CUDA include directories: ${CUDAToolkit_INCLUDE_DIRS}")
        target_compile_definitions(${module_name} PUBLIC CUDA_INCLUDE_DIRS="${CUDAToolkit_INCLUDE_DIRS}")
        set(LIBRAPID_HAS_CUDA true)
    else ()
        message(WARNING "CUDA support was requested, but a valid CUDA installation could not be found")
    endif ()
endif ()

if (LIBRAPID_USE_OPENCL)
    find_package(OpenCL QUIET)

    if (OpenCL_FOUND)
        message(STATUS "[ LIBRAPID ] Using OpenCL ${OpenCL_VERSION_STRING}")
        target_include_directories(${module_name} PUBLIC ${OpenCL_INCLUDE_DIRS})
        target_link_libraries(${module_name} PUBLIC ${OpenCL_LIBRARIES})
        target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_OPENCL)

        # Build static CLBlast library
        set(BUILD_SHARED_LIBS OFF)
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/CLBlast")
        target_link_libraries(${module_name} PUBLIC clblast)
        target_include_directories(${module_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/CLBlast/include")
        set(LIBRAPID_HAS_OPENCL true)

        if (IS_MACOS)
            # Add -framework OpenCL to the linker flags
            target_link_libraries(${module_name} PUBLIC "-framework OpenCL")

            # Define CL_HPP_TARGET_OPENCL_VERSION to the version of OpenCL being used
            message(STATUS "[ LIBRAPID ] OpenCL Version: ${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0")
            set(OpenCLVersion ${OpenCL_VERSION_MAJOR}${OpenCL_VERSION_MINOR}0)
            target_compile_definitions(${module_name} PUBLIC CL_HPP_TARGET_OPENCL_VERSION=${OpenCLVersion})
            target_compile_definitions(${module_name} PUBLIC CL_HPP_MINIMUM_OPENCL_VERSION=${OpenCLVersion})

            if (NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
                message(WARNING "[ LIBRAPID ] Accelerate OpenCL is designed for AppleClang. Relaxing some conditions")
                target_compile_options(${module_name} PUBLIC "-flax-vector-conversions")
            endif ()
        endif ()
    else ()
        message(WARNING "OpenCL support was requested, but a valid OpenCL installation could not be found")
    endif ()
endif ()

if (${LIBRAPID_OPTIMISE_SMALL_ARRAYS})
    message(STATUS "[ LIBRAPID ] Small array optimisations enabled")
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_OPTIMISE_SMALL_ARRAYS)
endif ()

# Add dependencies
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/fmt")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/xsimd")

target_compile_definitions(fmt PUBLIC FMT_HEADER_ONLY)
target_link_libraries(${module_name} PUBLIC fmt xsimd)

# Disable "parameter passing for argument of type ... changed to match ..." for xsimd
target_compile_options(xsimd INTERFACE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wno-psabi>)

if (${LIBRAPID_USE_MULTIPREC})
    # Load MPIR
    find_package(MPIR QUIET)
    if (NOT LIBRAPID_GET_MULTIPREC AND MPIR_FOUND AND MPIR_INCLUDE_DIR AND MPIR_LIBRARY)
        message(STATUS "[ LIBRAPID ] Found existing MPIR package")
        target_include_directories(${module_name} PUBLIC ${MPIR_INCLUDE_DIR})
        target_link_libraries(${module_name} PUBLIC ${MPIR_LIBRARY})
        set(LIBRAPID_MPIR_INCLUDE ${MPIR_INCLUDE_DIR} CACHE INTERNAL "Include path for MPIR used by LibRapid")
        set(LIBRAPID_MPIR_LIBRARY ${MPIR_LIBRARY} CACHE INTERNAL "Library path for MPIR used by LibRapid")
    else ()
        message(STATUS "[ LIBRAPID ] Cloning custom MPIR package")
        FetchContent_Declare(
                mpir
                GIT_REPOSITORY https://github.com/LibRapid/MPIR.git
        )
        FetchContent_MakeAvailable(mpir)
        target_link_libraries(${module_name} PUBLIC mpir)
        target_include_directories(${module_name} PUBLIC ${mpir_SOURCE_DIR}/mpir)

        message(STATUS "[ LIBRAPID ] MPIR Binary Directory: ${mpir_BINARY_DIR}")

        set(LIBRAPID_MPIR_INCLUDE ${mpir_SOURCE_DIR}/mpir CACHE INTERNAL "Include path for MPIR used by LibRapid")
        set(LIBRAPID_MPIR_LIBRARY ${mpir_BINARY_DIR} CACHE INTERNAL "Library path for MPIR used by LibRapid")
    endif ()

    # Load MPFR
    find_package(MPFR)
    if (NOT LIBRAPID_GET_MULTIPREC AND MPFR_FOUND)
        message(STATUS "[ LIBRAPID ] Found existing MPFR package")
        target_include_directories(${module_name} PUBLIC ${MPFR_INCLUDES})
        target_link_libraries(${module_name} PUBLIC ${MPFR_LIBRARIES})
    else ()
        message(STATUS "[ LIBRAPID ] Cloning custom MPFR package")
        FetchContent_Declare(
                mpfr
                GIT_REPOSITORY https://github.com/LibRapid/MPFR.git
        )
        FetchContent_MakeAvailable(mpfr)
        target_link_libraries(${module_name} PUBLIC mpfr)
        target_include_directories(${module_name} PUBLIC ${mpfr_SOURCE_DIR}/mpfr/src)
    endif ()

    target_compile_definitions(${module_name} PUBLIC LIBRAPID_USE_MULTIPREC)
endif ()

if (LIBRAPID_FAST_MATH)
    if (MSVC)
        target_compile_options(${module_name} PUBLIC /fp:fast)
    elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        target_compile_options(${module_name} PUBLIC -ffast-math)
    endif ()

    target_compile_definitions(${module_name} PUBLIC LIBRAPID_FAST_MATH)
endif ()

if (LIBRAPID_NATIVE_ARCH)
    message(STATUS "[ LIBRAPID ] Compiling for native architecture")

    include(ArchDetect2)
    target_compile_options(${module_name} PUBLIC ${LIBRAPID_ARCH_FLAGS})
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_NATIVE_ARCH)

    #    OptimizeForArchitecture()
    #    target_compile_options(${module_name} PUBLIC ${Vc_DEFINITIONS} ${Vc_ARCHITECTURE_FLAGS})
    #    target_compile_definitions(${module_name} PUBLIC LIBRAPID_NATIVE_ARCH)
    #    set(LIBRAPID_ARCH_FLAGS ${Vc_DEFINITIONS} ${Vc_ARCHITECTURE_FLAGS})
    #    message(STATUS "[ LIBRAPID ] Additional Definitions: ${Vc_DEFINITIONS}")
    #    message(STATUS "[ LIBRAPID ] Supported flags: ${Vc_ARCHITECTURE_FLAGS}")
endif ()

# Add defines for CUDA vector widths
if (LIBRAPID_HAS_CUDA)
    # Ensure vector widths are in range [1, 4]

    if (LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH GREATER 4)
        message(WARNING "LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH is greater than 4, setting to 4")
        set(LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH 4)
    elseif (LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH LESS 1)
        message(WARNING "LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH is less than 1, setting to 1")
        set(LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH 1)
    endif ()

    if (LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH GREATER 4)
        message(WARNING "LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH is greater than 4, setting to 4")
        set(LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH 4)
    elseif (LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH LESS 1)
        message(WARNING "LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH is less than 1, setting to 1")
        set(LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH 1)
    endif ()

    target_compile_definitions(${module_name} PUBLIC LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH=${LIBRAPID_CUDA_DOUBLE_VECTOR_WIDTH})
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH=${LIBRAPID_CUDA_FLOAT_VECTOR_WIDTH})
endif ()

target_include_directories(${module_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/librapid" "${CMAKE_CURRENT_SOURCE_DIR}/librapid/include")

if (${LIBRAPID_BUILD_TESTS})
    message(STATUS "[ LIBRAPID ] Building LibRapid Tests")
    include(CTest)

    message(STATUS "[ LIBRAPID ] Cloning custom MPFR package")

    FetchContent_Declare(
            Catch2
            GIT_REPOSITORY https://github.com/catchorg/Catch2.git
            GIT_TAG origin/devel
    )

    FetchContent_MakeAvailable(Catch2)
    target_link_libraries(${module_name} PUBLIC Catch2::Catch2WithMain)

    enable_testing()
    add_subdirectory(test)

    # Add VALGRIND test if possible
    include(valgrindTarget)
    addValgrind(${module_name})
endif ()

# Compile the example projects
if (${LIBRAPID_BUILD_EXAMPLES})
    message(STATUS "[ LIBRAPID ] Building LibRapid Examples")
    add_subdirectory(examples)
endif ()

# # Enable code coverage checking
# find_package(codecov)
# if (ENABLE_COVERAGE)
#     message(STATUS "[ LIBRAPID ] Enabling code coverage for ${module_name}")
#     add_coverage(${module_name})
# endif()

if (LIBRAPID_CODE_COV AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "[ LIBRAPID ] Enabling code coverage") # Requires gcovr -- sudo apt install gcovr
    include(CodeCoverage)
    append_coverage_compiler_flags_to_target(${module_name})
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_WITH_COVERAGE)
    target_compile_options(${module_name} PUBLIC
            -O0        # no optimization
            -g         # generate debug info
            --coverage # sets all required flags
    )

    set(COVERAGE_EXCLUDES
            "${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/*"
            "${CMAKE_CURRENT_SOURCE_DIR}/librapid/blas"
            "${CMAKE_CURRENT_SOURCE_DIR}/librapid/cxxblas")

    setup_target_for_coverage_gcovr_html(
            NAME ${module_name}_coverage
            EXECUTABLE ctest -C ${CMAKE_BUILD_TYPE}
            EXCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/*" "${CMAKE_CURRENT_SOURCE_DIR}/librapid/blas" "${CMAKE_CURRENT_SOURCE_DIR}/librapid/cxxblas"
    )
endif (LIBRAPID_CODE_COV AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")

if (IS_WINDOWS AND LIBRAPID_NO_WINDOWS_H)
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_NO_WINDOWS_H=${LIBRAPID_NO_WINDOWS_H})
    target_compile_definitions(${module_name} PUBLIC JITIFY_NO_WINDOWS_H=${LIBRAPID_NO_WINDOWS_H})

    target_compile_definitions(${module_name} PUBLIC
            WIN32_LEAN_AND_MEAN
            NOGDI
            NOUSER
            NOHELP
            NOMINMAX
            NOSERVIC
            NOSOUND
            NOKANJI
            NOKERNEL
            NOVIRTUALKEYBOARD
            NOIME
            NOKEYBOARDINFO
            NOSYSCOMMANDS
            NOSHOWWINDOW
            NOCLIPBOARD
            NOCOLOR
            NOCOMM
            NOCTLMGR
            NODRAWTEXT
            NOGDICAPMASKS
            NOICONS
            NOKEYSTATES
            NOMB
            NOOPENFILE
            NOSCROLL
            NOSERVICE
            NOSOUND
            NOTEXTMETRIC
            NOWH
            NOWINOFFSETS
            NOWINMESSAGES
    )
endif ()

if (LIBRAPID_GET_FFTW)
    message(STATUS "[ LIBRAPID ] Building LibRapid with FFTW")

    # FFTW
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTS OFF)
    set(ENABLE_OPENMP ${LIBRAPID_USE_OMP})
    set(ENABLE_THREADS ${LIBRAPID_USE_OMP})
    set(ENABLE_FLOAT OFF)
    set(ENABLE_DOUBLE OFF)
    set(ENABLE_LONG_DOUBLE OFF)
    set(ENABLE_QUAD_PRECISION OFF)

    set(ENABLE_AVX2 ON)
    set(ENABLE_AVX ON)
    set(ENABLE_SSE2 ON)
    set(ENABLE_SSE ON)

    set(DISABLE_FORTRAN ON)

    set(ENABLE_DOUBLE ON)
    if (NOT EXISTS "${FETCHCONTENT_BASE_DIR}/fftw_double")
        FetchContent_Declare(
                fftw_double
                GIT_REPOSITORY https://github.com/LibRapid/FFTW.git
        )
        FetchContent_MakeAvailable(fftw_double)
    endif ()

    set(ENABLE_DOUBLE OFF)
    set(ENABLE_FLOAT ON)
    if (NOT EXISTS "${FETCHCONTENT_BASE_DIR}/fftw_float")
        FetchContent_Declare(
                fftw_float
                GIT_REPOSITORY https://github.com/LibRapid/FFTW.git
        )
        FetchContent_MakeAvailable(fftw_float)
    endif ()

    if (LIBRAPID_USE_OMP AND LIBRAPID_HAS_OMP)
        target_link_libraries(${module_name} PUBLIC fftw3_omp fftw3f_omp)
    else ()
        target_link_libraries(${module_name} PUBLIC fftw3 fftw3f)
    endif ()
    target_include_directories(${module_name} PUBLIC "${FETCHCONTENT_BASE_DIR}/fftw_float-src/api")
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_FFTW)
endif ()

target_include_directories(${module_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/pocketfft")

include("warnings")

auto_disable_warnings(${module_name})

# target_compile_options(${module_name} PUBLIC "-Q" "-ftime-report")
