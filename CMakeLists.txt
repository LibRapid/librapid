cmake_minimum_required(VERSION 3.16)
project(librapid)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
cmake_policy(SET CMP0077 NEW)
set(CMAKE_CXX_STANDARD 17)

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

option(LIBRAPID_GET_FFTW "Clone and use FFTW -- WARNING: SEE DOCUMENTATION" OFF)
option(LIBRAPID_GET_BLAS "Download pre-built OpenBLAS binaries and use them" OFF)
option(LIBRAPID_GET_MULTIPREC "Download generic multiprecision libraries, as opposed to trying to find one on the system" OFF)

option(LIBRAPID_USE_BLAS "Attempt to use a BLAS library" ON)
option(LIBRAPID_USE_OMP "Attempt to use OpenMP to allow multithreading" ON)
option(LIBRAPID_USE_OPENCL "Search for OpenCL and use it if possible" ON)
option(LIBRAPID_USE_CUDA "Attempt to use CUDA" ON)
option(LIBRAPID_USE_MULTIPREC "Include MPIR and MPFR in the LibRapid build" OFF)
option(LIBRAPID_FAST_MATH "Use potentially less accurate operations to increase performance" OFF)
option(LIBRAPID_NATIVE_ARCH "Use the native architecture of the system" OFF)

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

# Attempt to locate the required packages
include(FetchContent)
if (LIBRAPID_USE_CUDA)
    find_package(CUDAToolkit QUIET)
endif ()

if (LIBRAPID_USE_BLAS)
    find_package(BLAS QUIET)
endif ()

if (LIBRAPID_USE_OMP)
    find_package(OpenMP REQUIRED)
endif ()

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

set(module_name "librapid")
add_library(${module_name} STATIC ${LIBRAPID_SOURCES})

# clang-format off
target_compile_definitions(${module_name} PUBLIC LIBRAPID_MAJOR=${LIBRAPID_MAJOR})
target_compile_definitions(${module_name} PUBLIC LIBRAPID_MINOR=${LIBRAPID_MINOR})
target_compile_definitions(${module_name} PUBLIC LIBRAPID_PATCH=${LIBRAPID_PATCH})

set(LIBRAPID_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/librapid")
target_compile_definitions(${module_name} PUBLIC LIBRAPID_SOURCE="${LIBRAPID_SOURCE_DIR}")
# clang-format on

# Precompiled Headers
target_precompile_headers(${module_name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/librapid/include/librapid/core/librapidPch.hpp")

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
else ()
    # Enable all warnings
    if (MSVC)
        target_compile_options(${module_name} PRIVATE /W4)
    else ()
        target_compile_options(${module_name} PRIVATE -Wall -Wextra -pedantic)
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
include(identifyBLAS)

# If LIBRAPID_MKL_CONFIG_PATH is set, use that instead of the default
if (LIBRAPID_MKL_CONFIG_PATH)
    message(STATUS "[ LIBRAPID ] Using MKLConfig.cmake from ${LIBRAPID_MKL_CONFIG_PATH}")
    include(${LIBRAPID_MKL_CONFIG_PATH})
    message(STATUS "[ LIBRAPID ] Linking against MKL")
    message(STATUS "[ LIBRAPID ] MKL Libraries: ${MKL_LIBRARIES}")
    message(STATUS "[ LIBRAPID ] MKL Include Directories: ${MKL_INCLUDE}")
    target_link_libraries(${module_name} PUBLIC MKL::MKL)
    target_include_directories(${module_name} PUBLIC ${MKL_INCLUDE})
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_BLAS)
    set(LIBRAPID_HAS_BLAS ON)
    set_blas_definition("MKL")
endif ()

if (LIBRAPID_GET_BLAS AND NOT LIBRAPID_HAS_BLAS)
    message(STATUS "[ LIBRAPID ] Downloading OpenBLAS Build...")

    FetchContent_Declare(
            BuildOpenBLAS
            GIT_REPOSITORY https://github.com/LibRapid/BuildOpenBLAS.git
    )

    FetchContent_MakeAvailable(BuildOpenBLAS)

    set(BLAS_FOUND TRUE)
    set(LIBRAPID_USE_BLAS TRUE)

    if (${IS_WINDOWS})
        # Use openblas-windows-latest
        set(BLAS_LIBRARIES "${FETCHCONTENT_BASE_DIR}/buildopenblas-src/openblas-windows-latest/lib/openblas.lib")
    elseif (${IS_MACOS})
        # Use openblas-macos-latest
        set(BLAS_LIBRARIES "${FETCHCONTENT_BASE_DIR}/buildopenblas-src/openblas-macos-latest/lib/libopenblas.a")
    else () # Linux and other systems
        # Use openblas-ubuntu-latest
        set(BLAS_LIBRARIES "${FETCHCONTENT_BASE_DIR}/buildopenblas-src/openblas-ubuntu-latest/lib/libopenblas.a")
    endif ()
endif ()

# See if BLAS should be linked against
if (LIBRAPID_USE_BLAS AND NOT LIBRAPID_HAS_BLAS)
    if (${BLAS_FOUND})
        message(STATUS "[ LIBRAPID ] BLAS located was ${BLAS_LIBRARIES}")

        list(GET ${BLAS_LIBRARIES} 0 LIBRAPID_BLAS)

        if (NOT ${LIBRAPID_BLAS})
            set(LIBRAPID_BLAS ${BLAS_LIBRARIES})
        endif ()

        message(STATUS "[ LIBRAPID ] Using BLAS (" ${LIBRAPID_BLAS} ")")

        get_filename_component(filepath ${LIBRAPID_BLAS} DIRECTORY)
        get_filename_component(filename ${LIBRAPID_BLAS} NAME)

        # Attempt to identify which BLAS library is being used and
        # set the appropriate compile definition
        set_blas_definition_from_file(filename)

        # Copy include files
        set(inc_path "${filepath}/../include")
        message(STATUS "[ LIBRAPID ] Checking path ${inc_path} for include files")
        FILE(GLOB_RECURSE files "${filepath}/..")
        message(STATUS "[ LIBRAPID ] Information: ${files}")
        if (NOT (EXISTS ${inc_path}))
            message(STATUS "[ LIBRAPID ] Could not locate include path for BLAS")
        endif ()

        set(has_cblas OFF)

        if (EXISTS "${inc_path}/openblas")
            FILE(GLOB_RECURSE include_files "${inc_path}/openblas/*.*")
            foreach (file IN LISTS include_files)
                get_filename_component(inc_file ${file} NAME)
                if (${inc_file} STREQUAL "cblas.h")
                    set(has_cblas ON)
                endif ()
            endforeach ()
        else ()
            FILE(GLOB_RECURSE include_files "${inc_path}/*.*")
            foreach (file IN LISTS include_files)
                get_filename_component(inc_file ${file} NAME)
                if (${inc_file} STREQUAL "cblas.h")
                    set(has_cblas ON)
                endif ()
            endforeach ()
        endif ()

        if (${has_cblas})
            if (EXISTS "${inc_path}/openblas")
                FILE(GLOB_RECURSE include_files "${inc_path}/openblas/*.*")
                foreach (file IN LISTS include_files)
                    message(STATUS "[ LIBRAPID ] Found OpenBLAS include file " ${file})
                    get_filename_component(inc_file ${file} NAME)
                    configure_file(${file} "${CMAKE_CURRENT_SOURCE_DIR}/librapid/blas/${inc_file}" COPYONLY)
                endforeach ()
            endif ()
            # else ()

            FILE(GLOB_RECURSE include_files "${inc_path}/*.*")
            foreach (file IN LISTS include_files)
                message(STATUS "[ LIBRAPID ] Found include file " ${file})
                get_filename_component(inc_file ${file} NAME)
                configure_file(${file} "${CMAKE_CURRENT_SOURCE_DIR}/librapid/blas/${inc_file}" COPYONLY)
            endforeach ()

            # endif ()

            # Copy library files
            get_filename_component(lib_name ${LIBRAPID_BLAS} NAME)
            # message(STATUS "[ LIBRAPID ] Found library file ${lib_name}")
            configure_file(${LIBRAPID_BLAS} "${CMAKE_CURRENT_SOURCE_DIR}/librapid/blas/${lib_name}" COPYONLY)
        endif ()

        # Copy binary files if on Windows
        if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
            set(bin_path "${filepath}/../bin")
            if (NOT (EXISTS ${bin_path}))
                message(WARNING "Could not locate <bin> folder for BLAS")
            else ()
                FILE(GLOB_RECURSE include_files "${bin_path}/*.dll")
                foreach (file IN LISTS include_files)
                    message(STATUS "[ LIBRAPID ] Found binary file " ${file})
                    get_filename_component(filename ${file} NAME)
                    configure_file(${file} "${CMAKE_CURRENT_SOURCE_DIR}/librapid/blas/${filename}" COPYONLY)
                endforeach ()

                FILE(GLOB_RECURSE bin_files "${CMAKE_CURRENT_SOURCE_DIR}/*.dll")
                foreach (file IN LISTS bin_files)
                    message(STATUS "[ LIBRAPID ] Found packaged binary file " ${file})
                    get_filename_component(filename ${file} NAME)
                    configure_file(${file} "${CMAKE_CURRENT_SOURCE_DIR}/librapid/blas/${filename}" COPYONLY)
                endforeach ()
            endif ()
        endif ()

        # Add the compile definition so LibRapid knows it has BLAS
        if (${has_cblas})
            # Link the required library
            target_link_libraries(${module_name} PUBLIC
                    "${CMAKE_CURRENT_SOURCE_DIR}/librapid/blas/${lib_name}"
            )

            target_include_directories(${module_name} PUBLIC
                    "${CMAKE_CURRENT_SOURCE_DIR}/librapid/blas"
            )

            target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_BLAS)
            set(LIBRAPID_HAS_BLAS true)
        else ()
            message(WARNING "Although BLAS was found, no cblas.h file was found, so BLAS support is not enabled")
        endif ()
    else ()
        message(WARNING "BLAS support was requested but a valid BLAS interface was not found")
    endif ()
endif ()

# Check if CUDA should be used
if (LIBRAPID_USE_CUDA)
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
                Dbghelp
        )

        target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_CUDA)
        target_compile_definitions(${module_name} PUBLIC LIBRAPID_CUDA_STREAM)
        message(STATUS "[ LIBRAPID ] CUDA include directories: ${CUDAToolkit_INCLUDE_DIRS}")
        target_compile_definitions(${module_name} PUBLIC CUDA_INCLUDE_DIRS="${CUDAToolkit_INCLUDE_DIRS}")
        set(LIBRAPID_HAS_CUDA true)
    else ()
        message(WARNING "CUDA support was requested, but a valid CUDA installation could not be found")
    endif ()
endif ()


if (LIBRAPID_USE_OPENCL AND IS_MACOS)
    message(WARNING "OpenCL support was requested, but OpenCL is not supported on macOS. Disabling OpenCL support")
    set(LIBRAPID_USE_OPENCL OFF)
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
# add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/Vc")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/xsimd")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/librapid/vendor/scnlib")

target_compile_definitions(fmt PUBLIC FMT_HEADER_ONLY)
# target_compile_definitions(Vc PRIVATE Vc_HACK_OSTREAM_FOR_TTY)
# target_link_libraries(${module_name} PUBLIC fmt scn Vc xsimd)
target_link_libraries(${module_name} PUBLIC fmt scn xsimd)

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

    if (LIBRAPID_USE_OMP)
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
