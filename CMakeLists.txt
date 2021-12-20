cmake_minimum_required(VERSION 3.10)
project(librapid VERSION "0.2.19")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake_scripts")

if ()
    message(STATUS "Building library for Python")
else ()
    message(STATUS "Building static C++ library")
endif ()

message(STATUS "CMake Version: ${CMAKE_VERSION}")

set(LIBRAPID_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

set(CMAKE_CXX_STANDARD 17)

# Attempt to locate the required packages
find_package(CUDAToolkit)
find_package(BLAS)
find_package(OpenMP)
find_package(AVX)

if (${SKBUILD})
    # Ensure PyBind11 is accessible
    if (EXISTS "${CMAKE_SOURCE_DIR}/pybind11")
        add_subdirectory("pybind11")
    elseif (EXISTS "${CMAKE_SOURCE_DIR}/src/librapid/pybind11")
        add_subdirectory("src/librapid/pybind11")
    else ()
        include(FetchContent)

        FetchContent_Declare(
                pybind11
                URL https://github.com/pybind/pybind11/archive/refs/tags/v2.6.2.tar.gz
                URL_HASH SHA256=8ff2fff22df038f5cd02cea8af56622bc67f5b64534f1b83b9f133b8366acff2
        )
        FetchContent_MakeAvailable(pybind11)
    endif ()
else ()
    message(STATUS "CMake Source Directory: ${CMAKE_CURRENT_SOURCE_DIR}")
    include_directories(
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/cudahelpers
    )
endif ()

option(USE_BLAS "Attempt to use a BLAS library" ON)
option(USE_CUDA "Attempt to use CUDA" ON)
option(USE_OMP "Attempt to use OpenMP to allow multithreading" ON)

set(sources # Array library
        "src/librapid/array/extent.cpp"
        "src/librapid/array/iterators.cpp"
        "src/librapid/array/stride.cpp"
        "src/librapid/array/multiarray.cpp"
        "src/librapid/array/multiarray_constructors.cpp"
        "src/librapid/array/multiarray_string_methods.cpp"
        "src/librapid/array/multiarray_indexing.cpp"
        "src/librapid/array/multiarray_manipulation.cpp"
        "src/librapid/array/multiarray_arithmetic.cpp"
        "src/librapid/array/multiarray_utils.cpp"
        "src/librapid/array/multiarray_matrixops.cpp"
        # Math library
        "src/librapid/math/core_math.cpp"
        # String methods
        "src/librapid/stringmethods/format_number.cpp"
        # Utilities
        "src/librapid/utils/color.cpp"
        "src/librapid/utils/console_utils.cpp"
        "src/librapid/utils/time_utils.cpp"
        # Tests
        "src/librapid/test/librapid_test.cpp"
        # Dependencies
        "version2/instrset_detect.cpp"
        src/librapid/array/cblas_api.hpp)

if (${SKBUILD})
    set(module_name "_librapid")
    pybind11_add_module(${module_name} MODULE
            "src/librapid/pybind_librapid.cpp"
            ${sources}
            )

    add_compile_definitions(LIBRAPID_PYTHON)
else ()
    set(module_name "librapid")
    add_library(${module_name} STATIC ${sources})
endif ()

# See if OpenMP should be linked against
if (${OpenMP_FOUND})
    if (${USE_OMP})
        message(STATUS "Linking against OpenMP")

        # Link the required library
        target_link_libraries(${module_name} PRIVATE
                OpenMP::OpenMP_CXX
                )

        # Add the compile definition so LibRapid knows it has OpenMP
        add_compile_definitions(LIBRAPID_HAS_OMP)
    else ()
        message(WARNING "OpenMP was found but is not being linked against (Value <USE_OMP> is " ${USE_OMP} ")")
    endif ()
endif ()

# Check if BLAS was built by CI for Python Wheels.
# If so, use this instead of any other BLAS install found
if (EXISTS "${CMAKE_SOURCE_DIR}/src/librapid/openblas_install")
    message(STATUS "Using OpenBLAS built by CI for Python Wheels")
    set(BLAS_FOUND TRUE)
    set(USE_BLAS TRUE)

    if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
        set(BLAS_LIBRARIES "${CMAKE_SOURCE_DIR}/src/librapid/openblas_install/lib/openblas.lib")
    else ()
        set(BLAS_LIBRARIES "${CMAKE_SOURCE_DIR}/src/librapid/openblas_install/lib/libopenblas.a")
    endif ()
endif ()

# See if BLAS should be linked against
if (${BLAS_FOUND})
    if (${USE_BLAS})
        message(STATUS "BLAS located was ${BLAS_LIBRARIES}")

        list(GET ${BLAS_LIBRARIES} 0 LIBRAPID_BLAS)

        if (NOT ${LIBRAPID_BLAS})
            set(LIBRAPID_BLAS ${BLAS_LIBRARIES})
        endif ()

        message(STATUS "Using BLAS (" ${LIBRAPID_BLAS} ")")

        get_filename_component(filepath ${LIBRAPID_BLAS} DIRECTORY)

        # Copy include files
        set(inc_path "${filepath}/../include")
        message(STATUS "Checking path ${inc_path} for include files")
        FILE(GLOB_RECURSE files "${filepath}/..")
        message(STATUS "Information: ${files}")
        if (NOT (EXISTS ${inc_path}))
            message(STATUS "Could not locate include path for BLAS")
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
                    message(STATUS "Found OpenBLAS include file " ${file})
                    get_filename_component(inc_file ${file} NAME)
                    configure_file(${file} "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas/${inc_file}" COPYONLY)
                endforeach ()
            else ()
                FILE(GLOB_RECURSE include_files "${inc_path}/*.*")
                foreach (file IN LISTS include_files)
                    message(STATUS "Found include file " ${file})
                    get_filename_component(inc_file ${file} NAME)
                    configure_file(${file} "${CMAKE_SOURCE_DIR}/src/librapid/blas/${inc_file}" COPYONLY)
                endforeach ()
            endif ()
        endif ()

        # Copy library files
        get_filename_component(lib_name ${LIBRAPID_BLAS} NAME)
        message(STATUS "Found library file ${lib_name}")
        configure_file(${LIBRAPID_BLAS} "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas/${lib_name}" COPYONLY)

        # Copy binary files if on Windows
        if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
            set(bin_path "${filepath}/../bin")
            if (NOT (EXISTS ${bin_path}))
                message(FATAL_ERROR "Could not locate <bin> folder for BLAS")
            endif ()

            FILE(GLOB_RECURSE include_files "${bin_path}/*.dll")
            foreach (file IN LISTS include_files)
                message(STATUS "Found binary file " ${file})
                get_filename_component(filename ${file} NAME)
                configure_file(${file} "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas/${filename}" COPYONLY)
            endforeach ()
        endif ()

        # Link the required library
        target_link_libraries(${module_name} PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas/${lib_name}"
                )

        target_include_directories(${module_name} PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas"
                )

        # Add the compile definition so LibRapid knows it has BLAS
        if (${has_cblas})
            add_compile_definitions(LIBRAPID_HAS_BLAS)
        else ()
            message(WARNING "Although BLAS was found, no cblas.h file was found, so BLAS support is not enabled")
        endif ()
    else ()
        message(WARNING "BLAS was found but is not being linked against (Value <USE_BLAS> is " ${USE_BLAS} ")")
    endif ()
endif ()

# Check if CUDA should be used
if (${CUDAToolkit_FOUND})
    if (${USE_CUDA})
        message(STATUS "Using CUDA ${CUDAToolkit_VERSION}")

        target_include_directories(${module_name} PUBLIC
                ${CUDAToolkit_INCLUDE_DIRS}
                "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/cudahelpers"
                )

        target_link_directories(${module_name} PUBLIC
                ${CUDA_LIBRARIES}
                ${CUDA_CUBLAS_LIBRARIES}
                )

        target_link_libraries(${module_name} PUBLIC
                CUDA::cudart
                CUDA::cuda_driver
                CUDA::nvrtc
                CUDA::cublas
                CUDA::cufft
                CUDA::cufftw
                CUDA::curand
                CUDA::cusolver
                CUDA::cusparse
                Dbghelp
                )

        # add_compile_definitions(LIBRAPID_HAS_CUDA)
        # add_compile_definitions(LIBRAPID_CUDA_STREAM)

        target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_CUDA)
        target_compile_definitions(${module_name} PUBLIC LIBRAPID_CUDA_STREAM)
        message(STATUS "CUDA include directories: ${CUDAToolkit_INCLUDE_DIRS}")
        target_compile_definitions(${module_name} PUBLIC CUDA_INCLUDE_DIRS="${CUDAToolkit_INCLUDE_DIRS}")
    else ()
        message(WARNING "CUDA was found but is not being linked against (Value <USE_CUDA> is " ${USE_CUDA} ")")
    endif ()
endif ()

target_include_directories(${module_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid" # For wheel build
        "${CMAKE_CURRENT_SOURCE_DIR}/src" # For source dist
        "${CMAKE_CURRENT_SOURCE_DIR}" # For source dist
        )

# include(ProcessorCount)
# ProcessorCount(N)
# if(NOT threads EQUAL 0)
# 	add_compile_definitions(NUM_THREADS=${N}/2)
# endif()

if (${SKBUILD})
    install(TARGETS ${module_name} DESTINATION .)
else ()
    target_include_directories(librapid INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR})
endif ()

if ($ENV{LIBRAPID_NO_ARCH})
    message(STATUS "Not optimising for architecture")

    set(GNU_OPTIM_FLAGS "-O2 ${ARCH_FLAGS} ${AVX_FLAGS}")
    set(MSVC_OPTIM_FLAGS "/O2 /Ob2 /Oi /Ot /GT /GL ${ARCH_FLAGS} ${AVX_FLAGS}")
else ()
    include("${CMAKE_CURRENT_SOURCE_DIR}/cmake_scripts/FindAVX.cmake")

    message(STATUS "Found AVX: ${AVX_FOUND}")
    message(STATUS "Compiler Options: ${AVX_FLAGS}")

    set(ARCH_FLAGS "-march=native -mtune=native")
    set(GNU_OPTIM_FLAGS "-O3 ${ARCH_FLAGS} ${AVX_FLAGS}")
    set(MSVC_OPTIM_FLAGS "/O2 /Ob2 /Oi /Ot /GT /GL ${AVX_FLAGS}")
endif ()

# Set the compiler to use maximum optimisations (SPEED!!!!)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # using Clang
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${GNU_OPTIM_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # using GCC
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${GNU_OPTIM_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # using Intel C++
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${GNU_OPTIM_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # using Visual Studio C++
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${MSVC_OPTIM_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
endif ()
