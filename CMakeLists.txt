cmake_minimum_required(VERSION 3.10)
project(librapid VERSION "0.5.0")
set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
cmake_policy(SET CMP0077 NEW)

# Optional LibRapid settings
option(LIBRAPID_BUILD_EXAMPLES "Compile LibRapid C++ Examples" OFF)
option(LIBRAPID_BUILD_TESTS "Compile LibRapid C++ Tests" OFF)

option(LIBRAPID_USE_BLAS "Attempt to use a BLAS library" ON)
option(LIBRAPID_USE_CUDA "Attempt to use CUDA" ON)
option(LIBRAPID_USE_OMP "Attempt to use OpenMP to allow multithreading" ON)
option(LIBRAPID_USE_VC "Use Vc for SIMD vectorisation on primitive array types" ON)
option(LIBRAPID_USE_MULTIPREC "Include MPIR and MPFR in the LibRapid build" OFF)
option(LIBRAPID_QUIET_BUILD "Hide warnings generated WITHIN LibRapid's source" ON)

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
if (${LIBRAPID_USE_CUDA})
    find_package(CUDAToolkit QUIET)
endif ()

if (${LIBRAPID_USE_BLAS})
    find_package(BLAS QUIET)
endif ()

if (${LIBRAPID_USE_OMP})
    find_package(OpenMP QUIET)
endif ()

set(LIBRAPID_HAS_OMP false)
set(LIBRAPID_HAS_BLAS false)
set(LIBRAPID_HAS_CUDA false)

set(LIBRAPID_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/cudaCodeLoader.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/kernelHelper.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/linalg/dot.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/linalg/gemm.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/fastMath.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/toString.cpp"
        )

if (${LIBRAPID_USE_MULTIPREC})
    list(APPEND LIBRAPID_SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/mpfr/trig.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/mpfr/expLogPow.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/mpfr/floorCeil.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/mpfr/modAbs.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/mpfr/casting.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/mpfr/hypot.cpp"
            )
endif ()

# {fmt} might fail to compile when building for python if this is not set
if (${SKBUILD})
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    # Ensure PyBind11 is accessible
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/vendor/pybind11")

    set(module_name "_librapid")
    pybind11_add_module(
            ${module_name} MODULE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/librapidPythonInterface.cpp"

            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayBGInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayCGInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayF16GInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayF32GInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayF64GInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayI16GInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayI32GInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayI64GInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayBInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayCInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayF16Interface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayF32Interface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayF64Interface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayI16Interface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayI32Interface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayI64Interface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayMPZInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayMPQInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayMPFRInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayCF32Interface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayCF64Interface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/ArrayCMPFRInterface.cpp"

            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/mpfrInterface.cpp"

            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/complexInterface.cpp"

            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/mathInterface.cpp"

            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/../autogen/Vec2iInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/../autogen/Vec2fInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/../autogen/Vec2dInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/../autogen/Vec3iInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/../autogen/Vec3fInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/../autogen/Vec3dInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/../autogen/Vec4iInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/../autogen/Vec4fInterface.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/../autogen/Vec4dInterface.cpp"

            ${LIBRAPID_SOURCES}
    )

    # add_compile_definitions(LIBRAPID_PYTHON)
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_PYTHON)

    target_compile_options(${module_name} PRIVATE
            $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
            -w>
            $<$<CXX_COMPILER_ID:MSVC>:
            /W0>)
else ()
    set(module_name "librapid")
    add_library(${module_name} STATIC ${LIBRAPID_SOURCES})
endif ()

# See if OpenMP should be linked against
if (${LIBRAPID_USE_OMP})
    if (${OpenMP_FOUND})
        message(STATUS "Linking against OpenMP")

        # Link the required library
        target_link_libraries(${module_name} PUBLIC OpenMP::OpenMP_CXX)

        # Add the compile definition so LibRapid knows it has OpenMP
        # add_compile_definitions(LIBRAPID_HAS_OMP)
        target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_OMP)
        set(LIBRAPID_HAS_OMP true)
    else ()
        message(WARNING "OpenMP support was requested, but OpenMP was not found")
    endif ()
endif ()

# Check if BLAS was built by CI for Python Wheels.
# If so, use this instead of any other BLAS install found
if (${LIBRAPID_USE_BLAS})
    if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/openblas_install" OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/openblas_install/lib/libopenblas.a")
        message(STATUS "Using OpenBLAS built by CI for Python Wheels")
        set(BLAS_FOUND TRUE)
        set(LIBRAPID_USE_BLAS TRUE)

        if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
            set(BLAS_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/openblas_install/lib/openblas.lib")
        else ()
            set(BLAS_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/openblas_install/lib/libopenblas.a")
        endif ()
    endif ()
endif ()

# See if BLAS should be linked against
if (${LIBRAPID_USE_BLAS})
    if (${BLAS_FOUND})
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
                    configure_file(${file} "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas/${inc_file}" COPYONLY)
                endforeach ()
            endif ()

            # Copy library files
            get_filename_component(lib_name ${LIBRAPID_BLAS} NAME)
            # message(STATUS "Found library file ${lib_name}")
            configure_file(${LIBRAPID_BLAS} "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas/${lib_name}" COPYONLY)
        endif ()

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

            FILE(GLOB_RECURSE bin_files "${CMAKE_CURRENT_SOURCE_DIR}/*.dll")
            foreach (file IN LISTS bin_files)
                message(STATUS "Found packaged binary file " ${file})
                get_filename_component(filename ${file} NAME)
                configure_file(${file} "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas/${filename}" COPYONLY)
            endforeach ()
        endif ()

        # Add the compile definition so LibRapid knows it has BLAS
        if (${has_cblas})
            # Link the required library
            target_link_libraries(${module_name} PUBLIC
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas/${lib_name}"
                    )

            target_include_directories(${module_name} PUBLIC
                    "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas"
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
if (${LIBRAPID_USE_CUDA})
    if (${CUDAToolkit_FOUND})
        message(STATUS "Using CUDA ${CUDAToolkit_VERSION}")

        # Ensure jitify is accessible
        if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/vendor/jitify")
            message(STATUS "Jitify exists in LibRapid source tree")
        else ()
            message(FATAL_ERROR "Jitify not found. Ensure the full LibRapid GitHub repo is cloned!")
        endif ()

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

        target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_CUDA)
        target_compile_definitions(${module_name} PUBLIC LIBRAPID_CUDA_STREAM)
        message(STATUS "CUDA include directories: ${CUDAToolkit_INCLUDE_DIRS}")
        target_compile_definitions(${module_name} PUBLIC CUDA_INCLUDE_DIRS="${CUDAToolkit_INCLUDE_DIRS}")
        set(LIBRAPID_HAS_CUDA true)
    else ()
        message(WARNING "CUDA support was requested, but a valid CUDA installation could not be found")
    endif ()
endif ()

# Write configuration settings to file
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "# LibRapid Configuration Settings\n\n")

if (${LIBRAPID_HAS_OMP})
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_HAS_OMP = TRUE\n")
else ()
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_HAS_OMP = FALSE\n")
endif ()

if (${LIBRAPID_HAS_BLAS})
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_HAS_BLAS = TRUE\n")
else ()
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_HAS_BLAS = FALSE\n")
endif ()

if (${LIBRAPID_HAS_CUDA})
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_HAS_CUDA = TRUE\n")
else ()
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_HAS_CUDA = FALSE\n")
endif ()

if (${LIBRAPID_USE_VC})
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_USE_VC = TRUE\n")
else ()
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_USE_VC = FALSE\n")
endif ()

if (${LIBRAPID_USE_MULTIPREC})
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_USE_MULTIPREC = TRUE\n")
else ()
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_USE_MULTIPREC = FALSE\n")
endif ()

# Add dependencies
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/vendor/fmt")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/vendor/scnlib")

target_link_libraries(${module_name} PUBLIC fmt scn)

if (${LIBRAPID_USE_VC})
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/vendor/Vc")
    target_compile_definitions(Vc PRIVATE Vc_HACK_OSTREAM_FOR_TTY)
    target_link_libraries(${module_name} PUBLIC Vc)
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_USE_VC)
endif ()

if (${LIBRAPID_USE_MULTIPREC})
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/vendor/mpfr")
    target_link_libraries(${module_name} PUBLIC mpfr)
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_USE_MULTIPREC)
endif ()

if (${LIBRAPID_QUIET_BUILD})
    # Disable warnings because they're really annoying
    target_compile_options(${module_name} PRIVATE
            $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
            -w>
            $<$<CXX_COMPILER_ID:MSVC>:
            /w>)
endif ()

target_include_directories(${module_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/alt" "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}")

if (${SKBUILD})
    install(TARGETS ${module_name} DESTINATION .)
else ()
    if (${LIBRAPID_BUILD_EXAMPLES})
        message(STATUS "Building LibRapid Examples")
        add_subdirectory(examples)
    endif ()

    if (${LIBRAPID_BUILD_TESTS})
        message(STATUS "Building LibRapid Tests")
        include(CTest)
        enable_testing()
        add_subdirectory(tests/cpp)
    endif ()
endif ()
