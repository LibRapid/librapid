cmake_minimum_required(VERSION 3.10)
project(librapid VERSION "0.3.3")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)

set(LIBRAPID_INCLUDE_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

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
find_package(CUDAToolkit QUIET)
find_package(BLAS QUIET)
find_package(OpenMP)

option(USE_BLAS "Attempt to use a BLAS library" ON)
option(USE_CUDA "Attempt to use CUDA" ON)
option(USE_OMP "Attempt to use OpenMP to allow multithreading" ON)

set(LIBRAPID_HAS_OMP false)
set(LIBRAPID_HAS_BLAS false)
set(LIBRAPID_HAS_CUDA false)

set(LIBRAPID_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/cudaCodeLoader.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/kernelHelper.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/src/dotCBlas.cpp"
        )

# {fmt} might fail to compile when building for python if this is not set
if (${SKBUILD})
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    # Ensure PyBind11 is accessible
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/vendor/pybind11")

    set(module_name "_librapid")
    pybind11_add_module(
            ${module_name} MODULE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/librapidPythonInterface.cpp"
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
if (${OpenMP_FOUND})
    if (${USE_OMP})
        message(STATUS "Linking against OpenMP")

        # Link the required library
        target_link_libraries(${module_name} PUBLIC OpenMP::OpenMP_CXX)

        # Add the compile definition so LibRapid knows it has OpenMP
        # add_compile_definitions(LIBRAPID_HAS_OMP)
        target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_OMP)
        set(LIBRAPID_HAS_OMP true)
    else ()
        message(WARNING "OpenMP was found but is not being linked against (Value <USE_OMP> is " ${USE_OMP} ")")
    endif ()
endif ()

# Check if BLAS was built by CI for Python Wheels.
# If so, use this instead of any other BLAS install found
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/openblas_install" OR EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/openblas_install/lib/libopenblas.a")
    message(STATUS "Using OpenBLAS built by CI for Python Wheels")
    set(BLAS_FOUND TRUE)
    set(USE_BLAS TRUE)

    if (${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
        set(BLAS_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/openblas_install/lib/openblas.lib")
    else ()
        set(BLAS_LIBRARIES "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/openblas_install/lib/libopenblas.a")
    endif ()
else ()
    message(STATUS "OpenBLAS for CI not found")
    SUBDIRLIST(listDir "${CMAKE_CURRENT_SOURCE_DIR}")
    message(STATUS "Files in Directory: ${listDir}")
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
                    # message(STATUS "Found OpenBLAS include file " ${file})
                    get_filename_component(inc_file ${file} NAME)
                    configure_file(${file} "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas/${inc_file}" COPYONLY)
                endforeach ()
            else ()
                FILE(GLOB_RECURSE include_files "${inc_path}/*.*")
                foreach (file IN LISTS include_files)
                    # message(STATUS "Found include file " ${file})
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
                # message(STATUS "Found binary file " ${file})
                get_filename_component(filename ${file} NAME)
                configure_file(${file} "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/blas/${filename}" COPYONLY)
            endforeach ()

            FILE(GLOB_RECURSE bin_files "${CMAKE_CURRENT_SOURCE_DIR}/*.dll")
            foreach (file IN LISTS bin_files)
                # message(STATUS "Found packaged binary file " ${file})
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
        message(WARNING "BLAS was found but is not being linked against (Value <USE_BLAS> is " ${USE_BLAS} ")")
    endif ()
endif ()

# Check if CUDA should be used
if (${CUDAToolkit_FOUND})
    if (${USE_CUDA})
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
        message(WARNING "CUDA was found but is not being linked against (Value <USE_CUDA> is " ${USE_CUDA} ")")
    endif ()
endif ()

# Write configuration settings to file
file(WRITE "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "# LibRapid Configuration Settings\n\n")

if (${LIBRAPID_HAS_OMP})
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_HAS_OMP = TRUE\n")
endif ()

if (${LIBRAPID_HAS_BLAS})
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_HAS_BLAS = TRUE\n")
endif ()

if (${LIBRAPID_HAS_CUDA})
    file(APPEND "${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/configuration.txt" "LIBRAPID_HAS_CUDA = TRUE\n")
endif ()

# Add dependencies
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/vendor/fmt")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/vendor/scnlib")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/vendor/Vc")

target_link_libraries(${module_name} PUBLIC fmt scn Vc)
target_include_directories(${module_name} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_SOURCE_DIR}")

if (${SKBUILD})
    install(TARGETS ${module_name} DESTINATION .)
else ()
    message(STATUS "Building LibRapid Examples")
    add_subdirectory(examples)
endif ()
