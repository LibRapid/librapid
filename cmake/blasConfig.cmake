# Identify which BLAS library is being used based on the filename or path
macro(identifyBlas filename)
    if ("${filename}" MATCHES "(openblas|/openblas/).*")
        set(BLAS_LIB "OPENBLAS")
    elseif ("${filename}" MATCHES "(mkl|/mkl/).*")
        set(BLAS_LIB "MKLBLAS")
    elseif ("${filename}" MATCHES "(atlas|/atlas/).*")
        set(BLAS_LIB "ATLAS")
    elseif ("${filename}" MATCHES "Accelerate\\.framework")
        set(BLAS_LIB "ACCELERATE")
    else ()
        set(BLAS_LIB "GENERIC")
    endif ()

    message(STATUS "[ LIBRAPID ] Identified BLAS Library: ${BLAS_LIB}")
endmacro()

macro(set_blas_definition_from_file filename)
    identifyBlas(filename)
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_BLAS_${BLAS_LIB})
endmacro()

macro(set_blas_definition name)
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_BLAS_${name})
endmacro()

macro(download_openblas)
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

    set_blas_definition("OPENBLAS")
endmacro()

macro(link_openblas)
    get_filename_component(filepath ${LIBRAPID_BLAS} DIRECTORY)
    get_filename_component(filename ${LIBRAPID_BLAS} NAME)

    set(include_path "${filepath}/../include")

    # todo

    set_blas_definition("OPENBLAS")
endmacro()

macro(link_accelerate)
    target_link_libraries(${module_name} PUBLIC "-framework Accelerate")

    # If not using apple-clang, we need to relax some conditions
    if (NOT CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        message(WARNING "[ LIBRAPID ] Accelerate is designed for AppleClang. Relaxing some conditions")
        target_compile_options(${module_name} PUBLIC "-flax-vector-conversions")
    endif ()

    set_blas_definition("ACCELERATE")
endmacro()

macro(configure_blas)
    if (NOT LIBRAPID_USE_BLAS)
        return()
    endif ()

    if (LIBRAPID_GET_BLAS)
        download_openblas()
    else ()
        find_package(BLAS QUIET)
    endif ()

    if (NOT BLAS_FOUND)
        message(STATUS "[ LIBRAPID ] BLAS library not found on system. Consider enabling LIBRAPID_GET_BLAS")
        return()
    endif ()

    message(STATUS "[ LIBRAPID ] Located BLAS at ${BLAS_LIBRARIES}")

    list(GET ${BLAS_LIBRARIES} 0 LIBRAPID_BLAS)

    if (NOT ${LIBRAPID_BLAS})
        set(LIBRAPID_BLAS ${BLAS_LIBRARIES})
    endif ()

    message(STATUS "[ LIBRAPID ] Using BLAS")

    identifyBlas("${LIBRAPID_BLAS}")

    # Configure BLAS (different steps are needed for each library)
    if (${BLAS_LIB} STREQUAL "OPENBLAS")
        link_openblas()
    elseif (${BLAS_LIB} STREQUAL "MKLBLAS")
        link_mkl()
    elseif (${BLAS_LIB} STREQUAL "ATLAS")
        link_atlas()
    elseif (${BLAS_LIB} STREQUAL "ACCELERATE")
        link_accelerate()
    else ()
        link_generic()
    endif ()
endmacro()
