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
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_HAS_BLAS)
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
    set(LIBRAPID_BLAS ${BLAS_LIBRARIES})
endmacro()

macro(link_openblas)

    get_filename_component(filepath ${LIBRAPID_BLAS} DIRECTORY)
    get_filename_component(filename ${LIBRAPID_BLAS} NAME)

    set(include_path "${filepath}/../include")
    target_include_directories(${module_name} PUBLIC "${include_path}")

    set(include_files "")
    if (EXISTS "${include_path}/openblas")
        FILE(GLOB_RECURSE include_files "${include_path}/openblas/*.*")
        target_include_directories(${module_name} PUBLIC "${include_path}/openblas")
    else ()
        FILE(GLOB_RECURSE include_files "${include_path}/*.*")
    endif ()

    set(has_cblas OFF)

    foreach (file IN LISTS include_files)
        get_filename_component(inc_file ${file} NAME)
        if (${inc_file} STREQUAL "cblas.h")
            set(has_cblas ON)
        endif ()
    endforeach ()

    if (${has_cblas})
        target_link_libraries(${module_name} PUBLIC ${LIBRAPID_BLAS})
        set_blas_definition("OPENBLAS")
    else ()
        message(WARNING "[ LIBRAPID ] OpenBLAS does not contain cblas.h. Consider enabling LIBRAPID_GET_BLAS")
    endif ()
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

macro(link_generic)
endmacro()

macro(configure_blas)
    if (LIBRAPID_USE_BLAS)
        if (LIBRAPID_GET_BLAS)
            download_openblas()
        else ()
            find_package(BLAS QUIET)
        endif ()

        if (BLAS_FOUND)
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
        else ()
            message(STATUS "[ LIBRAPID ] BLAS library not found on system. Consider enabling LIBRAPID_GET_BLAS")
        endif ()
    endif ()
endmacro()
