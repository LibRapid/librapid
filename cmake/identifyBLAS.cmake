# Identify which BLAS library is being used based on the
# *.lib filename

macro(identifyBlas filename)
    if (filename MATCHES "(openblas).*")
        set(BLAS_LIB "OPENBLAS")
    else ()
        set(BLAS_LIB "GENERIC")
    endif ()

    message(STATUS "Identified BLAS Library ${BLAS_LIB}")
endmacro()

macro(setBlasDefinition filename)
    identifyBlas(filename)

    target_compile_definitions(${module_name} PUBLIC LIBRAPID_BLAS_${BLAS_LIB})
endmacro()
