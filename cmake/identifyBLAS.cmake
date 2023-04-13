# Identify which BLAS library is being used based on the
# *.lib filename

macro(identifyBlas filename)
    if (filename MATCHES "(openblas).*")
        set(BLAS_LIB "OPENBLAS")
    elseif (filename MATCHES "(mkl).*")
        set(BLAS_LIB "MKLBLAS")
    elseif (filename MATCHES "(atlas).*")
        set(BLAS_LIB "ATLAS")
    elseif (filename MATCHES "(accelerate).*")
        set(BLAS_LIB "ACCELERATE")
    else ()
        set(BLAS_LIB "GENERIC")
    endif ()

    message(STATUS "[ LIBRAPID ] Identified BLAS Library ${BLAS_LIB}")
endmacro()

macro(set_blas_definition_from_file filename)
    identifyBlas(filename)
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_BLAS_${BLAS_LIB})
endmacro()

macro(set_blas_definition name)
    target_compile_definitions(${module_name} PUBLIC LIBRAPID_BLAS_${name})
endmacro()
