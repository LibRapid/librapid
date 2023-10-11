#ifndef CXXBLAS_DRIVERS_ACCELERATE_H
#define CXXBLAS_DRIVERS_ACCELERATE_H 1

#define HAVE_CBLAS 1
#ifdef BLASINT
#	define CBLAS_INT BLASINT
#else
#	define CBLAS_INT int
#endif
#define BLAS_IMPL "Accelerate.framework"
#ifndef CBLAS_INDEX
#	define CBLAS_INDEX size_t
#endif // CBLAS_INDEX

#define ACCELERATE_NEW_LAPACK 1
#include <Accelerate/Accelerate.h>

#ifdef negativeInfinity // This breaks things
#undef negativeInfinity
#endif

#endif // CXXBLAS_DRIVERS_ACCELERATE_H
