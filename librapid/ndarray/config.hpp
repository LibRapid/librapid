#ifndef NDARRAY_CONFIG
#define NDARRAY_CONFIG

#if defined(NDEBUG) || defined(NDARRAY_NDEBUG)
#define ND_NDEBUG
#define ND_INLINE inline
#else
#define ND_DEBUG
#define ND_INLINE
#endif // NDEBUG || NDARRAY_DEBUG

#ifdef _OPENMP
#define ND_HAS_OMP
#endif // _OPENMP

#ifdef ND_HAS_OMP
#include <omp.h>
#endif // NDARRAY_HAS_OMP

#ifndef ND_NUM_THREADS
#define ND_NUM_THREADS 4
#endif

#ifndef ND_MAX_DIMS
#define ND_MAX_DIMS 50
#endif // ND_MAX_DIMS

using nd_int = unsigned long long;

#endif // NDARRAY_CONFIG

/**
 * @brief This is a function with a purpose
 * 
 * @param val A value
 * @return int The value multiplied by 123
 */
int a_random_function(int val) {
    return val * 123;
}
