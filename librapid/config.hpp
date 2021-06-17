#ifndef NDARRAY_CONFIG
#define NDARRAY_CONFIG

#ifndef LIBRAPID_BUILD
// LIBRAPID_BUILD 0 == C++
// LIBRAPID_BUILD 1 == PYBIND
#define LIBRAPID_BUILD 0
#endif

#if LIBRAPID_BUILD == 1

// PyBind11 specific definitions and includes
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#endif

#ifdef LIBRAPID_CBLAS
#undef LIBRAPID_CBLAS
#define LIBRAPID_CBLAS 1
#include <cblas.h>
#endif

#if defined(NDEBUG) || defined(NDARRAY_NDEBUG)
#define ND_NDEBUG
#define ND_INLINE inline
#else
#define ND_DEBUG
#define ND_INLINE
#endif // NDEBUG || NDARRAY_DEBUG

#ifdef _OPENMP
#define ND_HAS_OMP
#include <omp.h>
#endif // _OPENMP

#ifndef ND_NUM_THREADS
#define ND_NUM_THREADS 4
#endif

#ifndef ND_MAX_DIMS
#define ND_MAX_DIMS 32
#endif // ND_MAX_DIMS

// Operating system defines

#if defined(_M_IX86)
#define LIBRAPID_X86
#elif defined(_M_X64)
#define LIBRAPID_X64
#else
#define LIBRAPID_BUILD_UNKNOWN
#endif

#if defined(_WIN32)
#define LIBRAPID_OS_WINDOWS // Windows
#define LIBRAPID_OS "windows"
#elif defined(_WIN64)
#define LIBRAPID_OS_WINDOWS // Windows
#define LIBRAPID_OS "windows"
#elif defined(__CYGWIN__) && !defined(_WIN32)
#define LIBRAPID_OS_WINDOWS  // Windows (Cygwin POSIX under Microsoft Window)
#define LIBRAPID_OS "windows"
#elif defined(__ANDROID__)
#define LIBRAPID_OS_ANDROID // Android (implies Linux, so it must come first)
#define LIBRAPID_OS "android"
#elif defined(__linux__)
#define LIBRAPID_OS_LINUX // Debian, Ubuntu, Gentoo, Fedora, openSUSE, RedHat, Centos and other
#define LIBRAPID_OS "linux"
#elif defined(__unix__) || !defined(__APPLE__) && defined(__MACH__)
#include <sys/param.h>
#if defined(BSD)
#define LIBRAPID_OS_BSD // FreeBSD, NetBSD, OpenBSD, DragonFly BSD
#define LIBRAPID_OS "bsd"
#endif
#elif defined(__hpux)
#define LIBRAPID_OS_HP_UX // HP-UX
#define LIBRAPID_OS "hp-ux"
#elif defined(_AIX)
#define LIBRAPID_OS_AIX // IBM AIX
#define LIBRAPID_OS "aix"
#elif defined(__APPLE__) && defined(__MACH__) // Apple OSX and iOS (Darwin)
#define LIBRAPID_OS_APPLE
#include <TargetConditionals.h>
#if TARGET_IPHONE_SIMULATOR == 1
#define LIBRAPID_OS_IOS // Apple iOS
#define LIBRAPID_OS "ios"
#elif TARGET_OS_IPHONE == 1
#define LIBRAPID_OS_IOS // Apple iOS
#define LIBRAPID_OS "ios"
#elif TARGET_OS_MAC == 1
#define LIBRAPID_OS_OSX // Apple OSX
#define LIBRAPID_OS "osx"
#endif
#elif defined(__sun) && defined(__SVR4)
#define LIBRAPID_OS_SOLARIS // Oracle Solaris, Open Indiana
#define LIBRAPID_OS "solaris"
#else
#define LIBRAPID_OS_UNKNOWN
#define LIBRAPID_OS "unknown"
#endif

using nd_int = long long;

#if defined(OPENBLAS_OPENMP) || defined(OPENBLAS_THREAD) || defined(OPENBLAS_SEQUENTIAL)
#define LIBRAPID_HAS_OPENBLAS
#endif

// BLAS config settings
namespace librapid {
    int has_blas() {
    #if defined(LIBRAPID_CBLAS) && (LIBRAPID_CBLAS == 1)
        return 1;
    #else
        return 0;
    #endif // LIBRAPID_CBLAS
    }

	void set_blas_threads(int num) {
    #ifdef LIBRAPID_HAS_OPENBLAS
		openblas_set_num_threads(num);
		goto_set_num_threads(num);

	#ifdef ND_HAS_OMP
		omp_set_num_threads(num);
	#endif // ND_HAS_OMP
	#else
		std::cout << "The function 'librapid_set_blas_threads' only works in C++ and Python when compiled with OpenBLAS" << "\n";
	#endif // LIBRAPID_HAS_OPENBLAS
    }

	int get_blas_threads() {
    #ifdef LIBRAPID_HAS_OPENBLAS
		return openblas_get_num_threads();
	#else
		std::cout << "The function 'librapid_set_blas_threads' only works in C++ and Python when compiled with OpenBLAS" << "\n";
		return -1;
	#endif // LIBRAPID_HAS_OPENBLAS
    }
}

#endif // NDARRAY_CONFIG