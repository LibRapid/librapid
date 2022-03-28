#ifndef LIBRAPID_CONFIG
#define LIBRAPID_CONFIG

// STL headers
#include <cstdlib>
#include <string>
#include <vector>
#include <type_traits>
#include <algorithm>
#include <complex>
#include <memory>

// Include some intrinsic functions
#ifdef _WIN32
#	include <intrin.h>
#else
#	include <x86intrin.h>
#endif

// Include OpenMP if possible
#ifndef LIBRAPID_HAS_OMP
#	ifdef _OPENMP
#		include <omp.h>
#		define LIBRAPID_HAS_OMP
#	endif // _OPENMP
#else	   // LIBRAPID_HAS_OMP
#	include <omp.h>
#endif // LIBRAPID_HAS_OMP

// Include CBLAS if possible
#ifdef LIBRAPID_HAS_BLAS
#	include <cblas.h>
#endif

#ifdef LIBRAPID_PYTHON
// PyBind11 specific definitions and includes
#	include <pybind11/pybind11.h>
#	include <pybind11/stl.h>
namespace py = pybind11;
#endif // LIBRAPID_PYTHON

// Provide definitions for Release and Debug builds
#if defined(NDEBUG) || defined(LIBRAPID_NDEBUG)
#	define LIBRAPID_RELEASE
#else
#	define LIBRAPID_DEBUG
#endif // NDEBUG || LIBRAPID_NDEBUG

#ifndef LIBRAPID_MAX_DIMS
#	define LIBRAPID_MAX_DIMS 32
#endif // LIBRAPID_MAX_DIMS

// Data alignment
#ifndef DATA_ALIGN
#	define DATA_ALIGN 32 // 1024
#endif

// For n < THREAD_THRESHOLD, code runs serially -- otherwise it'll run in
// parallel
#ifndef THREAD_THREASHOLD
#	define THREAD_THREASHOLD 10000
#endif // THREAD_THREASHOLD

// SIMD instructions
#define VCL_NAMESPACE vcl

#include <version2/vectorclass.h>

// Include {fmt}
#include <fmt/chrono.h>
#include <fmt/color.h>
#include <fmt/compile.h>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

// Operating system defines

// Check for windows
#if defined(_WIN32) || defined(_WIN64)
#	if defined(_WIN64)
#		define LIBRAPID_X64
#	else
#		define LIBRAPID_X86
#	endif
#endif

// Check for GCC

#if defined(_WIN32)
#	define LIBRAPID_OS_WINDOWS // Windows
#	define LIBRAPID_OS			"windows"
#elif defined(_WIN64)
#	define LIBRAPID_OS_WINDOWS // Windows
#	define LIBRAPID_OS			"windows"
#elif defined(__CYGWIN__) && !defined(_WIN32)
#	define LIBRAPID_OS_WINDOWS // Windows (Cygwin POSIX under Microsoft Window)
#	define LIBRAPID_OS			"windows"
#elif defined(__ANDROID__)
#	define LIBRAPID_OS_ANDROID // Android (implies Linux, so it must come
								// first)
#	define LIBRAPID_OS "android"
#elif defined(__linux__)
#	define LIBRAPID_OS_LINUX // Debian, Ubuntu, Gentoo, Fedora, openSUSE,
							  // RedHat, Centos and other
#	define LIBRAPID_OS_UNIX
#	define LIBRAPID_OS "linux"
#elif defined(__unix__) || !defined(__APPLE__) && defined(__MACH__)
#	include <sys/param.h>
#	if defined(BSD)
#		define LIBRAPID_OS_BSD // FreeBSD, NetBSD, OpenBSD, DragonFly BSD
#		define LIBRAPID_OS_UNIX
#		define LIBRAPID_OS "bsd"
#	endif
#elif defined(__hpux)
#	define LIBRAPID_OS_HP_UX // HP-UX
#	define LIBRAPID_OS		  "hp-ux"
#elif defined(_AIX)
#	define LIBRAPID_OS_AIX // IBM AIX
#	define LIBRAPID_OS		"aix"
#elif defined(__APPLE__) && defined(__MACH__) // Apple OSX and iOS (Darwin)
#	define LIBRAPID_OS_APPLE
#	define LIBRAPID_OS_UNIX
#	include <TargetConditionals.h>
#	if TARGET_IPHONE_SIMULATOR == 1
#		define LIBRAPID_OS_IOS // Apple iOS
#		define LIBRAPID_OS		"ios"
#	elif TARGET_OS_IPHONE == 1
#		define LIBRAPID_OS_IOS // Apple iOS
#		define LIBRAPID_OS		"ios"
#	elif TARGET_OS_MAC == 1
#		define LIBRAPID_OS_OSX // Apple OSX
#		define LIBRAPID_OS		"osx"
#	endif
#elif defined(__sun) && defined(__SVR4)
#	define LIBRAPID_OS_SOLARIS // Oracle Solaris, Open Indiana
#	define LIBRAPID_OS			"solaris"
#else
#	define LIBRAPID_OS_UNKNOWN
#	define LIBRAPID_OS "unknown"
#endif

namespace librapid {
	constexpr int64_t AUTO = -1;
}

#if defined(OPENBLAS_OPENMP) || defined(OPENBLAS_THREAD) ||                    \
  defined(OPENBLAS_SEQUENTIAL)
#	define LIBRAPID_HAS_OPENBLAS
#endif

#include <iostream>

// BLAS config settings
namespace librapid {
	inline bool hasBlas() {
#ifdef LIBRAPID_HAS_BLAS
		return true;
#else
		return false;
#endif // LIBRAPID_HAS_BLAS
	}

	inline bool hasCuda() {
#ifdef LIBRAPID_HAS_CUDA
		return true;
#else
		return false;
#endif
	}

	// inline void setBlasThreads(int num)
	// {
	// #ifdef LIBRAPID_HAS_OPENBLAS
	// 	openblas_set_num_threads(num);
	// 	goto_set_num_threads(num);
	//
	// #ifdef LR_HAS_OMP
	// 	omp_set_num_threads(num);
	// #endif // LR_HAS_OMP
	// #else
	// 	throw std::runtime_error("Cannot set BLAS threads because OpenBLAS was
	// not " 							 "linked against"); #endif //
	// LIBRAPID_HAS_OPENBLAS
	// }
	//
	// inline int getBlasThreads()
	// {
	// #ifdef LIBRAPID_HAS_OPENBLAS
	// 	return openblas_get_num_threads();
	// #else
	// 	throw std::runtime_error("Cannot set BLAS threads because OpenBLAS was
	// not " 							 "linked against"); 	return -1;
	// #endif
	// // LIBRAPID_HAS_OPENBLAS
	// }

	inline void setNumThreads(int64_t num) {
#if defined(LIBRAPID_HAS_OPENBLAS)
		openblas_set_num_threads((int)num);
		goto_set_num_threads((int)num);
#endif

#if defined(_OPENMP)
		omp_set_num_threads((int)num);
#endif
	}

	inline int getNumThreads() {
#if defined(LIBRAPID_HAS_OPENBLAS)
		return openblas_get_num_threads();
#elif defined(LR_HAS_OMP)
		return omp_get_num_threads();
#else
		return 1;
#endif
	}
} // namespace librapid

// CUDA enabled LibRapid
#ifdef LIBRAPID_HAS_CUDA

#	ifdef _MSC_VER
// Disable warnings about unsafe classes
#		pragma warning(disable : 4996)

// Disable zero division errors
#		pragma warning(disable : 4723)
#	endif

#	include <cublas_v2.h>
#	include <cuda.h>
#	include <curand.h>
#	include <curand_kernel.h>
#	include <jitify/jitify.hpp>

// cuBLAS API errors
static const char *getCublasErrorEnum_(cublasStatus_t error) {
	switch (error) {
		case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
		case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "UNKNOWN ERROR";
}

//********************//
// cuBLAS ERROR CHECK //
//********************//
#	ifndef cublasSafeCall
#		define cublasSafeCall(err) cublasSafeCall_(err, __FILE__, __LINE__)
#	endif

inline void cublasSafeCall_(cublasStatus_t err, const char *file,
							const int line) {
	if (err != CUBLAS_STATUS_SUCCESS)
		throw std::runtime_error("cuBLAS error at (" + std::string(file) +
								 ", line " + std::to_string(line) +
								 "): " + getCublasErrorEnum_(err));
}

//********************//
// CUDA ERROR CHECK //
//********************//
#	ifndef cudaSafeCall
#		define cudaSafeCall(err) cudaSafeCall_(err, __FILE__, __LINE__)
#	endif

inline void cudaSafeCall_(cudaError_t err, const char *file, const int line) {
	if (err != cudaSuccess)
		throw std::runtime_error("CUDA error at (" + std::string(file) +
								 ", line " + std::to_string(line) +
								 "): " + cudaGetErrorString(err));
}

#	define jitifyCall(call)                                                   \
		do {                                                                   \
			if (call != CUDA_SUCCESS) {                                        \
				const char *str;                                               \
				cuGetErrorName(call, &str);                                    \
				throw std::runtime_error(std::string("CUDA JIT failed: ") +    \
										 str);                                 \
			}                                                                  \
		} while (0)

#	ifdef _MSC_VER
#		pragma warning(default : 4996)
#	endif

#	include <helper_cuda.h>
#	include <helper_functions.h>

#endif // LIBRAPID_HAS_CUDA

#undef min
#undef max

// Disable "conversation from X to Y, possible loss of data"
#if defined(__clang__)
#	pragma GCC diagnostic push
#	pragma GCC diagnostic ignored "-Wconversion"
#elif defined(__GNUC__) || defined(__GNUG__)
#	pragma GCC diagnostic push
#	pragma GCC diagnostic ignored "-Wconversion"
#elif defined(_MSC_VER)
#	pragma warning(push)
#	pragma warning(disable : 4244)
#	pragma warning(disable : 6386)
#endif

namespace librapid {
	inline void *alignedMalloc(uint64_t requiredBytes,
							   uint64_t alignment = DATA_ALIGN) {
		void *p1;  // original block
		void **p2; // aligned block
		int64_t offset = alignment - 1 + sizeof(void *);

		if ((p1 = (void *)malloc(requiredBytes + offset)) == nullptr)
			throw std::bad_alloc();

		p2	   = (void **)(((int64_t)(p1) + offset) & ~(alignment - 1));
		p2[-1] = p1;
		return p2;
	}

	inline void alignedFree(void *p) { free(((void **)p)[-1]); }
} // namespace librapid

#include <librapid/autocast/custom_complex.hpp>
#include <type_traits>

namespace librapid {
	template<typename A, typename B>
	struct CommonType {
		using type = typename std::common_type<A, B>::type;
	};

	template<typename A, typename B>
	struct CommonType<Complex<A>, B> {
		using type = Complex<typename std::common_type<A, B>::type>;
	};

	template<typename A, typename B>
	struct CommonType<A, Complex<B>> {
		using type = Complex<typename std::common_type<A, B>::type>;
	};

	template<typename A, typename B>
	struct CommonType<Complex<A>, Complex<B>> {
		using type = Complex<typename std::common_type<A, B>::type>;
	};
} // namespace librapid

#endif // LIBRAPID_CONFIG