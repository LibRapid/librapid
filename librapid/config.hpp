#ifndef LIBRAPID_CONFIG
#define LIBRAPID_CONFIG

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

#ifdef LIBRAPID_PYTHON_FLOAT
	using python_dtype = float;
#else
	#ifdef LIBRAPID_PYTHON_DOUBLE
		using python_dtype = double;
	#else
		#error Unknown Python Datatype
	#endif
#endif

namespace librapid
{
	int python_bitness()
	{
		return sizeof(python_dtype) * 8;
	}
}

#endif

#ifdef LIBRAPID_CBLAS
#undef LIBRAPID_CBLAS
#define LIBRAPID_CBLAS 1
#include <cblas.h>
#endif

#if defined(NDEBUG) || defined(LIBRAPID_NDEBUG)
#define LR_NDEBUG
#define LR_INLINE inline
#else
#define LR_DEBUG
#define LR_INLINE
#endif // NDEBUG || NDARRAY_DEBUG

#ifdef _OPENMP
#define LR_HAS_OMP
#include <omp.h>
#endif // _OPENMP

#ifndef LIBRAPID_MAX_DIMS
#define LIBRAPID_MAX_DIMS 32
#endif // LIBRAPID_MAX_DIMS

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
#define LIBRAPID_OS_UNIX
#define LIBRAPID_OS "linux"
#elif defined(__unix__) || !defined(__APPLE__) && defined(__MACH__)
#include <sys/param.h>
#if defined(BSD)
#define LIBRAPID_OS_BSD // FreeBSD, NetBSD, OpenBSD, DragonFly BSD
#define LIBRAPID_OS_UNIX
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
#define LIBRAPID_OS_UNIX
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

using lr_int = long long;

#if defined(OPENBLAS_OPENMP) || defined(OPENBLAS_THREAD) || defined(OPENBLAS_SEQUENTIAL)
#define LIBRAPID_HAS_OPENBLAS
#endif

// Easy access to the current time (in ms)
#include <chrono>
#define TIME (double) std::chrono::high_resolution_clock().now().time_since_epoch().count() / 1000000000

namespace librapid
{
	LR_INLINE void sleep(double time)
	{
		auto start = TIME;

		while (TIME - start < time)
		{
		}
	}
}

// Console width and height information
#if defined(LIBRAPID_OS_WINDOWS)
#include <windows.h>
#elif defined(LIBRAPID_OS_UNIX)
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace librapid
{
	struct console_size
	{
		lr_int rows;
		lr_int cols;
	};

#if defined(LIBRAPID_OS_WINDOWS)
	console_size get_console_size()
	{
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		int cols, rows;

		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
		cols = csbi.srWindow.Right - csbi.srWindow.Left + 1;
		rows = csbi.srWindow.Bottom - csbi.srWindow.Top + 1;

		return {rows, cols};
	}
#elif defined(LIBRAPID_OS_UNIX)
	console_size get_console_size()
	{
		struct winsize w;
		ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);

		return {w.ws_row, w.ws_col};
	}
#else
	console_size get_console_size()
	{
		// Not a clue what this would run on, or how it would be done
		// correctly, so just return some arbitrary values...

		return {70, 120};
	}
#endif
}

// BLAS config settings
namespace librapid
{
	bool has_blas()
	{
	#if defined(LIBRAPID_CBLAS) && (LIBRAPID_CBLAS == 1)
		return true;
	#else
		return false;
	#endif // LIBRAPID_CBLAS
}

	void set_blas_threads(int num)
	{
	#ifdef LIBRAPID_HAS_OPENBLAS
		openblas_set_num_threads(num);
		goto_set_num_threads(num);

	#ifdef LR_HAS_OMP
		omp_set_num_threads(num);
	#endif // LR_HAS_OMP
	#else
		std::cout << "The function 'librapid_set_blas_threads' only works in C++"
			"and Python when compiled with OpenBLAS" << "\n";
	#endif // LIBRAPID_HAS_OPENBLAS
	}

	int get_blas_threads()
	{
	#ifdef LIBRAPID_HAS_OPENBLAS
		return openblas_get_num_threads();
	#else
		std::cout << "The function 'librapid_set_blas_threads' only works in C++"
			"and Python when compiled with OpenBLAS" << "\n";
		return -1;
	#endif // LIBRAPID_HAS_OPENBLAS
	}

	void set_num_threads(int num)
	{
	#if defined(LIBRAPID_HAS_OPENBLAS)
		set_blas_threads(num);
	#elif defined(LR_HAS_OMP)
		omp_set_num_threads(num);
	#else
		std::cout << "LibRapid does not have access to any multi-threaded components"
			"such as OpenMP or OpenBLAS, so the function \"set_num_threads\""
			"will not do anything" << "\n";
	#endif
	}

	int get_num_threads()
	{
	#if defined(LIBRAPID_HAS_OPENBLAS)
		return get_blas_threads();
	#elif defined(LR_HAS_OMP)
		return omp_get_num_threads();
	#else
		std::cout << "LibRapid does not have access to any multi-threaded components"
			"such as OpenMP or OpenBLAS, so the function \"set_num_threads\""
			"will not do anything" << "\n";
	#endif
	}
}

#undef min
#undef max

#endif // LIBRAPID_CONFIG