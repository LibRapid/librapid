#pragma once

// Include required headers
#include <cstdlib>
#include <cstddef>
#include <string>
#include <vector>
#include <type_traits>
#include <algorithm>
#include <complex>
#include <memory>
#include <iomanip>
#include <iostream>
#include <atomic>
#include <thread>

#if defined(_OPENMP)
#	include <omp.h>
#	define LIBRAPID_OPENMP
#	define LIBRAPID_OPENMP_VAL 1
#else
#	define LIBRAPID_OPENMP_VAL 0
#endif

// SIMD instructions
#define VCL_NAMESPACE vcl

#include "../vendor/version2/vectorclass.h"

// Include {fmt} -- fast IO and formatting
#define FMT_HEADER_ONLY

#include "../vendor/fmt/include/fmt/core.h"
#include "../vendor/fmt/include/fmt/format.h"
#include "../vendor/fmt/include/fmt/ranges.h"
#include "../vendor/fmt/include/fmt/chrono.h"
#include "../vendor/fmt/include/fmt/compile.h"
#include "../vendor/fmt/include/fmt/color.h"
#include "../vendor/fmt/include/fmt/os.h"
#include "../vendor/fmt/include/fmt/ostream.h"
#include "../vendor/fmt/include/fmt/printf.h"
#include "../vendor/fmt/include/fmt/xchar.h"

// LibRapid definitions

#if !defined(NDEBUG)
#	define LIBRAPID_DEBUG
#else
#	define LIBRAPID_RELEASE
#endif

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

// Compiler information
#if defined(__GNUC__)
#	define LIBRAPID_GNU_CXX
#	define LIBRAPID_COMPILER "GNU C/C++ Compiler"
#endif

#if defined(__MINGW32__)
#	define LIBRAPID_MINGW_CXX
#	define LIBRAPID_COMPILER "Mingw or GNU C/C++ Compiler ported for Windows NT"
#endif

#if defined(__MINGW64__)
#	define LIBRAPID_MINGW_CXX
#	define LIBRAPID_COMPILER "Mingw or GNU C/C++ Compiler ported for Windows NT - 64 bits only"
#endif

#if defined(__GFORTRAN__)
#	define LIBRAPID_FORTRAN_CXX
#	define LIBRAPID_COMPILER "Fortran / GNU Fortran Compiler"
#endif

#if defined(__clang__)
#	define LIBRAPID_CLANG_CXX
#	define LIBRAPID_COMPILER "Clang / LLVM Compiler"
#endif

#if defined(_MSC_VER)
#	define LIBRAPID_MSVC_CXX
#	define LIBRAPID_COMPILER "Microsoft Visual Studio Compiler MSVC"
#endif

#if defined(_MANAGED) || defined(__cplusplus_cli)
#	define LIBRAPID_DOTNET_CXX
#	define LIBRAPID_COMPILER "Compilation to C++/CLI .NET (CLR) bytecode"
#endif

#if defined(__INTEL_COMPILER)
#	define LIBRAPID_INTEL_CXX
#	define LIBRAPID_COMPILER "Intel C/C++ Compiler"
#endif

#if defined(__PGI) || defined(__PGIC__)
#	define LIBRAPID_PORTLAND_CXX
#	define LIBRAPID_COMPILER "Portland Group C/C++ Compiler"
#endif

#if defined(__BORLANDC__)
#	define LIBRAPID_BORLAND_CXX
#	define LIBRAPID_COMPILER "Borland C++ Compiler"
#endif

#if defined(__EMSCRIPTEN__)
#	define LIBRAPID_EMSCRIPTEN_CXX
#	define LIBRAPID_COMPILER "emscripten (asm.js - web assembly)"
#endif

#if defined(__asmjs__)
#	define LIBRAPID_ASMJS_CXX
#	define LIBRAPID_COMPILER "asm.js"
#endif

#if defined(__wasm__)
#	define LIBRAPID_WASM_CXX
#	define LIBRAPID_COMPILER "WebAssembly"
#endif

#if defined(__NVCC__)
#	define LIBRAPID_NVCC_CXX
#	define LIBRAPID_COMPILER "NVIDIA NVCC CUDA Compiler"
#endif

#if defined(__CLING__)
#	define LIBRAPID_CLING_CXX
#	define LIBRAPID_COMPILER "CERN's ROOT Cling C++ Interactive Shell"
#endif

#if defined(__GLIBCXX__)
#	define LIBRAPID_LIBSTDCPP
#	define LIBRAPID_STL "Libstdc++"
#endif

#if defined(_LBCPP_VERSION)
#	define LIBRAPID_LIBCPP
#	define LIBRAPID_STL "LibC++"
#endif

#if defined(_MSC_VER)
#	define LIBRAPID_MSVCSTD
#	define LIBRAPID_STL "MSVC C++ Library (Runtime)"
#endif

#if defined(__BIONIC__)
#	define LIBRAPID_BIONIC
#	define LIBRAPID_STL "Bionic LibC runtime. (Android's C-library modified from BSD)"
#endif

// Check for 32bit vs 64bit
// Check windows
#if _WIN32 || _WIN64
#	if _WIN64
#		define LIBRAPID_64BIT
#	else
#		define LIBRAPID_32BIT
#	endif
#endif

// Check GCC
#if __GNUC__
#	if __x86_64__ || __ppc64__
#		define LIBRAPID_64BIT
#	else
#		define LIBRAPID_32BIT
#	endif
#endif

// Check C++ Version
#if __cplusplus >= 201103L
#	define LIBRAPID_CXX_11
#elif __cplusplus >= 201402L
#	define LIBRAPID_CXX_14
#elif __cplusplus >= 201703L
#	define LIBRAPID_CXX_17
#elif __cplusplus >= 202002L
#	define LIBRAPID_CXX_20
#else
#	define LIBRAPID_CXX_11 // Assume we're using C++ 11???
#endif

// Nice-to-have macros
#ifdef LIBRAPID_CXX_20
#	define LIBRAPID_LIKELY	  [[likely]]
#	define LIBRAPID_UNLIKELY [[unlikely]]
#else
#	define LIBRAPID_LIKELY
#	define LIBRAPID_UNLIKELY
#endif

#ifndef __has_cpp_attribute
#	define __has_cpp_attribute(x) 0
#endif

#if __has_cpp_attribute(deprecated)
#	define LR_DEPRECATED(msg) [[deprecated(msg)]]
#else
#	define LR_DEPRECATED(msg)
#endif

#if __has_cpp_attribute(nodiscard)
#	if defined(LIBRAPID_CXX_20)
#		define LR_NODISCARD(msg) [[nodiscard(msg)]]
#	else
#		define LR_NODISCARD(msg) [[nodiscard]]
#	endif
#else
#	define LR_NODISCARD(msg)
#endif

#if defined(__cpp_consteval)
#	define LR_CONSTEVAL consteval
#else
#	define LR_CONSTEVAL
#endif

#if defined(__cpp_constexpr)
#	define LR_CONSTEXPR consteval
#else
#	define LR_CONSTEXPR
#endif

#if defined(FILENAME)
#	warning                                                                                        \
	  "The macro 'FILENAME' is already defined. LibRapid's logging system might not function correctly as a result"
#else
#	ifdef LIBRAPID_OS_WINDOWS
#		define FILENAME (strrchr(__FILE__, '\\') ? strrchr(__FILE__, '\\') + 1 : __FILE__)
#	else
#		define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#	endif
#endif

#if defined(FUNCTION)
#	warning                                                                                        \
	  "The macro 'FUNCTION' is already defined. LibRapid's logging system might not function correctly as a result"
#else
#	if defined(LIBRAPID_MSVC_CXX)
#		define FUNCTION __FUNCSIG__
#	elif defined(LIBRAPID_GNU_CXX) || defined(LIBRAPID_CLANG_CXX) || defined(LIBRAPID_CLING_CXX)
#		define FUNCTION __PRETTY_FUNCTION__
#	else
#		define FUNCTION "Function Signature Unknown"
#	endif
#endif

// Settings and macros for inline functions
#if defined(LIBRAPID_DEBUG) || defined(LIBRAPID_NO_INLINE)
#	define LR_INLINE
#	define LR_FORCE_INLINE
#else // LIBRAPID_DEBUG || LIBRAPID_NO_INLINE

#	define LR_INLINE inline

#	if defined(LIBRAPID_MSVC_CXX)
#		define LR_FORCE_INLINE __forceinline
#	elif defined(LIBRAPID_GNU_CXX) || defined(LIBRAPID_CLANG_CXX) ||                              \
	  defined(LIBRAPID_INTEL_CXX) || defined(LIBRAPID_CLING_CXX) || defined(LIBRAPID_MINGW_CXX)
#		define LR_FORCE_INLINE inline __attribute__((always_inline))
#	else
#		define LR_FORCE_INLINE inline
#	endif
#endif // LIBRAPID_DEBUG || LIBRAPID_NO_INLINE

// Settings for Debug configurations
#if defined(LIBRAPID_DEBUG)
#	define LIBRAPID_ASSERT
#	define LIBRAPID_LOG
#endif

// Settings for Release configurations
#if defined(LIBRAPID_RELEASE)
#endif

#if defined(LIBRAPID_TRACEBACK)
#	define LIBRAPID_ASSERT
#	define LIBRAPID_LOG
#endif

#ifdef LIBRAPID_OS_WINDOWS
#	include <windows.h>

// Construct a class to force ANSI sequences to work
namespace librapid::internal {
	class ForceANSI {
	public:
		ForceANSI() { system(("chcp " + std::to_string(CP_UTF8)).c_str()); }
	};

	const inline auto ansiForcer = ForceANSI();
} // namespace librapid::internal
#endif

#if defined(LIBRAPID_ASSERT) || defined(LIBRAPID_LOG) || defined(LIBRAPID_DEBUG)
#	ifdef LIBRAPID_MSVC_CXX
#		define LR_LOG_STATUS(msg, ...)                                                            \
			do {                                                                                   \
				fmt::print("{} {}\n",                                                              \
						   fmt::format(fmt::fg(fmt::color::green), "[STATUS]"),                    \
						   fmt::format(msg, __VA_ARGS__));                                         \
			} while (0)

#		define LR_LOG_WARN(msg, ...)                                                              \
			do {                                                                                   \
				fmt::print("{} {}\n",                                                              \
						   fmt::format(fmt::fg(fmt::color::yellow), "[WARNING]"),                  \
						   fmt::format(msg, __VA_ARGS__));                                         \
			} while (0)

#		define LR_LOG_ERROR(msg, ...)                                                             \
			do {                                                                                   \
				fmt::print("{} {}\n",                                                              \
						   fmt::format(fmt::fg(fmt::color::red), "[ERROR]"),                       \
						   fmt::format(msg, __VA_ARGS__));                                         \
				std::exit(1);                                                                      \
			} while (0)
#	else
#		define LR_LOG_STATUS(msg, ...)                                                            \
			do {                                                                                   \
				fmt::print("{} {}\n",                                                              \
						   fmt::format(fmt::fg(fmt::color::green), "[STATUS]"),                    \
						   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__));                           \
			} while (0)

#		define LR_LOG_WARN(msg, ...)                                                              \
			do {                                                                                   \
				fmt::print("{} {}\n",                                                              \
						   fmt::format(fmt::fg(fmt::color::yellow), "[WARNING]"),                  \
						   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__));                           \
			} while (0)

#		define LR_LOG_ERROR(msg, ...)                                                             \
			do {                                                                                   \
				fmt::print("{} {}\n",                                                              \
						   fmt::format(fmt::fg(fmt::color::red), "[ERROR]"),                       \
						   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__));                           \
				std::exit(1);                                                                      \
			} while (0)
#	endif // LIBRAPID_MSVC_CXX
#else
#	define LR_LOG_STATUS(...)
#	define LR_LOG_WARN(...)
#	define LR_LOG_ERROR(...)
#endif

// Lightweight min max functions
namespace librapid::internal {
	/*
	 * We can't use the librapid::max function here, because it's defined in
	 * math.hh, which also includes config.hh, so we'd have a circular include
	 * which would cause a LOT of issues
	 */

	template<typename T>
	T smallMax_internal(T val) {
		return val;
	}

	template<typename T, typename... Tn>
	T smallMax_internal(T val, Tn... vals) {
		auto maxOther = smallMax_internal(vals...);
		return val < maxOther ? maxOther : val;
	}
} // namespace librapid::internal

#if defined(LIBRAPID_ASSERT)

#	ifdef LIBRAPID_MSVC_CXX
#		define LR_STATUS(msg, ...)                                                                \
			do {                                                                                   \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)strlen(FUNCTION) + 6,                 \
														(int)strlen("WARN ASSERTION FAILED"));     \
				fmt::print(fmt::fg(fmt::color::green),                                             \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "STATUS",                                                               \
						   FILENAME,                                                               \
						   FUNCTION,                                                               \
						   __LINE__,                                                               \
						   fmt::format(msg, __VA_ARGS__),                                          \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen);                                                                \
			} while (0)

#		define LR_WARN(msg, ...)                                                                  \
			do {                                                                                   \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)strlen(FUNCTION) + 6,                 \
														(int)strlen("WARN ASSERTION FAILED"));     \
				fmt::print(fmt::fg(fmt::color::yellow),                                            \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "WARNING",                                                              \
						   FILENAME,                                                               \
						   FUNCTION,                                                               \
						   __LINE__,                                                               \
						   fmt::format(msg, __VA_ARGS__),                                          \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen);                                                                \
			} while (0)

#		define LR_ERROR(msg, ...)                                                                 \
			do {                                                                                   \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)strlen(FUNCTION) + 6,                 \
														(int)strlen("WARN ASSERTION FAILED"));     \
				fmt::print(fmt::fg(fmt::color::red),                                               \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "ERROR",                                                                \
						   FILENAME,                                                               \
						   FUNCTION,                                                               \
						   __LINE__,                                                               \
						   fmt::format(msg, __VA_ARGS__),                                          \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen);                                                                \
				std::exit(1);                                                                      \
			} while (0)

#		define LR_WASSERT(cond, msg, ...)                                                         \
			do {                                                                                   \
				if (!(cond)) {                                                                     \
					int maxLen = librapid::internal::smallMax_internal(                            \
					  (int)std::ceil(std::log(__LINE__)) + 6,                                      \
					  (int)strlen(FILENAME) + 6,                                                   \
					  (int)strlen(FUNCTION) + 6,                                                   \
					  (int)strlen(#cond) + 6,                                                      \
					  (int)strlen("WARN ASSERTION FAILED"));                                       \
					fmt::print(fmt::fg(fmt::color::yellow),                                        \
							   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                          \
							   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                           \
							   "{4:>{10}}]\n{5}\n",                                                \
							   "WARN ASSERTION FAILED",                                            \
							   FILENAME,                                                           \
							   FUNCTION,                                                           \
							   __LINE__,                                                           \
							   #cond,                                                              \
							   fmt::format(msg, __VA_ARGS__),                                      \
							   maxLen + 5,                                                         \
							   maxLen + 0,                                                         \
							   maxLen - 4,                                                         \
							   maxLen + 0,                                                         \
							   maxLen - 5);                                                        \
				}                                                                                  \
			} while (0)

#		define LR_ASSERT(cond, msg, ...)                                                          \
			do {                                                                                   \
				if (!(cond)) {                                                                     \
					int maxLen =                                                                   \
					  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)),    \
															(int)strlen(FILENAME),                 \
															(int)strlen(FUNCTION),                 \
															(int)strlen(#cond),                    \
															(int)strlen("ASSERTION FAILED"));      \
					fmt::print(fmt::fg(fmt::color::red),                                           \
							   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                          \
							   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                           \
							   "{4:>{10}}]\n{5}\n",                                                \
							   "ASSERTION FAILED",                                                 \
							   FILENAME,                                                           \
							   FUNCTION,                                                           \
							   __LINE__,                                                           \
							   #cond,                                                              \
							   fmt::format(msg, __VA_ARGS__),                                      \
							   maxLen + 14,                                                        \
							   maxLen + 9,                                                         \
							   maxLen + 5,                                                         \
							   maxLen + 9,                                                         \
							   maxLen + 4);                                                        \
					std::exit(1);                                                                  \
				}                                                                                  \
			} while (0)
#	else
#		define LR_STATUS(msg, ...)                                                                \
			do {                                                                                   \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)strlen(FUNCTION) + 6,                 \
														(int)strlen("WARN ASSERTION FAILED"));     \
				fmt::print(fmt::fg(fmt::color::green),                                             \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "STATUS",                                                               \
						   FILENAME,                                                               \
						   FUNCTION,                                                               \
						   __LINE__,                                                               \
						   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                            \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen);                                                                \
			} while (0)

#		define LR_WARN(msg, ...)                                                                  \
			do {                                                                                   \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)strlen(FUNCTION) + 6,                 \
														(int)strlen("WARN ASSERTION FAILED"));     \
				fmt::print(fmt::fg(fmt::color::yellow),                                            \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "WARNING",                                                              \
						   FILENAME,                                                               \
						   FUNCTION,                                                               \
						   __LINE__,                                                               \
						   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                            \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen);                                                                \
			} while (0)

#		define LR_ERROR(msg, ...)                                                                 \
			do {                                                                                   \
				int maxLen =                                                                       \
				  librapiod::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,   \
														 (int)strlen(FILENAME) + 6,                \
														 (int)strlen(FUNCTION) + 6,                \
														 (int)strlen("WARN ASSERTION FAILED"));    \
				fmt::print(fmt::fg(fmt::color::red),                                               \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "ERROR",                                                                \
						   FILENAME,                                                               \
						   FUNCTION,                                                               \
						   __LINE__,                                                               \
						   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                            \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen);                                                                \
				std::exit(1);                                                                      \
			} while (0)

#		define LR_WASSERT(cond, msg, ...)                                                         \
			do {                                                                                   \
				if (!(cond)) {                                                                     \
					int maxLen = librapid::internal::smallMax_internal(                            \
					  (int)std::ceil(std::log(__LINE__)) + 6,                                      \
					  (int)strlen(FILENAME) + 6,                                                   \
					  (int)strlen(FUNCTION) + 6,                                                   \
					  (int)strlen(#cond) + 6,                                                      \
					  (int)strlen("WARN ASSERTION FAILED"));                                       \
					fmt::print(fmt::fg(fmt::color::yellow),                                        \
							   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                          \
							   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                           \
							   "{4:>{10}}]\n{5}\n",                                                \
							   "WARN ASSERTION FAILED",                                            \
							   FILENAME,                                                           \
							   FUNCTION,                                                           \
							   __LINE__,                                                           \
							   #cond,                                                              \
							   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                        \
							   maxLen + 5,                                                         \
							   maxLen + 0,                                                         \
							   maxLen - 4,                                                         \
							   maxLen + 0,                                                         \
							   maxLen - 5);                                                        \
				}                                                                                  \
			} while (0)

#		define LR_ASSERT(cond, msg, ...)                                                          \
			do {                                                                                   \
				if (!(cond)) {                                                                     \
					int maxLen =                                                                   \
					  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)),    \
															(int)strlen(FILENAME),                 \
															(int)strlen(FUNCTION),                 \
															(int)strlen(#cond),                    \
															(int)strlen("ASSERTION FAILED"));      \
					fmt::print(fmt::fg(fmt::color::red),                                           \
							   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                          \
							   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                           \
							   "{4:>{10}}]\n{5}\n",                                                \
							   "ASSERTION FAILED",                                                 \
							   FILENAME,                                                           \
							   FUNCTION,                                                           \
							   __LINE__,                                                           \
							   #cond,                                                              \
							   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                        \
							   maxLen + 14,                                                        \
							   maxLen + 9,                                                         \
							   maxLen + 5,                                                         \
							   maxLen + 9,                                                         \
							   maxLen + 4);                                                        \
					std::exit(1);                                                                  \
				}                                                                                  \
			} while (0)
#	endif // LIBRAPID_MSVC_CXX
#else
#	define LR_STATUS(msg, ...)                                                                    \
		do {                                                                                       \
		} while (0)
#	define LR_WARN(msg, ...)                                                                      \
		do {                                                                                       \
		} while (0)
#	define LR_ERROR(msg, ...)                                                                     \
		do {                                                                                       \
		} while (0)
#	define LR_LOG(msg, ...)                                                                       \
		do {                                                                                       \
		} while (0)
#	define LR_WASSERT(cond, ...)                                                                  \
		do {                                                                                       \
		} while (0)
#	define LR_ASSERT(cond, ...)                                                                   \
		do {                                                                                       \
		} while (0)
#endif

#if defined(LIBRAPID_MSVC_CXX)
#	define LR_ASSERT_ALWAYS(cond, msg, ...)                                                       \
		do {                                                                                       \
			if (!(cond)) {                                                                         \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)),        \
														(int)strlen(FILENAME),                     \
														(int)strlen(FUNCTION),                     \
														(int)strlen(#cond),                        \
														(int)strlen("ASSERTION FAILED"));          \
				fmt::print(fmt::fg(fmt::color::red),                                               \
						   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                              \
						   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                               \
						   "{4:>{10}}]\n{5}\n",                                                    \
						   "ASSERTION FAILED",                                                     \
						   FILENAME,                                                               \
						   FUNCTION,                                                               \
						   __LINE__,                                                               \
						   #cond,                                                                  \
						   fmt::format(msg, __VA_ARGS__),                                          \
						   maxLen + 14,                                                            \
						   maxLen + 9,                                                             \
						   maxLen + 5,                                                             \
						   maxLen + 9,                                                             \
						   maxLen + 4);                                                            \
				std::exit(1);                                                                      \
			}                                                                                      \
		} while (0)
#else
#	define LR_ASSERT_ALWAYS(cond, msg, ...)                                                       \
		do {                                                                                       \
			if (!(cond)) {                                                                         \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)),        \
														(int)strlen(FILENAME),                     \
														(int)strlen(FUNCTION),                     \
														(int)strlen(#cond),                        \
														(int)strlen("ASSERTION FAILED"));          \
				fmt::print(fmt::fg(fmt::color::red),                                               \
						   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                              \
						   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                               \
						   "{4:>{10}}]\n{5}\n",                                                    \
						   "ASSERTION FAILED",                                                     \
						   FILENAME,                                                               \
						   FUNCTION,                                                               \
						   __LINE__,                                                               \
						   #cond,                                                                  \
						   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                            \
						   maxLen + 14,                                                            \
						   maxLen + 9,                                                             \
						   maxLen + 5,                                                             \
						   maxLen + 9,                                                             \
						   maxLen + 4);                                                            \
				std::exit(1);                                                                      \
			}                                                                                      \
		} while (0)
#endif

#if defined(LIBRAPID_TRACEBACK)
#	define LR_TRACE LR_STATUS("LIBRAPID TRACEBACK")
#else
#	define LR_TRACE                                                                               \
		do {                                                                                       \
		} while (0)
#endif

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
#	include "../vendor/jitify/jitify.hpp"

// cuBLAS API errors
static const char *getCublasErrorEnum_(cublasStatus_t error) {
	switch (error) {
		case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
		case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
		case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
	}

	return "UNKNOWN ERROR";
}

//********************//
// cuBLAS ERROR CHECK //
//********************//
// #	ifndef cublasSafeCall
// #		define cublasSafeCall(err) cublasSafeCall_(err, __FILE__, __LINE__)
// #	endif

#	if !defined(cublasSafeCall)
#		define cublasSafeCall(err)                                                                \
			LR_ASSERT_ALWAYS(                                                                      \
			  err == CUBLAS_STATUS_SUCCESS, "cuBLAS error: {}", getCublasErrorEnum_(err))
#	endif

// inline void cublasSafeCall_(cublasStatus_t err, const char *file,
//							const int line) {
//	if (err != CUBLAS_STATUS_SUCCESS)
//		throw std::runtime_error("cuBLAS error at (" + std::string(file) +
//								 ", line " + std::to_string(line) +
//								 "): " + getCublasErrorEnum_(err));
// }

//********************//
//  CUDA ERROR CHECK  //
//********************//
//#	ifndef cudaSafeCall
//#		define cudaSafeCall(err) cudaSafeCall_(err, __FILE__, __LINE__)
//#	endif
//
// inline void cudaSafeCall_(cudaError_t err, const char *file, const int line)
// { 	if (err != cudaSuccess) 		throw std::runtime_error("CUDA error at ("
// +
// std::string(file) +
//								 ", line " + std::to_string(line) +
//								 "): " + cudaGetErrorString(err));
//}

#	if !defined(cudaSafeCall)
#		define cudaSafeCall(err)                                                                  \
			LR_ASSERT_ALWAYS(                                                                      \
			  err == CUBLAS_STATUS_SUCCESS, "CUDA error: {}", cudaGetErrorString(err))
#	endif

#	define jitifyCall(call)                                                                       \
		do {                                                                                       \
			if (call != CUDA_SUCCESS) {                                                            \
				const char *str;                                                                   \
				cuGetErrorName(call, &str);                                                        \
				throw std::runtime_error(std::string("CUDA JIT failed: ") + str);                  \
			}                                                                                      \
		} while (0)

#	ifdef _MSC_VER
#		pragma warning(default : 4996)
#	endif

#	include "src/librapid/cuda/helper_cuda.h"
#	include "src/librapid/cuda/helper_functions.h"

#endif // LIBRAPID_HAS_CUDA

namespace librapid::device {
	struct CPU {};
	struct GPU {};
} // namespace librapid::device

// User Config Variables

namespace librapid {
#ifdef LIBRAPID_HAS_OMP
	inline static unsigned int numThreads = 8;
#else
	inline static unsigned int numThreads = 1;
#endif
} // namespace librapid

// Prefer using the GPU over the CPU -- promote arrays to the GPU where possible
#if !defined(LIBRAPID_PREFER_CPU)
#	define LIBRAPID_PREFER_GPU
#endif