#ifndef LIBRAPID_INCLUDE
#define LIBRAPID_INCLUDE

#pragma warning(push)
// Disable zero-division warnings for the vector library
#pragma warning(disable : 4723)

// Disable zero-division warnings for the vector library
#pragma warning(disable : 4804)

#ifndef LIBRAPID_VERSION
#define LIBRAPID_VERSION "0.3.19"
#endif

// Include required headers
#include <cstdlib>
#include <cmath>
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
#include <random>

#if defined(_OPENMP)
#	include <omp.h>
#	define LIBRAPID_OPENMP
#	define LIBRAPID_OPENMP_VAL 1
#else
#	define LIBRAPID_OPENMP_VAL 0
#endif

// Include {fmt} -- fast IO and formatting
#define FMT_HEADER_ONLY

#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/chrono.h>
#include <fmt/compile.h>
#include <fmt/color.h>
#include <fmt/os.h>
#include <fmt/ostream.h>
#include <fmt/printf.h>
#include <fmt/xchar.h>

// Include scnlib -- fast string scanning and IO
#include <scn/scn.h>
#include <scn/tuple_return/tuple_return.h>

// SIMD instructions
#include <Vc/Vc>
#include <Vc/algorithm>
#include <Vc/cpuid.h>

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

#if !defined(LIBRAPID_HAS_OMP) && defined(_OPENMP)
#	define LIBRAPID_HAS_OMP
#endif

#if defined(LIBRAPID_HAS_OMP)
#	define LIBRAPID_OMP_VAL 1
#else
#	define LIBRAPID_OMP_VAL 0
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

#if defined(LIBRAPID_OS_WINDOWS) && defined(LIBRAPID_MSVC_CXX)
#	define WIN32_LEAN_AND_MEAN
#	include <Windows.h>

// Construct a class to force ANSI sequences to work
namespace librapid { namespace internal {
	class ForceANSI {
	public:
		ForceANSI() { system(("chcp " + std::to_string(CP_UTF8)).c_str()); }
	};

	const auto ansiForcer = ForceANSI();
}} // namespace librapid::internal
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
namespace librapid { namespace internal {
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
}} // namespace librapid::internal

#if defined(LIBRAPID_ASSERT)

#	define LR_WARN_ONCE(msg, ...)                                                                 \
		do {                                                                                       \
			static bool _alerted = false;                                                          \
			if (!_alerted) {                                                                       \
				LR_WARN(msg, __VA_ARGS__);                                                         \
				_alerted = true;                                                                   \
			}                                                                                      \
		} while (false)

#	ifdef LIBRAPID_MSVC_CXX
#		define LR_STATUS(msg, ...)                                                                \
			do {                                                                                   \
				std::string funcName = FUNCTION;                                                   \
				if (funcName.length() > 75) funcName = "<Signature too Long>";                     \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)funcName.length() + 6,                \
														(int)strlen("WARN ASSERTION FAILED"));     \
				fmt::print(fmt::fg(fmt::color::green),                                             \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "STATUS",                                                               \
						   FILENAME,                                                               \
						   funcName,                                                               \
						   __LINE__,                                                               \
						   fmt::format(msg, __VA_ARGS__),                                          \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen);                                                                \
			} while (0)

#		define LR_WARN(msg, ...)                                                                  \
			do {                                                                                   \
				std::string funcName = FUNCTION;                                                   \
				if (funcName.length() > 75) funcName = "<Signature too Long>";                     \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)funcName.length() + 6,                \
														(int)strlen("WARN ASSERTION FAILED"));     \
				fmt::print(fmt::fg(fmt::color::yellow),                                            \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "WARNING",                                                              \
						   FILENAME,                                                               \
						   funcName,                                                               \
						   __LINE__,                                                               \
						   fmt::format(msg, __VA_ARGS__),                                          \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen);                                                                \
			} while (0)

#		define LR_ERROR(msg, ...)                                                                 \
			do {                                                                                   \
				std::string funcName = FUNCTION;                                                   \
				if (funcName.length() > 75) funcName = "<Signature too Long>";                     \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)funcName.length() + 6,                \
														(int)strlen("WARN ASSERTION FAILED"));     \
				std::string formatted = fmt::format(                                               \
				  "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                                       \
				  "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                             \
				  "ERROR",                                                                         \
				  FILENAME,                                                                        \
				  funcName,                                                                        \
				  __LINE__,                                                                        \
				  fmt::format(msg, __VA_ARGS__),                                                   \
				  maxLen + 5,                                                                      \
				  maxLen + 0,                                                                      \
				  maxLen - 4,                                                                      \
				  maxLen);                                                                         \
				if (librapid::throwOnAssert) {                                                     \
					throw std::runtime_error(formatted);                                           \
				} else {                                                                           \
					fmt::print(fmt::fg(fmt::color::red), formatted);                               \
					std::exit(1);                                                                  \
				}                                                                                  \
			} while (0)

#		define LR_WASSERT(cond, msg, ...)                                                         \
			std::string funcName = FUNCTION;                                                       \
			if (funcName.length() > 75) funcName = "<Signature too Long>";                         \
			do {                                                                                   \
				if (!(cond)) {                                                                     \
					int maxLen = librapid::internal::smallMax_internal(                            \
					  (int)std::ceil(std::log(__LINE__)) + 6,                                      \
					  (int)strlen(FILENAME) + 6,                                                   \
					  (int)funcName.length() + 6,                                                  \
					  (int)strlen(#cond) + 6,                                                      \
					  (int)strlen("WARN ASSERTION FAILED"));                                       \
					fmt::print(fmt::fg(fmt::color::yellow),                                        \
							   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                          \
							   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                           \
							   "{4:>{10}}]\n{5}\n",                                                \
							   "WARN ASSERTION FAILED",                                            \
							   FILENAME,                                                           \
							   funcName,                                                           \
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
					std::string funcName = FUNCTION;                                               \
					if (funcName.length() > 75) funcName = "<Signature too Long>";                 \
					int maxLen =                                                                   \
					  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)),    \
															(int)strlen(FILENAME),                 \
															(int)funcName.length(),                \
															(int)strlen(#cond),                    \
															(int)strlen("ASSERTION FAILED"));      \
					std::string formatted = fmt::format(                                           \
					  "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                                   \
					  "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                                    \
					  "{4:>{10}}]\n{5}\n",                                                         \
					  "ASSERTION FAILED",                                                          \
					  FILENAME,                                                                    \
					  funcName,                                                                    \
					  __LINE__,                                                                    \
					  #cond,                                                                       \
					  fmt::format(msg, __VA_ARGS__),                                               \
					  maxLen + 14,                                                                 \
					  maxLen + 9,                                                                  \
					  maxLen + 5,                                                                  \
					  maxLen + 9,                                                                  \
					  maxLen + 4);                                                                 \
					if (librapid::throwOnAssert) {                                                 \
						throw std::runtime_error(formatted);                                       \
					} else {                                                                       \
						fmt::print(fmt::fg(fmt::color::red), formatted);                           \
						std::exit(1);                                                              \
					}                                                                              \
				}                                                                                  \
			} while (0)
#	else
#		define LR_STATUS(msg, ...)                                                                \
			do {                                                                                   \
				std::string funcName = FUNCTION;                                                   \
				if (funcName.length() > 75) funcName = "<Signature too Long>";                     \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)funcName.length() + 6,                \
														(int)strlen("WARN ASSERTION FAILED"));     \
				fmt::print(fmt::fg(fmt::color::green),                                             \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "STATUS",                                                               \
						   FILENAME,                                                               \
						   funcName,                                                               \
						   __LINE__,                                                               \
						   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                            \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen);                                                                \
			} while (0)

#		define LR_WARN(msg, ...)                                                                  \
			do {                                                                                   \
				std::string funcName = FUNCTION;                                                   \
				if (funcName.length() > 75) funcName = "<Signature too Long>";                     \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,    \
														(int)strlen(FILENAME) + 6,                 \
														(int)funcName.length() + 6,                \
														(int)strlen("WARN ASSERTION FAILED"));     \
				fmt::print(fmt::fg(fmt::color::yellow),                                            \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "WARNING",                                                              \
						   FILENAME,                                                               \
						   funcName,                                                               \
						   __LINE__,                                                               \
						   fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                            \
						   maxLen + 5,                                                             \
						   maxLen + 0,                                                             \
						   maxLen - 4,                                                             \
						   maxLen);                                                                \
			} while (0)

#		define LR_ERROR(msg, ...)                                                                 \
			do {                                                                                   \
				std::string funcName = FUNCTION;                                                   \
				if (funcName.length() > 75) funcName = "<Signature too Long>";                     \
				int maxLen =                                                                       \
				  librapiod::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)) + 6,   \
														 (int)strlen(FILENAME) + 6,                \
														 (int)funcName.length() + 6,               \
														 (int)strlen("WARN ASSERTION FAILED"));    \
				fmt::print(fmt::fg(fmt::color::red),                                               \
						   "[{0:-^{5}}]\n[File {1:>{6}}]\n[Function "                              \
						   "{2:>{7}}]\n[Line {3:>{8}}]\n{4}\n",                                    \
						   "ERROR",                                                                \
						   FILENAME,                                                               \
						   funcName,                                                               \
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
					std::string funcName = FUNCTION;                                               \
					if (funcName.length() > 75) funcName = "<Signature too Long>";                 \
					\ int maxLen = librapid::internal::smallMax_internal(                          \
					  (int)std::ceil(std::log(__LINE__)) + 6,                                      \
					  (int)strlen(FILENAME) + 6,                                                   \
					  (int)funcName.length() + 6,                                                  \
					  (int)strlen(#cond) + 6,                                                      \
					  (int)strlen("WARN ASSERTION FAILED"));                                       \
					fmt::print(fmt::fg(fmt::color::yellow),                                        \
							   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                          \
							   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                           \
							   "{4:>{10}}]\n{5}\n",                                                \
							   "WARN ASSERTION FAILED",                                            \
							   FILENAME,                                                           \
							   funcName,                                                           \
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
				std::string funcName = FUNCTION;                                                   \
				if (funcName.length() > 75) funcName = "<Signature too Long>";                     \
				if (!(cond)) {                                                                     \
					int maxLen =                                                                   \
					  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)),    \
															(int)strlen(FILENAME),                 \
															(int)funcName.length(),                \
															(int)strlen(#cond),                    \
															(int)strlen("ASSERTION FAILED"));      \
					std::string formatted = fmt::format(                                           \
					  "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                                   \
					  "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                                    \
					  "{4:>{10}}]\n{5}\n",                                                         \
					  "ASSERTION FAILED",                                                          \
					  FILENAME,                                                                    \
					  funcName,                                                                    \
					  __LINE__,                                                                    \
					  #cond,                                                                       \
					  fmt::format(msg __VA_OPT__(, ) __VA_ARGS__),                                 \
					  maxLen + 14,                                                                 \
					  maxLen + 9,                                                                  \
					  maxLen + 5,                                                                  \
					  maxLen + 9,                                                                  \
					  maxLen + 4);                                                                 \
					if (librapid::throwOnAssert) {                                                 \
						throw std::runtime_error(formatted);                                       \
					} else {                                                                       \
						fmt::print(fmt::fg(fmt::color::red), formatted);                           \
						std::exit(1);                                                              \
					}                                                                              \
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
				std::string funcName = FUNCTION;                                                   \
				if (funcName.length() > 75) funcName = "<Signature too Long>";                     \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)),        \
														(int)strlen(FILENAME),                     \
														(int)funcName.length(),                    \
														(int)strlen(#cond),                        \
														(int)strlen("ASSERTION FAILED"));          \
				std::string formatted = fmt::format(                                               \
				  "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                                       \
				  "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                                        \
				  "{4:>{10}}]\n{5}\n",                                                             \
				  "ASSERTION FAILED",                                                              \
				  FILENAME,                                                                        \
				  funcName,                                                                        \
				  __LINE__,                                                                        \
				  #cond,                                                                           \
				  fmt::format(msg, __VA_ARGS__),                                                   \
				  maxLen + 14,                                                                     \
				  maxLen + 9,                                                                      \
				  maxLen + 5,                                                                      \
				  maxLen + 9,                                                                      \
				  maxLen + 4);                                                                     \
				if (librapid::throwOnAssert) {                                                     \
					throw std::runtime_error(formatted);                                           \
				} else {                                                                           \
					fmt::print(fmt::fg(fmt::color::red), formatted);                               \
					std::exit(1);                                                                  \
				}                                                                                  \
			}                                                                                      \
		} while (0)
#else
#	define LR_ASSERT_ALWAYS(cond, msg, ...)                                                       \
		do {                                                                                       \
			std::string funcName = FUNCTION;                                                       \
			if (funcName.length() > 75) funcName = "<Signature too Long>";                         \
			if (!(cond)) {                                                                         \
				int maxLen =                                                                       \
				  librapid::internal::smallMax_internal((int)std::ceil(std::log(__LINE__)),        \
														(int)strlen(FILENAME),                     \
														(int)funcName.length(),                    \
														(int)strlen(#cond),                        \
														(int)strlen("ASSERTION FAILED"));          \
				fmt::print(fmt::fg(fmt::color::red),                                               \
						   "[{0:-^{6}}]\n[File {1:>{7}}]\n[Function "                              \
						   "{2:>{8}}]\n[Line {3:>{9}}]\n[Condition "                               \
						   "{4:>{10}}]\n{5}\n",                                                    \
						   "ASSERTION FAILED",                                                     \
						   FILENAME,                                                               \
						   funcName,                                                               \
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
/*
 * Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 * * Neither the name of NVIDIA CORPORATION nor the names of its
 *   contributors may be used to endorse or promote products derived
 *   from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
  -----------
  Jitify 0.9
  -----------
  A C++ library for easy integration of CUDA runtime compilation into
  existing codes.

  --------------
  How to compile
  --------------
  Compiler dependencies: <jitify.hpp>, -std=c++11
  Linker dependencies:   dl cuda nvrtc

  --------------------------------------
  Embedding source files into executable
  --------------------------------------
  g++  ... -ldl -rdynamic -DJITIFY_ENABLE_EMBEDDED_FILES=1
  -Wl,-b,binary,my_kernel.cu,include/my_header.cuh,-b,default nvcc ... -ldl
  -Xcompiler "-rdynamic
  -Wl\,-b\,binary\,my_kernel.cu\,include/my_header.cuh\,-b\,default"
  JITIFY_INCLUDE_EMBEDDED_FILE(my_kernel_cu);
  JITIFY_INCLUDE_EMBEDDED_FILE(include_my_header_cuh);

  ----
  TODO
  ----
  Extract valid compile options and pass the rest to cuModuleLoadDataEx
  See if can have stringified headers automatically looked-up
    by having stringify add them to a (static) global map.
    The global map can be updated by creating a static class instance
      whose constructor performs the registration.
    Can then remove all headers from JitCache constructor in example code
  See other TODOs in code
*/

/*! \file jitify.hpp
 *  \brief The Jitify library header
 */

/*! \mainpage Jitify - A C++ library that simplifies the use of NVRTC
 *  \p Use class jitify::JitCache to manage and launch JIT-compiled CUDA
 *    kernels.
 *
 *  \p Use namespace jitify::reflection to reflect types and values into
 *    code-strings.
 *
 *  \p Use JITIFY_INCLUDE_EMBEDDED_FILE() to declare files that have been
 *  embedded into the executable using the GCC linker.
 *
 *  \p Use jitify::parallel_for and JITIFY_LAMBDA() to generate and launch
 *  simple kernels.
 */

#ifndef JITIFY_THREAD_SAFE
#define JITIFY_THREAD_SAFE 1
#endif

#if JITIFY_ENABLE_EMBEDDED_FILES
#include <dlfcn.h>
#endif
#include <stdint.h>
#include <algorithm>
#include <cctype>
#include <cstring>  // For strtok_r etc.
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#if JITIFY_THREAD_SAFE
#include <mutex>
#endif

#include <cuda.h>
#include <cuda_runtime_api.h>  // For dim3, cudaStream_t
#if CUDA_VERSION >= 8000
#define NVRTC_GET_TYPE_NAME 1
#endif
#include <nvrtc.h>

// For use by get_current_executable_path().
#ifdef __linux__
#include <linux/limits.h>  // For PATH_MAX

#include <cstdlib>  // For realpath
#define JITIFY_PATH_MAX PATH_MAX
#elif defined(_WIN32) || defined(_WIN64)
#include <windows.h>
#define JITIFY_PATH_MAX MAX_PATH
#else
#error "Unsupported platform"
#endif

#ifdef _MSC_VER       // MSVC compiler
#include <dbghelp.h>  // For UnDecorateSymbolName
#else
#include <cxxabi.h>  // For abi::__cxa_demangle
#endif

#if defined(_WIN32) || defined(_WIN64)
// WAR for strtok_r being called strtok_s on Windows
#pragma push_macro("strtok_r")
#undef strtok_r
#define strtok_r strtok_s
// WAR for min and max possibly being macros defined by windows.h
#pragma push_macro("min")
#pragma push_macro("max")
#undef min
#undef max
#endif

#ifndef JITIFY_PRINT_LOG
#define JITIFY_PRINT_LOG 1
#endif

#if JITIFY_PRINT_ALL
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LINKER_LOG 1
#define JITIFY_PRINT_LAUNCH 1
#define JITIFY_PRINT_HEADER_PATHS 1
#endif

#if JITIFY_ENABLE_EMBEDDED_FILES
#define JITIFY_FORCE_UNDEFINED_SYMBOL(x) void* x##_forced = (void*)&x
/*! Include a source file that has been embedded into the executable using the
 *    GCC linker.
 * \param name The name of the source file (<b>not</b> as a string), which must
 * be sanitized by replacing non-alpha-numeric characters with underscores.
 * E.g., \code{.cpp}JITIFY_INCLUDE_EMBEDDED_FILE(my_header_h)\endcode will
 * include the embedded file "my_header.h".
 * \note Files declared with this macro can be referenced using
 * their original (unsanitized) filenames when creating a \p
 * jitify::Program instance.
 */
#define JITIFY_INCLUDE_EMBEDDED_FILE(name)                                \
  extern "C" uint8_t _jitify_binary_##name##_start[] asm("_binary_" #name \
                                                         "_start");       \
  extern "C" uint8_t _jitify_binary_##name##_end[] asm("_binary_" #name   \
                                                       "_end");           \
  JITIFY_FORCE_UNDEFINED_SYMBOL(_jitify_binary_##name##_start);           \
  JITIFY_FORCE_UNDEFINED_SYMBOL(_jitify_binary_##name##_end)
#endif  // JITIFY_ENABLE_EMBEDDED_FILES

/*! Jitify library namespace
 */
namespace jitify {

/*! Source-file load callback.
 *
 *  \param filename The name of the requested source file.
 *  \param tmp_stream A temporary stream that can be used to hold source code.
 *  \return A pointer to an input stream containing the source code, or NULL
 *  to defer loading of the file to Jitify's file-loading mechanisms.
 */
typedef std::istream* (*file_callback_type)(std::string filename,
                                            std::iostream& tmp_stream);

// Exclude from Doxygen
//! \cond

class JitCache;

// Simple cache using LRU discard policy
template <typename KeyType, typename ValueType>
class ObjectCache {
 public:
  typedef KeyType key_type;
  typedef ValueType value_type;

 private:
  typedef std::map<key_type, value_type> object_map;
  typedef std::deque<key_type> key_rank;
  typedef typename key_rank::iterator rank_iterator;
  object_map _objects;
  key_rank _ranked_keys;
  size_t _capacity;

  inline void discard_old(size_t n = 0) {
    if (n > _capacity) {
      throw std::runtime_error("Insufficient capacity in cache");
    }
    while (_objects.size() > _capacity - n) {
      key_type discard_key = _ranked_keys.back();
      _ranked_keys.pop_back();
      _objects.erase(discard_key);
    }
  }

 public:
  inline ObjectCache(size_t capacity = 8) : _capacity(capacity) {}
  inline void resize(size_t capacity) {
    _capacity = capacity;
    this->discard_old();
  }
  inline bool contains(const key_type& k) const {
    return (bool)_objects.count(k);
  }
  inline void touch(const key_type& k) {
    if (!this->contains(k)) {
      throw std::runtime_error("Key not found in cache");
    }
    rank_iterator rank = std::find(_ranked_keys.begin(), _ranked_keys.end(), k);
    if (rank != _ranked_keys.begin()) {
      // Move key to front of ranks
      _ranked_keys.erase(rank);
      _ranked_keys.push_front(k);
    }
  }
  inline value_type& get(const key_type& k) {
    if (!this->contains(k)) {
      throw std::runtime_error("Key not found in cache");
    }
    this->touch(k);
    return _objects[k];
  }
  inline value_type& insert(const key_type& k,
                            const value_type& v = value_type()) {
    this->discard_old(1);
    _ranked_keys.push_front(k);
    return _objects.insert(std::make_pair(k, v)).first->second;
  }
  template <typename... Args>
  inline value_type& emplace(const key_type& k, Args&&... args) {
    this->discard_old(1);
    // Note: Use of piecewise_construct allows non-movable non-copyable types
    auto iter = _objects
                    .emplace(std::piecewise_construct, std::forward_as_tuple(k),
                             std::forward_as_tuple(args...))
                    .first;
    _ranked_keys.push_front(iter->first);
    return iter->second;
  }
};

namespace detail {

// Convenience wrapper for std::vector that provides handy constructors
template <typename T>
class vector : public std::vector<T> {
  typedef std::vector<T> super_type;

 public:
  vector() : super_type() {}
  vector(size_t n) : super_type(n) {}  // Note: Not explicit, allows =0
  vector(std::vector<T> const& vals) : super_type(vals) {}
  template <int N>
  vector(T const (&vals)[N]) : super_type(vals, vals + N) {}
  vector(std::vector<T>&& vals) : super_type(vals) {}
  vector(std::initializer_list<T> vals) : super_type(vals) {}
};

// Helper functions for parsing/manipulating source code

inline std::string replace_characters(std::string str,
                                      std::string const& oldchars,
                                      char newchar) {
  size_t i = str.find_first_of(oldchars);
  while (i != std::string::npos) {
    str[i] = newchar;
    i = str.find_first_of(oldchars, i + 1);
  }
  return str;
}
inline std::string sanitize_filename(std::string name) {
  return replace_characters(name, "/\\.-: ?%*|\"<>", '_');
}

#if JITIFY_ENABLE_EMBEDDED_FILES
class EmbeddedData {
  void* _app;
  EmbeddedData(EmbeddedData const&);
  EmbeddedData& operator=(EmbeddedData const&);

 public:
  EmbeddedData() {
    _app = dlopen(NULL, RTLD_LAZY);
    if (!_app) {
      throw std::runtime_error(std::string("dlopen failed: ") + dlerror());
    }
    dlerror();  // Clear any existing error
  }
  ~EmbeddedData() {
    if (_app) {
      dlclose(_app);
    }
  }
  const uint8_t* operator[](std::string key) const {
    key = sanitize_filename(key);
    key = "_binary_" + key;
    uint8_t const* data = (uint8_t const*)dlsym(_app, key.c_str());
    if (!data) {
      throw std::runtime_error(std::string("dlsym failed: ") + dlerror());
    }
    return data;
  }
  const uint8_t* begin(std::string key) const {
    return (*this)[key + "_start"];
  }
  const uint8_t* end(std::string key) const { return (*this)[key + "_end"]; }
};
#endif  // JITIFY_ENABLE_EMBEDDED_FILES

inline bool is_tokenchar(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
         (c >= '0' && c <= '9') || c == '_';
}
inline std::string replace_token(std::string src, std::string token,
                                 std::string replacement) {
  size_t i = src.find(token);
  while (i != std::string::npos) {
    if (i == 0 || i == src.size() - token.size() ||
        (!is_tokenchar(src[i - 1]) && !is_tokenchar(src[i + token.size()]))) {
      src.replace(i, token.size(), replacement);
      i += replacement.size();
    } else {
      i += token.size();
    }
    i = src.find(token, i);
  }
  return src;
}
inline std::string path_base(std::string p) {
  // "/usr/local/myfile.dat" -> "/usr/local"
  // "foo/bar"  -> "foo"
  // "foo/bar/" -> "foo/bar"
#if defined _WIN32 || defined _WIN64
  const char* sep = "\\/";
#else
  char sep = '/';
#endif
  size_t i = p.find_last_of(sep);
  if (i != std::string::npos) {
    return p.substr(0, i);
  } else {
    return "";
  }
}
inline std::string path_join(std::string p1, std::string p2) {
#ifdef _WIN32
  char sep = '\\';
#else
  char sep = '/';
#endif
  if (p1.size() && p2.size() && p2[0] == sep) {
    throw std::invalid_argument("Cannot join to absolute path");
  }
  if (p1.size() && p1[p1.size() - 1] != sep) {
    p1 += sep;
  }
  return p1 + p2;
}
// Elides "/." and "/.." tokens from path.
inline std::string path_simplify(const std::string& path) {
  std::vector<std::string> dirs;
  std::string cur_dir;
  bool after_slash = false;
  for (int i = 0; i < (int)path.size(); ++i) {
    if (path[i] == '/') {
      if (after_slash) continue;  // Ignore repeat slashes
      after_slash = true;
      if (cur_dir == ".." && !dirs.empty() && dirs.back() != "..") {
        if (dirs.size() == 1 && dirs.front().empty()) {
          throw std::runtime_error(
              "Invalid path: back-traversals exceed depth of absolute path");
        }
        dirs.pop_back();
      } else if (cur_dir != ".") {  // Ignore /./
        dirs.push_back(cur_dir);
      }
      cur_dir.clear();
    } else {
      after_slash = false;
      cur_dir.push_back(path[i]);
    }
  }
  if (!after_slash) {
    dirs.push_back(cur_dir);
  }
  std::stringstream ss;
  for (int i = 0; i < (int)dirs.size() - 1; ++i) {
    ss << dirs[i] << "/";
  }
  if (!dirs.empty()) ss << dirs.back();
  if (after_slash) ss << "/";
  return ss.str();
}
inline unsigned long long hash_larson64(const char* s,
                                        unsigned long long seed = 0) {
  unsigned long long hash = seed;
  while (*s) {
    hash = hash * 101 + *s++;
  }
  return hash;
}

inline uint64_t hash_combine(uint64_t a, uint64_t b) {
  // Note: The magic number comes from the golden ratio
  return a ^ (0x9E3779B97F4A7C17ull + b + (b >> 2) + (a << 6));
}

inline bool extract_include_info_from_compile_error(std::string log,
                                                    std::string& name,
                                                    std::string& parent,
                                                    int& line_num) {
  static const std::vector<std::string> pattern = {
      "could not open source file \"", "cannot open source file \""};

  for (auto& p : pattern) {
    size_t beg = log.find(p);
    if (beg != std::string::npos) {
      beg += p.size();
      size_t end = log.find("\"", beg);
      name = log.substr(beg, end - beg);

      size_t line_beg = log.rfind("\n", beg);
      if (line_beg == std::string::npos) {
        line_beg = 0;
      } else {
        line_beg += 1;
      }

      size_t split = log.find("(", line_beg);
      parent = log.substr(line_beg, split - line_beg);
      line_num =
          atoi(log.substr(split + 1, log.find(")", split + 1) - (split + 1))
                   .c_str());

      return true;
    }
  }

  return false;
}

inline bool is_include_directive_with_quotes(const std::string& source,
                                             int line_num) {
  // TODO: Check each find() for failure.
  size_t beg = 0;
  for (int i = 1; i < line_num; ++i) {
    beg = source.find("\n", beg) + 1;
  }
  beg = source.find("include", beg) + 7;
  beg = source.find_first_of("\"<", beg);
  return source[beg] == '"';
}

inline std::string comment_out_code_line(int line_num, std::string source) {
  size_t beg = 0;
  for (int i = 1; i < line_num; ++i) {
    beg = source.find("\n", beg) + 1;
  }
  return (source.substr(0, beg) + "//" + source.substr(beg));
}

inline void print_with_line_numbers(std::string const& source) {
  int linenum = 1;
  std::stringstream source_ss(source);
  std::stringstream output_ss;
  output_ss.imbue(std::locale::classic());
  for (std::string line; std::getline(source_ss, line); ++linenum) {
    output_ss << std::setfill(' ') << std::setw(3) << linenum << " " << line
              << std::endl;
  }
  std::cout << output_ss.str();
}

inline void print_compile_log(std::string program_name,
                              std::string const& log) {
  std::cout << "---------------------------------------------------"
            << std::endl;
  std::cout << "--- JIT compile log for " << program_name << " ---"
            << std::endl;
  std::cout << "---------------------------------------------------"
            << std::endl;
  std::cout << log << std::endl;
  std::cout << "---------------------------------------------------"
            << std::endl;
}

inline std::vector<std::string> split_string(std::string str,
                                             long maxsplit = -1,
                                             std::string delims = " \t") {
  std::vector<std::string> results;
  if (maxsplit == 0) {
    results.push_back(str);
    return results;
  }
  // Note: +1 to include NULL-terminator
  std::vector<char> v_str(str.c_str(), str.c_str() + (str.size() + 1));
  char* c_str = v_str.data();
  char* saveptr = c_str;
  char* token = nullptr;
  for (long i = 0; i != maxsplit; ++i) {
    token = ::strtok_r(c_str, delims.c_str(), &saveptr);
    c_str = 0;
    if (!token) {
      return results;
    }
    results.push_back(token);
  }
  // Check if there's a final piece
  token += ::strlen(token) + 1;
  if (token - v_str.data() < (ptrdiff_t)str.size()) {
    // Find the start of the final piece
    token += ::strspn(token, delims.c_str());
    if (*token) {
      results.push_back(token);
    }
  }
  return results;
}

static const std::map<std::string, std::string>& get_jitsafe_headers_map();

inline bool load_source(
    std::string filename, std::map<std::string, std::string>& sources,
    std::string current_dir = "",
    std::vector<std::string> include_paths = std::vector<std::string>(),
    file_callback_type file_callback = 0, std::string* program_name = nullptr,
    std::map<std::string, std::string>* fullpaths = nullptr,
    bool search_current_dir = true) {
  std::istream* source_stream = 0;
  std::stringstream string_stream;
  std::ifstream file_stream;
  // First detect direct source-code string ("my_program\nprogram_code...")
  size_t newline_pos = filename.find("\n");
  if (newline_pos != std::string::npos) {
    std::string source = filename.substr(newline_pos + 1);
    filename = filename.substr(0, newline_pos);
    string_stream << source;
    source_stream = &string_stream;
  }
  if (program_name) {
    *program_name = filename;
  }
  if (sources.count(filename)) {
    // Already got this one
    return true;
  }
  if (!source_stream) {
    std::string fullpath = path_join(current_dir, filename);
    // Try loading from callback
    if (!file_callback ||
        !((source_stream = file_callback(fullpath, string_stream)) != 0)) {
#if JITIFY_ENABLE_EMBEDDED_FILES
      // Try loading as embedded file
      EmbeddedData embedded;
      std::string source;
      try {
        source.assign(embedded.begin(fullpath), embedded.end(fullpath));
        string_stream << source;
        source_stream = &string_stream;
      } catch (std::runtime_error const&)
#endif  // JITIFY_ENABLE_EMBEDDED_FILES
      {
        // Try loading from filesystem
        bool found_file = false;
        if (search_current_dir) {
          file_stream.open(fullpath.c_str());
          if (file_stream) {
            source_stream = &file_stream;
            found_file = true;
          }
        }
        // Search include directories
        if (!found_file) {
          for (int i = 0; i < (int)include_paths.size(); ++i) {
            fullpath = path_join(include_paths[i], filename);
            file_stream.open(fullpath.c_str());
            if (file_stream) {
              source_stream = &file_stream;
              found_file = true;
              break;
            }
          }
          if (!found_file) {
            // Try loading from builtin headers
            fullpath = path_join("__jitify_builtin", filename);
            auto it = get_jitsafe_headers_map().find(filename);
            if (it != get_jitsafe_headers_map().end()) {
              string_stream << it->second;
              source_stream = &string_stream;
            } else {
              return false;
            }
          }
        }
      }
    }
    if (fullpaths) {
      // Record the full file path corresponding to this include name.
      (*fullpaths)[filename] = path_simplify(fullpath);
    }
  }
  sources[filename] = std::string();
  std::string& source = sources[filename];
  std::string line;
  size_t linenum = 0;
  unsigned long long hash = 0;
  bool pragma_once = false;
  bool remove_next_blank_line = false;
  while (std::getline(*source_stream, line)) {
    ++linenum;

    // HACK WAR for static variables not allowed on the device (unless
    // __shared__)
    // TODO: This breaks static member variables
    // line = replace_token(line, "static const", "/*static*/ const");

    // TODO: Need to watch out for /* */ comments too
    std::string cleanline =
        line.substr(0, line.find("//"));  // Strip line comments
    // if( cleanline.back() == "\r" ) { // Remove Windows line ending
    //	cleanline = cleanline.substr(0, cleanline.size()-1);
    //}
    // TODO: Should trim whitespace before checking .empty()
    if (cleanline.empty() && remove_next_blank_line) {
      remove_next_blank_line = false;
      continue;
    }
    // Maintain a file hash for use in #pragma once WAR
    hash = hash_larson64(line.c_str(), hash);
    if (cleanline.find("#pragma once") != std::string::npos) {
      pragma_once = true;
      // Note: This is an attempt to recover the original line numbering,
      //         which otherwise gets off-by-one due to the include guard.
      remove_next_blank_line = true;
      // line = "//" + line; // Comment out the #pragma once line
      continue;
    }

    // HACK WAR for Thrust using "#define FOO #pragma bar"
    // TODO: This is not robust to block comments, line continuations, or tabs.
    size_t pragma_beg = cleanline.find("#pragma ");
    if (pragma_beg != std::string::npos) {
      std::string line_after_pragma = line.substr(pragma_beg + 8);
      // TODO: Handle block comments (currently they cause a compilation error).
      size_t comment_start = line_after_pragma.find("//");
      std::string pragma_args = line_after_pragma.substr(0, comment_start);
      std::string comment = comment_start != std::string::npos
                                ? line_after_pragma.substr(comment_start)
                                : "";
      line = line.substr(0, pragma_beg) + "_Pragma(\"" + pragma_args + "\")" +
             comment;
    }

    source += line + "\n";
  }
  // HACK TESTING (WAR for cub)
  // source = "#define cudaDeviceSynchronize() cudaSuccess\n" + source;
  ////source = "cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }\n" +
  /// source;

  // WAR for #pragma once causing problems when there are multiple inclusions
  //   of the same header from different paths.
  if (pragma_once) {
    std::stringstream ss;
    ss.imbue(std::locale::classic());
    ss << std::uppercase << std::hex << std::setw(8) << std::setfill('0')
       << hash;
    std::string include_guard_name = "_JITIFY_INCLUDE_GUARD_" + ss.str() + "\n";
    std::string include_guard_header;
    include_guard_header += "#ifndef " + include_guard_name;
    include_guard_header += "#define " + include_guard_name;
    std::string include_guard_footer;
    include_guard_footer += "#endif // " + include_guard_name;
    source = include_guard_header + source + "\n" + include_guard_footer;
  }
  // return filename;
  return true;
}

}  // namespace detail

//! \endcond

/*! Jitify reflection utilities namespace
 */
namespace reflection {

//  Provides type and value reflection via a function 'reflect':
//    reflect<Type>()   -> "Type"
//    reflect(value)    -> "(T)value"
//    reflect<VAL>()    -> "VAL"
//    reflect<Type,VAL> -> "VAL"
//    reflect_template<float,NonType<int,7>,char>() -> "<float,7,char>"
//    reflect_template({"float", "7", "char"}) -> "<float,7,char>"

/*! A wrapper class for non-type template parameters.
 */
template <typename T, T VALUE_>
struct NonType {
  constexpr static T VALUE = VALUE_;
};

// Forward declaration
template <typename T>
inline std::string reflect(T const& value);

//! \cond

namespace detail {

template <typename T>
inline std::string value_string(const T& x) {
  std::stringstream ss;
  ss << x;
  return ss.str();
}
// WAR for non-printable characters
template <>
inline std::string value_string<char>(const char& x) {
  std::stringstream ss;
  ss << (int)x;
  return ss.str();
}
template <>
inline std::string value_string<signed char>(const signed char& x) {
  std::stringstream ss;
  ss << (int)x;
  return ss.str();
}
template <>
inline std::string value_string<unsigned char>(const unsigned char& x) {
  std::stringstream ss;
  ss << (int)x;
  return ss.str();
}
template <>
inline std::string value_string<wchar_t>(const wchar_t& x) {
  std::stringstream ss;
  ss << (long)x;
  return ss.str();
}
// Specialisation for bool true/false literals
template <>
inline std::string value_string<bool>(const bool& x) {
  return x ? "true" : "false";
}

// Removes all tokens that start with double underscores.
inline void strip_double_underscore_tokens(char* s) {
  using jitify::detail::is_tokenchar;
  char* w = s;
  do {
    if (*s == '_' && *(s + 1) == '_') {
      while (is_tokenchar(*++s))
        ;
    }
  } while ((*w++ = *s++));
}

//#if CUDA_VERSION < 8000
#ifdef _MSC_VER  // MSVC compiler
inline std::string demangle_cuda_symbol(const char* mangled_name) {
  // We don't have a way to demangle CUDA symbol names under MSVC.
  return mangled_name;
}
inline std::string demangle_native_type(const std::type_info& typeinfo) {
  // Get the decorated name and skip over the leading '.'.
  const char* decorated_name = typeinfo.raw_name() + 1;
  char undecorated_name[4096];
  if (UnDecorateSymbolName(
          decorated_name, undecorated_name,
          sizeof(undecorated_name) / sizeof(*undecorated_name),
          UNDNAME_NO_ARGUMENTS |          // Treat input as a type name
              UNDNAME_NAME_ONLY           // No "class" and "struct" prefixes
          /*UNDNAME_NO_MS_KEYWORDS*/)) {  // No "__cdecl", "__ptr64" etc.
    // WAR for UNDNAME_NO_MS_KEYWORDS messing up function types.
    strip_double_underscore_tokens(undecorated_name);
    return undecorated_name;
  }
  throw std::runtime_error("UnDecorateSymbolName failed");
}
#else  // not MSVC
inline std::string demangle_cuda_symbol(const char* mangled_name) {
  size_t bufsize = 0;
  char* buf = nullptr;
  std::string demangled_name;
  int status;
  auto demangled_ptr = std::unique_ptr<char, decltype(free)*>(
      abi::__cxa_demangle(mangled_name, buf, &bufsize, &status), free);
  if (status == 0) {
    demangled_name = demangled_ptr.get();  // all worked as expected
  } else if (status == -2) {
    demangled_name = mangled_name;  // we interpret this as plain C name
  } else if (status == -1) {
    throw std::runtime_error(
        std::string("memory allocation failure in __cxa_demangle"));
  } else if (status == -3) {
    throw std::runtime_error(std::string("invalid argument to __cxa_demangle"));
  }
  return demangled_name;
}
inline std::string demangle_native_type(const std::type_info& typeinfo) {
  return demangle_cuda_symbol(typeinfo.name());
}
#endif  // not MSVC
//#endif // CUDA_VERSION < 8000

template <typename>
class JitifyTypeNameWrapper_ {};

template <typename T>
struct type_reflection {
  inline static std::string name() {
    //#if CUDA_VERSION < 8000
    // TODO: Use nvrtcGetTypeName once it has the same behavior as this.
    // WAR for typeid discarding cv qualifiers on value-types
    // Wrap type in dummy template class to preserve cv-qualifiers, then strip
    // off the wrapper from the resulting string.

    // This fixes a bug where these two values will return empty strings
    if constexpr(std::is_same_v<T, int64_t>) return "signed long long";
    if constexpr(std::is_same_v<T, uint64_t>) return "unsigned long long";
	  
    std::string wrapped_name =
        demangle_native_type(typeid(JitifyTypeNameWrapper_<T>));
    // Note: The reflected name of this class also has namespace prefixes.
    const std::string wrapper_class_name = "JitifyTypeNameWrapper_<";
    size_t start = wrapped_name.find(wrapper_class_name);
    if (start == std::string::npos) {
      throw std::runtime_error("Type reflection failed: " + wrapped_name);
    }
    start += wrapper_class_name.size();
    std::string name =
        wrapped_name.substr(start, wrapped_name.size() - (start + 1));
    return name;
    //#else
    //         std::string ret;
    //         nvrtcResult status = nvrtcGetTypeName<T>(&ret);
    //         if( status != NVRTC_SUCCESS ) {
    //                 throw std::runtime_error(std::string("nvrtcGetTypeName
    // failed:
    //")+ nvrtcGetErrorString(status));
    //         }
    //         return ret;
    //#endif
  }
};  // namespace detail
template <typename T, T VALUE>
struct type_reflection<NonType<T, VALUE> > {
  inline static std::string name() {
    return jitify::reflection::reflect(VALUE);
  }
};

}  // namespace detail

//! \endcond

/*! Create an Instance object that contains a const reference to the
 *  value.  We use this to wrap abstract objects from which we want to extract
 *  their type at runtime (e.g., derived type).  This is used to facilitate
 *  templating on derived type when all we know at compile time is abstract
 * type.
 */
template <typename T>
struct Instance {
  const T& value;
  Instance(const T& value_arg) : value(value_arg) {}
};

/*! Create an Instance object from which we can extract the value's run-time
 * type.
 *  \param value The const value to be captured.
 */
template <typename T>
inline Instance<T const> instance_of(T const& value) {
  return Instance<T const>(value);
}

/*! A wrapper used for representing types as values.
 */
template <typename T>
struct Type {};

// Type reflection
// E.g., reflect<float>() -> "float"
// Note: This strips trailing const and volatile qualifiers
/*! Generate a code-string for a type.
 *  \code{.cpp}reflect<float>() --> "float"\endcode
 */
template <typename T>
inline std::string reflect() {
  return detail::type_reflection<T>::name();
}
// Value reflection
// E.g., reflect(3.14f) -> "(float)3.14"
/*! Generate a code-string for a value.
 *  \code{.cpp}reflect(3.14f) --> "(float)3.14"\endcode
 */
template <typename T>
inline std::string reflect(T const& value) {
  return "(" + reflect<T>() + ")" + detail::value_string(value);
}
// Non-type template arg reflection (implicit conversion to int64_t)
// E.g., reflect<7>() -> "(int64_t)7"
/*! Generate a code-string for an integer non-type template argument.
 *  \code{.cpp}reflect<7>() --> "(int64_t)7"\endcode
 */
template <int64_t N>
inline std::string reflect() {
  return reflect<NonType<int64_t, N> >();
}
// Non-type template arg reflection (explicit type)
// E.g., reflect<int,7>() -> "(int)7"
/*! Generate a code-string for a generic non-type template argument.
 *  \code{.cpp} reflect<int,7>() --> "(int)7" \endcode
 */
template <typename T, T N>
inline std::string reflect() {
  return reflect<NonType<T, N> >();
}
// Type reflection via value
// E.g., reflect(Type<float>()) -> "float"
/*! Generate a code-string for a type wrapped as a Type instance.
 *  \code{.cpp}reflect(Type<float>()) --> "float"\endcode
 */
template <typename T>
inline std::string reflect(jitify::reflection::Type<T>) {
  return reflect<T>();
}

/*! Generate a code-string for a type wrapped as an Instance instance.
 *  \code{.cpp}reflect(Instance<float>(3.1f)) --> "float"\endcode
 *  or more simply when passed to a instance_of helper
 *  \code{.cpp}reflect(instance_of(3.1f)) --> "float"\endcodei
 *  This is specifically for the case where we want to extract the run-time
 * type, e.g., derived type, of an object pointer.
 */
template <typename T>
inline std::string reflect(jitify::reflection::Instance<T>& value) {
  return detail::demangle_native_type(typeid(value.value));
}

// Type from value
// E.g., type_of(3.14f) -> Type<float>()
/*! Create a Type object representing a value's type.
 *  \param value The value whose type is to be captured.
 */
template <typename T>
inline Type<T> type_of(T&) {
  return Type<T>();
}
/*! Create a Type object representing a value's type.
 *  \param value The const value whose type is to be captured.
 */
template <typename T>
inline Type<T const> type_of(T const&) {
  return Type<T const>();
}

// Multiple value reflections one call, returning list of strings
template <typename... Args>
inline std::vector<std::string> reflect_all(Args... args) {
  return {reflect(args)...};
}

inline std::string reflect_list(jitify::detail::vector<std::string> const& args,
                                std::string opener = "",
                                std::string closer = "") {
  std::stringstream ss;
  ss << opener;
  for (int i = 0; i < (int)args.size(); ++i) {
    if (i > 0) ss << ",";
    ss << args[i];
  }
  ss << closer;
  return ss.str();
}

// Template instantiation reflection
// inline std::string reflect_template(std::vector<std::string> const& args) {
inline std::string reflect_template(
    jitify::detail::vector<std::string> const& args) {
  // Note: The space in " >" is a WAR to avoid '>>' appearing
  return reflect_list(args, "<", " >");
}
// TODO: See if can make this evaluate completely at compile-time
template <typename... Ts>
inline std::string reflect_template() {
  return reflect_template({reflect<Ts>()...});
  // return reflect_template<sizeof...(Ts)>({reflect<Ts>()...});
}

}  // namespace reflection

//! \cond

namespace detail {

// Demangles nested variable names using the PTX name mangling scheme
// (which follows the Itanium64 ABI). E.g., _ZN1a3Foo2bcE -> a::Foo::bc.
inline std::string demangle_ptx_variable_name(const char* name) {
  std::stringstream ss;
  const char* c = name;
  if (*c++ != '_' || *c++ != 'Z') return name;  // Non-mangled name
  if (*c++ != 'N') return "";  // Not a nested name, unsupported
  while (true) {
    // Parse identifier length.
    int n = 0;
    while (std::isdigit(*c)) {
      n = n * 10 + (*c - '0');
      c++;
    }
    if (!n) return "";  // Invalid or unsupported mangled name
    // Parse identifier.
    const char* c0 = c;
    while (n-- && *c) c++;
    if (!*c) return "";  // Mangled name is truncated
    std::string id(c0, c);
    // Identifiers starting with "_GLOBAL" are anonymous namespaces.
    ss << (id.substr(0, 7) == "_GLOBAL" ? "(anonymous namespace)" : id);
    // Nested name specifiers end with 'E'.
    if (*c == 'E') break;
    // There are more identifiers to come, add join token.
    ss << "::";
  }
  return ss.str();
}

static const char* get_current_executable_path() {
  static const char* path = []() -> const char* {
    static char buffer[JITIFY_PATH_MAX] = {};
#ifdef __linux__
    if (!::realpath("/proc/self/exe", buffer)) return nullptr;
#elif defined(_WIN32) || defined(_WIN64)
    if (!GetModuleFileNameA(nullptr, buffer, JITIFY_PATH_MAX)) return nullptr;
#endif
    return buffer;
  }();
  return path;
}

inline bool endswith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.substr(str.size() - suffix.size()) == suffix;
}

// Infers the JIT input type from the filename suffix. If no known suffix is
// present, the filename is assumed to refer to a library, and the associated
// suffix (and possibly prefix) is automatically added to the filename.
inline CUjitInputType get_cuda_jit_input_type(std::string* filename) {
  if (endswith(*filename, ".ptx")) {
    return CU_JIT_INPUT_PTX;
  } else if (endswith(*filename, ".cubin")) {
    return CU_JIT_INPUT_CUBIN;
  } else if (endswith(*filename, ".fatbin")) {
    return CU_JIT_INPUT_FATBINARY;
  } else if (endswith(*filename,
#if defined _WIN32 || defined _WIN64
                      ".obj"
#else  // Linux
                      ".o"
#endif
                      )) {
    return CU_JIT_INPUT_OBJECT;
  } else {  // Assume library
#if defined _WIN32 || defined _WIN64
    if (!endswith(*filename, ".lib")) {
      *filename += ".lib";
    }
#else  // Linux
    if (!endswith(*filename, ".a")) {
      *filename = "lib" + *filename + ".a";
    }
#endif
    return CU_JIT_INPUT_LIBRARY;
  }
}

class CUDAKernel {
  std::vector<std::string> _link_files;
  std::vector<std::string> _link_paths;
  CUlinkState _link_state;
  CUmodule _module;
  CUfunction _kernel;
  std::string _func_name;
  std::string _ptx;
  std::map<std::string, std::string> _global_map;
  std::vector<CUjit_option> _opts;
  std::vector<void*> _optvals;
#ifdef JITIFY_PRINT_LINKER_LOG
  static const unsigned int _log_size = 8192;
  char _error_log[_log_size];
  char _info_log[_log_size];
#endif

  inline void cuda_safe_call(CUresult res) const {
    if (res != CUDA_SUCCESS) {
      const char* msg;
      cuGetErrorName(res, &msg);
      throw std::runtime_error(msg);
    }
  }
  inline void create_module(std::vector<std::string> link_files,
                            std::vector<std::string> link_paths) {
    CUresult result;
#ifndef JITIFY_PRINT_LINKER_LOG
    // WAR since linker log does not seem to be constructed using a single call
    // to cuModuleLoadDataEx.
    if (link_files.empty()) {
      result =
          cuModuleLoadDataEx(&_module, _ptx.c_str(), (unsigned)_opts.size(),
                             _opts.data(), _optvals.data());
    } else
#endif
    {
      cuda_safe_call(cuLinkCreate((unsigned)_opts.size(), _opts.data(),
                                  _optvals.data(), &_link_state));
      cuda_safe_call(cuLinkAddData(_link_state, CU_JIT_INPUT_PTX,
                                   (void*)_ptx.c_str(), _ptx.size(),
                                   "jitified_source.ptx", 0, 0, 0));
      for (int i = 0; i < (int)link_files.size(); ++i) {
        std::string link_file = link_files[i];
        CUjitInputType jit_input_type;
        if (link_file == ".") {
          // Special case for linking to current executable.
          link_file = get_current_executable_path();
          jit_input_type = CU_JIT_INPUT_OBJECT;
        } else {
          // Infer based on filename.
          jit_input_type = get_cuda_jit_input_type(&link_file);
        }
        result = cuLinkAddFile(_link_state, jit_input_type, link_file.c_str(),
                               0, 0, 0);
        int path_num = 0;
        while (result == CUDA_ERROR_FILE_NOT_FOUND &&
               path_num < (int)link_paths.size()) {
          std::string filename = path_join(link_paths[path_num++], link_file);
          result = cuLinkAddFile(_link_state, jit_input_type, filename.c_str(),
                                 0, 0, 0);
        }
#if JITIFY_PRINT_LINKER_LOG
        if (result == CUDA_ERROR_FILE_NOT_FOUND) {
          std::cerr << "Linker error: Device library not found: " << link_file
                    << std::endl;
        } else if (result != CUDA_SUCCESS) {
          std::cerr << "Linker error: Failed to add file: " << link_file
                    << std::endl;
          std::cerr << _error_log << std::endl;
        }
#endif
        cuda_safe_call(result);
      }
      size_t cubin_size;
      void* cubin;
      result = cuLinkComplete(_link_state, &cubin, &cubin_size);
      if (result == CUDA_SUCCESS) {
        result = cuModuleLoadData(&_module, cubin);
      }
    }
#ifdef JITIFY_PRINT_LINKER_LOG
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "--- Linker for "
              << reflection::detail::demangle_cuda_symbol(_func_name.c_str())
              << " ---" << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    std::cout << _info_log << std::endl;
    std::cout << std::endl;
    std::cout << _error_log << std::endl;
    std::cout << "---------------------------------------" << std::endl;
#endif
    cuda_safe_call(result);
    // Allow _func_name to be empty to support cases where we want to generate
    // PTX containing extern symbol definitions but no kernels.
    if (!_func_name.empty()) {
      cuda_safe_call(
          cuModuleGetFunction(&_kernel, _module, _func_name.c_str()));
    }
  }
  inline void destroy_module() {
    if (_link_state) {
      cuda_safe_call(cuLinkDestroy(_link_state));
    }
    _link_state = 0;
    if (_module) {
      cuModuleUnload(_module);
    }
    _module = 0;
  }

  // create a map of __constant__ and __device__ variables in the ptx file
  // mapping demangled to mangled name
  inline void create_global_variable_map() {
    size_t pos = 0;
    while (pos < _ptx.size()) {
      pos = std::min(_ptx.find(".const .align", pos),
                     _ptx.find(".global .align", pos));
      if (pos == std::string::npos) break;
      size_t end = _ptx.find_first_of(";=", pos);
      if (_ptx[end] == '=') --end;
      std::string line = _ptx.substr(pos, end - pos);
      pos = end;
      size_t symbol_start = line.find_last_of(" ") + 1;
      size_t symbol_end = line.find_last_of("[");
      std::string entry = line.substr(symbol_start, symbol_end - symbol_start);
      std::string key = detail::demangle_ptx_variable_name(entry.c_str());
      // Skip unsupported mangled names. E.g., a static variable defined inside
      // a function (such variables are not directly addressable from outside
      // the function, so skipping them is the correct behavior).
      if (key == "") continue;
      _global_map[key] = entry;
    }
  }

  inline void set_linker_log() {
#ifdef JITIFY_PRINT_LINKER_LOG
    _opts.push_back(CU_JIT_INFO_LOG_BUFFER);
    _optvals.push_back((void*)_info_log);
    _opts.push_back(CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES);
    _optvals.push_back((void*)(long)_log_size);
    _opts.push_back(CU_JIT_ERROR_LOG_BUFFER);
    _optvals.push_back((void*)_error_log);
    _opts.push_back(CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES);
    _optvals.push_back((void*)(long)_log_size);
    _opts.push_back(CU_JIT_LOG_VERBOSE);
    _optvals.push_back((void*)1);
#endif
  }

 public:
  inline CUDAKernel() : _link_state(0), _module(0), _kernel(0) {}
  inline CUDAKernel(const CUDAKernel& other) = delete;
  inline CUDAKernel& operator=(const CUDAKernel& other) = delete;
  inline CUDAKernel(CUDAKernel&& other) = delete;
  inline CUDAKernel& operator=(CUDAKernel&& other) = delete;
  inline CUDAKernel(const char* func_name, const char* ptx,
                    std::vector<std::string> link_files,
                    std::vector<std::string> link_paths, unsigned int nopts = 0,
                    CUjit_option* opts = 0, void** optvals = 0)
      : _link_files(link_files),
        _link_paths(link_paths),
        _link_state(0),
        _module(0),
        _kernel(0),
        _func_name(func_name),
        _ptx(ptx),
        _opts(opts, opts + nopts),
        _optvals(optvals, optvals + nopts) {
    this->set_linker_log();
    this->create_module(link_files, link_paths);
    this->create_global_variable_map();
  }

  inline CUDAKernel& set(const char* func_name, const char* ptx,
                         std::vector<std::string> link_files,
                         std::vector<std::string> link_paths,
                         unsigned int nopts = 0, CUjit_option* opts = 0,
                         void** optvals = 0) {
    this->destroy_module();
    _func_name = func_name;
    _ptx = ptx;
    _link_files = link_files;
    _link_paths = link_paths;
    _opts.assign(opts, opts + nopts);
    _optvals.assign(optvals, optvals + nopts);
    this->set_linker_log();
    this->create_module(link_files, link_paths);
    this->create_global_variable_map();
    return *this;
  }
  inline ~CUDAKernel() { this->destroy_module(); }
  inline operator CUfunction() const { return _kernel; }

  inline CUresult launch(dim3 grid, dim3 block, unsigned int smem,
                         CUstream stream, std::vector<void*> arg_ptrs) const {
    return cuLaunchKernel(_kernel, grid.x, grid.y, grid.z, block.x, block.y,
                          block.z, smem, stream, arg_ptrs.data(), NULL);
  }

  inline void safe_launch(dim3 grid, dim3 block, unsigned int smem,
                          CUstream stream, std::vector<void*> arg_ptrs) const {
    return cuda_safe_call(cuLaunchKernel(_kernel, grid.x, grid.y, grid.z,
                                         block.x, block.y, block.z, smem,
                                         stream, arg_ptrs.data(), NULL));
  }

  inline int get_func_attribute(CUfunction_attribute attribute) const {
    int value;
    cuda_safe_call(cuFuncGetAttribute(&value, attribute, _kernel));
    return value;
  }

  inline void set_func_attribute(CUfunction_attribute attribute,
                                 int value) const {
    cuda_safe_call(cuFuncSetAttribute(_kernel, attribute, value));
  }

  inline CUdeviceptr get_global_ptr(const char* name,
                                    size_t* size = nullptr) const {
    CUdeviceptr global_ptr = 0;
    auto global = _global_map.find(name);
    if (global != _global_map.end()) {
      cuda_safe_call(cuModuleGetGlobal(&global_ptr, size, _module,
                                       global->second.c_str()));
    } else {
      throw std::runtime_error(std::string("failed to look up global ") + name);
    }
    return global_ptr;
  }

  template <typename T>
  inline CUresult get_global_data(const char* name, T* data, size_t count,
                                  CUstream stream = 0) const {
    size_t size_bytes;
    CUdeviceptr ptr = get_global_ptr(name, &size_bytes);
    size_t given_size_bytes = count * sizeof(T);
    if (given_size_bytes != size_bytes) {
      throw std::runtime_error(
          std::string("Value for global variable ") + name +
          " has wrong size: got " + std::to_string(given_size_bytes) +
          " bytes, expected " + std::to_string(size_bytes));
    }
    return cuMemcpyDtoHAsync(data, ptr, size_bytes, stream);
  }

  template <typename T>
  inline CUresult set_global_data(const char* name, const T* data, size_t count,
                                  CUstream stream = 0) const {
    size_t size_bytes;
    CUdeviceptr ptr = get_global_ptr(name, &size_bytes);
    size_t given_size_bytes = count * sizeof(T);
    if (given_size_bytes != size_bytes) {
      throw std::runtime_error(
          std::string("Value for global variable ") + name +
          " has wrong size: got " + std::to_string(given_size_bytes) +
          " bytes, expected " + std::to_string(size_bytes));
    }
    return cuMemcpyHtoDAsync(ptr, data, size_bytes, stream);
  }

  const std::string& function_name() const { return _func_name; }
  const std::string& ptx() const { return _ptx; }
  const std::vector<std::string>& link_files() const { return _link_files; }
  const std::vector<std::string>& link_paths() const { return _link_paths; }
};

static const char* jitsafe_header_preinclude_h = R"(
//// WAR for Thrust (which appears to have forgotten to include this in result_of_adaptable_function.h
//#include <type_traits>

//// WAR for Thrust (which appear to have forgotten to include this in error_code.h)
//#include <string>

// WAR for generics/shfl.h
#define THRUST_STATIC_ASSERT(x)

// WAR for CUB
#ifdef __host__
#undef __host__
#endif
#define __host__

// WAR to allow exceptions to be parsed
#define try
#define catch(...)
)"
#if defined(_WIN32) || defined(_WIN64)
// WAR for NVRTC <= 11.0 not defining _WIN64.
R"(
#ifndef _WIN64
#define _WIN64 1
#endif
)"
#endif
;

static const char* jitsafe_header_float_h = R"(
#pragma once

#define FLT_RADIX       2
#define FLT_MANT_DIG    24
#define DBL_MANT_DIG    53
#define FLT_DIG         6
#define DBL_DIG         15
#define FLT_MIN_EXP     -125
#define DBL_MIN_EXP     -1021
#define FLT_MIN_10_EXP  -37
#define DBL_MIN_10_EXP  -307
#define FLT_MAX_EXP     128
#define DBL_MAX_EXP     1024
#define FLT_MAX_10_EXP  38
#define DBL_MAX_10_EXP  308
#define FLT_MAX         3.4028234e38f
#define DBL_MAX         1.7976931348623157e308
#define FLT_EPSILON     1.19209289e-7f
#define DBL_EPSILON     2.220440492503130e-16
#define FLT_MIN         1.1754943e-38f
#define DBL_MIN         2.2250738585072013e-308
#define FLT_ROUNDS      1
#if defined __cplusplus && __cplusplus >= 201103L
#define FLT_EVAL_METHOD 0
#define DECIMAL_DIG     21
#endif
)";

static const char* jitsafe_header_limits_h = R"(
#pragma once

#if defined _WIN32 || defined _WIN64
 #define __WORDSIZE 32
#else
 #if defined __x86_64__ && !defined __ILP32__
  #define __WORDSIZE 64
 #else
  #define __WORDSIZE 32
 #endif
#endif
#define MB_LEN_MAX  16
#define CHAR_BIT    8
#define SCHAR_MIN   (-128)
#define SCHAR_MAX   127
#define UCHAR_MAX   255
enum {
  _JITIFY_CHAR_IS_UNSIGNED = (char)-1 >= 0,
  CHAR_MIN = _JITIFY_CHAR_IS_UNSIGNED ? 0 : SCHAR_MIN,
  CHAR_MAX = _JITIFY_CHAR_IS_UNSIGNED ? UCHAR_MAX : SCHAR_MAX,
};
#define SHRT_MIN    (-32768)
#define SHRT_MAX    32767
#define USHRT_MAX   65535
#define INT_MIN     (-INT_MAX - 1)
#define INT_MAX     2147483647
#define UINT_MAX    4294967295U
#if __WORDSIZE == 64
 # define LONG_MAX  9223372036854775807L
#else
 # define LONG_MAX  2147483647L
#endif
#define LONG_MIN    (-LONG_MAX - 1L)
#if __WORDSIZE == 64
 #define ULONG_MAX  18446744073709551615UL
#else
 #define ULONG_MAX  4294967295UL
#endif
#define LLONG_MAX  9223372036854775807LL
#define LLONG_MIN  (-LLONG_MAX - 1LL)
#define ULLONG_MAX 18446744073709551615ULL
)";

static const char* jitsafe_header_iterator = R"(
#pragma once

namespace std {
struct output_iterator_tag {};
struct input_iterator_tag {};
struct forward_iterator_tag {};
struct bidirectional_iterator_tag {};
struct random_access_iterator_tag {};
template<class Iterator>
struct iterator_traits {
  typedef typename Iterator::iterator_category iterator_category;
  typedef typename Iterator::value_type        value_type;
  typedef typename Iterator::difference_type   difference_type;
  typedef typename Iterator::pointer           pointer;
  typedef typename Iterator::reference         reference;
};
template<class T>
struct iterator_traits<T*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T*                         pointer;
  typedef T&                         reference;
};
template<class T>
struct iterator_traits<T const*> {
  typedef random_access_iterator_tag iterator_category;
  typedef T                          value_type;
  typedef ptrdiff_t                  difference_type;
  typedef T const*                   pointer;
  typedef T const&                   reference;
};
}  // namespace std
)";

// TODO: This is incomplete; need floating point limits
//   Joe Eaton: added IEEE float and double types, none of the smaller types
//              using type specific structs since we can't template on floats.
static const char* jitsafe_header_limits = R"(
#pragma once
#include <cfloat>
#include <climits>
#include <cstdint>
// TODO: epsilon(), infinity(), etc
namespace std {
namespace __jitify_detail {
#if __cplusplus >= 201103L
#define JITIFY_CXX11_CONSTEXPR constexpr
#define JITIFY_CXX11_NOEXCEPT noexcept
#else
#define JITIFY_CXX11_CONSTEXPR
#define JITIFY_CXX11_NOEXCEPT
#endif

struct FloatLimits {
#if __cplusplus >= 201103L
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          float lowest() JITIFY_CXX11_NOEXCEPT {   return -FLT_MAX;}
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          float min() JITIFY_CXX11_NOEXCEPT {      return FLT_MIN; }
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          float max() JITIFY_CXX11_NOEXCEPT {      return FLT_MAX; }
#endif  // __cplusplus >= 201103L
   enum {
   is_specialized    = true,
   is_signed         = true,
   is_integer        = false,
   is_exact          = false,
   has_infinity      = true,
   has_quiet_NaN     = true,
   has_signaling_NaN = true,
   has_denorm        = 1,
   has_denorm_loss   = true,
   round_style       = 1,
   is_iec559         = true,
   is_bounded        = true,
   is_modulo         = false,
   digits            = 24,
   digits10          = 6,
   max_digits10      = 9,
   radix             = 2,
   min_exponent      = -125,
   min_exponent10    = -37,
   max_exponent      = 128,
   max_exponent10    = 38,
   tinyness_before   = false,
   traps             = false
   };
};
struct DoubleLimits {
#if __cplusplus >= 201103L
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          double lowest() noexcept { return -DBL_MAX; }
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          double min() noexcept { return DBL_MIN; }
   static JITIFY_CXX11_CONSTEXPR inline __host__ __device__ 
          double max() noexcept { return DBL_MAX; }
#endif  // __cplusplus >= 201103L
   enum {
   is_specialized    = true,
   is_signed         = true,
   is_integer        = false,
   is_exact          = false,
   has_infinity      = true,
   has_quiet_NaN     = true,
   has_signaling_NaN = true,
   has_denorm        = 1,
   has_denorm_loss   = true,
   round_style       = 1,
   is_iec559         = true,
   is_bounded        = true,
   is_modulo         = false,
   digits            = 53,
   digits10          = 15,
   max_digits10      = 17,
   radix             = 2,
   min_exponent      = -1021,
   min_exponent10    = -307,
   max_exponent      = 1024,
   max_exponent10    = 308,
   tinyness_before   = false,
   traps             = false
   };
};
template<class T, T Min, T Max, int Digits=-1>
struct IntegerLimits {
	static inline __host__ __device__ T min() { return Min; }
	static inline __host__ __device__ T max() { return Max; }
#if __cplusplus >= 201103L
	static constexpr inline __host__ __device__ T lowest() noexcept {
		return Min;
	}
#endif  // __cplusplus >= 201103L
	enum {
       is_specialized = true,
       digits            = (Digits == -1) ? (int)(sizeof(T)*8 - (Min != 0)) : Digits,
       digits10          = (digits * 30103) / 100000,
       is_signed         = ((T)(-1)<0),
       is_integer        = true,
       is_exact          = true,
       has_infinity      = false,
       has_quiet_NaN     = false,
       has_signaling_NaN = false,
       has_denorm        = 0,
       has_denorm_loss   = false,
       round_style       = 0,
       is_iec559         = false,
       is_bounded        = true,
       is_modulo         = !(is_signed || Max == 1 /*is bool*/),
       max_digits10      = 0,
       radix             = 2,
       min_exponent      = 0,
       min_exponent10    = 0,
       max_exponent      = 0,
       max_exponent10    = 0,
       tinyness_before   = false,
       traps             = false
	};
};
} // namespace __jitify_detail
template<typename T> struct numeric_limits {
    enum { is_specialized = false };
};
template<> struct numeric_limits<bool>               : public 
__jitify_detail::IntegerLimits<bool,              false,    true,1> {};
template<> struct numeric_limits<char>               : public 
__jitify_detail::IntegerLimits<char,              CHAR_MIN, CHAR_MAX> 
{};
template<> struct numeric_limits<signed char>        : public 
__jitify_detail::IntegerLimits<signed char,       SCHAR_MIN,SCHAR_MAX> 
{};
template<> struct numeric_limits<unsigned char>      : public 
__jitify_detail::IntegerLimits<unsigned char,     0,        UCHAR_MAX> 
{};
template<> struct numeric_limits<wchar_t>            : public 
__jitify_detail::IntegerLimits<wchar_t,           WCHAR_MIN, WCHAR_MAX> {};
template<> struct numeric_limits<short>              : public 
__jitify_detail::IntegerLimits<short,             SHRT_MIN, SHRT_MAX> 
{};
template<> struct numeric_limits<unsigned short>     : public 
__jitify_detail::IntegerLimits<unsigned short,    0,        USHRT_MAX> 
{};
template<> struct numeric_limits<int>                : public 
__jitify_detail::IntegerLimits<int,               INT_MIN,  INT_MAX> {};
template<> struct numeric_limits<unsigned int>       : public 
__jitify_detail::IntegerLimits<unsigned int,      0,        UINT_MAX> 
{};
template<> struct numeric_limits<long>               : public 
__jitify_detail::IntegerLimits<long,              LONG_MIN, LONG_MAX> 
{};
template<> struct numeric_limits<unsigned long>      : public 
__jitify_detail::IntegerLimits<unsigned long,     0,        ULONG_MAX> 
{};
template<> struct numeric_limits<long long>          : public 
__jitify_detail::IntegerLimits<long long,         LLONG_MIN,LLONG_MAX> 
{};
template<> struct numeric_limits<unsigned long long> : public 
__jitify_detail::IntegerLimits<unsigned long long,0,        ULLONG_MAX> 
{};
//template<typename T> struct numeric_limits { static const bool 
//is_signed = ((T)(-1)<0); };
template<> struct numeric_limits<float>              : public 
__jitify_detail::FloatLimits 
{};
template<> struct numeric_limits<double>             : public 
__jitify_detail::DoubleLimits 
{};
}  // namespace std
)";

// TODO: This is highly incomplete
static const char* jitsafe_header_type_traits = R"(
    #pragma once
    #if __cplusplus >= 201103L
    namespace std {

    template<bool B, class T = void> struct enable_if {};
    template<class T>                struct enable_if<true, T> { typedef T type; };
    #if __cplusplus >= 201402L
    template< bool B, class T = void > using enable_if_t = typename enable_if<B,T>::type;
    #endif

    struct true_type  {
      enum { value = true };
      operator bool() const { return true; }
    };
    struct false_type {
      enum { value = false };
      operator bool() const { return false; }
    };

    template<typename T> struct is_floating_point    : false_type {};
    template<> struct is_floating_point<float>       :  true_type {};
    template<> struct is_floating_point<double>      :  true_type {};
    template<> struct is_floating_point<long double> :  true_type {};

    template<class T> struct is_integral              : false_type {};
    template<> struct is_integral<bool>               :  true_type {};
    template<> struct is_integral<char>               :  true_type {};
    template<> struct is_integral<signed char>        :  true_type {};
    template<> struct is_integral<unsigned char>      :  true_type {};
    template<> struct is_integral<short>              :  true_type {};
    template<> struct is_integral<unsigned short>     :  true_type {};
    template<> struct is_integral<int>                :  true_type {};
    template<> struct is_integral<unsigned int>       :  true_type {};
    template<> struct is_integral<long>               :  true_type {};
    template<> struct is_integral<unsigned long>      :  true_type {};
    template<> struct is_integral<long long>          :  true_type {};
    template<> struct is_integral<unsigned long long> :  true_type {};

    template<typename T> struct is_signed    : false_type {};
    template<> struct is_signed<float>       :  true_type {};
    template<> struct is_signed<double>      :  true_type {};
    template<> struct is_signed<long double> :  true_type {};
    template<> struct is_signed<signed char> :  true_type {};
    template<> struct is_signed<short>       :  true_type {};
    template<> struct is_signed<int>         :  true_type {};
    template<> struct is_signed<long>        :  true_type {};
    template<> struct is_signed<long long>   :  true_type {};

    template<typename T> struct is_unsigned             : false_type {};
    template<> struct is_unsigned<unsigned char>      :  true_type {};
    template<> struct is_unsigned<unsigned short>     :  true_type {};
    template<> struct is_unsigned<unsigned int>       :  true_type {};
    template<> struct is_unsigned<unsigned long>      :  true_type {};
    template<> struct is_unsigned<unsigned long long> :  true_type {};

    template<typename T, typename U> struct is_same      : false_type {};
    template<typename T>             struct is_same<T,T> :  true_type {};

    template<class T> struct is_array : false_type {};
    template<class T> struct is_array<T[]> : true_type {};
    template<class T, size_t N> struct is_array<T[N]> : true_type {};

    //partial implementation only of is_function
    template<class> struct is_function : false_type { };
    template<class Ret, class... Args> struct is_function<Ret(Args...)> : true_type {}; //regular
    template<class Ret, class... Args> struct is_function<Ret(Args......)> : true_type {}; // variadic

    template<class> struct result_of;
    template<class F, typename... Args>
    struct result_of<F(Args...)> {
    // TODO: This is a hack; a proper implem is quite complicated.
    typedef typename F::result_type type;
    };

    template <class T> struct remove_reference { typedef T type; };
    template <class T> struct remove_reference<T&> { typedef T type; };
    template <class T> struct remove_reference<T&&> { typedef T type; };
    #if __cplusplus >= 201402L
    template< class T > using remove_reference_t = typename remove_reference<T>::type;
    #endif

    template<class T> struct remove_extent { typedef T type; };
    template<class T> struct remove_extent<T[]> { typedef T type; };
    template<class T, size_t N> struct remove_extent<T[N]> { typedef T type; };
    #if __cplusplus >= 201402L
    template< class T > using remove_extent_t = typename remove_extent<T>::type;
    #endif

    template< class T > struct remove_const          { typedef T type; };
    template< class T > struct remove_const<const T> { typedef T type; };
    template< class T > struct remove_volatile             { typedef T type; };
    template< class T > struct remove_volatile<volatile T> { typedef T type; };
    template< class T > struct remove_cv { typedef typename remove_volatile<typename remove_const<T>::type>::type type; };
    #if __cplusplus >= 201402L
    template< class T > using remove_cv_t       = typename remove_cv<T>::type;
    template< class T > using remove_const_t    = typename remove_const<T>::type;
    template< class T > using remove_volatile_t = typename remove_volatile<T>::type;
    #endif

    template<bool B, class T, class F> struct conditional { typedef T type; };
    template<class T, class F> struct conditional<false, T, F> { typedef F type; };
    #if __cplusplus >= 201402L
    template< bool B, class T, class F > using conditional_t = typename conditional<B,T,F>::type;
    #endif

    namespace __jitify_detail {
    template< class T, bool is_function_type = false > struct add_pointer { using type = typename remove_reference<T>::type*; };
    template< class T > struct add_pointer<T, true> { using type = T; };
    template< class T, class... Args > struct add_pointer<T(Args...), true> { using type = T(*)(Args...); };
    template< class T, class... Args > struct add_pointer<T(Args..., ...), true> { using type = T(*)(Args..., ...); };
    }  // namespace __jitify_detail
    template< class T > struct add_pointer : __jitify_detail::add_pointer<T, is_function<T>::value> {};
    #if __cplusplus >= 201402L
    template< class T > using add_pointer_t = typename add_pointer<T>::type;
    #endif

    template< class T > struct decay {
    private:
      typedef typename remove_reference<T>::type U;
    public:
      typedef typename conditional<is_array<U>::value, typename remove_extent<U>::type*,
        typename conditional<is_function<U>::value,typename add_pointer<U>::type,typename remove_cv<U>::type
        >::type>::type type;
    };
    #if __cplusplus >= 201402L
    template< class T > using decay_t = typename decay<T>::type;
    #endif

    template<class T, T v>
    struct integral_constant {
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant type; // using injected-class-name
    constexpr operator value_type() const noexcept { return value; }
    #if __cplusplus >= 201402L
    constexpr value_type operator()() const noexcept { return value; }
    #endif
    };

    template<class T> struct is_lvalue_reference : false_type {};
    template<class T> struct is_lvalue_reference<T&> : true_type {};

    template<class T> struct is_rvalue_reference : false_type {};
    template<class T> struct is_rvalue_reference<T&&> : true_type {};

    namespace __jitify_detail {
    template <class T> struct type_identity { using type = T; };
    template <class T> auto add_lvalue_reference(int) -> type_identity<T&>;
    template <class T> auto add_lvalue_reference(...) -> type_identity<T>;
    template <class T> auto add_rvalue_reference(int) -> type_identity<T&&>;
    template <class T> auto add_rvalue_reference(...) -> type_identity<T>;
    } // namespace _jitify_detail

    template <class T> struct add_lvalue_reference : decltype(__jitify_detail::add_lvalue_reference<T>(0)) {};
    template <class T> struct add_rvalue_reference : decltype(__jitify_detail::add_rvalue_reference<T>(0)) {};
    #if __cplusplus >= 201402L
    template <class T> using add_lvalue_reference_t = typename add_lvalue_reference<T>::type;
    template <class T> using add_rvalue_reference_t = typename add_rvalue_reference<T>::type;
    #endif

    template<typename T> struct is_const          : public false_type {};
    template<typename T> struct is_const<const T> : public true_type {};

    template<typename T> struct is_volatile             : public false_type {};
    template<typename T> struct is_volatile<volatile T> : public true_type {};

    template<typename T> struct is_void             : public false_type {};
    template<>           struct is_void<void>       : public true_type {};
    template<>           struct is_void<const void> : public true_type {};

    template<typename T> struct is_reference     : public false_type {};
    template<typename T> struct is_reference<T&> : public true_type {};

    template<typename _Tp, bool = (is_void<_Tp>::value || is_reference<_Tp>::value)>
    struct __add_reference_helper { typedef _Tp&    type; };

    template<typename _Tp> struct __add_reference_helper<_Tp, true> { typedef _Tp     type; };
    template<typename _Tp> struct add_reference : public __add_reference_helper<_Tp>{};

    namespace __jitify_detail {
    template<typename T> struct is_int_or_cref {
    typedef typename remove_reference<T>::type type_sans_ref;
    static const bool value = (is_integral<T>::value || (is_integral<type_sans_ref>::value
      && is_const<type_sans_ref>::value && !is_volatile<type_sans_ref>::value));
    }; // end is_int_or_cref
    template<typename From, typename To> struct is_convertible_sfinae {
    private:
    typedef char                          yes;
    typedef struct { char two_chars[2]; } no;
    static inline yes   test(To) { return yes(); }
    static inline no    test(...) { return no(); }
    static inline typename remove_reference<From>::type& from() { typename remove_reference<From>::type* ptr = 0; return *ptr; }
    public:
    static const bool value = sizeof(test(from())) == sizeof(yes);
    }; // end is_convertible_sfinae
    template<typename From, typename To> struct is_convertible_needs_simple_test {
    static const bool from_is_void      = is_void<From>::value;
    static const bool to_is_void        = is_void<To>::value;
    static const bool from_is_float     = is_floating_point<typename remove_reference<From>::type>::value;
    static const bool to_is_int_or_cref = is_int_or_cref<To>::value;
    static const bool value = (from_is_void || to_is_void || (from_is_float && to_is_int_or_cref));
    }; // end is_convertible_needs_simple_test
    template<typename From, typename To, bool = is_convertible_needs_simple_test<From,To>::value>
    struct is_convertible {
    static const bool value = (is_void<To>::value || (is_int_or_cref<To>::value && !is_void<From>::value));
    }; // end is_convertible
    template<typename From, typename To> struct is_convertible<From, To, false> {
    static const bool value = (is_convertible_sfinae<typename add_reference<From>::type, To>::value);
    }; // end is_convertible
    } // end __jitify_detail
    // implementation of is_convertible taken from thrust's pre C++11 path
    template<typename From, typename To> struct is_convertible
    : public integral_constant<bool, __jitify_detail::is_convertible<From, To>::value>
    { }; // end is_convertible

    template<class A, class B> struct is_base_of { };

    template<size_t len, size_t alignment> struct aligned_storage { struct type { alignas(alignment) char data[len]; }; };
    template <class T> struct alignment_of : std::integral_constant<size_t,alignof(T)> {};

    template <typename T> struct make_unsigned;
    template <> struct make_unsigned<signed char>        { typedef unsigned char type; };
    template <> struct make_unsigned<signed short>       { typedef unsigned short type; };
    template <> struct make_unsigned<signed int>         { typedef unsigned int type; };
    template <> struct make_unsigned<signed long>        { typedef unsigned long type; };
    template <> struct make_unsigned<signed long long>   { typedef unsigned long long type; };
    template <> struct make_unsigned<unsigned char>      { typedef unsigned char type; };
    template <> struct make_unsigned<unsigned short>     { typedef unsigned short type; };
    template <> struct make_unsigned<unsigned int>       { typedef unsigned int type; };
    template <> struct make_unsigned<unsigned long>      { typedef unsigned long type; };
    template <> struct make_unsigned<unsigned long long> { typedef unsigned long long type; };
    template <> struct make_unsigned<char>               { typedef unsigned char type; };
    #if defined _WIN32 || defined _WIN64
    template <> struct make_unsigned<wchar_t>            { typedef unsigned short type; };
    #else
    template <> struct make_unsigned<wchar_t>            { typedef unsigned int type; };
    #endif

    template <typename T> struct make_signed;
    template <> struct make_signed<signed char>        { typedef signed char type; };
    template <> struct make_signed<signed short>       { typedef signed short type; };
    template <> struct make_signed<signed int>         { typedef signed int type; };
    template <> struct make_signed<signed long>        { typedef signed long type; };
    template <> struct make_signed<signed long long>   { typedef signed long long type; };
    template <> struct make_signed<unsigned char>      { typedef signed char type; };
    template <> struct make_signed<unsigned short>     { typedef signed short type; };
    template <> struct make_signed<unsigned int>       { typedef signed int type; };
    template <> struct make_signed<unsigned long>      { typedef signed long type; };
    template <> struct make_signed<unsigned long long> { typedef signed long long type; };
    template <> struct make_signed<char>               { typedef signed char type; };
    #if defined _WIN32 || defined _WIN64
    template <> struct make_signed<wchar_t>            { typedef signed short type; };
    #else
    template <> struct make_signed<wchar_t>            { typedef signed int type; };
    #endif

    }  // namespace std
    #endif // c++11
)";

// TODO: INT_FAST8_MAX et al. and a few other misc constants
static const char* jitsafe_header_stdint_h =
    "#pragma once\n"
    "#include <climits>\n"
    "namespace __jitify_stdint_ns {\n"
    "typedef signed char      int8_t;\n"
    "typedef signed short     int16_t;\n"
    "typedef signed int       int32_t;\n"
    "typedef signed long long int64_t;\n"
    "typedef signed char      int_fast8_t;\n"
    "typedef signed short     int_fast16_t;\n"
    "typedef signed int       int_fast32_t;\n"
    "typedef signed long long int_fast64_t;\n"
    "typedef signed char      int_least8_t;\n"
    "typedef signed short     int_least16_t;\n"
    "typedef signed int       int_least32_t;\n"
    "typedef signed long long int_least64_t;\n"
    "typedef signed long long intmax_t;\n"
    "typedef signed long      intptr_t; //optional\n"
    "typedef unsigned char      uint8_t;\n"
    "typedef unsigned short     uint16_t;\n"
    "typedef unsigned int       uint32_t;\n"
    "typedef unsigned long long uint64_t;\n"
    "typedef unsigned char      uint_fast8_t;\n"
    "typedef unsigned short     uint_fast16_t;\n"
    "typedef unsigned int       uint_fast32_t;\n"
    "typedef unsigned long long uint_fast64_t;\n"
    "typedef unsigned char      uint_least8_t;\n"
    "typedef unsigned short     uint_least16_t;\n"
    "typedef unsigned int       uint_least32_t;\n"
    "typedef unsigned long long uint_least64_t;\n"
    "typedef unsigned long long uintmax_t;\n"
    "#define INT8_MIN    SCHAR_MIN\n"
    "#define INT16_MIN   SHRT_MIN\n"
    "#if defined _WIN32 || defined _WIN64\n"
    "#define WCHAR_MIN   0\n"
    "#define WCHAR_MAX   USHRT_MAX\n"
    "typedef unsigned long long uintptr_t; //optional\n"
    "#else\n"
    "#define WCHAR_MIN   INT_MIN\n"
    "#define WCHAR_MAX   INT_MAX\n"
    "typedef unsigned long      uintptr_t; //optional\n"
    "#endif\n"
    "#define INT32_MIN   INT_MIN\n"
    "#define INT64_MIN   LLONG_MIN\n"
    "#define INT8_MAX    SCHAR_MAX\n"
    "#define INT16_MAX   SHRT_MAX\n"
    "#define INT32_MAX   INT_MAX\n"
    "#define INT64_MAX   LLONG_MAX\n"
    "#define UINT8_MAX   UCHAR_MAX\n"
    "#define UINT16_MAX  USHRT_MAX\n"
    "#define UINT32_MAX  UINT_MAX\n"
    "#define UINT64_MAX  ULLONG_MAX\n"
    "#define INTPTR_MIN  LONG_MIN\n"
    "#define INTMAX_MIN  LLONG_MIN\n"
    "#define INTPTR_MAX  LONG_MAX\n"
    "#define INTMAX_MAX  LLONG_MAX\n"
    "#define UINTPTR_MAX ULONG_MAX\n"
    "#define UINTMAX_MAX ULLONG_MAX\n"
    "#define PTRDIFF_MIN INTPTR_MIN\n"
    "#define PTRDIFF_MAX INTPTR_MAX\n"
    "#define SIZE_MAX    UINT64_MAX\n"
    "} // namespace __jitify_stdint_ns\n"
    "namespace std { using namespace __jitify_stdint_ns; }\n"
    "using namespace __jitify_stdint_ns;\n";

// TODO: offsetof
static const char* jitsafe_header_stddef_h =
    "#pragma once\n"
    "#include <climits>\n"
    "namespace __jitify_stddef_ns {\n"
    "#if __cplusplus >= 201103L\n"
    "typedef decltype(nullptr) nullptr_t;\n"
    "#if defined(_MSC_VER)\n"
    "  typedef double max_align_t;\n"
    "#elif defined(__APPLE__)\n"
    "  typedef long double max_align_t;\n"
    "#else\n"
    "  // Define max_align_t to match the GCC definition.\n"
    "  typedef struct {\n"
    "    long long __jitify_max_align_nonce1\n"
    "        __attribute__((__aligned__(__alignof__(long long))));\n"
    "    long double __jitify_max_align_nonce2\n"
    "        __attribute__((__aligned__(__alignof__(long double))));\n"
    "  } max_align_t;\n"
    "#endif\n"
    "#endif  // __cplusplus >= 201103L\n"
    "#if __cplusplus >= 201703L\n"
    "enum class byte : unsigned char {};\n"
    "#endif  // __cplusplus >= 201703L\n"
    "} // namespace __jitify_stddef_ns\n"
    "namespace std {\n"
    "  // NVRTC provides built-in definitions of ::size_t and ::ptrdiff_t.\n"
    "  using ::size_t;\n"
    "  using ::ptrdiff_t;\n"
    "  using namespace __jitify_stddef_ns;\n"
    "} // namespace std\n"
    "using namespace __jitify_stddef_ns;\n";

static const char* jitsafe_header_stdlib_h =
    "#pragma once\n"
    "#include <stddef.h>\n";
static const char* jitsafe_header_stdio_h =
    "#pragma once\n"
    "#include <stddef.h>\n"
    "#define FILE int\n"
    "int fflush ( FILE * stream );\n"
    "int fprintf ( FILE * stream, const char * format, ... );\n";

static const char* jitsafe_header_string_h =
    "#pragma once\n"
    "char* strcpy ( char * destination, const char * source );\n"
    "int strcmp ( const char * str1, const char * str2 );\n"
    "char* strerror( int errnum );\n";

static const char* jitsafe_header_cstring =
    "#pragma once\n"
    "\n"
    "namespace __jitify_cstring_ns {\n"
    "char* strcpy ( char * destination, const char * source );\n"
    "int strcmp ( const char * str1, const char * str2 );\n"
    "char* strerror( int errnum );\n"
    "} // namespace __jitify_cstring_ns\n"
    "namespace std { using namespace __jitify_cstring_ns; }\n"
    "using namespace __jitify_cstring_ns;\n";

// HACK TESTING (WAR for cub)
static const char* jitsafe_header_iostream =
    "#pragma once\n"
    "#include <ostream>\n"
    "#include <istream>\n";
// HACK TESTING (WAR for Thrust)
static const char* jitsafe_header_ostream =
    "#pragma once\n"
    "\n"
    "namespace std {\n"
    "template<class CharT,class Traits=void>\n"  // = std::char_traits<CharT>
                                                 // >\n"
    "struct basic_ostream {\n"
    "};\n"
    "typedef basic_ostream<char> ostream;\n"
    "ostream& endl(ostream& os);\n"
    "ostream& operator<<( ostream&, ostream& (*f)( ostream& ) );\n"
    "template< class CharT, class Traits > basic_ostream<CharT, Traits>& endl( "
    "basic_ostream<CharT, Traits>& os );\n"
    "template< class CharT, class Traits > basic_ostream<CharT, Traits>& "
    "operator<<( basic_ostream<CharT,Traits>& os, const char* c );\n"
    "#if __cplusplus >= 201103L\n"
    "template< class CharT, class Traits, class T > basic_ostream<CharT, "
    "Traits>& operator<<( basic_ostream<CharT,Traits>&& os, const T& value );\n"
    "#endif  // __cplusplus >= 201103L\n"
    "}  // namespace std\n";

static const char* jitsafe_header_istream =
    "#pragma once\n"
    "\n"
    "namespace std {\n"
    "template<class CharT,class Traits=void>\n"  // = std::char_traits<CharT>
                                                 // >\n"
    "struct basic_istream {\n"
    "};\n"
    "typedef basic_istream<char> istream;\n"
    "}  // namespace std\n";

static const char* jitsafe_header_sstream =
    "#pragma once\n"
    "#include <ostream>\n"
    "#include <istream>\n";

static const char* jitsafe_header_utility =
    "#pragma once\n"
    "namespace std {\n"
    "template<class T1, class T2>\n"
    "struct pair {\n"
    "	T1 first;\n"
    "	T2 second;\n"
    "	inline pair() {}\n"
    "	inline pair(T1 const& first_, T2 const& second_)\n"
    "		: first(first_), second(second_) {}\n"
    "	// TODO: Standard includes many more constructors...\n"
    "	// TODO: Comparison operators\n"
    "};\n"
    "template<class T1, class T2>\n"
    "pair<T1,T2> make_pair(T1 const& first, T2 const& second) {\n"
    "	return pair<T1,T2>(first, second);\n"
    "}\n"
    "}  // namespace std\n";

// TODO: incomplete
static const char* jitsafe_header_vector =
    "#pragma once\n"
    "namespace std {\n"
    "template<class T, class Allocator=void>\n"  // = std::allocator> \n"
    "struct vector {\n"
    "};\n"
    "}  // namespace std\n";

// TODO: incomplete
static const char* jitsafe_header_string =
    "#pragma once\n"
    "namespace std {\n"
    "template<class CharT,class Traits=void,class Allocator=void>\n"
    "struct basic_string {\n"
    "basic_string();\n"
    "basic_string( const CharT* s );\n"  //, const Allocator& alloc =
                                         // Allocator() );\n"
    "const CharT* c_str() const;\n"
    "bool empty() const;\n"
    "void operator+=(const char *);\n"
    "void operator+=(const basic_string &);\n"
    "};\n"
    "typedef basic_string<char> string;\n"
    "}  // namespace std\n";

// TODO: incomplete
static const char* jitsafe_header_stdexcept =
    "#pragma once\n"
    "namespace std {\n"
    "struct runtime_error {\n"
    "explicit runtime_error( const std::string& what_arg );"
    "explicit runtime_error( const char* what_arg );"
    "virtual const char* what() const;\n"
    "};\n"
    "}  // namespace std\n";

// TODO: incomplete
static const char* jitsafe_header_complex =
    "#pragma once\n"
    "namespace std {\n"
    "template<typename T>\n"
    "class complex {\n"
    "	T _real;\n"
    "	T _imag;\n"
    "public:\n"
    "	complex() : _real(0), _imag(0) {}\n"
    "	complex(T const& real, T const& imag)\n"
    "		: _real(real), _imag(imag) {}\n"
    "	complex(T const& real)\n"
    "               : _real(real), _imag(static_cast<T>(0)) {}\n"
    "	T const& real() const { return _real; }\n"
    "	T&       real()       { return _real; }\n"
    "	void real(const T &r) { _real = r; }\n"
    "	T const& imag() const { return _imag; }\n"
    "	T&       imag()       { return _imag; }\n"
    "	void imag(const T &i) { _imag = i; }\n"
    "       complex<T>& operator+=(const complex<T> z)\n"
    "         { _real += z.real(); _imag += z.imag(); return *this; }\n"
    "};\n"
    "template<typename T>\n"
    "complex<T> operator*(const complex<T>& lhs, const complex<T>& rhs)\n"
    "  { return complex<T>(lhs.real()*rhs.real()-lhs.imag()*rhs.imag(),\n"
    "                      lhs.real()*rhs.imag()+lhs.imag()*rhs.real()); }\n"
    "template<typename T>\n"
    "complex<T> operator*(const complex<T>& lhs, const T & rhs)\n"
    "  { return complexs<T>(lhs.real()*rhs,lhs.imag()*rhs); }\n"
    "template<typename T>\n"
    "complex<T> operator*(const T& lhs, const complex<T>& rhs)\n"
    "  { return complexs<T>(rhs.real()*lhs,rhs.imag()*lhs); }\n"
    "}  // namespace std\n";

// TODO: This is incomplete (missing binary and integer funcs, macros,
// constants, types)
static const char* jitsafe_header_math_h =
    "#pragma once\n"
    "namespace __jitify_math_ns {\n"
    "#if __cplusplus >= 201103L\n"
    "#define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\\n"
    "	inline double      f(double x)         { return ::f(x); } \\\n"
    "	inline float       f##f(float x)       { return ::f(x); } \\\n"
    "	/*inline long double f##l(long double x) { return ::f(x); }*/ \\\n"
    "	inline float       f(float x)          { return ::f(x); } \\\n"
    "	/*inline long double f(long double x)    { return ::f(x); }*/\n"
    "#else\n"
    "#define DEFINE_MATH_UNARY_FUNC_WRAPPER(f) \\\n"
    "	inline double      f(double x)         { return ::f(x); } \\\n"
    "	inline float       f##f(float x)       { return ::f(x); } \\\n"
    "	/*inline long double f##l(long double x) { return ::f(x); }*/\n"
    "#endif\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cos)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sin)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tan)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(acos)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(asin)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(atan)\n"
    "template<typename T> inline T atan2(T y, T x) { return ::atan2(y, x); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cosh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sinh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tanh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(exp)\n"
    "template<typename T> inline T frexp(T x, int* exp) { return ::frexp(x, "
    "exp); }\n"
    "template<typename T> inline T ldexp(T x, int  exp) { return ::ldexp(x, "
    "exp); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log10)\n"
    "template<typename T> inline T modf(T x, T* intpart) { return ::modf(x, "
    "intpart); }\n"
    "template<typename T> inline T pow(T x, T y) { return ::pow(x, y); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(sqrt)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(ceil)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(floor)\n"
    "template<typename T> inline T fmod(T n, T d) { return ::fmod(n, d); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(fabs)\n"
    "template<typename T> inline T abs(T x) { return ::abs(x); }\n"
    "#if __cplusplus >= 201103L\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(acosh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(asinh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(atanh)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(exp2)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(expm1)\n"
    "template<typename T> inline int ilogb(T x) { return ::ilogb(x); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log1p)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(log2)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(logb)\n"
    "template<typename T> inline T scalbn (T x, int n)  { return ::scalbn(x, "
    "n); }\n"
    "template<typename T> inline T scalbln(T x, long n) { return ::scalbn(x, "
    "n); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(cbrt)\n"
    "template<typename T> inline T hypot(T x, T y) { return ::hypot(x, y); }\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(erf)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(erfc)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(tgamma)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(lgamma)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(trunc)\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(round)\n"
    "template<typename T> inline long lround(T x) { return ::lround(x); }\n"
    "template<typename T> inline long long llround(T x) { return ::llround(x); "
    "}\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(rint)\n"
    "template<typename T> inline long lrint(T x) { return ::lrint(x); }\n"
    "template<typename T> inline long long llrint(T x) { return ::llrint(x); "
    "}\n"
    "DEFINE_MATH_UNARY_FUNC_WRAPPER(nearbyint)\n"
    // TODO: remainder, remquo, copysign, nan, nextafter, nexttoward, fdim,
    // fmax, fmin, fma
    "#endif\n"
    "#undef DEFINE_MATH_UNARY_FUNC_WRAPPER\n"
    "} // namespace __jitify_math_ns\n"
    "namespace std { using namespace __jitify_math_ns; }\n"
    "#define M_PI 3.14159265358979323846\n"
    // Note: Global namespace already includes CUDA math funcs
    "//using namespace __jitify_math_ns;\n";

static const char* jitsafe_header_memory_h = R"(
    #pragma once
    #include <string.h>
 )";

// TODO: incomplete
static const char* jitsafe_header_mutex = R"(
    #pragma once
    #if __cplusplus >= 201103L
    namespace std {
    class mutex {
    public:
    void lock();
    bool try_lock();
    void unlock();
    };
    }  // namespace std
    #endif
 )";

static const char* jitsafe_header_algorithm = R"(
    #pragma once
    #if __cplusplus >= 201103L
    namespace std {

    #if __cplusplus == 201103L
    #define JITIFY_CXX14_CONSTEXPR
    #else
    #define JITIFY_CXX14_CONSTEXPR constexpr
    #endif

    template<class T> JITIFY_CXX14_CONSTEXPR const T& max(const T& a, const T& b)
    {
      return (b > a) ? b : a;
    }
    template<class T> JITIFY_CXX14_CONSTEXPR const T& min(const T& a, const T& b)
    {
      return (b < a) ? b : a;
    }

    }  // namespace std
    #endif
 )";

static const char* jitsafe_header_time_h = R"(
    #pragma once
    #define NULL 0
    #define CLOCKS_PER_SEC 1000000
    namespace __jitify_time_ns {
    typedef long time_t;
    struct tm {
      int tm_sec;
      int tm_min;
      int tm_hour;
      int tm_mday;
      int tm_mon;
      int tm_year;
      int tm_wday;
      int tm_yday;
      int tm_isdst;
    };
    #if __cplusplus >= 201703L
    struct timespec {
      time_t tv_sec;
      long tv_nsec;
    };
    #endif
    }  // namespace __jitify_time_ns
    namespace std {
      // NVRTC provides built-in definitions of ::size_t and ::clock_t.
      using ::size_t;
      using ::clock_t;
      using namespace __jitify_time_ns;
    }
    using namespace __jitify_time_ns;
 )";

static const char* jitsafe_header_tuple = R"(
    #pragma once
    #if __cplusplus >= 201103L
    namespace std {
    template<class... Types > class tuple;
    } // namespace std
    #endif
 )";

static const char* jitsafe_header_assert = R"(
    #pragma once
 )";

// WAR: These need to be pre-included as a workaround for NVRTC implicitly using
// /usr/include as an include path. The other built-in headers will be included
// lazily as needed.
static const char* preinclude_jitsafe_header_names[] = {"jitify_preinclude.h",
                                                        "limits.h",
                                                        "math.h",
                                                        "memory.h",
                                                        "stdint.h",
                                                        "stdlib.h",
                                                        "stdio.h",
                                                        "string.h",
                                                        "time.h",
                                                        "assert.h"};

template <class T, int N>
int array_size(T (&)[N]) {
  return N;
}
const int preinclude_jitsafe_headers_count =
    array_size(preinclude_jitsafe_header_names);

static const std::map<std::string, std::string>& get_jitsafe_headers_map() {
  static const std::map<std::string, std::string> jitsafe_headers_map = {
      {"jitify_preinclude.h", jitsafe_header_preinclude_h},
      {"float.h", jitsafe_header_float_h},
      {"cfloat", jitsafe_header_float_h},
      {"limits.h", jitsafe_header_limits_h},
      {"climits", jitsafe_header_limits_h},
      {"stdint.h", jitsafe_header_stdint_h},
      {"cstdint", jitsafe_header_stdint_h},
      {"stddef.h", jitsafe_header_stddef_h},
      {"cstddef", jitsafe_header_stddef_h},
      {"stdlib.h", jitsafe_header_stdlib_h},
      {"cstdlib", jitsafe_header_stdlib_h},
      {"stdio.h", jitsafe_header_stdio_h},
      {"cstdio", jitsafe_header_stdio_h},
      {"string.h", jitsafe_header_string_h},
      {"cstring", jitsafe_header_cstring},
      {"iterator", jitsafe_header_iterator},
      {"limits", jitsafe_header_limits},
      {"type_traits", jitsafe_header_type_traits},
      {"utility", jitsafe_header_utility},
      {"math.h", jitsafe_header_math_h},
      {"cmath", jitsafe_header_math_h},
      {"memory.h", jitsafe_header_memory_h},
      {"complex", jitsafe_header_complex},
      {"iostream", jitsafe_header_iostream},
      {"ostream", jitsafe_header_ostream},
      {"istream", jitsafe_header_istream},
      {"sstream", jitsafe_header_sstream},
      {"vector", jitsafe_header_vector},
      {"string", jitsafe_header_string},
      {"stdexcept", jitsafe_header_stdexcept},
      {"mutex", jitsafe_header_mutex},
      {"algorithm", jitsafe_header_algorithm},
      {"time.h", jitsafe_header_time_h},
      {"ctime", jitsafe_header_time_h},
      {"tuple", jitsafe_header_tuple},
      {"assert.h", jitsafe_header_assert},
      {"cassert", jitsafe_header_assert}};
  return jitsafe_headers_map;
}

inline void add_options_from_env(std::vector<std::string>& options) {
  // Add options from environment variable
  const char* env_options = std::getenv("JITIFY_OPTIONS");
  if (env_options) {
    std::stringstream ss;
    ss << env_options;
    std::string opt;
    while (!(ss >> opt).fail()) {
      options.push_back(opt);
    }
  }
  // Add options from JITIFY_OPTIONS macro
#ifdef JITIFY_OPTIONS
#define JITIFY_TOSTRING_IMPL(x) #x
#define JITIFY_TOSTRING(x) JITIFY_TOSTRING_IMPL(x)
  std::stringstream ss;
  ss << JITIFY_TOSTRING(JITIFY_OPTIONS);
  std::string opt;
  while (!(ss >> opt).fail()) {
    options.push_back(opt);
  }
#undef JITIFY_TOSTRING
#undef JITIFY_TOSTRING_IMPL
#endif  // JITIFY_OPTIONS
}

inline void detect_and_add_cuda_arch(std::vector<std::string>& options) {
  for (int i = 0; i < (int)options.size(); ++i) {
    // Note that this will also match the middle of "--gpu-architecture".
    if (options[i].find("-arch") != std::string::npos) {
      // Arch already specified in options
      return;
    }
  }
  // Use the compute capability of the current device
  // TODO: Check these API calls for errors
  cudaError_t status;
  int device;
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string(
            "Failed to detect GPU architecture: cudaGetDevice failed: ") +
        cudaGetErrorString(status));
  }
  int cc_major;
  cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device);
  int cc_minor;
  cudaDeviceGetAttribute(&cc_minor, cudaDevAttrComputeCapabilityMinor, device);
  int cc = cc_major * 10 + cc_minor;
  // Note: We must limit the architecture to the max supported by the current
  //         version of NVRTC, otherwise newer hardware will cause errors
  //         on older versions of CUDA.
  // TODO: It would be better to detect this somehow, rather than hard-coding it

  // Tegra chips do not have forwards compatibility so we need to special case
  // them.
  bool is_tegra = ((cc_major == 3 && cc_minor == 2) ||  // Logan
                   (cc_major == 5 && cc_minor == 3) ||  // Erista
                   (cc_major == 6 && cc_minor == 2) ||  // Parker
                   (cc_major == 7 && cc_minor == 2));   // Xavier
  if (!is_tegra) {
    // ensure that future CUDA versions just work (even if suboptimal)
    const int cuda_major = std::min(10, CUDA_VERSION / 1000);
    // clang-format off
    switch (cuda_major) {
      case 10: cc = std::min(cc, 75); break; // Turing
      case  9: cc = std::min(cc, 70); break; // Volta
      case  8: cc = std::min(cc, 61); break; // Pascal
      case  7: cc = std::min(cc, 52); break; // Maxwell
      default:
        throw std::runtime_error("Unexpected CUDA major version " +
                                 std::to_string(cuda_major));
    }
    // clang-format on
  }

  std::stringstream ss;
  ss << cc;
  options.push_back("-arch=compute_" + ss.str());
}

inline void detect_and_add_cxx11_flag(std::vector<std::string>& options) {
  // Reverse loop so we can erase on the fly.
  for (int i = (int)options.size() - 1; i >= 0; --i) {
    if (options[i].find("-std=c++98") != std::string::npos) {
      // NVRTC doesn't support specifying c++98 explicitly, so we remove it.
      options.erase(options.begin() + i);
      return;
    } else if (options[i].find("-std") != std::string::npos) {
      // Some other standard was explicitly specified, don't change anything.
      return;
    }
  }
  // Jitify must be compiled with C++11 support, so we default to enabling it
  // for the JIT-compiled code too.
  options.push_back("-std=c++11");
}

inline void split_compiler_and_linker_options(
    std::vector<std::string> options,
    std::vector<std::string>* compiler_options,
    std::vector<std::string>* linker_files,
    std::vector<std::string>* linker_paths) {
  for (int i = 0; i < (int)options.size(); ++i) {
    std::string opt = options[i];
    std::string flag = opt.substr(0, 2);
    std::string value = opt.substr(2);
    if (flag == "-l") {
      linker_files->push_back(value);
    } else if (flag == "-L") {
      linker_paths->push_back(value);
    } else {
      compiler_options->push_back(opt);
    }
  }
}

inline bool pop_remove_unused_globals_flag(std::vector<std::string>* options) {
  auto it = std::remove_if(
      options->begin(), options->end(), [](const std::string& opt) {
        return opt.find("-remove-unused-globals") != std::string::npos;
      });
  if (it != options->end()) {
    options->resize(it - options->begin());
    return true;
  }
  return false;
}

inline std::string ptx_parse_decl_name(const std::string& line) {
  size_t name_end = line.find_first_of("[;");
  if (name_end == std::string::npos) {
    throw std::runtime_error(
        "Failed to parse .global/.const declaration in PTX: expected a "
        "semicolon");
  }
  size_t name_start_minus1 = line.find_last_of(" \t", name_end);
  if (name_start_minus1 == std::string::npos) {
    throw std::runtime_error(
        "Failed to parse .global/.const declaration in PTX: expected "
        "whitespace");
  }
  size_t name_start = name_start_minus1 + 1;
  std::string name = line.substr(name_start, name_end - name_start);
  return name;
}

inline void ptx_remove_unused_globals(std::string* ptx) {
  std::istringstream iss(*ptx);
  std::vector<std::string> lines;
  std::unordered_map<size_t, std::string> line_num_to_global_name;
  std::unordered_set<std::string> name_set;
  for (std::string line; std::getline(iss, line);) {
    size_t line_num = lines.size();
    lines.push_back(line);
    auto terms = split_string(line);
    if (terms.size() <= 1) continue;  // Ignore lines with no arguments
    if (terms[0].substr(0, 2) == "//") continue;  // Ignore comment lines
    if (terms[0].substr(0, 7) == ".global" ||
        terms[0].substr(0, 6) == ".const") {
      line_num_to_global_name.emplace(line_num, ptx_parse_decl_name(line));
      continue;
    }
    if (terms[0][0] == '.') continue;  // Ignore .version, .reg, .param etc.
    // Note: The first term will always be an instruction name; starting at 1
    // also allows unchecked inspection of the previous term.
    for (int i = 1; i < (int)terms.size(); ++i) {
      if (terms[i].substr(0, 2) == "//") break;  // Ignore comments
      // Note: The characters '.' and '%' are not treated as delimiters.
      const char* token_delims = " \t()[]{},;+-*/~&|^?:=!<>\"'\\";
      for (auto token : split_string(terms[i], -1, token_delims)) {
        if (  // Ignore non-names
            !(std::isalpha(token[0]) || token[0] == '_' || token[0] == '$') ||
            token.find('.') != std::string::npos ||
            // Ignore variable/parameter declarations
            terms[i - 1][0] == '.' ||
            // Ignore branch instructions
            (token == "bra" && terms[i - 1][0] == '@') ||
            // Ignore branch labels
            (token.substr(0, 2) == "BB" &&
             terms[i - 1].substr(0, 3) == "bra")) {
          continue;
        }
        name_set.insert(token);
      }
    }
  }
  std::ostringstream oss;
  for (size_t line_num = 0; line_num < lines.size(); ++line_num) {
    auto it = line_num_to_global_name.find(line_num);
    if (it != line_num_to_global_name.end()) {
      const std::string& name = it->second;
      if (!name_set.count(name)) {
        continue;  // Remove unused .global declaration.
      }
    }
    oss << lines[line_num] << '\n';
  }
  *ptx = oss.str();
}

inline nvrtcResult compile_kernel(std::string program_name,
                                  std::map<std::string, std::string> sources,
                                  std::vector<std::string> options,
                                  std::string instantiation = "",
                                  std::string* log = 0, std::string* ptx = 0,
                                  std::string* mangled_instantiation = 0) {
  std::string program_source = sources[program_name];
  // Build arrays of header names and sources
  std::vector<const char*> header_names_c;
  std::vector<const char*> header_sources_c;
  int num_headers = (int)(sources.size() - 1);
  header_names_c.reserve(num_headers);
  header_sources_c.reserve(num_headers);
  typedef std::map<std::string, std::string> source_map;
  for (source_map::const_iterator iter = sources.begin(); iter != sources.end();
       ++iter) {
    std::string const& name = iter->first;
    std::string const& code = iter->second;
    if (name == program_name) {
      continue;
    }
    header_names_c.push_back(name.c_str());
    header_sources_c.push_back(code.c_str());
  }

  // TODO: This WAR is expected to be unnecessary as of CUDA > 10.2.
  bool should_remove_unused_globals =
      detail::pop_remove_unused_globals_flag(&options);

  std::vector<const char*> options_c(options.size() + 2);
  options_c[0] = "--device-as-default-execution-space";
  options_c[1] = "--pre-include=jitify_preinclude.h";
  for (int i = 0; i < (int)options.size(); ++i) {
    options_c[i + 2] = options[i].c_str();
  }

#if CUDA_VERSION < 8000
  std::string inst_dummy;
  if (!instantiation.empty()) {
    // WAR for no nvrtcAddNameExpression before CUDA 8.0
    // Force template instantiation by adding dummy reference to kernel
    inst_dummy = "__jitify_instantiation";
    program_source +=
        "\nvoid* " + inst_dummy + " = (void*)" + instantiation + ";\n";
  }
#endif

#define CHECK_NVRTC(call)                         \
  do {                                            \
    nvrtcResult check_nvrtc_macro_ret = call;     \
    if (check_nvrtc_macro_ret != NVRTC_SUCCESS) { \
      return check_nvrtc_macro_ret;               \
    }                                             \
  } while (0)

  nvrtcProgram nvrtc_program;
  CHECK_NVRTC(nvrtcCreateProgram(
      &nvrtc_program, program_source.c_str(), program_name.c_str(), num_headers,
      header_sources_c.data(), header_names_c.data()));

  // Ensure nvrtc_program gets destroyed.
  struct ScopedNvrtcProgramDestroyer {
    nvrtcProgram& nvrtc_program_;
    ScopedNvrtcProgramDestroyer(nvrtcProgram& nvrtc_program)
        : nvrtc_program_(nvrtc_program) {}    
    ~ScopedNvrtcProgramDestroyer() { nvrtcDestroyProgram(&nvrtc_program_); }
    ScopedNvrtcProgramDestroyer(const ScopedNvrtcProgramDestroyer&) = delete;
    ScopedNvrtcProgramDestroyer& operator=(const ScopedNvrtcProgramDestroyer&) =
        delete;
  } nvrtc_program_scope_guard{nvrtc_program};

#if CUDA_VERSION >= 8000
  if (!instantiation.empty()) {
    CHECK_NVRTC(nvrtcAddNameExpression(nvrtc_program, instantiation.c_str()));
  }
#endif

  nvrtcResult ret = nvrtcCompileProgram(nvrtc_program, (int)options_c.size(),
                                        options_c.data());
  if (log) {
    size_t logsize;
    CHECK_NVRTC(nvrtcGetProgramLogSize(nvrtc_program, &logsize));
    std::vector<char> vlog(logsize, 0);
    CHECK_NVRTC(nvrtcGetProgramLog(nvrtc_program, vlog.data()));
    log->assign(vlog.data(), logsize);
  }
  if (ret != NVRTC_SUCCESS) {
    return ret;
  }

  if (ptx) {
    size_t ptxsize;
    CHECK_NVRTC(nvrtcGetPTXSize(nvrtc_program, &ptxsize));
    std::vector<char> vptx(ptxsize);
    CHECK_NVRTC(nvrtcGetPTX(nvrtc_program, vptx.data()));
    ptx->assign(vptx.data(), ptxsize);
    if (should_remove_unused_globals) {
      detail::ptx_remove_unused_globals(ptx);
    }
  }

  if (!instantiation.empty() && mangled_instantiation) {
#if CUDA_VERSION >= 8000
    const char* mangled_instantiation_cstr;
    // Note: The returned string pointer becomes invalid after
    //         nvrtcDestroyProgram has been called, so we save it.
    CHECK_NVRTC(nvrtcGetLoweredName(nvrtc_program, instantiation.c_str(),
                                    &mangled_instantiation_cstr));
    *mangled_instantiation = mangled_instantiation_cstr;
#else
    // Extract mangled kernel template instantiation from PTX
    inst_dummy += " = ";  // Note: This must match how the PTX is generated
    int mi_beg = ptx->find(inst_dummy) + inst_dummy.size();
    int mi_end = ptx->find(";", mi_beg);
    *mangled_instantiation = ptx->substr(mi_beg, mi_end - mi_beg);
#endif
  }

#undef CHECK_NVRTC
  return NVRTC_SUCCESS;
}

inline void load_program(std::string const& cuda_source,
                         std::vector<std::string> const& headers,
                         file_callback_type file_callback,
                         std::vector<std::string>* include_paths,
                         std::map<std::string, std::string>* program_sources,
                         std::vector<std::string>* program_options,
                         std::string* program_name) {
  // Extract include paths from compile options
  std::vector<std::string>::iterator iter = program_options->begin();
  while (iter != program_options->end()) {
    std::string const& opt = *iter;
    if (opt.substr(0, 2) == "-I") {
      include_paths->push_back(opt.substr(2));
      iter = program_options->erase(iter);
    } else {
      ++iter;
    }
  }

  // Load program source
  if (!detail::load_source(cuda_source, *program_sources, "", *include_paths,
                           file_callback, program_name)) {
    throw std::runtime_error("Source not found: " + cuda_source);
  }

  // Maps header include names to their full file paths.
  std::map<std::string, std::string> header_fullpaths;

  // Load header sources
  for (std::string const& header : headers) {
    if (!detail::load_source(header, *program_sources, "", *include_paths,
                             file_callback, nullptr, &header_fullpaths)) {
      // **TODO: Deal with source not found
      throw std::runtime_error("Source not found: " + header);
    }
  }

#if JITIFY_PRINT_SOURCE
  std::string& program_source = (*program_sources)[*program_name];
  std::cout << "---------------------------------------" << std::endl;
  std::cout << "--- Source of " << *program_name << " ---" << std::endl;
  std::cout << "---------------------------------------" << std::endl;
  detail::print_with_line_numbers(program_source);
  std::cout << "---------------------------------------" << std::endl;
#endif

  std::vector<std::string> compiler_options, linker_files, linker_paths;
  detail::split_compiler_and_linker_options(*program_options, &compiler_options,
                                            &linker_files, &linker_paths);

  // If no arch is specified at this point we use whatever the current
  // context is. This ensures we pick up the correct internal headers
  // for arch-dependent compilation, e.g., some intrinsics are only
  // present for specific architectures.
  detail::detect_and_add_cuda_arch(compiler_options);
  detail::detect_and_add_cxx11_flag(compiler_options);

  // Iteratively try to compile the sources, and use the resulting errors to
  // identify missing headers.
  std::string log;
  nvrtcResult ret;
  while ((ret = detail::compile_kernel(*program_name, *program_sources,
                                       compiler_options, "", &log)) ==
         NVRTC_ERROR_COMPILATION) {
    std::string include_name;
    std::string include_parent;
    int line_num = 0;
    if (!detail::extract_include_info_from_compile_error(
            log, include_name, include_parent, line_num)) {
#if JITIFY_PRINT_LOG
      detail::print_compile_log(*program_name, log);
#endif
      // There was a non include-related compilation error
      // TODO: How to handle error?
      throw std::runtime_error("Runtime compilation failed");
    }

    bool is_included_with_quotes = false;
    if (program_sources->count(include_parent)) {
      const std::string& parent_source = (*program_sources)[include_parent];
      is_included_with_quotes =
          is_include_directive_with_quotes(parent_source, line_num);
    }

    // Try to load the new header
    // Note: This fullpath lookup is needed because the compiler error
    // messages have the include name of the header instead of its full path.
    std::string include_parent_fullpath = header_fullpaths[include_parent];
    std::string include_path = detail::path_base(include_parent_fullpath);
    if (detail::load_source(include_name, *program_sources, include_path,
                            *include_paths, file_callback, nullptr,
                            &header_fullpaths, is_included_with_quotes)) {
#if JITIFY_PRINT_HEADER_PATHS
      std::cout << "Found #include " << include_name << " from "
                << include_parent << ":" << line_num << " ["
                << include_parent_fullpath << "]"
                << " at:\n  " << header_fullpaths[include_name] << std::endl;
#endif
    } else {  // Failed to find header file.
      // Comment-out the include line and print a warning
      if (!program_sources->count(include_parent)) {
        // ***TODO: Unless there's another mechanism (e.g., potentially
        //            the parent path vs. filename problem), getting
        //            here means include_parent was found automatically
        //            in a system include path.
        //            We need a WAR to zap it from *its parent*.

        typedef std::map<std::string, std::string> source_map;
        for (source_map::const_iterator it = program_sources->begin();
             it != program_sources->end(); ++it) {
          std::cout << "  " << it->first << std::endl;
        }
        throw std::out_of_range(include_parent +
                                " not in loaded sources!"
                                " This may be due to a header being loaded by"
                                " NVRTC without Jitify's knowledge.");
      }
      std::string& parent_source = (*program_sources)[include_parent];
      parent_source = detail::comment_out_code_line(line_num, parent_source);
#if JITIFY_PRINT_LOG
      std::cout << include_parent << "(" << line_num
                << "): warning: " << include_name << ": [jitify] File not found"
                << std::endl;
#endif
    }
  }
  if (ret != NVRTC_SUCCESS) {
#if JITIFY_PRINT_LOG
    if (ret == NVRTC_ERROR_INVALID_OPTION) {
      std::cout << "Compiler options: ";
      for (int i = 0; i < (int)compiler_options.size(); ++i) {
        std::cout << compiler_options[i] << " ";
      }
      std::cout << std::endl;
    }
#endif
    throw std::runtime_error(std::string("NVRTC error: ") +
                             nvrtcGetErrorString(ret));
  }
}

inline void instantiate_kernel(
    std::string const& program_name,
    std::map<std::string, std::string> const& program_sources,
    std::string const& instantiation, std::vector<std::string> const& options,
    std::string* log, std::string* ptx, std::string* mangled_instantiation,
    std::vector<std::string>* linker_files,
    std::vector<std::string>* linker_paths) {
  std::vector<std::string> compiler_options;
  detail::split_compiler_and_linker_options(options, &compiler_options,
                                            linker_files, linker_paths);

  nvrtcResult ret =
      detail::compile_kernel(program_name, program_sources, compiler_options,
                             instantiation, log, ptx, mangled_instantiation);
#if JITIFY_PRINT_LOG
  if (log->size() > 1) {
    detail::print_compile_log(program_name, *log);
  }
#endif
  if (ret != NVRTC_SUCCESS) {
    throw std::runtime_error(std::string("NVRTC error: ") +
                             nvrtcGetErrorString(ret));
  }

#if JITIFY_PRINT_PTX
  std::cout << "---------------------------------------" << std::endl;
  std::cout << *mangled_instantiation << std::endl;
  std::cout << "---------------------------------------" << std::endl;
  std::cout << "--- PTX for " << mangled_instantiation << " in " << program_name
            << " ---" << std::endl;
  std::cout << "---------------------------------------" << std::endl;
  std::cout << *ptx << std::endl;
  std::cout << "---------------------------------------" << std::endl;
#endif
}

inline void get_1d_max_occupancy(CUfunction func,
                                 CUoccupancyB2DSize smem_callback,
                                 unsigned int* smem, int max_block_size,
                                 unsigned int flags, int* grid, int* block) {
  if (!func) {
    throw std::runtime_error(
        "Kernel pointer is NULL; you may need to define JITIFY_THREAD_SAFE "
        "1");
  }
  CUresult res = cuOccupancyMaxPotentialBlockSizeWithFlags(
      grid, block, func, smem_callback, *smem, max_block_size, flags);
  if (res != CUDA_SUCCESS) {
    const char* msg;
    cuGetErrorName(res, &msg);
    throw std::runtime_error(msg);
  }
  if (smem_callback) {
    *smem = (unsigned int)smem_callback(*block);
  }
}

}  // namespace detail

//! \endcond

class KernelInstantiation;
class Kernel;
class Program;
class JitCache;

struct ProgramConfig {
  std::vector<std::string> options;
  std::vector<std::string> include_paths;
  std::string name;
  typedef std::map<std::string, std::string> source_map;
  source_map sources;
};

class JitCache_impl {
  friend class Program_impl;
  friend class KernelInstantiation_impl;
  friend class KernelLauncher_impl;
  typedef uint64_t key_type;
  jitify::ObjectCache<key_type, detail::CUDAKernel> _kernel_cache;
  jitify::ObjectCache<key_type, ProgramConfig> _program_config_cache;
  std::vector<std::string> _options;
#if JITIFY_THREAD_SAFE
  std::mutex _kernel_cache_mutex;
  std::mutex _program_cache_mutex;
#endif
 public:
  inline JitCache_impl(size_t cache_size)
      : _kernel_cache(cache_size), _program_config_cache(cache_size) {
    detail::add_options_from_env(_options);

    // Bootstrap the cuda context to avoid errors
    cudaFree(0);
  }
};

class Program_impl {
  // A friendly class
  friend class Kernel_impl;
  friend class KernelLauncher_impl;
  friend class KernelInstantiation_impl;
  // TODO: This can become invalid if JitCache is destroyed before the
  //         Program object is. However, this can't happen if JitCache
  //           instances are static.
  JitCache_impl& _cache;
  uint64_t _hash;
  ProgramConfig* _config;
  void load_sources(std::string source, std::vector<std::string> headers,
                    std::vector<std::string> options,
                    file_callback_type file_callback);

 public:
  inline Program_impl(JitCache_impl& cache, std::string source,
                      jitify::detail::vector<std::string> headers = 0,
                      jitify::detail::vector<std::string> options = 0,
                      file_callback_type file_callback = 0);
  inline Program_impl(Program_impl const&) = default;
  inline Program_impl(Program_impl&&) = default;
  inline std::vector<std::string> const& options() const {
    return _config->options;
  }
  inline std::string const& name() const { return _config->name; }
  inline ProgramConfig::source_map const& sources() const {
    return _config->sources;
  }
  inline std::vector<std::string> const& include_paths() const {
    return _config->include_paths;
  }
};

class Kernel_impl {
  friend class KernelLauncher_impl;
  friend class KernelInstantiation_impl;
  Program_impl _program;
  std::string _name;
  std::vector<std::string> _options;
  uint64_t _hash;

 public:
  inline Kernel_impl(Program_impl const& program, std::string name,
                     jitify::detail::vector<std::string> options = 0);
  inline Kernel_impl(Kernel_impl const&) = default;
  inline Kernel_impl(Kernel_impl&&) = default;
};

class KernelInstantiation_impl {
  friend class KernelLauncher_impl;
  Kernel_impl _kernel;
  uint64_t _hash;
  std::string _template_inst;
  std::vector<std::string> _options;
  detail::CUDAKernel* _cuda_kernel;
  inline void print() const;
  void build_kernel();

 public:
  inline KernelInstantiation_impl(
      Kernel_impl const& kernel, std::vector<std::string> const& template_args);
  inline KernelInstantiation_impl(KernelInstantiation_impl const&) = default;
  inline KernelInstantiation_impl(KernelInstantiation_impl&&) = default;
  detail::CUDAKernel const& cuda_kernel() const { return *_cuda_kernel; }
};

class KernelLauncher_impl {
  KernelInstantiation_impl _kernel_inst;
  dim3 _grid;
  dim3 _block;
  unsigned int _smem;
  cudaStream_t _stream;

 public:
  inline KernelLauncher_impl(KernelInstantiation_impl const& kernel_inst,
                             dim3 grid, dim3 block, unsigned int smem = 0,
                             cudaStream_t stream = 0)
      : _kernel_inst(kernel_inst),
        _grid(grid),
        _block(block),
        _smem(smem),
        _stream(stream) {}
  inline KernelLauncher_impl(KernelLauncher_impl const&) = default;
  inline KernelLauncher_impl(KernelLauncher_impl&&) = default;
  inline CUresult launch(
      jitify::detail::vector<void*> arg_ptrs,
      jitify::detail::vector<std::string> arg_types = 0) const;
  inline void safe_launch(
      jitify::detail::vector<void*> arg_ptrs,
      jitify::detail::vector<std::string> arg_types = 0) const;

 private:
  inline void pre_launch(
      jitify::detail::vector<std::string> arg_types = 0) const;
};

/*! An object representing a configured and instantiated kernel ready
 *    for launching.
 */
class KernelLauncher {
  std::unique_ptr<KernelLauncher_impl const> _impl;

 public:
  KernelLauncher() = default;
  inline KernelLauncher(KernelInstantiation const& kernel_inst, dim3 grid,
                        dim3 block, unsigned int smem = 0,
                        cudaStream_t stream = 0);

  // Note: It's important that there is no implicit conversion required
  //         for arg_ptrs, because otherwise the parameter pack version
  //         below gets called instead (probably resulting in a segfault).
  /*! Launch the kernel.
   *
   *  \param arg_ptrs  A vector of pointers to each function argument for the
   *    kernel.
   *  \param arg_types A vector of function argument types represented
   *    as code-strings. This parameter is optional and is only used to print
   *    out the function signature.
   */
  inline CUresult launch(
      std::vector<void*> arg_ptrs = std::vector<void*>(),
      jitify::detail::vector<std::string> arg_types = 0) const {
    return _impl->launch(arg_ptrs, arg_types);
  }

  /*! Launch the kernel and check for cuda errors.
   *
   *  \see launch
   */
  inline void safe_launch(
      std::vector<void*> arg_ptrs = std::vector<void*>(),
      jitify::detail::vector<std::string> arg_types = 0) const {
    _impl->safe_launch(arg_ptrs, arg_types);
  }

  // Regular function call syntax
  /*! Launch the kernel.
   *
   *  \see launch
   */
  template <typename... ArgTypes>
  inline CUresult operator()(const ArgTypes&... args) const {
    return this->launch(args...);
  }
  /*! Launch the kernel.
   *
   *  \param args Function arguments for the kernel.
   */
  template <typename... ArgTypes>
  inline CUresult launch(const ArgTypes&... args) const {
    return this->launch(std::vector<void*>({(void*)&args...}),
                        {reflection::reflect<ArgTypes>()...});
  }
  /*! Launch the kernel and check for cuda errors.
   *
   *  \param args Function arguments for the kernel.
   */
  template <typename... ArgTypes>
  inline void safe_launch(const ArgTypes&... args) const {
    this->safe_launch(std::vector<void*>({(void*)&args...}),
                      {reflection::reflect<ArgTypes>()...});
  }
};

/*! An object representing a kernel instantiation made up of a Kernel and
 *    template arguments.
 */
class KernelInstantiation {
  friend class KernelLauncher;
  std::unique_ptr<KernelInstantiation_impl const> _impl;

 public:
  KernelInstantiation() = default;
  inline KernelInstantiation(Kernel const& kernel,
                             std::vector<std::string> const& template_args);

  /*! Implicit conversion to the underlying CUfunction object.
   *
   * \note This allows use of CUDA APIs like
   *   cuOccupancyMaxActiveBlocksPerMultiprocessor.
   */
  inline operator CUfunction() const { return _impl->cuda_kernel(); }

  /*! Configure the kernel launch.
   *
   *  \see configure
   */
  inline KernelLauncher operator()(dim3 grid, dim3 block, unsigned int smem = 0,
                                   cudaStream_t stream = 0) const {
    return this->configure(grid, block, smem, stream);
  }
  /*! Configure the kernel launch.
   *
   *  \param grid   The thread grid dimensions for the launch.
   *  \param block  The thread block dimensions for the launch.
   *  \param smem   The amount of shared memory to dynamically allocate, in
   * bytes.
   *  \param stream The CUDA stream to launch the kernel in.
   */
  inline KernelLauncher configure(dim3 grid, dim3 block, unsigned int smem = 0,
                                  cudaStream_t stream = 0) const {
    return KernelLauncher(*this, grid, block, smem, stream);
  }
  /*! Configure the kernel launch with a 1-dimensional block and grid chosen
   *  automatically to maximise occupancy.
   *
   * \param max_block_size  The upper limit on the block size, or 0 for no
   * limit.
   * \param smem  The amount of shared memory to dynamically allocate, in bytes.
   * \param smem_callback  A function returning smem for a given block size (overrides \p smem).
   * \param stream The CUDA stream to launch the kernel in.
   * \param flags The flags to pass to cuOccupancyMaxPotentialBlockSizeWithFlags.
   */
  inline KernelLauncher configure_1d_max_occupancy(
      int max_block_size = 0, unsigned int smem = 0,
      CUoccupancyB2DSize smem_callback = 0, cudaStream_t stream = 0,
      unsigned int flags = 0) const {
    int grid;
    int block;
    CUfunction func = _impl->cuda_kernel();
    detail::get_1d_max_occupancy(func, smem_callback, &smem, max_block_size,
                                 flags, &grid, &block);
    return this->configure(grid, block, smem, stream);
  }

  /*
   * Returns the function attribute requested from the kernel
   */
  inline int get_func_attribute(CUfunction_attribute attribute) const {
    return _impl->cuda_kernel().get_func_attribute(attribute);
  }

  /*
   * Set the function attribute requested for the kernel
   */
  inline void set_func_attribute(CUfunction_attribute attribute,
                                 int value) const {
    _impl->cuda_kernel().set_func_attribute(attribute, value);
  }

  /*
   * \deprecated Use \p get_global_ptr instead.
   */
  inline CUdeviceptr get_constant_ptr(const char* name,
                                      size_t* size = nullptr) const {
    return get_global_ptr(name, size);
  }

  /*
   * Get a device pointer to a global __constant__ or __device__ variable using
   * its un-mangled name. If provided, *size is set to the size of the variable
   * in bytes.
   */
  inline CUdeviceptr get_global_ptr(const char* name,
                                    size_t* size = nullptr) const {
    return _impl->cuda_kernel().get_global_ptr(name, size);
  }

  /*
   * Copy data from a global __constant__ or __device__ array to the host using
   * its un-mangled name.
   */
  template <typename T>
  inline CUresult get_global_array(const char* name, T* data, size_t count,
                                   CUstream stream = 0) const {
    return _impl->cuda_kernel().get_global_data(name, data, count, stream);
  }

  /*
   * Copy a value from a global __constant__ or __device__ variable to the host
   * using its un-mangled name.
   */
  template <typename T>
  inline CUresult get_global_value(const char* name, T* value,
                                   CUstream stream = 0) const {
    return get_global_array(name, value, 1, stream);
  }

  /*
   * Copy data from the host to a global __constant__ or __device__ array using
   * its un-mangled name.
   */
  template <typename T>
  inline CUresult set_global_array(const char* name, const T* data,
                                   size_t count, CUstream stream = 0) const {
    return _impl->cuda_kernel().set_global_data(name, data, count, stream);
  }

  /*
   * Copy a value from the host to a global __constant__ or __device__ variable
   * using its un-mangled name.
   */
  template <typename T>
  inline CUresult set_global_value(const char* name, const T& value,
                                   CUstream stream = 0) const {
    return set_global_array(name, &value, 1, stream);
  }

  const std::string& mangled_name() const {
    return _impl->cuda_kernel().function_name();
  }

  const std::string& ptx() const { return _impl->cuda_kernel().ptx(); }

  const std::vector<std::string>& link_files() const {
    return _impl->cuda_kernel().link_files();
  }

  const std::vector<std::string>& link_paths() const {
    return _impl->cuda_kernel().link_paths();
  }
};

/*! An object representing a kernel made up of a Program, a name and options.
 */
class Kernel {
  friend class KernelInstantiation;
  std::unique_ptr<Kernel_impl const> _impl;

 public:
  Kernel() = default;
  Kernel(Program const& program, std::string name,
         jitify::detail::vector<std::string> options = 0);

  /*! Instantiate the kernel.
   *
   *  \param template_args A vector of template arguments represented as
   *    code-strings. These can be generated using
   *    \code{.cpp}jitify::reflection::reflect<type>()\endcode or
   *    \code{.cpp}jitify::reflection::reflect(value)\endcode
   *
   *  \note Template type deduction is not possible, so all types must be
   *    explicitly specified.
   */
  // inline KernelInstantiation instantiate(std::vector<std::string> const&
  // template_args) const {
  inline KernelInstantiation instantiate(
      std::vector<std::string> const& template_args =
          std::vector<std::string>()) const {
    return KernelInstantiation(*this, template_args);
  }

  // Regular template instantiation syntax (note limited flexibility)
  /*! Instantiate the kernel.
   *
   *  \note The template arguments specified on this function are
   *    used to instantiate the kernel. Non-type template arguments must
   *    be wrapped with
   *    \code{.cpp}jitify::reflection::NonType<type,value>\endcode
   *
   *  \note Template type deduction is not possible, so all types must be
   *    explicitly specified.
   */
  template <typename... TemplateArgs>
  inline KernelInstantiation instantiate() const {
    return this->instantiate(
        std::vector<std::string>({reflection::reflect<TemplateArgs>()...}));
  }
  // Template-like instantiation syntax
  //   E.g., instantiate(myvar,Type<MyType>())(grid,block)
  /*! Instantiate the kernel.
   *
   *  \param targs The template arguments for the kernel, represented as
   *    values. Types must be wrapped with
   *    \code{.cpp}jitify::reflection::Type<type>()\endcode or
   *    \code{.cpp}jitify::reflection::type_of(value)\endcode
   *
   *  \note Template type deduction is not possible, so all types must be
   *    explicitly specified.
   */
  template <typename... TemplateArgs>
  inline KernelInstantiation instantiate(TemplateArgs... targs) const {
    return this->instantiate(
        std::vector<std::string>({reflection::reflect(targs)...}));
  }
};

/*! An object representing a program made up of source code, headers
 *    and options.
 */
class Program {
  friend class Kernel;
  std::unique_ptr<Program_impl const> _impl;

 public:
  Program() = default;
  Program(JitCache& cache, std::string source,
          jitify::detail::vector<std::string> headers = 0,
          jitify::detail::vector<std::string> options = 0,
          file_callback_type file_callback = 0);

  /*! Select a kernel.
   *
   * \param name The name of the kernel (unmangled and without
   * template arguments).
   * \param options A vector of options to be passed to the NVRTC
   * compiler when compiling this kernel.
   */
  inline Kernel kernel(std::string name,
                       jitify::detail::vector<std::string> options = 0) const {
    return Kernel(*this, name, options);
  }
  /*! Select a kernel.
   *
   *  \see kernel
   */
  inline Kernel operator()(
      std::string name, jitify::detail::vector<std::string> options = 0) const {
    return this->kernel(name, options);
  }
};

/*! An object that manages a cache of JIT-compiled CUDA kernels.
 *
 */
class JitCache {
  friend class Program;
  std::unique_ptr<JitCache_impl> _impl;

 public:
  /*! JitCache constructor.
   *  \param cache_size The number of kernels to hold in the cache
   *    before overwriting the least-recently-used ones.
   */
  enum { DEFAULT_CACHE_SIZE = 128 };
  JitCache(size_t cache_size = DEFAULT_CACHE_SIZE)
      : _impl(new JitCache_impl(cache_size)) {}

  /*! Create a program.
   *
   *  \param source A string containing either the source filename or
   *    the source itself; in the latter case, the first line must be
   *    the name of the program.
   *  \param headers A vector of strings representing the source of
   *    each header file required by the program. Each entry can be
   *    either the header filename or the header source itself; in
   *    the latter case, the first line must be the name of the header
   *    (i.e., the name by which the header is #included).
   *  \param options A vector of options to be passed to the
   *    NVRTC compiler. Include paths specified with \p -I
   *    are added to the search paths used by Jitify. The environment
   *    variable JITIFY_OPTIONS can also be used to define additional
   *    options.
   *  \param file_callback A pointer to a callback function that is
   *    invoked whenever a source file needs to be loaded. Inside this
   *    function, the user can either load/specify the source themselves
   *    or defer to Jitify's file-loading mechanisms.
   *  \note Program or header source files referenced by filename are
   *  looked-up using the following mechanisms (in this order):
   *  \note 1) By calling file_callback.
   *  \note 2) By looking for the file embedded in the executable via the GCC
   * linker.
   *  \note 3) By looking for the file in the filesystem.
   *
   *  \note Jitify recursively scans all source files for \p #include
   *  directives and automatically adds them to the set of headers needed
   *  by the program.
   *  If a \p #include directive references a header that cannot be found,
   *  the directive is automatically removed from the source code to prevent
   *  immediate compilation failure. This may result in compilation errors
   *  if the header was required by the program.
   *
   *  \note Jitify automatically includes NVRTC-safe versions of some
   *  standard library headers.
   */
  inline Program program(std::string source,
                         jitify::detail::vector<std::string> headers = 0,
                         jitify::detail::vector<std::string> options = 0,
                         file_callback_type file_callback = 0) {
    return Program(*this, source, headers, options, file_callback);
  }
};

inline Program::Program(JitCache& cache, std::string source,
                        jitify::detail::vector<std::string> headers,
                        jitify::detail::vector<std::string> options,
                        file_callback_type file_callback)
    : _impl(new Program_impl(*cache._impl, source, headers, options,
                             file_callback)) {}

inline Kernel::Kernel(Program const& program, std::string name,
                      jitify::detail::vector<std::string> options)
    : _impl(new Kernel_impl(*program._impl, name, options)) {}

inline KernelInstantiation::KernelInstantiation(
    Kernel const& kernel, std::vector<std::string> const& template_args)
    : _impl(new KernelInstantiation_impl(*kernel._impl, template_args)) {}

inline KernelLauncher::KernelLauncher(KernelInstantiation const& kernel_inst,
                                      dim3 grid, dim3 block, unsigned int smem,
                                      cudaStream_t stream)
    : _impl(new KernelLauncher_impl(*kernel_inst._impl, grid, block, smem,
                                    stream)) {}

inline std::ostream& operator<<(std::ostream& stream, dim3 d) {
  if (d.y == 1 && d.z == 1) {
    stream << d.x;
  } else {
    stream << "(" << d.x << "," << d.y << "," << d.z << ")";
  }
  return stream;
}

inline void KernelLauncher_impl::pre_launch(
    jitify::detail::vector<std::string> arg_types) const {
  (void)arg_types;
#if JITIFY_PRINT_LAUNCH
  Kernel_impl const& kernel = _kernel_inst._kernel;
  std::string arg_types_string =
      (arg_types.empty() ? "..." : reflection::reflect_list(arg_types));
  std::cout << "Launching " << kernel._name << _kernel_inst._template_inst
            << "<<<" << _grid << "," << _block << "," << _smem << "," << _stream
            << ">>>"
            << "(" << arg_types_string << ")" << std::endl;
#endif
  if (!_kernel_inst._cuda_kernel) {
    throw std::runtime_error(
        "Kernel pointer is NULL; you may need to define JITIFY_THREAD_SAFE 1");
  }
}

inline CUresult KernelLauncher_impl::launch(
    jitify::detail::vector<void*> arg_ptrs,
    jitify::detail::vector<std::string> arg_types) const {
  pre_launch(arg_types);
  return _kernel_inst._cuda_kernel->launch(_grid, _block, _smem, _stream,
                                           arg_ptrs);
}

inline void KernelLauncher_impl::safe_launch(
    jitify::detail::vector<void*> arg_ptrs,
    jitify::detail::vector<std::string> arg_types) const {
  pre_launch(arg_types);
  _kernel_inst._cuda_kernel->safe_launch(_grid, _block, _smem, _stream,
                                         arg_ptrs);
}

inline KernelInstantiation_impl::KernelInstantiation_impl(
    Kernel_impl const& kernel, std::vector<std::string> const& template_args)
    : _kernel(kernel), _options(kernel._options) {
  _template_inst =
      (template_args.empty() ? ""
                             : reflection::reflect_template(template_args));
  using detail::hash_combine;
  using detail::hash_larson64;
  _hash = _kernel._hash;
  _hash = hash_combine(_hash, hash_larson64(_template_inst.c_str()));
  JitCache_impl& cache = _kernel._program._cache;
  uint64_t cache_key = _hash;
#if JITIFY_THREAD_SAFE
  std::lock_guard<std::mutex> lock(cache._kernel_cache_mutex);
#endif
  if (cache._kernel_cache.contains(cache_key)) {
#if JITIFY_PRINT_INSTANTIATION
    std::cout << "Found ";
    this->print();
#endif
    _cuda_kernel = &cache._kernel_cache.get(cache_key);
  } else {
#if JITIFY_PRINT_INSTANTIATION
    std::cout << "Building ";
    this->print();
#endif
    _cuda_kernel = &cache._kernel_cache.emplace(cache_key);
    this->build_kernel();
  }
}

inline void KernelInstantiation_impl::print() const {
  std::string options_string = reflection::reflect_list(_options);
  std::cout << _kernel._name << _template_inst << " [" << options_string << "]"
            << std::endl;
}

inline void KernelInstantiation_impl::build_kernel() {
  Program_impl const& program = _kernel._program;

  std::string instantiation = _kernel._name + _template_inst;

  std::string log, ptx, mangled_instantiation;
  std::vector<std::string> linker_files, linker_paths;
  detail::instantiate_kernel(program.name(), program.sources(), instantiation,
                             _options, &log, &ptx, &mangled_instantiation,
                             &linker_files, &linker_paths);

  _cuda_kernel->set(mangled_instantiation.c_str(), ptx.c_str(), linker_files,
                    linker_paths);
}

Kernel_impl::Kernel_impl(Program_impl const& program, std::string name,
                         jitify::detail::vector<std::string> options)
    : _program(program), _name(name), _options(options) {
  // Merge options from parent
  _options.insert(_options.end(), _program.options().begin(),
                  _program.options().end());
  detail::detect_and_add_cuda_arch(_options);
  detail::detect_and_add_cxx11_flag(_options);
  std::string options_string = reflection::reflect_list(_options);
  using detail::hash_combine;
  using detail::hash_larson64;
  _hash = _program._hash;
  _hash = hash_combine(_hash, hash_larson64(_name.c_str()));
  _hash = hash_combine(_hash, hash_larson64(options_string.c_str()));
}

Program_impl::Program_impl(JitCache_impl& cache, std::string source,
                           jitify::detail::vector<std::string> headers,
                           jitify::detail::vector<std::string> options,
                           file_callback_type file_callback)
    : _cache(cache) {
  // Compute hash of source, headers and options
  std::string options_string = reflection::reflect_list(options);
  using detail::hash_combine;
  using detail::hash_larson64;
  _hash = hash_combine(hash_larson64(source.c_str()),
                       hash_larson64(options_string.c_str()));
  for (size_t i = 0; i < headers.size(); ++i) {
    _hash = hash_combine(_hash, hash_larson64(headers[i].c_str()));
  }
  _hash = hash_combine(_hash, (uint64_t)file_callback);
  // Add pre-include built-in JIT-safe headers
  for (int i = 0; i < detail::preinclude_jitsafe_headers_count; ++i) {
    const char* hdr_name = detail::preinclude_jitsafe_header_names[i];
    const std::string& hdr_source =
        detail::get_jitsafe_headers_map().at(hdr_name);
    headers.push_back(std::string(hdr_name) + "\n" + hdr_source);
  }
  // Merge options from parent
  options.insert(options.end(), _cache._options.begin(), _cache._options.end());
  // Load sources
#if JITIFY_THREAD_SAFE
  std::lock_guard<std::mutex> lock(cache._program_cache_mutex);
#endif
  if (!cache._program_config_cache.contains(_hash)) {
    _config = &cache._program_config_cache.insert(_hash);
    this->load_sources(source, headers, options, file_callback);
  } else {
    _config = &cache._program_config_cache.get(_hash);
  }
}

inline void Program_impl::load_sources(std::string source,
                                       std::vector<std::string> headers,
                                       std::vector<std::string> options,
                                       file_callback_type file_callback) {
  _config->options = options;
  detail::load_program(source, headers, file_callback, &_config->include_paths,
                       &_config->sources, &_config->options, &_config->name);
}

enum Location { HOST, DEVICE };

/*! Specifies location and parameters for execution of an algorithm.
 *  \param stream        The CUDA stream on which to execute.
 *  \param headers       A vector of headers to include in the code.
 *  \param options       Options to pass to the NVRTC compiler.
 *  \param file_callback See jitify::Program.
 *  \param block_size    The size of the CUDA thread block with which to
 * execute.
 *  \param cache_size    The number of kernels to store in the cache
 * before overwriting the least-recently-used ones.
 */
struct ExecutionPolicy {
  /*! Location (HOST or DEVICE) on which to execute.*/
  Location location;
  /*! List of headers to include when compiling the algorithm.*/
  std::vector<std::string> headers;
  /*! List of compiler options.*/
  std::vector<std::string> options;
  /*! Optional callback for loading source files.*/
  file_callback_type file_callback;
  /*! CUDA stream on which to execute.*/
  cudaStream_t stream;
  /*! CUDA device on which to execute.*/
  int device;
  /*! CUDA block size with which to execute.*/
  int block_size;
  /*! The number of instantiations to store in the cache before overwriting
   *  the least-recently-used ones.*/
  size_t cache_size;
  ExecutionPolicy(Location location_ = DEVICE,
                  jitify::detail::vector<std::string> headers_ = 0,
                  jitify::detail::vector<std::string> options_ = 0,
                  file_callback_type file_callback_ = 0,
                  cudaStream_t stream_ = 0, int device_ = 0,
                  int block_size_ = 256,
                  size_t cache_size_ = JitCache::DEFAULT_CACHE_SIZE)
      : location(location_),
        headers(headers_),
        options(options_),
        file_callback(file_callback_),
        stream(stream_),
        device(device_),
        block_size(block_size_),
        cache_size(cache_size_) {}
};

template <class Func>
class Lambda;

/*! An object that captures a set of variables for use in a parallel_for
 *    expression. See JITIFY_CAPTURE().
 */
class Capture {
 public:
  std::vector<std::string> _arg_decls;
  std::vector<void*> _arg_ptrs;

 public:
  template <typename... Args>
  inline Capture(std::vector<std::string> arg_names, Args const&... args)
      : _arg_ptrs{(void*)&args...} {
    std::vector<std::string> arg_types = {reflection::reflect<Args>()...};
    _arg_decls.resize(arg_names.size());
    for (int i = 0; i < (int)arg_names.size(); ++i) {
      _arg_decls[i] = arg_types[i] + " " + arg_names[i];
    }
  }
};

/*! An object that captures the instantiated Lambda function for use
    in a parallel_for expression and the function string for NVRTC
    compilation
 */
template <class Func>
class Lambda {
 public:
  Capture _capture;
  std::string _func_string;
  Func _func;

 public:
  inline Lambda(Capture const& capture, std::string func_string, Func func)
      : _capture(capture), _func_string(func_string), _func(func) {}
};

template <typename T>
inline Lambda<T> make_Lambda(Capture const& capture, std::string func,
                             T lambda) {
  return Lambda<T>(capture, func, lambda);
}

#define JITIFY_CAPTURE(...)                                            \
  jitify::Capture(jitify::detail::split_string(#__VA_ARGS__, -1, ","), \
                  __VA_ARGS__)

#define JITIFY_MAKE_LAMBDA(capture, x, ...)               \
  jitify::make_Lambda(capture, std::string(#__VA_ARGS__), \
                      [x](int i) { __VA_ARGS__; })

#define JITIFY_ARGS(...) __VA_ARGS__

#define JITIFY_LAMBDA_(x, ...) \
  JITIFY_MAKE_LAMBDA(JITIFY_CAPTURE(x), JITIFY_ARGS(x), __VA_ARGS__)

// macro sequence to strip surrounding brackets
#define JITIFY_STRIP_PARENS(X) X
#define JITIFY_PASS_PARAMETERS(X) JITIFY_STRIP_PARENS(JITIFY_ARGS X)

/*! Creates a Lambda object with captured variables and a function
 *    definition.
 *  \param capture A bracket-enclosed list of variables to capture.
 *  \param ...     The function definition.
 *
 *  \code{.cpp}
 *  float* capture_me;
 *  int    capture_me_too;
 *  auto my_lambda = JITIFY_LAMBDA( (capture_me, capture_me_too),
 *                                  capture_me[i] = i*capture_me_too );
 *  \endcode
 */
#define JITIFY_LAMBDA(capture, ...)                            \
  JITIFY_LAMBDA_(JITIFY_ARGS(JITIFY_PASS_PARAMETERS(capture)), \
                 JITIFY_ARGS(__VA_ARGS__))

// TODO: Try to implement for_each that accepts iterators instead of indices
//       Add compile guard for NOCUDA compilation
/*! Call a function for a range of indices
 *
 *  \param policy Determines the location and device parameters for
 *  execution of the parallel_for.
 *  \param begin  The starting index.
 *  \param end    The ending index.
 *  \param lambda A Lambda object created using the JITIFY_LAMBDA() macro.
 *
 *  \code{.cpp}
 *  char const* in;
 *  float*      out;
 *  parallel_for(0, 100, JITIFY_LAMBDA( (in, out), {char x = in[i]; out[i] =
 * x*x; } ); \endcode
 */
template <typename IndexType, class Func>
CUresult parallel_for(ExecutionPolicy policy, IndexType begin, IndexType end,
                      Lambda<Func> const& lambda) {
  using namespace jitify;

  if (policy.location == HOST) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (IndexType i = begin; i < end; i++) {
      lambda._func(i);
    }
    return CUDA_SUCCESS;  // FIXME - replace with non-CUDA enum type?
  }

  thread_local static JitCache kernel_cache(policy.cache_size);

  std::vector<std::string> arg_decls;
  arg_decls.push_back("I begin, I end");
  arg_decls.insert(arg_decls.end(), lambda._capture._arg_decls.begin(),
                   lambda._capture._arg_decls.end());

  std::stringstream source_ss;
  source_ss << "parallel_for_program\n";
  for (auto const& header : policy.headers) {
    std::string header_name = header.substr(0, header.find("\n"));
    source_ss << "#include <" << header_name << ">\n";
  }
  source_ss << "template<typename I>\n"
               "__global__\n"
               "void parallel_for_kernel("
            << reflection::reflect_list(arg_decls)
            << ") {\n"
               "	I i0 = threadIdx.x + blockDim.x*blockIdx.x;\n"
               "	for( I i=i0+begin; i<end; i+=blockDim.x*gridDim.x ) {\n"
               "	"
            << "\t" << lambda._func_string << ";\n"
            << "	}\n"
               "}\n";

  Program program = kernel_cache.program(source_ss.str(), policy.headers,
                                         policy.options, policy.file_callback);

  std::vector<void*> arg_ptrs;
  arg_ptrs.push_back(&begin);
  arg_ptrs.push_back(&end);
  arg_ptrs.insert(arg_ptrs.end(), lambda._capture._arg_ptrs.begin(),
                  lambda._capture._arg_ptrs.end());

  size_t n = end - begin;
  dim3 block(policy.block_size);
  dim3 grid((unsigned int)std::min((n - 1) / block.x + 1, size_t(65535)));
  cudaSetDevice(policy.device);
  return program.kernel("parallel_for_kernel")
      .instantiate<IndexType>()
      .configure(grid, block, 0, policy.stream)
      .launch(arg_ptrs);
}

namespace experimental {

using jitify::file_callback_type;

namespace serialization {

namespace detail {

// This should be incremented whenever the serialization format changes in any
// incompatible way.
static constexpr const size_t kSerializationVersion = 2;

inline void serialize(std::ostream& stream, size_t u) {
  uint64_t u64 = u;
  char bytes[8];
  for (int i = 0; i < (int)sizeof(bytes); ++i) {
    // Convert to little-endian bytes.
    bytes[i] = (unsigned char)(u64 >> (i * CHAR_BIT));
  }
  stream.write(bytes, sizeof(bytes));
}

inline bool deserialize(std::istream& stream, size_t* size) {
  char bytes[8];
  stream.read(bytes, sizeof(bytes));
  uint64_t u64 = 0;
  for (int i = 0; i < (int)sizeof(bytes); ++i) {
    // Convert from little-endian bytes.
    u64 |= uint64_t((unsigned char)(bytes[i])) << (i * CHAR_BIT);
  }
  *size = u64;
  return stream.good();
}

inline void serialize(std::ostream& stream, std::string const& s) {
  serialize(stream, s.size());
  stream.write(s.data(), s.size());
}

inline bool deserialize(std::istream& stream, std::string* s) {
  size_t size;
  if (!deserialize(stream, &size)) return false;
  s->resize(size);
  if (s->size()) {
    stream.read(&(*s)[0], s->size());
  }
  return stream.good();
}

inline void serialize(std::ostream& stream, std::vector<std::string> const& v) {
  serialize(stream, v.size());
  for (auto const& s : v) {
    serialize(stream, s);
  }
}

inline bool deserialize(std::istream& stream, std::vector<std::string>* v) {
  size_t size;
  if (!deserialize(stream, &size)) return false;
  v->resize(size);
  for (auto& s : *v) {
    if (!deserialize(stream, &s)) return false;
  }
  return true;
}

inline void serialize(std::ostream& stream,
                      std::map<std::string, std::string> const& m) {
  serialize(stream, m.size());
  for (auto const& kv : m) {
    serialize(stream, kv.first);
    serialize(stream, kv.second);
  }
}

inline bool deserialize(std::istream& stream,
                        std::map<std::string, std::string>* m) {
  size_t size;
  if (!deserialize(stream, &size)) return false;
  for (size_t i = 0; i < size; ++i) {
    std::string key;
    if (!deserialize(stream, &key)) return false;
    if (!deserialize(stream, &(*m)[key])) return false;
  }
  return true;
}

template <typename T, typename... Rest>
inline void serialize(std::ostream& stream, T const& value, Rest... rest) {
  serialize(stream, value);
  serialize(stream, rest...);
}

template <typename T, typename... Rest>
inline bool deserialize(std::istream& stream, T* value, Rest... rest) {
  if (!deserialize(stream, value)) return false;
  return deserialize(stream, rest...);
}

inline void serialize_magic_number(std::ostream& stream) {
  stream.write("JTFY", 4);
  serialize(stream, kSerializationVersion);
}

inline bool deserialize_magic_number(std::istream& stream) {
  char magic_number[4] = {0, 0, 0, 0};
  stream.read(&magic_number[0], 4);
  if (!(magic_number[0] == 'J' && magic_number[1] == 'T' &&
        magic_number[2] == 'F' && magic_number[3] == 'Y')) {
    return false;
  }
  size_t serialization_version;
  if (!deserialize(stream, &serialization_version)) return false;
  return serialization_version == kSerializationVersion;
}

}  // namespace detail

template <typename... Values>
inline std::string serialize(Values const&... values) {
  std::ostringstream ss(std::stringstream::out | std::stringstream::binary);
  detail::serialize_magic_number(ss);
  detail::serialize(ss, values...);
  return ss.str();
}

template <typename... Values>
inline bool deserialize(std::string const& serialized, Values*... values) {
  std::istringstream ss(serialized,
                        std::stringstream::in | std::stringstream::binary);
  if (!detail::deserialize_magic_number(ss)) return false;
  return detail::deserialize(ss, values...);
}

}  // namespace serialization

class Program;
class Kernel;
class KernelInstantiation;
class KernelLauncher;

/*! An object representing a program made up of source code, headers
 *    and options.
 */
class Program {
 private:
  friend class KernelInstantiation;
  std::string _name;
  std::vector<std::string> _options;
  std::map<std::string, std::string> _sources;

  // Private constructor used by deserialize()
  Program() {}

 public:
  /*! Create a program.
   *
   *  \param source A string containing either the source filename or
   *    the source itself; in the latter case, the first line must be
   *    the name of the program.
   *  \param headers A vector of strings representing the source of
   *    each header file required by the program. Each entry can be
   *    either the header filename or the header source itself; in
   *    the latter case, the first line must be the name of the header
   *    (i.e., the name by which the header is #included).
   *  \param options A vector of options to be passed to the
   *    NVRTC compiler. Include paths specified with \p -I
   *    are added to the search paths used by Jitify. The environment
   *    variable JITIFY_OPTIONS can also be used to define additional
   *    options.
   *  \param file_callback A pointer to a callback function that is
   *    invoked whenever a source file needs to be loaded. Inside this
   *    function, the user can either load/specify the source themselves
   *    or defer to Jitify's file-loading mechanisms.
   *  \note Program or header source files referenced by filename are
   *  looked-up using the following mechanisms (in this order):
   *  \note 1) By calling file_callback.
   *  \note 2) By looking for the file embedded in the executable via the GCC
   * linker.
   *  \note 3) By looking for the file in the filesystem.
   *
   *  \note Jitify recursively scans all source files for \p #include
   *  directives and automatically adds them to the set of headers needed
   *  by the program.
   *  If a \p #include directive references a header that cannot be found,
   *  the directive is automatically removed from the source code to prevent
   *  immediate compilation failure. This may result in compilation errors
   *  if the header was required by the program.
   *
   *  \note Jitify automatically includes NVRTC-safe versions of some
   *  standard library headers.
   */
  Program(std::string const& cuda_source,
          std::vector<std::string> const& given_headers = {},
          std::vector<std::string> const& given_options = {},
          file_callback_type file_callback = nullptr) {
    // Add pre-include built-in JIT-safe headers
    std::vector<std::string> headers = given_headers;
    for (int i = 0; i < detail::preinclude_jitsafe_headers_count; ++i) {
      const char* hdr_name = detail::preinclude_jitsafe_header_names[i];
      const std::string& hdr_source =
          detail::get_jitsafe_headers_map().at(hdr_name);
      headers.push_back(std::string(hdr_name) + "\n" + hdr_source);
    }

    _options = given_options;
    detail::add_options_from_env(_options);
    std::vector<std::string> include_paths;
    detail::load_program(cuda_source, headers, file_callback, &include_paths,
                         &_sources, &_options, &_name);
  }

  /*! Restore a serialized program.
   *
   * \param serialized_program The serialized program to restore.
   *
   * \see serialize
   */
  static Program deserialize(std::string const& serialized_program) {
    Program program;
    if (!serialization::deserialize(serialized_program, &program._name,
                                    &program._options, &program._sources)) {
      throw std::runtime_error("Failed to deserialize program");
    }
    return program;
  }

  /*! Save the program.
   *
   * \see deserialize
   */
  std::string serialize() const {
    // Note: Must update kSerializationVersion if this is changed.
    return serialization::serialize(_name, _options, _sources);
  };

  /*! Select a kernel.
   *
   * \param name The name of the kernel (unmangled and without
   * template arguments).
   * \param options A vector of options to be passed to the NVRTC
   * compiler when compiling this kernel.
   */
  Kernel kernel(std::string const& name,
                std::vector<std::string> const& options = {}) const;
};

class Kernel {
  friend class KernelInstantiation;
  Program const* _program;
  std::string _name;
  std::vector<std::string> _options;

 public:
  Kernel(Program const* program, std::string const& name,
         std::vector<std::string> const& options = {})
      : _program(program), _name(name), _options(options) {}

  /*! Instantiate the kernel.
   *
   *  \param template_args A vector of template arguments represented as
   *    code-strings. These can be generated using
   *    \code{.cpp}jitify::reflection::reflect<type>()\endcode or
   *    \code{.cpp}jitify::reflection::reflect(value)\endcode
   *
   *  \note Template type deduction is not possible, so all types must be
   *    explicitly specified.
   */
  KernelInstantiation instantiate(
      std::vector<std::string> const& template_args =
          std::vector<std::string>()) const;

  // Regular template instantiation syntax (note limited flexibility)
  /*! Instantiate the kernel.
   *
   *  \note The template arguments specified on this function are
   *    used to instantiate the kernel. Non-type template arguments must
   *    be wrapped with
   *    \code{.cpp}jitify::reflection::NonType<type,value>\endcode
   *
   *  \note Template type deduction is not possible, so all types must be
   *    explicitly specified.
   */
  template <typename... TemplateArgs>
  KernelInstantiation instantiate() const;

  // Template-like instantiation syntax
  //   E.g., instantiate(myvar,Type<MyType>())(grid,block)
  /*! Instantiate the kernel.
   *
   *  \param targs The template arguments for the kernel, represented as
   *    values. Types must be wrapped with
   *    \code{.cpp}jitify::reflection::Type<type>()\endcode or
   *    \code{.cpp}jitify::reflection::type_of(value)\endcode
   *
   *  \note Template type deduction is not possible, so all types must be
   *    explicitly specified.
   */
  template <typename... TemplateArgs>
  KernelInstantiation instantiate(TemplateArgs... targs) const;
};

class KernelInstantiation {
  friend class KernelLauncher;
  std::unique_ptr<detail::CUDAKernel> _cuda_kernel;

  // Private constructor used by deserialize()
  KernelInstantiation(std::string const& func_name, std::string const& ptx,
                      std::vector<std::string> const& link_files,
                      std::vector<std::string> const& link_paths)
      : _cuda_kernel(new detail::CUDAKernel(func_name.c_str(), ptx.c_str(),
                                            link_files, link_paths)) {}

 public:
  KernelInstantiation(Kernel const& kernel,
                      std::vector<std::string> const& template_args) {
    Program const* program = kernel._program;

    std::string template_inst =
        (template_args.empty() ? ""
                               : reflection::reflect_template(template_args));
    std::string instantiation = kernel._name + template_inst;

    std::vector<std::string> options;
    options.insert(options.begin(), program->_options.begin(),
                   program->_options.end());
    options.insert(options.begin(), kernel._options.begin(),
                   kernel._options.end());
    detail::detect_and_add_cuda_arch(options);
    detail::detect_and_add_cxx11_flag(options);

    std::string log, ptx, mangled_instantiation;
    std::vector<std::string> linker_files, linker_paths;
    detail::instantiate_kernel(program->_name, program->_sources, instantiation,
                               options, &log, &ptx, &mangled_instantiation,
                               &linker_files, &linker_paths);

    _cuda_kernel.reset(new detail::CUDAKernel(mangled_instantiation.c_str(),
                                              ptx.c_str(), linker_files,
                                              linker_paths));
  }

  /*! Implicit conversion to the underlying CUfunction object.
   *
   * \note This allows use of CUDA APIs like
   *   cuOccupancyMaxActiveBlocksPerMultiprocessor.
   */
  operator CUfunction() const { return *_cuda_kernel; }

  /*! Restore a serialized kernel instantiation.
   *
   * \param serialized_kernel_inst The serialized kernel instantiation to
   * restore.
   *
   * \see serialize
   */
  static KernelInstantiation deserialize(
      std::string const& serialized_kernel_inst) {
    std::string func_name, ptx;
    std::vector<std::string> link_files, link_paths;
    if (!serialization::deserialize(serialized_kernel_inst, &func_name, &ptx,
                                    &link_files, &link_paths)) {
      throw std::runtime_error("Failed to deserialize kernel instantiation");
    }
    return KernelInstantiation(func_name, ptx, link_files, link_paths);
  }

  /*! Save the program.
   *
   * \see deserialize
   */
  std::string serialize() const {
    // Note: Must update kSerializationVersion if this is changed.
    return serialization::serialize(
        _cuda_kernel->function_name(), _cuda_kernel->ptx(),
        _cuda_kernel->link_files(), _cuda_kernel->link_paths());
  }

  /*! Configure the kernel launch.
   *
   *  \param grid   The thread grid dimensions for the launch.
   *  \param block  The thread block dimensions for the launch.
   *  \param smem   The amount of shared memory to dynamically allocate, in
   * bytes.
   *  \param stream The CUDA stream to launch the kernel in.
   */
  KernelLauncher configure(dim3 grid, dim3 block, unsigned int smem = 0,
                           cudaStream_t stream = 0) const;

  /*! Configure the kernel launch with a 1-dimensional block and grid chosen
   *  automatically to maximise occupancy.
   *
   * \param max_block_size  The upper limit on the block size, or 0 for no
   * limit.
   * \param smem  The amount of shared memory to dynamically allocate, in bytes.
   * \param smem_callback  A function returning smem for a given block size
   * (overrides \p smem).
   * \param stream The CUDA stream to launch the kernel in.
   * \param flags The flags to pass to
   * cuOccupancyMaxPotentialBlockSizeWithFlags.
   */
  KernelLauncher configure_1d_max_occupancy(
      int max_block_size = 0, unsigned int smem = 0,
      CUoccupancyB2DSize smem_callback = 0, cudaStream_t stream = 0,
      unsigned int flags = 0) const;

  /*
   * Returns the function attribute requested from the kernel
   */
  inline int get_func_attribute(CUfunction_attribute attribute) const {
    return _cuda_kernel->get_func_attribute(attribute);
  }

  /*
   * Set the function attribute requested for the kernel
   */
  inline void set_func_attribute(CUfunction_attribute attribute,
                                 int value) const {
    _cuda_kernel->set_func_attribute(attribute, value);
  }

  /*
   * \deprecated Use \p get_global_ptr instead.
   */
  CUdeviceptr get_constant_ptr(const char* name, size_t* size = nullptr) const {
    return get_global_ptr(name, size);
  }

  /*
   * Get a device pointer to a global __constant__ or __device__ variable using
   * its un-mangled name. If provided, *size is set to the size of the variable
   * in bytes.
   */
  CUdeviceptr get_global_ptr(const char* name, size_t* size = nullptr) const {
    return _cuda_kernel->get_global_ptr(name, size);
  }

  /*
   * Copy data from a global __constant__ or __device__ array to the host using
   * its un-mangled name.
   */
  template <typename T>
  CUresult get_global_array(const char* name, T* data, size_t count,
                            CUstream stream = 0) const {
    return _cuda_kernel->get_global_data(name, data, count, stream);
  }

  /*
   * Copy a value from a global __constant__ or __device__ variable to the host
   * using its un-mangled name.
   */
  template <typename T>
  CUresult get_global_value(const char* name, T* value,
                            CUstream stream = 0) const {
    return get_global_array(name, value, 1, stream);
  }

  /*
   * Copy data from the host to a global __constant__ or __device__ array using
   * its un-mangled name.
   */
  template <typename T>
  CUresult set_global_array(const char* name, const T* data, size_t count,
                            CUstream stream = 0) const {
    return _cuda_kernel->set_global_data(name, data, count, stream);
  }

  /*
   * Copy a value from the host to a global __constant__ or __device__ variable
   * using its un-mangled name.
   */
  template <typename T>
  CUresult set_global_value(const char* name, const T& value,
                            CUstream stream = 0) const {
    return set_global_array(name, &value, 1, stream);
  }

  const std::string& mangled_name() const {
    return _cuda_kernel->function_name();
  }

  const std::string& ptx() const { return _cuda_kernel->ptx(); }

  const std::vector<std::string>& link_files() const {
    return _cuda_kernel->link_files();
  }

  const std::vector<std::string>& link_paths() const {
    return _cuda_kernel->link_paths();
  }
};

class KernelLauncher {
  KernelInstantiation const* _kernel_inst;
  dim3 _grid;
  dim3 _block;
  unsigned int _smem;
  cudaStream_t _stream;

 private:
  void pre_launch(std::vector<std::string> arg_types = {}) const {
    (void)arg_types;
#if JITIFY_PRINT_LAUNCH
    std::string arg_types_string =
        (arg_types.empty() ? "..." : reflection::reflect_list(arg_types));
    std::cout << "Launching " << _kernel_inst->_cuda_kernel->function_name()
              << "<<<" << _grid << "," << _block << "," << _smem << ","
              << _stream << ">>>"
              << "(" << arg_types_string << ")" << std::endl;
#endif
  }

 public:
  KernelLauncher(KernelInstantiation const* kernel_inst, dim3 grid, dim3 block,
                 unsigned int smem = 0, cudaStream_t stream = 0)
      : _kernel_inst(kernel_inst),
        _grid(grid),
        _block(block),
        _smem(smem),
        _stream(stream) {}

  // Note: It's important that there is no implicit conversion required
  //         for arg_ptrs, because otherwise the parameter pack version
  //         below gets called instead (probably resulting in a segfault).
  /*! Launch the kernel.
   *
   *  \param arg_ptrs  A vector of pointers to each function argument for the
   *    kernel.
   *  \param arg_types A vector of function argument types represented
   *    as code-strings. This parameter is optional and is only used to print
   *    out the function signature.
   */
  CUresult launch(std::vector<void*> arg_ptrs = {},
                  std::vector<std::string> arg_types = {}) const {
    pre_launch(arg_types);
    return _kernel_inst->_cuda_kernel->launch(_grid, _block, _smem, _stream,
                                              arg_ptrs);
  }

  void safe_launch(std::vector<void*> arg_ptrs = {},
                   std::vector<std::string> arg_types = {}) const {
    pre_launch(arg_types);
    _kernel_inst->_cuda_kernel->safe_launch(_grid, _block, _smem, _stream,
                                            arg_ptrs);
  }

  /*! Launch the kernel.
   *
   *  \param args Function arguments for the kernel.
   */
  template <typename... ArgTypes>
  CUresult launch(const ArgTypes&... args) const {
    return this->launch(std::vector<void*>({(void*)&args...}),
                        {reflection::reflect<ArgTypes>()...});
  }

  /*! Launch the kernel and check for cuda errors.
   *
   *  \param args Function arguments for the kernel.
   */
  template <typename... ArgTypes>
  void safe_launch(const ArgTypes&... args) const {
    return this->safe_launch(std::vector<void*>({(void*)&args...}),
                             {reflection::reflect<ArgTypes>()...});
  }
};

inline Kernel Program::kernel(std::string const& name,
                              std::vector<std::string> const& options) const {
  return Kernel(this, name, options);
}

inline KernelInstantiation Kernel::instantiate(
    std::vector<std::string> const& template_args) const {
  return KernelInstantiation(*this, template_args);
}

template <typename... TemplateArgs>
inline KernelInstantiation Kernel::instantiate() const {
  return this->instantiate(
      std::vector<std::string>({reflection::reflect<TemplateArgs>()...}));
}

template <typename... TemplateArgs>
inline KernelInstantiation Kernel::instantiate(TemplateArgs... targs) const {
  return this->instantiate(
      std::vector<std::string>({reflection::reflect(targs)...}));
}

inline KernelLauncher KernelInstantiation::configure(
    dim3 grid, dim3 block, unsigned int smem, cudaStream_t stream) const {
  return KernelLauncher(this, grid, block, smem, stream);
}

inline KernelLauncher KernelInstantiation::configure_1d_max_occupancy(
    int max_block_size, unsigned int smem, CUoccupancyB2DSize smem_callback,
    cudaStream_t stream, unsigned int flags) const {
  int grid;
  int block;
  CUfunction func = *_cuda_kernel;
  detail::get_1d_max_occupancy(func, smem_callback, &smem, max_block_size,
                               flags, &grid, &block);
  return this->configure(grid, block, smem, stream);
}

}  // namespace experimental

}  // namespace jitify

#if defined(_WIN32) || defined(_WIN64)
#pragma pop_macro("max")
#pragma pop_macro("min")
#pragma pop_macro("strtok_r")
#endif

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

#	if !defined(cublasSafeCall)
#		define cublasSafeCall(err)                                                                \
			LR_ASSERT_ALWAYS(                                                                      \
			  (err) == CUBLAS_STATUS_SUCCESS, "cuBLAS error: {}", getCublasErrorEnum_(err))
#	endif

//********************//
//  CUDA ERROR CHECK  //
//********************//

#	if !defined(cudaSafeCall)
#		define cudaSafeCall(err)                                                                  \
			LR_ASSERT_ALWAYS(!(err), "CUDA error: {}", cudaGetErrorString(err))
#	endif

#	define jitifyCall(call)                                                                       \
		do {                                                                                       \
			if ((call) != CUDA_SUCCESS) {                                                          \
				const char *str;                                                                   \
				cuGetErrorName(call, &str);                                                        \
				throw std::runtime_error(std::string("CUDA JIT failed: ") + str);                  \
			}                                                                                      \
		} while (0)

#	ifdef _MSC_VER
#		pragma warning(default : 4996)
#	endif

/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions for initialization and error checking

#ifndef COMMON_HELPER_CUDA_H_
#define COMMON_HELPER_CUDA_H_

/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// These are helper functions for the SDK samples (string parsing, timers, etc)
#ifndef COMMON_HELPER_STRING_H_
#define COMMON_HELPER_STRING_H_

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#	ifndef _CRT_SECURE_NO_DEPRECATE
#		define _CRT_SECURE_NO_DEPRECATE
#	endif
#	ifndef STRCASECMP
#		define STRCASECMP _stricmp
#	endif
#	ifndef STRNCASECMP
#		define STRNCASECMP _strnicmp
#	endif
#	ifndef STRCPY
#		define STRCPY(sFilePath, nLength, sPath) strcpy_s(sFilePath, nLength, sPath)
#	endif

#	ifndef FOPEN
#		define FOPEN(fHandle, filename, mode) fopen_s(&fHandle, filename, mode)
#	endif
#	ifndef FOPEN_FAIL
#		define FOPEN_FAIL(result) (result != 0)
#	endif
#	ifndef SSCANF
#		define SSCANF sscanf_s
#	endif
#	ifndef SPRINTF
#		define SPRINTF sprintf_s
#	endif
#else // Linux Includes
#	include <string.h>
#	include <strings.h>

#	ifndef STRCASECMP
#		define STRCASECMP strcasecmp
#	endif
#	ifndef STRNCASECMP
#		define STRNCASECMP strncasecmp
#	endif
#	ifndef STRCPY
#		define STRCPY(sFilePath, nLength, sPath) strcpy(sFilePath, sPath)
#	endif

#	ifndef FOPEN
#		define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#	endif
#	ifndef FOPEN_FAIL
#		define FOPEN_FAIL(result) (result == NULL)
#	endif
#	ifndef SSCANF
#		define SSCANF sscanf
#	endif
#	ifndef SPRINTF
#		define SPRINTF sprintf
#	endif
#endif

#ifndef EXIT_WAIVED
#	define EXIT_WAIVED 2
#endif

// CUDA Utility Helper Functions
inline int stringRemoveDelimiter(char delimiter, const char *string) {
	int string_start = 0;

	while (string[string_start] == delimiter) { string_start++; }

	if (string_start >= static_cast<int>(strlen(string) - 1)) { return 0; }

	return string_start;
}

inline int getFileExtension(char *filename, char **extension) {
	int string_length = static_cast<int>(strlen(filename));

	while (filename[string_length--] != '.') {
		if (string_length == 0) break;
	}

	if (string_length > 0) string_length += 2;

	if (string_length == 0)
		*extension = NULL;
	else
		*extension = &filename[string_length];

	return string_length;
}

inline bool checkCmdLineFlag(const int argc, const char **argv, const char *string_ref) {
	bool bFound = false;

	if (argc >= 1) {
		for (int i = 1; i < argc; i++) {
			int string_start		= stringRemoveDelimiter('-', argv[i]);
			const char *string_argv = &argv[i][string_start];

			const char *equal_pos = strchr(string_argv, '=');
			int argv_length =
			  static_cast<int>(equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

			int length = static_cast<int>(strlen(string_ref));

			if (length == argv_length && !STRNCASECMP(string_argv, string_ref, length)) {
				bFound = true;
				continue;
			}
		}
	}

	return bFound;
}

// This function wraps the CUDA Driver API into a template function
template<class T>
inline bool getCmdLineArgumentValue(const int argc, const char **argv, const char *string_ref,
									T *value) {
	bool bFound = false;

	if (argc >= 1) {
		for (int i = 1; i < argc; i++) {
			int string_start		= stringRemoveDelimiter('-', argv[i]);
			const char *string_argv = &argv[i][string_start];
			int length				= static_cast<int>(strlen(string_ref));

			if (!STRNCASECMP(string_argv, string_ref, length)) {
				if (length + 1 <= static_cast<int>(strlen(string_argv))) {
					int auto_inc = (string_argv[length] == '=') ? 1 : 0;
					*value		 = (T)atoi(&string_argv[length + auto_inc]);
				}

				bFound = true;
				i	   = argc;
			}
		}
	}

	return bFound;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref) {
	bool bFound = false;
	int value	= -1;

	if (argc >= 1) {
		for (int i = 1; i < argc; i++) {
			int string_start		= stringRemoveDelimiter('-', argv[i]);
			const char *string_argv = &argv[i][string_start];
			int length				= static_cast<int>(strlen(string_ref));

			if (!STRNCASECMP(string_argv, string_ref, length)) {
				if (length + 1 <= static_cast<int>(strlen(string_argv))) {
					int auto_inc = (string_argv[length] == '=') ? 1 : 0;
					value		 = atoi(&string_argv[length + auto_inc]);
				} else {
					value = 0;
				}

				bFound = true;
				continue;
			}
		}
	}

	if (bFound) {
		return value;
	} else {
		return 0;
	}
}

inline float getCmdLineArgumentFloat(const int argc, const char **argv, const char *string_ref) {
	bool bFound = false;
	float value = -1;

	if (argc >= 1) {
		for (int i = 1; i < argc; i++) {
			int string_start		= stringRemoveDelimiter('-', argv[i]);
			const char *string_argv = &argv[i][string_start];
			int length				= static_cast<int>(strlen(string_ref));

			if (!STRNCASECMP(string_argv, string_ref, length)) {
				if (length + 1 <= static_cast<int>(strlen(string_argv))) {
					int auto_inc = (string_argv[length] == '=') ? 1 : 0;
					value		 = static_cast<float>(atof(&string_argv[length + auto_inc]));
				} else {
					value = 0.f;
				}

				bFound = true;
				continue;
			}
		}
	}

	if (bFound) {
		return value;
	} else {
		return 0;
	}
}

inline bool getCmdLineArgumentString(const int argc, const char **argv, const char *string_ref,
									 char **string_retval) {
	bool bFound = false;

	if (argc >= 1) {
		for (int i = 1; i < argc; i++) {
			int string_start  = stringRemoveDelimiter('-', argv[i]);
			char *string_argv = const_cast<char *>(&argv[i][string_start]);
			int length		  = static_cast<int>(strlen(string_ref));

			if (!STRNCASECMP(string_argv, string_ref, length)) {
				*string_retval = &string_argv[length + 1];
				bFound		   = true;
				continue;
			}
		}
	}

	if (!bFound) { *string_retval = NULL; }

	return bFound;
}

//////////////////////////////////////////////////////////////////////////////
//! Find the path for a file assuming that
//! files are found in the searchPath.
//!
//! @return the path if succeeded, otherwise 0
//! @param filename         name of the file
//! @param executable_path  optional absolute path of the executable
//////////////////////////////////////////////////////////////////////////////
inline char *sdkFindFilePath(const char *filename, const char *executable_path) {
	// <executable_name> defines a variable that is replaced with the name of
	// the executable

	// Typical relative search paths to locate needed companion files (e.g.
	// sample input data, or JIT source files) The origin for the relative
	// search may be the .exe file, a .bat file launching an .exe, a browser
	// .exe launching the .exe or .bat, etc
	const char *searchPath[] = {
	  "./",											 // same dir
	  "./data/",									 // same dir
	  "../../../../Samples/<executable_name>/",		 // up 4 in tree
	  "../../../Samples/<executable_name>/",		 // up 3 in tree
	  "../../Samples/<executable_name>/",			 // up 2 in tree
	  "../../../../Samples/<executable_name>/data/", // up 4 in tree
	  "../../../Samples/<executable_name>/data/",	 // up 3 in tree
	  "../../Samples/<executable_name>/data/",		 // up 2 in tree
	  "../../../../Common/data/",					 // up 4 in tree
	  "../../../Common/data/",						 // up 3 in tree
	  "../../Common/data/"							 // up 2 in tree
	};

	// Extract the executable name
	std::string executable_name;

	if (executable_path != 0) {
		executable_name = std::string(executable_path);

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
		// Windows path delimiter
		size_t delimiter_pos = executable_name.find_last_of('\\');
		executable_name.erase(0, delimiter_pos + 1);

		if (executable_name.rfind(".exe") != std::string::npos) {
			// we strip .exe, only if the .exe is found
			executable_name.resize(executable_name.size() - 4);
		}

#else
		// Linux & OSX path delimiter
		size_t delimiter_pos = executable_name.find_last_of('/');
		executable_name.erase(0, delimiter_pos + 1);
#endif
	}

	// Loop over all search paths and return the first hit
	for (unsigned int i = 0; i < sizeof(searchPath) / sizeof(char *); ++i) {
		std::string path(searchPath[i]);
		size_t executable_name_pos = path.find("<executable_name>");

		// If there is executable_name variable in the searchPath
		// replace it with the value
		if (executable_name_pos != std::string::npos) {
			if (executable_path != 0) {
				path.replace(executable_name_pos, strlen("<executable_name>"), executable_name);
			} else {
				// Skip this path entry if no executable argument is given
				continue;
			}
		}

#ifdef _DEBUG
		printf("sdkFindFilePath <%s> in %s\n", filename, path.c_str());
#endif

		// Test if the file exists
		path.append(filename);
		FILE *fp;
		FOPEN(fp, path.c_str(), "rb");

		if (fp != NULL) {
			fclose(fp);
			// File found
			// returning an allocated array here for backwards compatibility
			// reasons
			char *file_path = reinterpret_cast<char *>(malloc(path.length() + 1));
			STRCPY(file_path, path.length() + 1, path.c_str());
			return file_path;
		}

		if (fp) { fclose(fp); }
	}

	// File not found
	return 0;
}

#endif // COMMON_HELPER_STRING_H_

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef EXIT_WAIVED
#	define EXIT_WAIVED 2
#endif

// Note, it is required that your SDK sample to include the proper header
// files, please refer the CUDA examples for examples of the needed CUDA
// headers, which may change depending on which CUDA functions are used.

// CUDA Runtime error messages
#ifdef __DRIVER_TYPES_H__

static const char *_cudaGetErrorEnum(cudaError_t error) { return cudaGetErrorName(error); }

#endif

#ifdef CUDA_DRIVER_API
// CUDA Driver API errors
static const char *_cudaGetErrorEnum(CUresult error) {
	static char unknown[] = "<unknown>";
	const char *ret		  = NULL;
	cuGetErrorName(error, &ret);
	return ret ? ret : unknown;
}
#endif

#ifdef CUBLAS_API_H_

// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error) {
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

	return "<unknown>";
}

#endif

#ifdef _CUFFT_H_
// cuFFT API errors
static const char *_cudaGetErrorEnum(cufftResult error) {
	switch (error) {
		case CUFFT_SUCCESS: return "CUFFT_SUCCESS";

		case CUFFT_INVALID_PLAN: return "CUFFT_INVALID_PLAN";

		case CUFFT_ALLOC_FAILED: return "CUFFT_ALLOC_FAILED";

		case CUFFT_INVALID_TYPE: return "CUFFT_INVALID_TYPE";

		case CUFFT_INVALID_VALUE: return "CUFFT_INVALID_VALUE";

		case CUFFT_INTERNAL_ERROR: return "CUFFT_INTERNAL_ERROR";

		case CUFFT_EXEC_FAILED: return "CUFFT_EXEC_FAILED";

		case CUFFT_SETUP_FAILED: return "CUFFT_SETUP_FAILED";

		case CUFFT_INVALID_SIZE: return "CUFFT_INVALID_SIZE";

		case CUFFT_UNALIGNED_DATA: return "CUFFT_UNALIGNED_DATA";

		case CUFFT_INCOMPLETE_PARAMETER_LIST: return "CUFFT_INCOMPLETE_PARAMETER_LIST";

		case CUFFT_INVALID_DEVICE: return "CUFFT_INVALID_DEVICE";

		case CUFFT_PARSE_ERROR: return "CUFFT_PARSE_ERROR";

		case CUFFT_NO_WORKSPACE: return "CUFFT_NO_WORKSPACE";

		case CUFFT_NOT_IMPLEMENTED: return "CUFFT_NOT_IMPLEMENTED";

		case CUFFT_LICENSE_ERROR: return "CUFFT_LICENSE_ERROR";

		case CUFFT_NOT_SUPPORTED: return "CUFFT_NOT_SUPPORTED";
	}

	return "<unknown>";
}
#endif

#ifdef CUSPARSEAPI
// cuSPARSE API errors
static const char *_cudaGetErrorEnum(cusparseStatus_t error) {
	switch (error) {
		case CUSPARSE_STATUS_SUCCESS: return "CUSPARSE_STATUS_SUCCESS";

		case CUSPARSE_STATUS_NOT_INITIALIZED: return "CUSPARSE_STATUS_NOT_INITIALIZED";

		case CUSPARSE_STATUS_ALLOC_FAILED: return "CUSPARSE_STATUS_ALLOC_FAILED";

		case CUSPARSE_STATUS_INVALID_VALUE: return "CUSPARSE_STATUS_INVALID_VALUE";

		case CUSPARSE_STATUS_ARCH_MISMATCH: return "CUSPARSE_STATUS_ARCH_MISMATCH";

		case CUSPARSE_STATUS_MAPPING_ERROR: return "CUSPARSE_STATUS_MAPPING_ERROR";

		case CUSPARSE_STATUS_EXECUTION_FAILED: return "CUSPARSE_STATUS_EXECUTION_FAILED";

		case CUSPARSE_STATUS_INTERNAL_ERROR: return "CUSPARSE_STATUS_INTERNAL_ERROR";

		case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
	}

	return "<unknown>";
}
#endif

#ifdef CUSOLVER_COMMON_H_
// cuSOLVER API errors
static const char *_cudaGetErrorEnum(cusolverStatus_t error) {
	switch (error) {
		case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
		case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
		case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
		case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
		case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
		case CUSOLVER_STATUS_MAPPING_ERROR: return "CUSOLVER_STATUS_MAPPING_ERROR";
		case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
		case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
		case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
		case CUSOLVER_STATUS_NOT_SUPPORTED: return "CUSOLVER_STATUS_NOT_SUPPORTED ";
		case CUSOLVER_STATUS_ZERO_PIVOT: return "CUSOLVER_STATUS_ZERO_PIVOT";
		case CUSOLVER_STATUS_INVALID_LICENSE: return "CUSOLVER_STATUS_INVALID_LICENSE";
	}

	return "<unknown>";
}
#endif

#ifdef CURAND_H_

// cuRAND API errors
static const char *_cudaGetErrorEnum(curandStatus_t error) {
	switch (error) {
		case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";

		case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";

		case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";

		case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";

		case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";

		case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";

		case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";

		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
			return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";

		case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";

		case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";

		case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";

		case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";

		case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}

#endif

#ifdef NVJPEGAPI
// nvJPEG API errors
static const char *_cudaGetErrorEnum(nvjpegStatus_t error) {
	switch (error) {
		case NVJPEG_STATUS_SUCCESS: return "NVJPEG_STATUS_SUCCESS";

		case NVJPEG_STATUS_NOT_INITIALIZED: return "NVJPEG_STATUS_NOT_INITIALIZED";

		case NVJPEG_STATUS_INVALID_PARAMETER: return "NVJPEG_STATUS_INVALID_PARAMETER";

		case NVJPEG_STATUS_BAD_JPEG: return "NVJPEG_STATUS_BAD_JPEG";

		case NVJPEG_STATUS_JPEG_NOT_SUPPORTED: return "NVJPEG_STATUS_JPEG_NOT_SUPPORTED";

		case NVJPEG_STATUS_ALLOCATOR_FAILURE: return "NVJPEG_STATUS_ALLOCATOR_FAILURE";

		case NVJPEG_STATUS_EXECUTION_FAILED: return "NVJPEG_STATUS_EXECUTION_FAILED";

		case NVJPEG_STATUS_ARCH_MISMATCH: return "NVJPEG_STATUS_ARCH_MISMATCH";

		case NVJPEG_STATUS_INTERNAL_ERROR: return "NVJPEG_STATUS_INTERNAL_ERROR";
	}

	return "<unknown>";
}
#endif

#ifdef NV_NPPIDEFS_H
// NPP API errors
static const char *_cudaGetErrorEnum(NppStatus error) {
	switch (error) {
		case NPP_NOT_SUPPORTED_MODE_ERROR: return "NPP_NOT_SUPPORTED_MODE_ERROR";

		case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR: return "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";

		case NPP_RESIZE_NO_OPERATION_ERROR: return "NPP_RESIZE_NO_OPERATION_ERROR";

		case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY: return "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";

#	if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

		case NPP_BAD_ARG_ERROR: return "NPP_BAD_ARGUMENT_ERROR";

		case NPP_COEFF_ERROR: return "NPP_COEFFICIENT_ERROR";

		case NPP_RECT_ERROR: return "NPP_RECTANGLE_ERROR";

		case NPP_QUAD_ERROR: return "NPP_QUADRANGLE_ERROR";

		case NPP_MEM_ALLOC_ERR: return "NPP_MEMORY_ALLOCATION_ERROR";

		case NPP_HISTO_NUMBER_OF_LEVELS_ERROR: return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

		case NPP_INVALID_INPUT: return "NPP_INVALID_INPUT";

		case NPP_POINTER_ERROR: return "NPP_POINTER_ERROR";

		case NPP_WARNING: return "NPP_WARNING";

		case NPP_ODD_ROI_WARNING: return "NPP_ODD_ROI_WARNING";
#	else

		// These are for CUDA 5.5 or higher
		case NPP_BAD_ARGUMENT_ERROR: return "NPP_BAD_ARGUMENT_ERROR";

		case NPP_COEFFICIENT_ERROR: return "NPP_COEFFICIENT_ERROR";

		case NPP_RECTANGLE_ERROR: return "NPP_RECTANGLE_ERROR";

		case NPP_QUADRANGLE_ERROR: return "NPP_QUADRANGLE_ERROR";

		case NPP_MEMORY_ALLOCATION_ERR: return "NPP_MEMORY_ALLOCATION_ERROR";

		case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR: return "NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR";

		case NPP_INVALID_HOST_POINTER_ERROR: return "NPP_INVALID_HOST_POINTER_ERROR";

		case NPP_INVALID_DEVICE_POINTER_ERROR: return "NPP_INVALID_DEVICE_POINTER_ERROR";
#	endif

		case NPP_LUT_NUMBER_OF_LEVELS_ERROR: return "NPP_LUT_NUMBER_OF_LEVELS_ERROR";

		case NPP_TEXTURE_BIND_ERROR: return "NPP_TEXTURE_BIND_ERROR";

		case NPP_WRONG_INTERSECTION_ROI_ERROR: return "NPP_WRONG_INTERSECTION_ROI_ERROR";

		case NPP_NOT_EVEN_STEP_ERROR: return "NPP_NOT_EVEN_STEP_ERROR";

		case NPP_INTERPOLATION_ERROR: return "NPP_INTERPOLATION_ERROR";

		case NPP_RESIZE_FACTOR_ERROR: return "NPP_RESIZE_FACTOR_ERROR";

		case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR: return "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";

#	if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) <= 0x5000

		case NPP_MEMFREE_ERR: return "NPP_MEMFREE_ERR";

		case NPP_MEMSET_ERR: return "NPP_MEMSET_ERR";

		case NPP_MEMCPY_ERR: return "NPP_MEMCPY_ERROR";

		case NPP_MIRROR_FLIP_ERR: return "NPP_MIRROR_FLIP_ERR";
#	else

		case NPP_MEMFREE_ERROR: return "NPP_MEMFREE_ERROR";

		case NPP_MEMSET_ERROR: return "NPP_MEMSET_ERROR";

		case NPP_MEMCPY_ERROR: return "NPP_MEMCPY_ERROR";

		case NPP_MIRROR_FLIP_ERROR: return "NPP_MIRROR_FLIP_ERROR";
#	endif

		case NPP_ALIGNMENT_ERROR: return "NPP_ALIGNMENT_ERROR";

		case NPP_STEP_ERROR: return "NPP_STEP_ERROR";

		case NPP_SIZE_ERROR: return "NPP_SIZE_ERROR";

		case NPP_NULL_POINTER_ERROR: return "NPP_NULL_POINTER_ERROR";

		case NPP_CUDA_KERNEL_EXECUTION_ERROR: return "NPP_CUDA_KERNEL_EXECUTION_ERROR";

		case NPP_NOT_IMPLEMENTED_ERROR: return "NPP_NOT_IMPLEMENTED_ERROR";

		case NPP_ERROR: return "NPP_ERROR";

		case NPP_SUCCESS: return "NPP_SUCCESS";

		case NPP_WRONG_INTERSECTION_QUAD_WARNING: return "NPP_WRONG_INTERSECTION_QUAD_WARNING";

		case NPP_MISALIGNED_DST_ROI_WARNING: return "NPP_MISALIGNED_DST_ROI_WARNING";

		case NPP_AFFINE_QUAD_INCORRECT_WARNING: return "NPP_AFFINE_QUAD_INCORRECT_WARNING";

		case NPP_DOUBLE_SIZE_WARNING: return "NPP_DOUBLE_SIZE_WARNING";

		case NPP_WRONG_INTERSECTION_ROI_WARNING: return "NPP_WRONG_INTERSECTION_ROI_WARNING";

#	if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x6000
		/* These are 6.0 or higher */
		case NPP_LUT_PALETTE_BITSIZE_ERROR: return "NPP_LUT_PALETTE_BITSIZE_ERROR";

		case NPP_ZC_MODE_NOT_SUPPORTED_ERROR: return "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";

		case NPP_QUALITY_INDEX_ERROR: return "NPP_QUALITY_INDEX_ERROR";

		case NPP_CHANNEL_ORDER_ERROR: return "NPP_CHANNEL_ORDER_ERROR";

		case NPP_ZERO_MASK_VALUE_ERROR: return "NPP_ZERO_MASK_VALUE_ERROR";

		case NPP_NUMBER_OF_CHANNELS_ERROR: return "NPP_NUMBER_OF_CHANNELS_ERROR";

		case NPP_COI_ERROR: return "NPP_COI_ERROR";

		case NPP_DIVISOR_ERROR: return "NPP_DIVISOR_ERROR";

		case NPP_CHANNEL_ERROR: return "NPP_CHANNEL_ERROR";

		case NPP_STRIDE_ERROR: return "NPP_STRIDE_ERROR";

		case NPP_ANCHOR_ERROR: return "NPP_ANCHOR_ERROR";

		case NPP_MASK_SIZE_ERROR: return "NPP_MASK_SIZE_ERROR";

		case NPP_MOMENT_00_ZERO_ERROR: return "NPP_MOMENT_00_ZERO_ERROR";

		case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR: return "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";

		case NPP_THRESHOLD_ERROR: return "NPP_THRESHOLD_ERROR";

		case NPP_CONTEXT_MATCH_ERROR: return "NPP_CONTEXT_MATCH_ERROR";

		case NPP_FFT_FLAG_ERROR: return "NPP_FFT_FLAG_ERROR";

		case NPP_FFT_ORDER_ERROR: return "NPP_FFT_ORDER_ERROR";

		case NPP_SCALE_RANGE_ERROR: return "NPP_SCALE_RANGE_ERROR";

		case NPP_DATA_TYPE_ERROR: return "NPP_DATA_TYPE_ERROR";

		case NPP_OUT_OFF_RANGE_ERROR: return "NPP_OUT_OFF_RANGE_ERROR";

		case NPP_DIVIDE_BY_ZERO_ERROR: return "NPP_DIVIDE_BY_ZERO_ERROR";

		case NPP_RANGE_ERROR: return "NPP_RANGE_ERROR";

		case NPP_NO_MEMORY_ERROR: return "NPP_NO_MEMORY_ERROR";

		case NPP_ERROR_RESERVED: return "NPP_ERROR_RESERVED";

		case NPP_NO_OPERATION_WARNING: return "NPP_NO_OPERATION_WARNING";

		case NPP_DIVIDE_BY_ZERO_WARNING: return "NPP_DIVIDE_BY_ZERO_WARNING";
#	endif

#	if ((NPP_VERSION_MAJOR << 12) + (NPP_VERSION_MINOR << 4)) >= 0x7000
		/* These are 7.0 or higher */
		case NPP_OVERFLOW_ERROR: return "NPP_OVERFLOW_ERROR";

		case NPP_CORRUPTED_DATA_ERROR: return "NPP_CORRUPTED_DATA_ERROR";
#	endif
	}

	return "<unknown>";
}
#endif

template<typename T>
void check(T result, char const *const func, const char *const file, int const line) {
	if (result) {
		fprintf(stderr,
				"CUDA error at %s:%d code=%d(%s) \"%s\" \n",
				file,
				line,
				static_cast<unsigned int>(result),
				_cudaGetErrorEnum(result),
				func);
		exit(EXIT_FAILURE);
	}
}

#ifdef __DRIVER_TYPES_H__
// This will output the proper CUDA error strings in the event
// that a CUDA host call returns an error
#	define checkCudaErrors(val) check((val), #	  val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#	define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
				"%s(%i) : getLastCudaError() CUDA error :"
				" %s : (%d) %s.\n",
				file,
				line,
				errorMessage,
				static_cast<int>(err),
				cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}

// This will only print the proper error string when calling cudaGetLastError
// but not exit program incase error detected.
#	define printLastCudaError(msg) __printLastCudaError(msg, __FILE__, __LINE__)

inline void __printLastCudaError(const char *errorMessage, const char *file, const int line) {
	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {
		fprintf(stderr,
				"%s(%i) : getLastCudaError() CUDA error :"
				" %s : (%d) %s.\n",
				file,
				line,
				errorMessage,
				static_cast<int>(err),
				cudaGetErrorString(err));
	}
}

#endif

#ifndef MAX
#	define MAX(a, b) (a > b ? a : b)
#endif

// Float To Int conversion
inline int ftoi(float value) {
	return (value >= 0 ? static_cast<int>(value + 0.5) : static_cast<int>(value - 0.5));
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the # of cores per SM
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		int Cores;
	} sSMtoCores;

	sSMtoCores nGpuArchCoresPerSM[] = {{0x30, 192},
									   {0x32, 192},
									   {0x35, 192},
									   {0x37, 192},
									   {0x50, 128},
									   {0x52, 128},
									   {0x53, 128},
									   {0x60, 64},
									   {0x61, 128},
									   {0x62, 128},
									   {0x70, 64},
									   {0x72, 64},
									   {0x75, 64},
									   {0x80, 64},
									   {0x86, 128},
									   {-1, -1}};

	int index = 0;

	while (nGpuArchCoresPerSM[index].SM != -1) {
		if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchCoresPerSM[index].Cores;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
	  "MapSMtoCores for SM %d.%d is undefined."
	  "  Default to use %d Cores/SM\n",
	  major,
	  minor,
	  nGpuArchCoresPerSM[index - 1].Cores);
	return nGpuArchCoresPerSM[index - 1].Cores;
}

inline const char *_ConvertSMVer2ArchName(int major, int minor) {
	// Defines for GPU Architecture types (using the SM version to determine
	// the GPU Arch name)
	typedef struct {
		int SM; // 0xMm (hexidecimal notation), M = SM Major version,
		// and m = SM minor version
		const char *name;
	} sSMtoArchName;

	sSMtoArchName nGpuArchNameSM[] = {{0x30, "Kepler"},
									  {0x32, "Kepler"},
									  {0x35, "Kepler"},
									  {0x37, "Kepler"},
									  {0x50, "Maxwell"},
									  {0x52, "Maxwell"},
									  {0x53, "Maxwell"},
									  {0x60, "Pascal"},
									  {0x61, "Pascal"},
									  {0x62, "Pascal"},
									  {0x70, "Volta"},
									  {0x72, "Xavier"},
									  {0x75, "Turing"},
									  {0x80, "Ampere"},
									  {0x86, "Ampere"},
									  {-1, "Graphics Device"}};

	int index = 0;

	while (nGpuArchNameSM[index].SM != -1) {
		if (nGpuArchNameSM[index].SM == ((major << 4) + minor)) {
			return nGpuArchNameSM[index].name;
		}

		index++;
	}

	// If we don't find the values, we default use the previous one
	// to run properly
	printf(
	  "MapSMtoArchName for SM %d.%d is undefined."
	  "  Default to use %s\n",
	  major,
	  minor,
	  nGpuArchNameSM[index - 1].name);
	return nGpuArchNameSM[index - 1].name;
}
// end of GPU Architecture definitions

#ifdef __CUDA_RUNTIME_H__

// General GPU Device CUDA Initialization
inline int gpuDeviceInit(int devID) {
	int device_count;
	checkCudaErrors(cudaGetDeviceCount(&device_count));

	if (device_count == 0) {
		fprintf(stderr,
				"gpuDeviceInit() CUDA error: "
				"no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	if (devID < 0) { devID = 0; }

	if (devID > device_count - 1) {
		fprintf(stderr, "\n");
		fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", device_count);
		fprintf(stderr,
				">> gpuDeviceInit (-device=%d) is not a valid"
				" GPU device. <<\n",
				devID);
		fprintf(stderr, "\n");
		return -devID;
	}

	int computeMode = -1, major = 0, minor = 0;
	checkCudaErrors(cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, devID));
	checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
	checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
	if (computeMode == cudaComputeModeProhibited) {
		fprintf(stderr,
				"Error: device is running in <Compute Mode "
				"Prohibited>, no threads can use cudaSetDevice().\n");
		return -1;
	}

	if (major < 1) {
		fprintf(stderr, "gpuDeviceInit(): GPU device does not support CUDA.\n");
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaSetDevice(devID));
	printf("gpuDeviceInit() CUDA Device [%d]: \"%s\n", devID, _ConvertSMVer2ArchName(major, minor));

	return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId() {
	int current_device = 0, sm_per_multiproc = 0;
	int max_perf_device	   = 0;
	int device_count	   = 0;
	int devices_prohibited = 0;

	uint64_t max_compute_perf = 0;
	checkCudaErrors(cudaGetDeviceCount(&device_count));

	if (device_count == 0) {
		fprintf(stderr,
				"gpuGetMaxGflopsDeviceId() CUDA error:"
				" no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	// Find the best CUDA capable GPU device
	current_device = 0;

	while (current_device < device_count) {
		int computeMode = -1, major = 0, minor = 0;
		checkCudaErrors(
		  cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
		checkCudaErrors(
		  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
		checkCudaErrors(
		  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));

		// If this GPU is not running on Compute Mode prohibited,
		// then we can add it to the list
		if (computeMode != cudaComputeModeProhibited) {
			if (major == 9999 && minor == 9999) {
				sm_per_multiproc = 1;
			} else {
				sm_per_multiproc = _ConvertSMVer2Cores(major, minor);
			}
			int multiProcessorCount = 0, clockRate = 0;
			checkCudaErrors(cudaDeviceGetAttribute(
			  &multiProcessorCount, cudaDevAttrMultiProcessorCount, current_device));
			cudaError_t result =
			  cudaDeviceGetAttribute(&clockRate, cudaDevAttrClockRate, current_device);
			if (result != cudaSuccess) {
				// If cudaDevAttrClockRate attribute is not supported we
				// set clockRate as 1, to consider GPU with most SMs and CUDA
				// Cores.
				if (result == cudaErrorInvalidValue) {
					clockRate = 1;
				} else {
					fprintf(stderr,
							"CUDA error at %s:%d code=%d(%s) \n",
							__FILE__,
							__LINE__,
							static_cast<unsigned int>(result),
							_cudaGetErrorEnum(result));
					exit(EXIT_FAILURE);
				}
			}
			uint64_t compute_perf = (uint64_t)multiProcessorCount * sm_per_multiproc * clockRate;

			if (compute_perf > max_compute_perf) {
				max_compute_perf = compute_perf;
				max_perf_device	 = current_device;
			}
		} else {
			devices_prohibited++;
		}

		++current_device;
	}

	if (devices_prohibited == device_count) {
		fprintf(stderr,
				"gpuGetMaxGflopsDeviceId() CUDA error:"
				" all devices have compute mode prohibited.\n");
		exit(EXIT_FAILURE);
	}

	return max_perf_device;
}

// Initialization code to find the best CUDA Device
inline int findCudaDevice(int argc, const char **argv) {
	int devID = 0;

	// If the command-line has a device number specified, use it
	if (checkCmdLineFlag(argc, argv, "device")) {
		devID = getCmdLineArgumentInt(argc, argv, "device=");

		if (devID < 0) {
			printf("Invalid command line parameter\n ");
			exit(EXIT_FAILURE);
		} else {
			devID = gpuDeviceInit(devID);

			if (devID < 0) {
				printf("exiting...\n");
				exit(EXIT_FAILURE);
			}
		}
	} else {
		// Otherwise pick the device with highest Gflops/s
		devID = gpuGetMaxGflopsDeviceId();
		checkCudaErrors(cudaSetDevice(devID));
		int major = 0, minor = 0;
		checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, devID));
		checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, devID));
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
			   devID,
			   _ConvertSMVer2ArchName(major, minor),
			   major,
			   minor);
	}

	return devID;
}

inline int findIntegratedGPU() {
	int current_device	   = 0;
	int device_count	   = 0;
	int devices_prohibited = 0;

	checkCudaErrors(cudaGetDeviceCount(&device_count));

	if (device_count == 0) {
		fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
		exit(EXIT_FAILURE);
	}

	// Find the integrated GPU which is compute capable
	while (current_device < device_count) {
		int computeMode = -1, integrated = -1;
		checkCudaErrors(
		  cudaDeviceGetAttribute(&computeMode, cudaDevAttrComputeMode, current_device));
		checkCudaErrors(cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, current_device));
		// If GPU is integrated and is not running on Compute Mode prohibited,
		// then cuda can map to GLES resource
		if (integrated && (computeMode != cudaComputeModeProhibited)) {
			checkCudaErrors(cudaSetDevice(current_device));

			int major = 0, minor = 0;
			checkCudaErrors(
			  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, current_device));
			checkCudaErrors(
			  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, current_device));
			printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n",
				   current_device,
				   _ConvertSMVer2ArchName(major, minor),
				   major,
				   minor);

			return current_device;
		} else {
			devices_prohibited++;
		}

		current_device++;
	}

	if (devices_prohibited == device_count) {
		fprintf(stderr,
				"CUDA error:"
				" No GLES-CUDA Interop capable GPU found.\n");
		exit(EXIT_FAILURE);
	}

	return -1;
}

// General check for CUDA GPU SM Capabilities
inline bool checkCudaCapabilities(int major_version, int minor_version) {
	int dev;
	int major = 0, minor = 0;

	checkCudaErrors(cudaGetDevice(&dev));
	checkCudaErrors(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
	checkCudaErrors(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));

	if ((major > major_version) || (major == major_version && minor >= minor_version)) {
		printf("  Device %d: <%16s >, Compute SM %d.%d detected\n",
			   dev,
			   _ConvertSMVer2ArchName(major, minor),
			   major,
			   minor);
		return true;
	} else {
		printf(
		  "  No GPU device was found that can support "
		  "CUDA compute capability %d.%d.\n",
		  major_version,
		  minor_version);
		return false;
	}
}

#endif

// end of CUDA Helper Functions

#endif // COMMON_HELPER_CUDA_H_

/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// These are helper functions for the SDK samples (string parsing,
// timers, image helpers, etc)
#ifndef COMMON_HELPER_FUNCTIONS_H_
#define COMMON_HELPER_FUNCTIONS_H_

#ifdef WIN32
#	pragma warning(disable : 4996)
#endif

// includes, project
#include <algorithm>
#include <assert.h>
/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* CUda UTility Library */
#ifndef COMMON_EXCEPTION_H_
#define COMMON_EXCEPTION_H_

// includes, system
#include <exception>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>
#include <string>

//! Exception wrapper.
//! @param Std_Exception Exception out of namespace std for easy typing.
template<class Std_Exception>
class Exception : public Std_Exception {
public:
	//! @brief Static construction interface
	//! @return Alwayss throws ( Located_Exception<Exception>)
	//! @param file file in which the Exception occurs
	//! @param line line in which the Exception occurs
	//! @param detailed details on the code fragment causing the Exception
	static void throw_it(const char *file, const int line, const char *detailed = "-");

	//! Static construction interface
	//! @return Alwayss throws ( Located_Exception<Exception>)
	//! @param file file in which the Exception occurs
	//! @param line line in which the Exception occurs
	//! @param detailed details on the code fragment causing the Exception
	static void throw_it(const char *file, const int line, const std::string &detailed);

	//! Destructor
	virtual ~Exception() throw();

private:
	//! Constructor, default (private)
	Exception();

	//! Constructor, standard
	//! @param str string returned by what()
	explicit Exception(const std::string &str);
};

////////////////////////////////////////////////////////////////////////////////
//! Exception handler function for arbitrary exceptions
//! @param ex exception to handle
////////////////////////////////////////////////////////////////////////////////
template<class Exception_Typ>
inline void handleException(const Exception_Typ &ex) {
	std::cerr << ex.what() << std::endl;

	exit(EXIT_FAILURE);
}

//! Convenience macros

//! Exception caused by dynamic program behavior, e.g. file does not exist
#define RUNTIME_EXCEPTION(msg) Exception<std::runtime_error>::throw_it(__FILE__, __LINE__, msg)

//! Logic exception in program, e.g. an assert failed
#define LOGIC_EXCEPTION(msg) Exception<std::logic_error>::throw_it(__FILE__, __LINE__, msg)

//! Out of range exception
#define RANGE_EXCEPTION(msg) Exception<std::range_error>::throw_it(__FILE__, __LINE__, msg)

////////////////////////////////////////////////////////////////////////////////
//! Implementation

// includes, system
#include <sstream>

////////////////////////////////////////////////////////////////////////////////
//! Static construction interface.
//! @param  Exception causing code fragment (file and line) and detailed infos.
////////////////////////////////////////////////////////////////////////////////
/*static*/ template<class Std_Exception>
void Exception<Std_Exception>::throw_it(const char *file, const int line, const char *detailed) {
	std::stringstream s;

	// Quiet heavy-weight but exceptions are not for
	// performance / release versions
	s << "Exception in file '" << file << "' in line " << line << "\n"
	  << "Detailed description: " << detailed << "\n";

	throw Exception(s.str());
}

////////////////////////////////////////////////////////////////////////////////
//! Static construction interface.
//! @param  Exception causing code fragment (file and line) and detailed infos.
////////////////////////////////////////////////////////////////////////////////
/*static*/ template<class Std_Exception>
void Exception<Std_Exception>::throw_it(const char *file, const int line, const std::string &msg) {
	throw_it(file, line, msg.c_str());
}

////////////////////////////////////////////////////////////////////////////////
//! Constructor, default (private).
////////////////////////////////////////////////////////////////////////////////
template<class Std_Exception>
Exception<Std_Exception>::Exception() : Std_Exception("Unknown Exception.\n") {}

////////////////////////////////////////////////////////////////////////////////
//! Constructor, standard (private).
//! String returned by what().
////////////////////////////////////////////////////////////////////////////////
template<class Std_Exception>
Exception<Std_Exception>::Exception(const std::string &s) : Std_Exception(s) {}

////////////////////////////////////////////////////////////////////////////////
//! Destructor
////////////////////////////////////////////////////////////////////////////////
template<class Std_Exception>
Exception<Std_Exception>::~Exception() throw() {}

// functions, exported

#endif // COMMON_EXCEPTION_H_

#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

// includes, timer, string parsing, image helpers
/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// These are helper functions for the SDK samples (image,bitmap)
#ifndef COMMON_HELPER_IMAGE_H_
#define COMMON_HELPER_IMAGE_H_

#include <algorithm>
#include <assert.h>

#include <fstream>
#include <iostream>
#include <math.h>
#include <stdint.h>
#include <string>
#include <vector>

#ifndef MIN
#	define MIN(a, b) ((a < b) ? a : b)
#endif
#ifndef MAX
#	define MAX(a, b) ((a > b) ? a : b)
#endif

#ifndef EXIT_WAIVED
#	define EXIT_WAIVED 2
#endif

// namespace unnamed (internal)
namespace helper_image_internal {
	//! size of PGM file header
	const unsigned int PGMHeaderSize = 0x40;

	// types

	//! Data converter from unsigned char / unsigned byte to type T
	template<class T>
	struct ConverterFromUByte;

	//! Data converter from unsigned char / unsigned byte
	template<>
	struct ConverterFromUByte<unsigned char> {
		//! Conversion operator
		//! @return converted value
		//! @param  val  value to convert
		float operator()(const unsigned char &val) { return static_cast<unsigned char>(val); }
	};

	//! Data converter from unsigned char / unsigned byte to float
	template<>
	struct ConverterFromUByte<float> {
		//! Conversion operator
		//! @return converted value
		//! @param  val  value to convert
		float operator()(const unsigned char &val) { return static_cast<float>(val) / 255.0f; }
	};

	//! Data converter from unsigned char / unsigned byte to type T
	template<class T>
	struct ConverterToUByte;

	//! Data converter from unsigned char / unsigned byte to unsigned int
	template<>
	struct ConverterToUByte<unsigned char> {
		//! Conversion operator (essentially a passthru
		//! @return converted value
		//! @param  val  value to convert
		unsigned char operator()(const unsigned char &val) { return val; }
	};

	//! Data converter from unsigned char / unsigned byte to unsigned int
	template<>
	struct ConverterToUByte<float> {
		//! Conversion operator
		//! @return converted value
		//! @param  val  value to convert
		unsigned char operator()(const float &val) {
			return static_cast<unsigned char>(val * 255.0f);
		}
	};
} // namespace helper_image_internal

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#	ifndef FOPEN
#		define FOPEN(fHandle, filename, mode) fopen_s(&fHandle, filename, mode)
#	endif
#	ifndef FOPEN_FAIL
#		define FOPEN_FAIL(result) (result != 0)
#	endif
#	ifndef SSCANF
#		define SSCANF sscanf_s
#	endif
#else
#	ifndef FOPEN
#		define FOPEN(fHandle, filename, mode) (fHandle = fopen(filename, mode))
#	endif
#	ifndef FOPEN_FAIL
#		define FOPEN_FAIL(result) (result == NULL)
#	endif
#	ifndef SSCANF
#		define SSCANF sscanf
#	endif
#endif

inline bool __loadPPM(const char *file, unsigned char **data, unsigned int *w, unsigned int *h,
					  unsigned int *channels) {
	FILE *fp = NULL;

	if (FOPEN_FAIL(FOPEN(fp, file, "rb"))) {
		std::cerr << "__LoadPPM() : Failed to open file: " << file << std::endl;
		return false;
	}

	// check header
	char header[helper_image_internal::PGMHeaderSize];

	if (fgets(header, helper_image_internal::PGMHeaderSize, fp) == NULL) {
		std::cerr << "__LoadPPM() : reading PGM header returned NULL" << std::endl;
		return false;
	}

	if (strncmp(header, "P5", 2) == 0) {
		*channels = 1;
	} else if (strncmp(header, "P6", 2) == 0) {
		*channels = 3;
	} else {
		std::cerr << "__LoadPPM() : File is not a PPM or PGM image" << std::endl;
		*channels = 0;
		return false;
	}

	// parse header, read maxval, width and height
	unsigned int width	= 0;
	unsigned int height = 0;
	unsigned int maxval = 0;
	unsigned int i		= 0;

	while (i < 3) {
		if (fgets(header, helper_image_internal::PGMHeaderSize, fp) == NULL) {
			std::cerr << "__LoadPPM() : reading PGM header returned NULL" << std::endl;
			return false;
		}

		if (header[0] == '#') { continue; }

		if (i == 0) {
			i += SSCANF(header, "%u %u %u", &width, &height, &maxval);
		} else if (i == 1) {
			i += SSCANF(header, "%u %u", &height, &maxval);
		} else if (i == 2) {
			i += SSCANF(header, "%u", &maxval);
		}
	}

	// check if given handle for the data is initialized
	if (NULL != *data) {
		if (*w != width || *h != height) {
			std::cerr << "__LoadPPM() : Invalid image dimensions." << std::endl;
		}
	} else {
		*data = (unsigned char *)malloc(sizeof(unsigned char) * width * height * *channels);
		*w	  = width;
		*h	  = height;
	}

	// read and close file
	if (fread(*data, sizeof(unsigned char), width * height * *channels, fp) == 0) {
		std::cerr << "__LoadPPM() read data returned error." << std::endl;
	}

	fclose(fp);

	return true;
}

template<class T>
inline bool sdkLoadPGM(const char *file, T **data, unsigned int *w, unsigned int *h) {
	unsigned char *idata = NULL;
	unsigned int channels;

	if (true != __loadPPM(file, &idata, w, h, &channels)) { return false; }

	unsigned int size = *w * *h * channels;

	// initialize mem if necessary
	// the correct size is checked / set in loadPGMc()
	if (NULL == *data) { *data = reinterpret_cast<T *>(malloc(sizeof(T) * size)); }

	// copy and cast data
	std::transform(idata, idata + size, *data, helper_image_internal::ConverterFromUByte<T>());

	free(idata);

	return true;
}

template<class T>
inline bool sdkLoadPPM4(const char *file, T **data, unsigned int *w, unsigned int *h) {
	unsigned char *idata = 0;
	unsigned int channels;

	if (__loadPPM(file, &idata, w, h, &channels)) {
		// pad 4th component
		int size = *w * *h;
		// keep the original pointer
		unsigned char *idata_orig = idata;
		*data					  = reinterpret_cast<T *>(malloc(sizeof(T) * size * 4));
		unsigned char *ptr		  = *data;

		for (int i = 0; i < size; i++) {
			*ptr++ = *idata++;
			*ptr++ = *idata++;
			*ptr++ = *idata++;
			*ptr++ = 0;
		}

		free(idata_orig);
		return true;
	} else {
		free(idata);
		return false;
	}
}

inline bool __savePPM(const char *file, unsigned char *data, unsigned int w, unsigned int h,
					  unsigned int channels) {
	assert(NULL != data);
	assert(w > 0);
	assert(h > 0);

	std::fstream fh(file, std::fstream::out | std::fstream::binary);

	if (fh.bad()) {
		std::cerr << "__savePPM() : Opening file failed." << std::endl;
		return false;
	}

	if (channels == 1) {
		fh << "P5\n";
	} else if (channels == 3) {
		fh << "P6\n";
	} else {
		std::cerr << "__savePPM() : Invalid number of channels." << std::endl;
		return false;
	}

	fh << w << "\n" << h << "\n" << 0xff << std::endl;

	for (unsigned int i = 0; (i < (w * h * channels)) && fh.good(); ++i) { fh << data[i]; }

	fh.flush();

	if (fh.bad()) {
		std::cerr << "__savePPM() : Writing data failed." << std::endl;
		return false;
	}

	fh.close();

	return true;
}

template<class T>
inline bool sdkSavePGM(const char *file, T *data, unsigned int w, unsigned int h) {
	unsigned int size	 = w * h;
	unsigned char *idata = (unsigned char *)malloc(sizeof(unsigned char) * size);

	std::transform(data, data + size, idata, helper_image_internal::ConverterToUByte<T>());

	// write file
	bool result = __savePPM(file, idata, w, h, 1);

	// cleanup
	free(idata);

	return result;
}

inline bool sdkSavePPM4ub(const char *file, unsigned char *data, unsigned int w, unsigned int h) {
	// strip 4th component
	int size			 = w * h;
	unsigned char *ndata = (unsigned char *)malloc(sizeof(unsigned char) * size * 3);
	unsigned char *ptr	 = ndata;

	for (int i = 0; i < size; i++) {
		*ptr++ = *data++;
		*ptr++ = *data++;
		*ptr++ = *data++;
		data++;
	}

	bool result = __savePPM(file, ndata, w, h, 3);
	free(ndata);
	return result;
}

//////////////////////////////////////////////////////////////////////////////
//! Read file \filename and return the data
//! @return bool if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
//////////////////////////////////////////////////////////////////////////////
template<class T>
inline bool sdkReadFile(const char *filename, T **data, unsigned int *len, bool verbose) {
	// check input arguments
	assert(NULL != filename);
	assert(NULL != len);

	// intermediate storage for the data read
	std::vector<T> data_read;

	// open file for reading
	FILE *fh = NULL;

	// check if filestream is valid
	if (FOPEN_FAIL(FOPEN(fh, filename, "r"))) {
		printf("Unable to open input file: %s\n", filename);
		return false;
	}

	// read all data elements
	T token;

	while (!feof(fh)) {
		fscanf(fh, "%f", &token);
		data_read.push_back(token);
	}

	// the last element is read twice
	data_read.pop_back();
	fclose(fh);

	// check if the given handle is already initialized
	if (NULL != *data) {
		if (*len != data_read.size()) {
			std::cerr << "sdkReadFile() : Initialized memory given but "
					  << "size  mismatch with signal read "
					  << "(data read / data init = " << (unsigned int)data_read.size() << " / "
					  << *len << ")" << std::endl;

			return false;
		}
	} else {
		// allocate storage for the data read
		*data = reinterpret_cast<T *>(malloc(sizeof(T) * data_read.size()));
		// store signal size
		*len = static_cast<unsigned int>(data_read.size());
	}

	// copy data
	memcpy(*data, &data_read.front(), sizeof(T) * data_read.size());

	return true;
}

//////////////////////////////////////////////////////////////////////////////
//! Read file \filename and return the data
//! @return bool if reading the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  uninitialized pointer, returned initialized and pointing to
//!        the data read
//! @param len  number of data elements in data, -1 on error
//////////////////////////////////////////////////////////////////////////////
template<class T>
inline bool sdkReadFileBlocks(const char *filename, T **data, unsigned int *len,
							  unsigned int block_num, unsigned int block_size, bool verbose) {
	// check input arguments
	assert(NULL != filename);
	assert(NULL != len);

	// open file for reading
	FILE *fh = fopen(filename, "rb");

	if (fh == NULL && verbose) {
		std::cerr << "sdkReadFile() : Opening file failed." << std::endl;
		return false;
	}

	// check if the given handle is already initialized
	// allocate storage for the data read
	data[block_num] = reinterpret_cast<T *>(malloc(block_size));

	// read all data elements
	fseek(fh, block_num * block_size, SEEK_SET);
	*len = fread(data[block_num], sizeof(T), block_size / sizeof(T), fh);

	fclose(fh);

	return true;
}

//////////////////////////////////////////////////////////////////////////////
//! Write a data file \filename
//! @return true if writing the file succeeded, otherwise false
//! @param filename name of the source file
//! @param data  data to write
//! @param len  number of data elements in data, -1 on error
//! @param epsilon  epsilon for comparison
//////////////////////////////////////////////////////////////////////////////
template<class T, class S>
inline bool sdkWriteFile(const char *filename, const T *data, unsigned int len, const S epsilon,
						 bool verbose, bool append = false) {
	assert(NULL != filename);
	assert(NULL != data);

	// open file for writing
	//    if (append) {
	std::fstream fh(filename, std::fstream::out | std::fstream::ate);

	if (verbose) {
		std::cerr << "sdkWriteFile() : Open file " << filename << " for write/append." << std::endl;
	}

	/*    } else {
			std::fstream fh(filename, std::fstream::out);
			if (verbose) {
				std::cerr << "sdkWriteFile() : Open file " << filename << " for
	   write." << std::endl;
			}
		}
	*/

	// check if filestream is valid
	if (!fh.good()) {
		if (verbose) { std::cerr << "sdkWriteFile() : Opening file failed." << std::endl; }

		return false;
	}

	// first write epsilon
	fh << "# " << epsilon << "\n";

	// write data
	for (unsigned int i = 0; (i < len) && (fh.good()); ++i) { fh << data[i] << ' '; }

	// Check if writing succeeded
	if (!fh.good()) {
		if (verbose) { std::cerr << "sdkWriteFile() : Writing file failed." << std::endl; }

		return false;
	}

	// file ends with nl
	fh << std::endl;

	return true;
}

//////////////////////////////////////////////////////////////////////////////
//! Compare two arrays of arbitrary type
//! @return  true if \a reference and \a data are identical, otherwise false
//! @param reference  timer_interface to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
//////////////////////////////////////////////////////////////////////////////
template<class T, class S>
inline bool compareData(const T *reference, const T *data, const unsigned int len, const S epsilon,
						const float threshold) {
	assert(epsilon >= 0);

	bool result				 = true;
	unsigned int error_count = 0;

	for (unsigned int i = 0; i < len; ++i) {
		float diff = static_cast<float>(reference[i]) - static_cast<float>(data[i]);
		bool comp  = (diff <= epsilon) && (diff >= -epsilon);
		result &= comp;

		error_count += !comp;

#if 0

		if (!comp) {
		  std::cerr << "ERROR, i = " << i << ",\t "
					<< reference[i] << " / "
					<< data[i]
					<< " (reference / data)\n";
		}

#endif
	}

	if (threshold == 0.0f) {
		return (result) ? true : false;
	} else {
		if (error_count) {
			printf("%4.2f(%%) of bytes mismatched (count=%d)\n",
				   static_cast<float>(error_count) * 100 / static_cast<float>(len),
				   error_count);
		}

		return (len * threshold > error_count) ? true : false;
	}
}

#ifndef __MIN_EPSILON_ERROR
#	define __MIN_EPSILON_ERROR 1e-3f
#endif

//////////////////////////////////////////////////////////////////////////////
//! Compare two arrays of arbitrary type
//! @return  true if \a reference and \a data are identical, otherwise false
//! @param reference  handle to the reference data / gold image
//! @param data       handle to the computed data
//! @param len        number of elements in reference and data
//! @param epsilon    epsilon to use for the comparison
//! @param epsilon    threshold % of (# of bytes) for pass/fail
//////////////////////////////////////////////////////////////////////////////
template<class T, class S>
inline bool compareDataAsFloatThreshold(const T *reference, const T *data, const unsigned int len,
										const S epsilon, const float threshold) {
	assert(epsilon >= 0);

	// If we set epsilon to be 0, let's set a minimum threshold
	float max_error = MAX((float)epsilon, __MIN_EPSILON_ERROR);
	int error_count = 0;
	bool result		= true;

	for (unsigned int i = 0; i < len; ++i) {
		float diff = fabs(static_cast<float>(reference[i]) - static_cast<float>(data[i]));
		bool comp  = (diff < max_error);
		result &= comp;

		if (!comp) { error_count++; }
	}

	if (threshold == 0.0f) {
		if (error_count) { printf("total # of errors = %d\n", error_count); }

		return (error_count == 0) ? true : false;
	} else {
		if (error_count) {
			printf("%4.2f(%%) of bytes mismatched (count=%d)\n",
				   static_cast<float>(error_count) * 100 / static_cast<float>(len),
				   error_count);
		}

		return ((len * threshold > error_count) ? true : false);
	}
}

inline void sdkDumpBin(void *data, unsigned int bytes, const char *filename) {
	printf("sdkDumpBin: <%s>\n", filename);
	FILE *fp;
	FOPEN(fp, filename, "wb");
	fwrite(data, bytes, 1, fp);
	fflush(fp);
	fclose(fp);
}

inline bool sdkCompareBin2BinUint(const char *src_file, const char *ref_file,
								  unsigned int nelements, const float epsilon,
								  const float threshold, char *exec_path) {
	unsigned int *src_buffer, *ref_buffer;
	FILE *src_fp = NULL, *ref_fp = NULL;

	uint64_t error_count = 0;
	size_t fsize		 = 0;

	if (FOPEN_FAIL(FOPEN(src_fp, src_file, "rb"))) {
		printf("compareBin2Bin <unsigned int> unable to open src_file: %s\n", src_file);
		error_count++;
	}

	char *ref_file_path = sdkFindFilePath(ref_file, exec_path);

	if (ref_file_path == NULL) {
		printf("compareBin2Bin <unsigned int>  unable to find <%s> in <%s>\n", ref_file, exec_path);
		printf(">>> Check info.xml and [project//data] folder <%s> <<<\n", ref_file);
		printf("Aborting comparison!\n");
		printf("  FAILED\n");
		error_count++;

		if (src_fp) { fclose(src_fp); }

		if (ref_fp) { fclose(ref_fp); }
	} else {
		if (FOPEN_FAIL(FOPEN(ref_fp, ref_file_path, "rb"))) {
			printf(
			  "compareBin2Bin <unsigned int>"
			  " unable to open ref_file: %s\n",
			  ref_file_path);
			error_count++;
		}

		if (src_fp && ref_fp) {
			src_buffer = (unsigned int *)malloc(nelements * sizeof(unsigned int));
			ref_buffer = (unsigned int *)malloc(nelements * sizeof(unsigned int));

			fsize = fread(src_buffer, nelements, sizeof(unsigned int), src_fp);
			fsize = fread(ref_buffer, nelements, sizeof(unsigned int), ref_fp);

			printf(
			  "> compareBin2Bin <unsigned int> nelements=%d,"
			  " epsilon=%4.2f, threshold=%4.2f\n",
			  nelements,
			  epsilon,
			  threshold);
			printf("   src_file <%s>, size=%d bytes\n", src_file, static_cast<int>(fsize));
			printf("   ref_file <%s>, size=%d bytes\n", ref_file_path, static_cast<int>(fsize));

			if (!compareData<unsigned int, float>(
				  ref_buffer, src_buffer, nelements, epsilon, threshold)) {
				error_count++;
			}

			fclose(src_fp);
			fclose(ref_fp);

			free(src_buffer);
			free(ref_buffer);
		} else {
			if (src_fp) { fclose(src_fp); }

			if (ref_fp) { fclose(ref_fp); }
		}
	}

	if (error_count == 0) {
		printf("  OK\n");
	} else {
		printf("  FAILURE: %d errors...\n", (unsigned int)error_count);
	}

	return (error_count == 0); // returns true if all pixels pass
}

inline bool sdkCompareBin2BinFloat(const char *src_file, const char *ref_file,
								   unsigned int nelements, const float epsilon,
								   const float threshold, char *exec_path) {
	float *src_buffer = NULL, *ref_buffer = NULL;
	FILE *src_fp = NULL, *ref_fp = NULL;
	size_t fsize = 0;

	uint64_t error_count = 0;

	if (FOPEN_FAIL(FOPEN(src_fp, src_file, "rb"))) {
		printf("compareBin2Bin <float> unable to open src_file: %s\n", src_file);
		error_count = 1;
	}

	char *ref_file_path = sdkFindFilePath(ref_file, exec_path);

	if (ref_file_path == NULL) {
		printf("compareBin2Bin <float> unable to find <%s> in <%s>\n", ref_file, exec_path);
		printf(">>> Check info.xml and [project//data] folder <%s> <<<\n", exec_path);
		printf("Aborting comparison!\n");
		printf("  FAILED\n");
		error_count++;

		if (src_fp) { fclose(src_fp); }

		if (ref_fp) { fclose(ref_fp); }
	} else {
		if (FOPEN_FAIL(FOPEN(ref_fp, ref_file_path, "rb"))) {
			printf("compareBin2Bin <float> unable to open ref_file: %s\n", ref_file_path);
			error_count = 1;
		}

		if (src_fp && ref_fp) {
			src_buffer = reinterpret_cast<float *>(malloc(nelements * sizeof(float)));
			ref_buffer = reinterpret_cast<float *>(malloc(nelements * sizeof(float)));

			printf(
			  "> compareBin2Bin <float> nelements=%d, epsilon=%4.2f,"
			  " threshold=%4.2f\n",
			  nelements,
			  epsilon,
			  threshold);
			fsize = fread(src_buffer, sizeof(float), nelements, src_fp);
			printf("   src_file <%s>, size=%d bytes\n",
				   src_file,
				   static_cast<int>(fsize * sizeof(float)));
			fsize = fread(ref_buffer, sizeof(float), nelements, ref_fp);
			printf("   ref_file <%s>, size=%d bytes\n",
				   ref_file_path,
				   static_cast<int>(fsize * sizeof(float)));

			if (!compareDataAsFloatThreshold<float, float>(
				  ref_buffer, src_buffer, nelements, epsilon, threshold)) {
				error_count++;
			}

			fclose(src_fp);
			fclose(ref_fp);

			free(src_buffer);
			free(ref_buffer);
		} else {
			if (src_fp) { fclose(src_fp); }

			if (ref_fp) { fclose(ref_fp); }
		}
	}

	if (error_count == 0) {
		printf("  OK\n");
	} else {
		printf("  FAILURE: %d errors...\n", (unsigned int)error_count);
	}

	return (error_count == 0); // returns true if all pixels pass
}

inline bool sdkCompareL2fe(const float *reference, const float *data, const unsigned int len,
						   const float epsilon) {
	assert(epsilon >= 0);

	float error = 0;
	float ref	= 0;

	for (unsigned int i = 0; i < len; ++i) {
		float diff = reference[i] - data[i];
		error += diff * diff;
		ref += reference[i] * reference[i];
	}

	float normRef = sqrtf(ref);

	if (fabs(ref) < 1e-7) {
#ifdef _DEBUG
		std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
		return false;
	}

	float normError = sqrtf(error);
	error			= normError / normRef;
	bool result		= error < epsilon;
#ifdef _DEBUG

	if (!result) {
		std::cerr << "ERROR, l2-norm error " << error << " is greater than epsilon " << epsilon
				  << "\n";
	}

#endif

	return result;
}

inline bool sdkLoadPPMub(const char *file, unsigned char **data, unsigned int *w, unsigned int *h) {
	unsigned int channels;
	return __loadPPM(file, data, w, h, &channels);
}

inline bool sdkLoadPPM4ub(const char *file, unsigned char **data, unsigned int *w,
						  unsigned int *h) {
	unsigned char *idata = 0;
	unsigned int channels;

	if (__loadPPM(file, &idata, w, h, &channels)) {
		// pad 4th component
		int size = *w * *h;
		// keep the original pointer
		unsigned char *idata_orig = idata;
		*data					  = (unsigned char *)malloc(sizeof(unsigned char) * size * 4);
		unsigned char *ptr		  = *data;

		for (int i = 0; i < size; i++) {
			*ptr++ = *idata++;
			*ptr++ = *idata++;
			*ptr++ = *idata++;
			*ptr++ = 0;
		}

		free(idata_orig);
		return true;
	} else {
		free(idata);
		return false;
	}
}

inline bool sdkComparePPM(const char *src_file, const char *ref_file, const float epsilon,
						  const float threshold, bool verboseErrors) {
	unsigned char *src_data, *ref_data;
	uint64_t error_count = 0;
	unsigned int ref_width, ref_height;
	unsigned int src_width, src_height;

	if (src_file == NULL || ref_file == NULL) {
		if (verboseErrors) {
			std::cerr << "PPMvsPPM: src_file or ref_file is NULL."
						 "  Aborting comparison\n";
		}

		return false;
	}

	if (verboseErrors) {
		std::cerr << "> Compare (a)rendered:  <" << src_file << ">\n";
		std::cerr << ">         (b)reference: <" << ref_file << ">\n";
	}

	if (sdkLoadPPM4ub(ref_file, &ref_data, &ref_width, &ref_height) != true) {
		if (verboseErrors) {
			std::cerr << "PPMvsPPM: unable to load ref image file: " << ref_file << "\n";
		}

		return false;
	}

	if (sdkLoadPPM4ub(src_file, &src_data, &src_width, &src_height) != true) {
		std::cerr << "PPMvsPPM: unable to load src image file: " << src_file << "\n";
		return false;
	}

	if (src_height != ref_height || src_width != ref_width) {
		if (verboseErrors) {
			std::cerr << "PPMvsPPM: source and ref size mismatch (" << src_width << ","
					  << src_height << ")vs(" << ref_width << "," << ref_height << ")\n";
		}
	}

	if (verboseErrors) {
		std::cerr << "PPMvsPPM: comparing images size (" << src_width << "," << src_height
				  << ") epsilon(" << epsilon << "), threshold(" << threshold * 100 << "%)\n";
	}

	if (compareData(ref_data, src_data, src_width * src_height * 4, epsilon, threshold) == false) {
		error_count = 1;
	}

	if (error_count == 0) {
		if (verboseErrors) { std::cerr << "    OK\n\n"; }
	} else {
		if (verboseErrors) { std::cerr << "    FAILURE!  " << error_count << " errors...\n\n"; }
	}

	// returns true if all pixels pass
	return (error_count == 0) ? true : false;
}

inline bool sdkComparePGM(const char *src_file, const char *ref_file, const float epsilon,
						  const float threshold, bool verboseErrors) {
	unsigned char *src_data = 0, *ref_data = 0;
	uint64_t error_count = 0;
	unsigned int ref_width, ref_height;
	unsigned int src_width, src_height;

	if (src_file == NULL || ref_file == NULL) {
		if (verboseErrors) {
			std::cerr << "PGMvsPGM: src_file or ref_file is NULL."
						 "  Aborting comparison\n";
		}

		return false;
	}

	if (verboseErrors) {
		std::cerr << "> Compare (a)rendered:  <" << src_file << ">\n";
		std::cerr << ">         (b)reference: <" << ref_file << ">\n";
	}

	if (sdkLoadPPMub(ref_file, &ref_data, &ref_width, &ref_height) != true) {
		if (verboseErrors) {
			std::cerr << "PGMvsPGM: unable to load ref image file: " << ref_file << "\n";
		}

		return false;
	}

	if (sdkLoadPPMub(src_file, &src_data, &src_width, &src_height) != true) {
		std::cerr << "PGMvsPGM: unable to load src image file: " << src_file << "\n";
		return false;
	}

	if (src_height != ref_height || src_width != ref_width) {
		if (verboseErrors) {
			std::cerr << "PGMvsPGM: source and ref size mismatch (" << src_width << ","
					  << src_height << ")vs(" << ref_width << "," << ref_height << ")\n";
		}
	}

	if (verboseErrors)
		std::cerr << "PGMvsPGM: comparing images size (" << src_width << "," << src_height
				  << ") epsilon(" << epsilon << "), threshold(" << threshold * 100 << "%)\n";

	if (compareData(ref_data, src_data, src_width * src_height, epsilon, threshold) == false) {
		error_count = 1;
	}

	if (error_count == 0) {
		if (verboseErrors) { std::cerr << "    OK\n\n"; }
	} else {
		if (verboseErrors) { std::cerr << "    FAILURE!  " << error_count << " errors...\n\n"; }
	}

	// returns true if all pixels pass
	return (error_count == 0) ? true : false;
}

#endif // COMMON_HELPER_IMAGE_H_

/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Helper Timing Functions
#ifndef COMMON_HELPER_TIMER_H_
#define COMMON_HELPER_TIMER_H_

#ifndef EXIT_WAIVED
#	define EXIT_WAIVED 2
#endif

// includes, system
#include <vector>

// includes, project

// Definition of the StopWatch Interface, this is used if we don't want to use
// the CUT functions But rather in a self contained class interface
class StopWatchInterface {
public:
	StopWatchInterface() {}

	virtual ~StopWatchInterface() {}

public:
	//! Start time measurement
	virtual void start() = 0;

	//! Stop time measurement
	virtual void stop() = 0;

	//! Reset time counters to zero
	virtual void reset() = 0;

	//! Time in msec. after start. If the stop watch is still running (i.e.
	//! there was no call to stop()) then the elapsed time is returned,
	//! otherwise the time between the last start() and stop call is returned
	virtual float getTime() = 0;

	//! Mean time to date based on the number of times the stopwatch has been
	//! _stopped_ (ie finished sessions) and the current total time
	virtual float getAverageTime() = 0;
};

//////////////////////////////////////////////////////////////////
// Begin Stopwatch timer class definitions for all OS platforms //
//////////////////////////////////////////////////////////////////
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// includes, system
#	define WINDOWS_LEAN_AND_MEAN

#	include <Windows.h>

#	undef min
#	undef max

//! Windows specific implementation of StopWatch
class StopWatchWin : public StopWatchInterface {
public:
	//! Constructor, default
	StopWatchWin() :
			start_time(), end_time(), diff_time(0.0f), total_time(0.0f), running(false),
			clock_sessions(0), freq(0), freq_set(false) {
		if (!freq_set) {
			// helper variable
			LARGE_INTEGER temp;

			// get the tick frequency from the OS
			QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER *>(&temp));

			// convert to type in which it is needed
			freq = (static_cast<double>(temp.QuadPart)) / 1000.0;

			// rememeber query
			freq_set = true;
		}
	}

	// Destructor
	~StopWatchWin() {}

public:
	//! Start time measurement
	inline void start();

	//! Stop time measurement
	inline void stop();

	//! Reset time counters to zero
	inline void reset();

	//! Time in msec. after start. If the stop watch is still running (i.e.
	//! there was no call to stop()) then the elapsed time is returned,
	//! otherwise the time between the last start() and stop call is returned
	inline float getTime();

	//! Mean time to date based on the number of times the stopwatch has been
	//! _stopped_ (ie finished sessions) and the current total time
	inline float getAverageTime();

private:
	// member variables

	//! Start of measurement
	LARGE_INTEGER start_time;
	//! End of measurement
	LARGE_INTEGER end_time;

	//! Time difference between the last start and stop
	float diff_time;

	//! TOTAL time difference between starts and stops
	float total_time;

	//! flag if the stop watch is running
	bool running;

	//! Number of times clock has been started
	//! and stopped to allow averaging
	int clock_sessions;

	//! tick frequency
	double freq;

	//! flag if the frequency has been set
	bool freq_set;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchWin::start() {
	QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&start_time));
	running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchWin::stop() {
	QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&end_time));
	diff_time = static_cast<float>(
	  ((static_cast<double>(end_time.QuadPart) - static_cast<double>(start_time.QuadPart)) / freq));

	total_time += diff_time;
	clock_sessions++;
	running = false;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchWin::reset() {
	diff_time	   = 0;
	total_time	   = 0;
	clock_sessions = 0;

	if (running) { QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&start_time)); }
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline float StopWatchWin::getTime() {
	// Return the TOTAL time to date
	float retval = total_time;

	if (running) {
		LARGE_INTEGER temp;
		QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&temp));
		retval += static_cast<float>(
		  ((static_cast<double>(temp.QuadPart) - static_cast<double>(start_time.QuadPart)) / freq));
	}

	return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline float StopWatchWin::getAverageTime() {
	return (clock_sessions > 0) ? (total_time / clock_sessions) : 0.0f;
}

#else
// Declarations for Stopwatch on Linux and Mac OSX
// includes, system
#	include <ctime>
#	include <sys/time.h>

//! Windows specific implementation of StopWatch
class StopWatchLinux : public StopWatchInterface {
public:
	//! Constructor, default
	StopWatchLinux() :
			start_time(), diff_time(0.0), total_time(0.0), running(false), clock_sessions(0) {}

	// Destructor
	virtual ~StopWatchLinux() {}

public:
	//! Start time measurement
	inline void start();

	//! Stop time measurement
	inline void stop();

	//! Reset time counters to zero
	inline void reset();

	//! Time in msec. after start. If the stop watch is still running (i.e.
	//! there was no call to stop()) then the elapsed time is returned,
	//! otherwise the time between the last start() and stop call is returned
	inline float getTime();

	//! Mean time to date based on the number of times the stopwatch has been
	//! _stopped_ (ie finished sessions) and the current total time
	inline float getAverageTime();

private:
	// helper functions

	//! Get difference between start time and current time
	inline float getDiffTime();

private:
	// member variables

	//! Start of measurement
	struct timeval start_time;

	//! Time difference between the last start and stop
	float diff_time;

	//! TOTAL time difference between starts and stops
	float total_time;

	//! flag if the stop watch is running
	bool running;

	//! Number of times clock has been started
	//! and stopped to allow averaging
	int clock_sessions;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchLinux::start() {
	gettimeofday(&start_time, 0);
	running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchLinux::stop() {
	diff_time = getDiffTime();
	total_time += diff_time;
	running = false;
	clock_sessions++;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchLinux::reset() {
	diff_time	   = 0;
	total_time	   = 0;
	clock_sessions = 0;

	if (running) { gettimeofday(&start_time, 0); }
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline float StopWatchLinux::getTime() {
	// Return the TOTAL time to date
	float retval = total_time;

	if (running) { retval += getDiffTime(); }

	return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline float StopWatchLinux::getAverageTime() {
	return (clock_sessions > 0) ? (total_time / clock_sessions) : 0.0f;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
inline float StopWatchLinux::getDiffTime() {
	struct timeval t_time;
	gettimeofday(&t_time, 0);

	// time difference in milli-seconds
	return static_cast<float>(1000.0 * (t_time.tv_sec - start_time.tv_sec) +
							  (0.001 * (t_time.tv_usec - start_time.tv_usec)));
}
#endif // WIN32

////////////////////////////////////////////////////////////////////////////////
//! Timer functionality exported

////////////////////////////////////////////////////////////////////////////////
//! Create a new timer
//! @return true if a time has been created, otherwise false
//! @param  name of the new timer, 0 if the creation failed
////////////////////////////////////////////////////////////////////////////////
inline bool sdkCreateTimer(StopWatchInterface **timer_interface) {
// printf("sdkCreateTimer called object %08x\n", (void *)*timer_interface);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
	*timer_interface = reinterpret_cast<StopWatchInterface *>(new StopWatchWin());
#else
	*timer_interface = reinterpret_cast<StopWatchInterface *>(new StopWatchLinux());
#endif
	return (*timer_interface != NULL) ? true : false;
}

////////////////////////////////////////////////////////////////////////////////
//! Delete a timer
//! @return true if a time has been deleted, otherwise false
//! @param  name of the timer to delete
////////////////////////////////////////////////////////////////////////////////
inline bool sdkDeleteTimer(StopWatchInterface **timer_interface) {
	// printf("sdkDeleteTimer called object %08x\n", (void *)*timer_interface);
	if (*timer_interface) {
		delete *timer_interface;
		*timer_interface = NULL;
	}

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Start the time with name \a name
//! @param name  name of the timer to start
////////////////////////////////////////////////////////////////////////////////
inline bool sdkStartTimer(StopWatchInterface **timer_interface) {
	// printf("sdkStartTimer called object %08x\n", (void *)*timer_interface);
	if (*timer_interface) { (*timer_interface)->start(); }

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop the time with name \a name. Does not reset.
//! @param name  name of the timer to stop
////////////////////////////////////////////////////////////////////////////////
inline bool sdkStopTimer(StopWatchInterface **timer_interface) {
	// printf("sdkStopTimer called object %08x\n", (void *)*timer_interface);
	if (*timer_interface) { (*timer_interface)->stop(); }

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Resets the timer's counter.
//! @param name  name of the timer to reset.
////////////////////////////////////////////////////////////////////////////////
inline bool sdkResetTimer(StopWatchInterface **timer_interface) {
	// printf("sdkResetTimer called object %08x\n", (void *)*timer_interface);
	if (*timer_interface) { (*timer_interface)->reset(); }

	return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Return the average time for timer execution as the total time
//! for the timer dividied by the number of completed (stopped) runs the timer
//! has made.
//! Excludes the current running time if the timer is currently running.
//! @param name  name of the timer to return the time of
////////////////////////////////////////////////////////////////////////////////
inline float sdkGetAverageTimerValue(StopWatchInterface **timer_interface) {
	//  printf("sdkGetAverageTimerValue called object %08x\n", (void
	//  *)*timer_interface);
	if (*timer_interface) {
		return (*timer_interface)->getAverageTime();
	} else {
		return 0.0f;
	}
}

////////////////////////////////////////////////////////////////////////////////
//! Total execution time for the timer over all runs since the last reset
//! or timer creation.
//! @param name  name of the timer to obtain the value of.
////////////////////////////////////////////////////////////////////////////////
inline float sdkGetTimerValue(StopWatchInterface **timer_interface) {
	// printf("sdkGetTimerValue called object %08x\n", (void
	// *)*timer_interface);
	if (*timer_interface) {
		return (*timer_interface)->getTime();
	} else {
		return 0.0f;
	}
}

#endif // COMMON_HELPER_TIMER_H_

#ifndef EXIT_WAIVED
#	define EXIT_WAIVED 2
#endif

#endif // COMMON_HELPER_FUNCTIONS_H_

#endif // LIBRAPID_HAS_CUDA

namespace librapid { namespace device {
	struct CPU {};
	struct GPU {};
}} // namespace librapid::device

// User Config Variables

namespace librapid {
#ifdef LIBRAPID_HAS_OMP
	static unsigned int numThreads	  = 8;
	static unsigned int matrixThreads = 8;
#else
	static unsigned int numThreads	  = 1;
	static unsigned int matrixThreads = 1;
#endif
	static bool throwOnAssert					= false;
	static std::vector<std::string> cudaHeaders = {};
	static std::vector<std::string> nvccOptions = {};
} // namespace librapid

// Prefer using the GPU over the CPU -- promote arrays to the GPU where possible
#if !defined(LIBRAPID_PREFER_CPU)
#	define LIBRAPID_PREFER_GPU
#endif

namespace librapid {
	template<bool B, typename T = void>
	using enable_if_t = typename std::enable_if<B, T>::type;

	template<class T, class U>
	constexpr bool is_same_v = std::is_same<T, U>::value;
} // namespace librapid

// I DO NOT TAKE CREDIT FOR THIS CODE -- I SIMPLY MODIFIED IT

// Dynamically strength-reduced div and mod
//
// Ideas taken from Sean Baxter's MGPU library.
// These classes provide for reduced complexity division and modulus
// on integers, for the case where the same divisor or modulus will
// be used repeatedly.

#define IS_POW_2(x) (0 == ((x) & ((x)-1)))

namespace librapid { namespace detail {
	void findDivisor(unsigned int denom, unsigned int &mulCoeff, unsigned int &shiftCoeff);

	LR_FORCE_INLINE unsigned int umulhi(unsigned int x, unsigned int y) {
		unsigned long long z = (unsigned long long)x * (unsigned long long)y;
		return (unsigned int)(z >> 32);
	}

	template<typename U>
	struct ReducedDivisorImpl {
		U mulCoeff;
		unsigned int shiftCoeff;
		U y;

		ReducedDivisorImpl(U _y) : y(_y) { detail::findDivisor(y, mulCoeff, shiftCoeff); }
		LR_FORCE_INLINE U div(U x) const {
			return (mulCoeff) ? detail::umulhi(x, mulCoeff) >> shiftCoeff : x;
		}

		LR_FORCE_INLINE U mod(U x) const { return (mulCoeff) ? x - (div(x) * y) : 0; }

		LR_FORCE_INLINE void divMod(U x, U &q, U &mod) {
			if (y == 1) {
				q	= x;
				mod = 0;
			} else {
				q	= div(x);
				mod = x - (q * y);
			}
		}

		LR_FORCE_INLINE U get() const { return y; }
	};

	using ReducedDivisor   = ReducedDivisorImpl<uint32_t>;
	using ReducedDivisor64 = ReducedDivisorImpl<uint64_t>;

	// Count leading zeroes
	LR_FORCE_INLINE int clz(int x) {
		for (int i = 31; i >= 0; --i)
			if ((1 << i) & x) return 31 - i;
		return 32;
	}

	LR_FORCE_INLINE int clz(long long x) {
		for (int i = 63; i >= 0; --i)
			if ((1ll << i) & x) return 63 - i;
		return 32;
	}

	int intLog2(int x, bool round_up = false) {
		int a = 31 - clz(x);
		if (round_up) a += !IS_POW_2(x);
		return a;
	}

	int intLog2(long long x, bool round_up = false) {
		int a = 63 - clz(x);
		if (round_up) a += !IS_POW_2(x);
		return a;
	}

	void findDivisor(unsigned int denom, unsigned int &mulCoeff, unsigned int &shiftCoeff) {
		LR_ASSERT(denom != 0, "Trying to find reduced divisor for zero is invalid");

		if (denom == 1) {
			mulCoeff   = 0;
			shiftCoeff = 0;
			return;
		}

		unsigned int p = 31 + intLog2((int)denom, true);
		unsigned int m = ((1ull << p) + denom - 1) / denom;
		mulCoeff	   = m;
		shiftCoeff	   = p - 32;
	}
}} // namespace librapid::detail

namespace librapid { namespace detail {
	template<typename T>
	struct ColumnMajorOrder {
		typedef T result_type;

		int m;
		int n;

		ColumnMajorOrder(const int &_m, const int &_n) : m(_m), n(_n) {}

		T operator()(const int &idx) const {
			int row = idx % m;
			int col = idx / m;
			return row * m + col;
		}
	};

	template<typename T>
	struct RowMajorOrder {
		typedef T ResultType;

		int m;
		int n;

		RowMajorOrder(const int &_m, const int &_n) : m(_m), n(_n) {}

		T operator()(const int &idx) const {
			int row = idx % n;
			int col = idx / n;
			return col * n + row;
		}
	};

	template<typename T>
	struct TxColumnMajorOrder {
		typedef T ResultType;

		int m;
		int n;

		TxColumnMajorOrder(const int &_m, const int &_n) : m(_m), n(_n) {}

		T operator()(const int &idx) const {
			int row = idx / m;
			int col = idx % m;
			return col * n + row;
		}
	};

	template<typename T>
	struct TxRowMajorOrder {
		typedef T ResultType;

		int m;
		int n;

		TxRowMajorOrder(const int &_m, const int &_n) : m(_m), n(_n) {}

		T operator()(const int &idx) const {
			int row = idx % n;
			int col = idx / n;
			return row * m + col;
		}
	};

	struct ColumnMajorIndex {
		int m;
		int n;

		ColumnMajorIndex(const int &_m, const int &_n) : m(_m), n(_n) {}

		int operator()(const int &i, const int &j) const { return i + j * m; }
	};

	struct RowMajorIndex {
		int m;
		int n;

		RowMajorIndex(const int &_m, const int &_n) : m(_m), n(_n) {}

		RowMajorIndex(const ReducedDivisor &_m, const int &_n) :
				m(_m.get()), n(_n) {}

		int operator()(const int &i, const int &j) const { return j + i * n; }
	};
} } // namespace librapid::detail

namespace librapid {
	template<typename T>
	LR_FORCE_INLINE void extendedGCD(T a, T b, T &gcd, T &mmi) {
		T x		= 0;
		T lastX = 1;
		T y		= 1;
		T lastY = 0;
		T origB = b;
		while (b != 0) {
			T quotient = a / b;
			T newB	   = a % b;
			a		   = b;
			b		   = newB;
			T newX	   = lastX - quotient * x;
			lastX	   = x;
			x		   = newX;
			T newY	   = lastY - quotient * y;
			lastY	   = y;
			y		   = newY;
		}
		gcd = a;
		mmi = 0;
		if (gcd == 1) {
			if (lastX < 0) {
				mmi = lastX + origB;
			} else {
				mmi = lastX;
			}
		}
	}

	template<typename T>
	LR_NODISCARD("")
	LR_FORCE_INLINE std::pair<T, T> extendedGCD(T a, T b) {
		T gcd, mmi;
		extendedGCD(a, b, gcd, mmi);
		return {gcd, mmi};
	}
} // namespace librapid

namespace librapid {
	namespace memory {
		template<typename T, typename d>
		class ValueReference;

		template<typename T = unsigned char, typename d = device::CPU>
		class DenseStorage;
	}

	namespace internal {
		template<typename T>
		struct traits;
	}

	template<typename T, int64_t maxDims, int64_t align>
	class ExtentType;

	template<typename ArrT>
	class CommaInitializer;

	template<typename Derived, typename device>
	class ArrayBase;

	namespace unary {
		template<typename DST, typename OtherDerived>
		class Cast;
	}

	namespace binop {
		template<typename Binop, typename Derived, typename OtherDerived>
		class CWiseBinop;
	}

	namespace unop {
		template<typename Unop, typename Derived>
		class CWiseUnop;
	}

	template<typename Scalar_, typename Device_>
	class Array;
} // namespace librapid

// Memory alignment adapted from
// https://gist.github.com/dblalock/255e76195676daa5cbc57b9b36d1c99a

namespace librapid { namespace memory {
	constexpr uint64_t memAlign = 32;

	template<typename T = char, typename d = device::CPU,
			 typename std::enable_if_t<std::is_same_v<d, device::CPU>, int> = 0>
	LR_NODISCARD("Do not leave a dangling pointer")
	LR_FORCE_INLINE T *malloc(size_t num, size_t alignment = memAlign, bool zero = false) {
		size_t size		   = sizeof(T) * num;
		size_t requestSize = size + alignment;
		auto *buf		   = (unsigned char *)(zero ? calloc(1, requestSize) : std::malloc(requestSize));

		LR_ASSERT(buf != nullptr,
				  "Memory allocation failed. Cannot allocate {} items of size "
				  "{} ({} bytes total)!",
				  num,
				  sizeof(T),
				  requestSize);

		size_t remainder = ((size_t)buf) % alignment;
		size_t offset	 = alignment - remainder;
		unsigned char *ret		 = buf + (unsigned char)offset;

		// store how many extra unsigned chars we allocated in the unsigned char just before
		// the pointer we return
		*(unsigned char *)(ret - 1) = (unsigned char)offset;

// Slightly altered traceback call to log unsigned chars being allocated
#ifdef LIBRAPID_TRACEBACK
		LR_STATUS("LIBRAPID TRACEBACK -- MALLOC {} unsigned charS -> {}", size, (void *)buf);
#endif

		return (T *)ret;
	}

	template<typename T = char, typename d = device::CPU,
			 typename std::enable_if_t<std::is_same_v<d, device::CPU>, int> = 0>
	LR_FORCE_INLINE void free(T *alignedPtr) {
#ifdef LIBRAPID_TRACEBACK
		LR_STATUS("LIBRAPID TRACEBACK -- FREE {}", (void *)alignedPtr);
#endif

		int offset = *(((unsigned char *)alignedPtr) - 1);
		std::free(((unsigned char *)alignedPtr) - offset);
	}

	// Only supports copying between host pointers
	template<typename T, typename d, typename T_, typename d_,
			 typename std::enable_if_t<
			   std::is_same_v<d, device::CPU> && std::is_same_v<d_, device::CPU>, int> = 0>
	LR_FORCE_INLINE void memcpy(T *dst, T_ *src, int64_t size) {
		if constexpr (std::is_same_v<T, T_>) {
			std::copy(src, src + size, dst);
		} else {
			// TODO: Optimise this?
			for (int64_t i = 0; i < size; ++i) { dst[i] = src[i]; }
		}
	}

	template<typename A, typename B>
	struct PromoteDevice {};

	template<>
	struct PromoteDevice<device::CPU, device::CPU> {
		using type = device::CPU;
	};

	template<>
	struct PromoteDevice<device::CPU, device::GPU> {
#if defined(LIBRAPID_PREFER_GPU)
		using type = device::GPU;
#else
		using type = device::CPU;
#endif
	};

	template<>
	struct PromoteDevice<device::GPU, device::CPU> {
#if defined(LIBRAPID_PREFER_GPU)
		using type = device::GPU;
#else
		using type = device::CPU;
#endif
	};

	template<>
	struct PromoteDevice<device::GPU, device::GPU> {
		using type = device::GPU;
	};
} } // namespace librapid::memory

namespace librapid { namespace internal {
	namespace flags {
		/**
		 * Flag Configuration:
		 *
		 * [0, 9]     -> Requirement flags
		 * [10, 31]   -> Operation type flags
		 * [32]       -> Unary operation
		 * [33]       -> Binary operation
		 * [34]       -> Matrix operation
		 */

		constexpr uint64_t Evaluated	 = 1ll << 0; // Result is already evaluated
		constexpr uint64_t RequireEval	 = 1ll << 1; // Result must be evaluated
		constexpr uint64_t RequireInput	 = 1ll << 2; // Requires the entire array (not scalar)
		constexpr uint64_t HasCustomEval = 1ll << 3; // Has a custom eval function

		constexpr uint64_t Bitwise	  = 1ll << 10; // Bitwise functions
		constexpr uint64_t Arithmetic = 1ll << 11; // Arithmetic functions
		constexpr uint64_t Logical	  = 1ll << 12; // Logical functions
		constexpr uint64_t Matrix	  = 1ll << 13; // Matrix operation

		// Extract only operation information
		constexpr uint64_t OperationMask = 0b1111111111111111100000000000000;

		constexpr uint64_t PacketBitwise	= 1ll << 14; // Packet needs bitwise
		constexpr uint64_t PacketArithmetic = 1ll << 15; // Packet needs arithmetic
		constexpr uint64_t PacketLogical	= 1ll << 16; // Packet needs logical

		constexpr uint64_t ScalarBitwise	= 1ll << 17; // Scalar needs bitwise
		constexpr uint64_t ScalarArithmetic = 1ll << 18; // Scalar needs arithmetic
		constexpr uint64_t ScalarLogical	= 1ll << 19; // Scalar needs logical

		constexpr uint64_t Unary  = 1ll << 32; // Operation takes one argument
		constexpr uint64_t Binary = 1ll << 33; // Operation takes two arguments

	} // namespace flags

	template<typename T>
	struct traits {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = T;
		using BaseScalar					 = T;
		using StorageType					 = memory::DenseStorage<T, device::CPU>;
		using Packet						 = std::false_type;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "[NO DEFINED TYPE]";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- Just a  Character -----------------------------------------------
	template<>
	struct traits<char> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = char;
		using BaseScalar					 = char;
		using StorageType					 = memory::DenseStorage<char, device::CPU>;
		using Packet						 = std::false_type;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "char";
		static constexpr uint64_t Flags =
		  flags::ScalarBitwise | flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- Boolean ---------------------------------------------------------
	template<>
	struct traits<bool> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = bool;
		using BaseScalar					 = uint32_t;
		using StorageType					 = memory::DenseStorage<bool, device::CPU>;
		using Packet						 = Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "bool";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 8bit Signed Integer ---------------------------------------------
	template<>
	struct traits<int8_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int8_t;
		using BaseScalar					 = int8_t;
		using StorageType					 = memory::DenseStorage<int8_t, device::CPU>;
		using Packet						 = Vc::Vector<BaseScalar>; // vcl::Vec64c;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		;
		static constexpr char Name[]	= "int8_t";
		static constexpr uint64_t Flags = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 8bit Unsigned Integer -------------------------------------------
	template<>
	struct traits<uint8_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint8_t;
		using BaseScalar					 = uint8_t;
		using StorageType					 = memory::DenseStorage<uint8_t>;
		using Packet						 = Vc::Vector<BaseScalar>; // vcl::Vec64uc;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		;
		static constexpr char Name[]	= "uint8_t";
		static constexpr uint64_t Flags = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 16bit Signed Integer --------------------------------------------
	template<>
	struct traits<int16_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int16_t;
		using BaseScalar					 = int16_t;
		using StorageType					 = memory::DenseStorage<int16_t>;
		using Packet						 = Vc::Vector<BaseScalar>; // vcl::Vec32s;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		;
		static constexpr char Name[]	= "int16_t";
		static constexpr uint64_t Flags = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 16bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint16_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint16_t;
		using BaseScalar					 = uint16_t;
		using StorageType					 = memory::DenseStorage<uint16_t>;
		using Packet						 = Vc::Vector<BaseScalar>; // vcl::Vec32us;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		;
		static constexpr char Name[]	= "uint16_t";
		static constexpr uint64_t Flags = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 32bit Signed Integer --------------------------------------------
	template<>
	struct traits<int32_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int32_t;
		using BaseScalar					 = int32_t;
		using StorageType					 = memory::DenseStorage<int32_t>;
		using Packet						 = Vc::Vector<BaseScalar>; // vcl::Vec8i;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		;
		static constexpr char Name[]	= "int32_t";
		static constexpr uint64_t Flags = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 32bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint32_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint32_t;
		using BaseScalar					 = uint32_t;
		using StorageType					 = memory::DenseStorage<uint32_t>;
		using Packet						 = Vc::Vector<BaseScalar>; // vcl::Vec8ui;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		;
		static constexpr char Name[]	= "uint32_t";
		static constexpr uint64_t Flags = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 64bit Signed Integer --------------------------------------------
	template<>
	struct traits<int64_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int64_t;
		using BaseScalar					 = int64_t;
		using StorageType					 = memory::DenseStorage<int64_t>;
		using Packet						 = Vc::Vector<BaseScalar>; // vcl::Vec8q;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		;
		static constexpr char Name[]	= "int64_t";
		static constexpr uint64_t Flags = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 64bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint64_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint64_t;
		using BaseScalar					 = uint64_t;
		using StorageType					 = memory::DenseStorage<uint64_t>;
		using Packet						 = Vc::Vector<BaseScalar>; // vcl::Vec8uq;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		;
		static constexpr char Name[]	= "uint64_t";
		static constexpr uint64_t Flags = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 32bit Floating Point --------------------------------------------
	template<>
	struct traits<float> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = float;
		using BaseScalar					 = float;
		using StorageType					 = memory::DenseStorage<float>;
		using Packet						 = Vc::Vector<BaseScalar>; // vcl::Vec16f;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		;
		static constexpr char Name[]	= "float";
		static constexpr uint64_t Flags = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 64bit Floating Point --------------------------------------------
	template<>
	struct traits<double> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = double;
		using BaseScalar					 = double;
		using StorageType					 = memory::DenseStorage<double>;
		using Packet						 = Vc::Vector<BaseScalar>; // vcl::Vec8d;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		;
		static constexpr char Name[]	= "double";
		static constexpr uint64_t Flags = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	template<typename LHS, typename RHS>
	struct PropagateDeviceType {
		using DeviceLHS = typename traits<LHS>::Device;
		using DeviceRHS = typename traits<RHS>::Device;
		using Device	= typename memory::PromoteDevice<DeviceLHS, DeviceRHS>::type;
	};

	template<typename LHS, typename RHS>
	struct ReturnType {
		using LhsType = LHS;
		using RhsType = RHS;
		using RetType = typename std::common_type<LhsType, RhsType>::type;
	};

	template<typename T>
	using StripQualifiers = typename std::remove_cv_t<typename std::remove_reference_t<T>>;
}} // namespace librapid::internal

namespace librapid { namespace detail {
	struct Prerotator {
		ReducedDivisor m, b;

		Prerotator() : m(1), b(1) {}
		Prerotator(int _m, int _b) : m(_m), b(_b) {}

		int x {};
		void setJ(const int &j) { x = b.div(j); }
		int operator()(const int &i) const { return m.mod(i + x); }
	};

	struct Postpermuter {
		ReducedDivisor m;
		int n;
		ReducedDivisor a;
		int j;

		Postpermuter() : m(1), a(1) {}
		Postpermuter(int _m, int _n, int _a) : m(_m), n(_n), a(_a) {}

		void setJ(const int &_j) { j = _j; }

		int operator()(const int &i) const { return m.mod((i * n + j - a.div(i))); }
	};

	struct Shuffle {
		int m, n, k, i;
		ReducedDivisor b;
		ReducedDivisor c;

		Shuffle() : b(1), c(1) {}
		Shuffle(int _m, int _n, int _c, int _k) : m(_m), n(_n), k(_k), b(_n / _c), c(_c) {}

		void setI(const int &_i) { i = _i; }

		LR_NODISCARD("") int f(const int &j) const {
			int r = j + i * (n - 1);
			// The (int) casts here prevent unsigned promotion
			// and the subsequent underflow: c implicitly casts
			// int - unsigned int to
			// unsigned int - unsigned int
			// rather than to
			// int - int
			// Which leads to underflow if the result is negative.
			if (i - (int)c.mod(j) <= m - (int)c.get()) {
				return r;
			} else {
				return r + m;
			}
		}

		LR_NODISCARD("") int operator()(const int &j) {
			int fij = f(j);
			unsigned int fijdivc, fijmodc;
			c.divMod(fij, fijdivc, fijmodc);
			// The extra mod in here prevents overflowing 32-bit int
			int term1 = b.mod(k * b.mod(fijdivc));
			int term2 = ((int)fijmodc) * (int)b.get();
			return term1 + term2;
		}
	};

	template<typename T, typename F>
	void colShuffle(int m, int n, T *d, T *tmp, F fn) {
		using Packet						 = typename internal::traits<T>::Packet;
		static constexpr int64_t packetWidth = internal::traits<T>::PacketWidth;

		RowMajorIndex rm(m, n);

		T *privTmp;
		F privFn;
		int tid;
		int i;

#pragma omp parallel private(tid, privTmp, privFn, i) num_threads(matrixThreads)
		{
#if defined(LIBRAPID_HAS_OMP)
			tid = omp_get_thread_num();
#else
			tid = 0;
#endif
			privFn	= fn;
			privTmp = tmp + m * tid;
#pragma omp for
			for (int j = 0; j < n; j++) {
				privFn.setJ(j);

				for (i = 0; i < m; i++) { privTmp[i] = d[rm(privFn(i), j)]; }
				for (i = 0; i < m; i++) { d[rm(i, j)] = privTmp[i]; }
			}
		}
	}

	template<typename T, typename F>
	void rowShuffle(int m, int n, T *d, T *tmp, F fn) {
		RowMajorIndex rm(m, n);
		T *privTmp;
		F privFn;
		int tid;
		int j;

#pragma omp parallel private(tid, privTmp, privFn, j) num_threads(matrixThreads)
		{
#if defined(LIBRAPID_HAS_OMP)
			tid = omp_get_thread_num();
#else
			tid = 0;
#endif
			privFn	= fn;
			privTmp = tmp + n * tid;
#pragma omp for
			for (int i = 0; i < m; i++) {
				privFn.setI(i);
				for (j = 0; j < n; j++) { privTmp[j] = d[rm(i, privFn(j))]; }
				for (j = 0; j < n; j++) { d[rm(i, j)] = privTmp[j]; }
			}
		}
	}

	template<typename T>
	void transpose(bool rowMajor, T *data, int m, int n, T *tmp) {
		if (!rowMajor) { std::swap(m, n); }

		int c = 0, t = 0, k = 0;
		extendedGCD(m, n, c, t);
		if (c > 1) {
			extendedGCD(m / c, n / c, t, k);
		} else {
			k = t;
		}

		if (c > 1) { colShuffle(m, n, data, tmp, Prerotator(m, n / c)); }
		rowShuffle(m, n, data, tmp, Shuffle(m, n, c, k));
		colShuffle(m, n, data, tmp, Postpermuter(m, n, m / c));
	}
}} // namespace librapid::detail

#include <utility>

namespace librapid {
	template<typename T>
	T mean(const std::vector<T> &vals) {
		T sum = T();
		for (const auto &val : vals) sum += val;
		return sum / vals.size();
	}

	template<typename T>
	T standardDeviation(const std::vector<T> &vals) {
		T x = mean(vals);
		std::vector<T> variance2;
		for (const auto &val : vals) variance2.emplace_back((x - val) * (x - val));
		return sqrt(mean(variance2));
	}
} // namespace librapid

namespace librapid {
	// Forward declare this function
	template<typename T>
	T map(T val, T start1, T stop1, T start2, T stop2);

	namespace time {
		constexpr int64_t day		  = 86400e9;
		constexpr int64_t hour		  = 3600e9;
		constexpr int64_t minute	  = 60e9;
		constexpr int64_t second	  = 1e9;
		constexpr int64_t millisecond = 1e6;
		constexpr int64_t microsecond = 1e3;
		constexpr int64_t nanosecond  = 1;
	} // namespace time

	template<int64_t scale = time::second>
	LR_NODISCARD("")
	LR_FORCE_INLINE double now() {
		using namespace std::chrono;
		return (double)high_resolution_clock::now().time_since_epoch().count() / (double)scale;
	}

	static double sleepOffset = 0;

	template<int64_t scale = time::second>
	LR_FORCE_INLINE void sleep(double time) {
		using namespace std::chrono;
		time *= scale;
		auto start = now<time::nanosecond>();
		while (now<time::nanosecond>() - start < time - sleepOffset) {}
	}

	template<int64_t scale = time::second>
	std::string formatTime(double time, const std::string &format = "{:3f}") {
		double ns					= time * scale;
		int numUnits				= 8;
		static std::string prefix[] = {"ns", "µs", "ms", "s", "m", "h", "d", "y"};
		static double divisor[]		= {1000, 1000, 1000, 60, 60, 24, 365, 1e300};
		for (int i = 0; i < numUnits; ++i) {
			if (ns < divisor[i]) return fmt::format(format + "{}", ns, prefix[i]);
			ns /= divisor[i];
		}
		return fmt::format("{}ns", time * ns);
	}

	class Timer {
	public:
		explicit Timer(std::string name = "Timer") :
				m_name(std::move(name)), m_start(now<time::nanosecond>()) {}

		~Timer() {
			double end = now<time::nanosecond>();
			fmt::print("[ TIMER ] {} : {}\n", m_name, formatTime<time::nanosecond>(end - m_start));
		}

	private:
		std::string m_name;
		double m_start;
	};

	template<typename LAMBDA, typename... Args>
	double timeFunction(const LAMBDA &op, int64_t samples = -1, int64_t iters = -1,
						double time = -1, Args... vals) {
		if (samples < 1) samples = 10;

		// Call the function to compile any kernels
		op(vals...);

		double loopTime		   = 1e300;
		int64_t itersCompleted = 0;
		if (iters < 1) {
			loopTime	   = 5e9 / (double)samples;
			iters		   = 1e10;
			itersCompleted = 0;
		}

		if (time > 0) {
			loopTime = time * time::second;
			loopTime /= (double)samples;
		}

		std::vector<double> times;

		for (int64_t sample = 0; sample < samples; ++sample) {
			itersCompleted = 0;
			double start   = now<time::nanosecond>();
			while (itersCompleted++ < iters && now<time::nanosecond>() - start < loopTime) {
				op(vals...);
			}
			double end = now<time::nanosecond>();
			times.emplace_back((end - start) / (double)itersCompleted);
		}

		// Calculate average (mean) time and standard deviation
		double avg = mean(times);
		double std = standardDeviation(times);
		fmt::print("Mean {:>9} ± {:>9} after {} samples, each with {} iterations\n",
				   formatTime<time::nanosecond>(avg, "{:.3f}"),
				   formatTime<time::nanosecond>(std, "{:.3f}"),
				   samples,
				   itersCompleted - 1);
		return avg;
	}
} // namespace librapid

#ifndef LIBRAPID_VECTOR
#define LIBRAPID_VECTOR

namespace librapid {
#ifndef LIBRAPID_DOXYGEN_BUILD
#	define MIN_DIM_CLAMP(_dims, _tmpDims) (((_dims) < (_tmpDims)) ? (_dims) : (_tmpDims))
#else
#	define MIN_DIM_CLAMP(_dims, _tmpDims) _dims
#endif

#ifndef LIBRAPID_DOXYGEN_BUILD
#	define MAX_DIM_CLAMP(_dims, _tmpDims) (((_dims) > (_tmpDims)) ? (_dims) : (_tmpDims))
#else
#	define MAX_DIM_CLAMP(_dims, _tmpDims) _dims
#endif

	template<typename DTYPE, int64_t dims>
	class Vec {
		template<typename T>
		using Common = typename std::common_type<DTYPE, T>::type;

	public:
		Vec() = default;

		template<typename X, typename... YZ>
		explicit Vec(X x, YZ... yz) : m_components {(DTYPE)x, (DTYPE)yz...} {
			static_assert(1 + sizeof...(YZ) <= dims, "Parameters cannot exceed vector dimensions");
		}

		template<typename T, int64_t d>
		explicit Vec(const Vec<T, d> &other) {
			int64_t i;
			for (i = 0; i < MIN_DIM_CLAMP(dims, d); ++i) { m_components[i] = other[i]; }
		}

		template<typename T>
		Vec(const Vec<T, 3> &other) {
			x = other.x;
			y = other.y;
			z = other.z;
		}

		Vec(const Vec<DTYPE, dims> &other) {
			int64_t i;
			for (i = 0; i < dims; ++i) { m_components[i] = other[i]; }
		}

		Vec<DTYPE, dims> &operator=(const Vec<DTYPE, dims> &other) {
			if (this == &other) { return *this; }
			for (int64_t i = 0; i < dims; ++i) { m_components[i] = other[i]; }
			return *this;
		}

		// Implement conversion to and from GLM datatypes
#ifdef GLM_VERSION

		template<typename T, int tmpDim, glm::qualifier p = glm::defaultp>
		Vec(const glm::vec<tmpDim, T, p> &vec) {
			for (int64_t i = 0; i < tmpDim; ++i) {
				m_components[i] = (i < dims) ? ((T)vec[i]) : (T());
			}
		}

		template<typename T = DTYPE, int tmpDim = dims, glm::qualifier p = glm::defaultp>
		operator glm::vec<tmpDim, T, p>() const {
			glm::vec<tmpDim, T, p> res;
			for (int64_t i = 0; i < dims; ++i) {
				res[i] = (i < dims) ? ((T)m_components[i]) : (T());
			}
			return res;
		}

#endif // GLM_VERSION

		/**
		 * Implement indexing (const and non-const)
		 * Functions take a single index and return a scalar value
		 */

		const DTYPE &operator[](int64_t index) const { return m_components[index]; }

		DTYPE &operator[](int64_t index) { return m_components[index]; }

		template<typename T, int64_t tmpDims>
		bool operator==(const Vec<T, tmpDims> &other) const {
			// For vectors with different dimensions, return true if the excess
			// values are all zero
			for (int64_t i = 0; i < MIN_DIM_CLAMP(dims, tmpDims); ++i) {
				if (m_components[i] != other[i]) return false;
			}

			// Quick return to avoid excess checks
			if (dims == tmpDims) return true;

			for (int64_t i = MIN_DIM_CLAMP(dims, tmpDims); i < MAX_DIM_CLAMP(dims, tmpDims); ++i) {
				if (i < dims && m_components[i]) return false;
				if (i < tmpDims && other[i]) return false;
			}

			return true;
		}

		template<typename T, int64_t tmpDims>
		bool operator!=(const Vec<T, tmpDims> &other) const {
			return !(*this == other);
		}

		/**
		 * Implement simple arithmetic operators + - * /
		 *
		 * Operations take two Vec objects and return a new vector (with common
		 * type) containing the result of the element-wise operation.
		 *
		 * Vectors must have same dimensions. To cast, use Vec.as<TYPE, DIMS>()
		 */
		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator+(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) + ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator-(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) - ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator*(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) * ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> operator/(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(dims, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(dims, tmpDims)); ++i) {
				res[i] = ((i < dims) ? m_components[i] : 0) / ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		/**
		 * Implement simple arithmetic operators + - * /
		 *
		 * Operations take a vector and a scalar, and return a new vector (with
		 * common type) containing the result of the element-wise operation.
		 */

		template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
		Vec<Common<T>, dims> operator+(const T &other) const {
			Vec<Common<T>, dims> res;
			for (int64_t i = 0; i < dims; ++i) { res[i] = m_components[i] + other; }
			return res;
		}

		template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
		Vec<Common<T>, dims> operator-(const T &other) const {
			Vec<Common<T>, dims> res;
			for (int64_t i = 0; i < dims; ++i) { res[i] = m_components[i] - other; }
			return res;
		}

		template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
		Vec<Common<T>, dims> operator*(const T &other) const {
			Vec<Common<T>, dims> res;
			for (int64_t i = 0; i < dims; ++i) { res[i] = m_components[i] * other; }
			return res;
		}

		template<typename T, typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
		Vec<Common<T>, dims> operator/(const T &other) const {
			Vec<Common<T>, dims> res;
			for (int64_t i = 0; i < dims; ++i) { res[i] = m_components[i] / other; }
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, dims> &operator+=(const Vec<T, tmpDims> &other) {
			for (int64_t i = 0; i < dims; ++i) {
				m_components[i] += (i < tmpDims) ? (other[i]) : (0);
			}
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, dims> &operator-=(const Vec<T, tmpDims> &other) {
			for (int64_t i = 0; i < dims; ++i) {
				m_components[i] -= (i < tmpDims) ? (other[i]) : (0);
			}
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, dims> &operator*=(const Vec<T, tmpDims> &other) {
			for (int64_t i = 0; i < dims; ++i) {
				m_components[i] *= (i < tmpDims) ? (other[i]) : (0);
			}
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, dims> &operator/=(const Vec<T, tmpDims> &other) {
			for (int64_t i = 0; i < dims; ++i) {
				m_components[i] /= (i < tmpDims) ? (other[i]) : (0);
			}
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator+=(const T &other) {
			for (int64_t i = 0; i < dims; ++i) { m_components[i] += other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator-=(const T &other) {
			for (int64_t i = 0; i < dims; ++i) { m_components[i] -= other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator*=(const T &other) {
			for (int64_t i = 0; i < dims; ++i) { m_components[i] *= other; }
			return *this;
		}

		template<typename T>
		Vec<DTYPE, dims> &operator/=(const T &other) {
			for (int64_t i = 0; i < dims; ++i) { m_components[i] /= other; }
			return *this;
		}

		/**
		 * Return the magnitude squared of a vector
		 */
		DTYPE mag2() const {
			DTYPE res = 0;
			for (int64_t i = 0; i < dims; ++i) { res += m_components[i] * m_components[i]; }
			return res;
		}

		/**
		 * Return the magnitude of a vector
		 */
		DTYPE mag() const { return sqrt(mag2()); }

		DTYPE invMag() const { return DTYPE(1) / sqrt(mag2()); }

		template<typename T, int64_t tmpDims>
		typename std::common_type<DTYPE, T>::type dist2(const Vec<T, tmpDims> &other) const {
			using RET	= typename std::common_type<DTYPE, T>::type;
			RET squared = 0;
			int64_t i	= 0;

			// Compute the squares of the differences for the matching
			// components
			for (; i < MIN_DIM_CLAMP(dims, tmpDims); ++i) {
				squared += (m_components[i] - other[i]) * (m_components[i] - other[i]);
			}

			// Compute the squares of the values for the remaining values.
			// This just enables calculating the distance between two vectors
			// with different dimensions
			for (; i < MAX_DIM_CLAMP(dims, tmpDims); ++i) {
				if (i < dims)
					squared += m_components[i] * m_components[i];
				else
					squared += other[i] * other[i];
			}

			return squared;
		}

		template<typename T, int64_t tmpDims>
		typename std::common_type<DTYPE, T>::type dist(const Vec<T, tmpDims> &other) const {
			return sqrt(dist2(other));
		}

		/**
		 * Compute the vector dot product
		 * AxBx + AyBy + AzCz + ...
		 */
		template<typename T>
		Common<T> dot(const Vec<T, dims> &other) const {
			Common<T> res = 0;
			for (int64_t i = 0; i < dims; ++i) { res += m_components[i] * other[i]; }
			return res;
		}

		/**
		 * Compute the vector cross product
		 */
		template<typename T>
		Vec<Common<T>, dims> cross(const Vec<T, dims> &other) const {
			static_assert(dims == 2 || dims == 3,
						  "Only 2D and 3D vectors support the cross product");

			Vec<Common<T>, dims> res;

			if constexpr (dims == 2) {
				m_components[2] = 0;
				other[2]		= 0;
			}

			res.x = y * other.z - z * other.y;
			res.y = z * other.x - x * other.z;
			res.z = x * other.y - y * other.x;

			return res;
		}

		inline Vec<DTYPE, 2> xy() const { return {x, y}; }

		inline Vec<DTYPE, 2> yx() const { return {y, x}; }

		inline Vec<DTYPE, 3> xyz() const { return {x, y, z}; }

		inline Vec<DTYPE, 3> xzy() const { return {x, z, y}; }

		inline Vec<DTYPE, 3> yxz() const { return {y, x, z}; }

		inline Vec<DTYPE, 3> yzx() const { return {y, z, x}; }

		inline Vec<DTYPE, 3> zxy() const { return {z, x, y}; }

		inline Vec<DTYPE, 3> zyx() const { return {z, y, x}; }

		inline Vec<DTYPE, 4> xyzw() const { return {x, y, z, w}; }

		inline Vec<DTYPE, 4> xywz() const { return {x, y, w, z}; }

		inline Vec<DTYPE, 4> xzyw() const { return {x, z, y, w}; }

		inline Vec<DTYPE, 4> xzwy() const { return {x, z, w, y}; }

		inline Vec<DTYPE, 4> xwyz() const { return {x, w, y, z}; }

		inline Vec<DTYPE, 4> xwzy() const { return {x, w, z, y}; }

		inline Vec<DTYPE, 4> yxzw() const { return {y, x, z, w}; }

		inline Vec<DTYPE, 4> yxwz() const { return {y, x, w, z}; }

		inline Vec<DTYPE, 4> yzxw() const { return {y, z, x, w}; }

		inline Vec<DTYPE, 4> yzwx() const { return {y, z, w, x}; }

		inline Vec<DTYPE, 4> ywxz() const { return {y, w, x, z}; }

		inline Vec<DTYPE, 4> ywzx() const { return {y, w, z, x}; }

		inline Vec<DTYPE, 4> zxyw() const { return {z, x, y, w}; }

		inline Vec<DTYPE, 4> zxwy() const { return {z, x, w, y}; }

		inline Vec<DTYPE, 4> zyxw() const { return {z, y, x, w}; }

		inline Vec<DTYPE, 4> zywx() const { return {z, y, w, x}; }

		inline Vec<DTYPE, 4> zwxy() const { return {z, w, x, y}; }

		inline Vec<DTYPE, 4> zwyx() const { return {z, w, y, x}; }

		inline Vec<DTYPE, 4> wxyz() const { return {w, x, y, z}; }

		inline Vec<DTYPE, 4> wxzy() const { return {w, x, z, y}; }

		inline Vec<DTYPE, 4> wyxz() const { return {w, y, x, z}; }

		inline Vec<DTYPE, 4> wyzx() const { return {w, y, z, x}; }

		inline Vec<DTYPE, 4> wzxy() const { return {w, z, x, y}; }

		inline Vec<DTYPE, 4> wzyx() const { return {w, z, y, x}; }

		[[nodiscard]] std::string str() const {
			std::string res = "(";
			for (int64_t i = 0; i < dims; ++i) {
				res += std::to_string(m_components[i]) + (i == dims - 1 ? ")" : ", ");
			}
			return res;
		}

		void setX(DTYPE val) { x = val; }

		void setY(DTYPE val) { y = val; }

		void setZ(DTYPE val) { z = val; }

		void setW(DTYPE val) { w = val; }

		DTYPE getX() { return x; }

		DTYPE getY() { return y; }

		DTYPE getZ() { return z; }

		DTYPE getW() { return w; }

		DTYPE &x = m_components[0];
		DTYPE &y = m_components[1];
		DTYPE &z = m_components[2];
		DTYPE &w = m_components[3];

	private:
		DTYPE m_components[dims < 4 ? 4 : dims];
	};

	/**
	 * Implement simple arithmetic operators + - * /
	 *
	 * Operations take a scalar and a vector and return a new vector (with
	 * common type) containing the result of the element-wise operation.
	 */

	template<typename T, typename DTYPE, int64_t dims,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator+(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (int64_t i = 0; i < dims; ++i) { res[i] = value + vec[i]; }
		return res;
	}

	template<typename T, typename DTYPE, int64_t dims,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator-(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (int64_t i = 0; i < dims; ++i) { res[i] = value - vec[i]; }
		return res;
	}

	template<typename T, typename DTYPE, int64_t dims,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator*(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (int64_t i = 0; i < dims; ++i) { res[i] = value * vec[i]; }
		return res;
	}

	template<typename T, typename DTYPE, int64_t dims,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, dims> operator/(const T &value,
																   const Vec<DTYPE, dims> &vec) {
		Vec<typename std::common_type<T, DTYPE>::type, dims> res;
		for (int64_t i = 0; i < dims; ++i) { res[i] = value / vec[i]; }
		return res;
	}

	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================
	// ==============================================================================================

	template<typename DTYPE>
	class Vec<DTYPE, 3> {
		template<typename T>
		using Common = typename std::common_type<DTYPE, T>::type;

	public:
		Vec() = default;

		template<typename X = DTYPE, typename Y = DTYPE, typename Z = DTYPE>
		explicit Vec(X x, Y y = 0, Z z = 0) : x(x), y(y), z(z) {}

		template<typename T, int64_t d>
		explicit Vec(const Vec<T, d> &other) {
			x = other.x;
			y = other.y;
			z = other.z;
		}

		Vec(const Vec<DTYPE, 3> &other) {
			x = other.x;
			y = other.y;
			z = other.z;
		}

		Vec<DTYPE, 3> &operator=(const Vec<DTYPE, 3> &other) {
			if (this == &other) { return *this; }
			x = other.x;
			y = other.y;
			z = other.z;
			return *this;
		}

		// Implement conversion to and from GLM datatypes
#ifdef GLM_VERSION

		template<typename T, glm::qualifier p = glm::defaultp>
		Vec(const glm::vec<2, T, p> &vec) {
			x = vec.x;
			y = vec.y;
		}

		template<typename T, glm::qualifier p = glm::defaultp>
		Vec(const glm::vec<3, T, p> &vec) {
			x = vec.x;
			y = vec.y;
			z = vec.z;
		}

		template<typename T, int tmpDim, glm::qualifier p = glm::defaultp>
		operator glm::vec<tmpDim, T, p>() const {
			glm::vec<tmpDim, T, p> res;
			for (int64_t i = 0; i < tmpDim; ++i) { res[i] = (i < 3) ? ((&x)[i]) : (T(0)); }
			return res;
		}

#endif // GLM_VERSION

		/**
		 * Implement indexing (const and non-const)
		 * Functions take a single index and return a scalar value
		 */

		const DTYPE &operator[](int64_t index) const { return (&x)[index]; }

		DTYPE &operator[](int64_t index) { return (&x)[index]; }

		template<typename T, int64_t tmpDims>
		bool operator==(const Vec<T, tmpDims> &other) const {
			if (tmpDims <= 3) {
				return x == other.x && y == other.y && z == other.z && w == other.w;
			}

			for (int64_t i = 3; i < tmpDims; ++i) {
				if (other[i]) return false;
			}

			return true;
		}

		template<typename T, int64_t tmpDims>
		bool operator!=(const Vec<T, tmpDims> &other) const {
			return !(*this == other);
		}

		/**
		 * Implement simple arithmetic operators + - * /
		 *
		 * Operations take two Vec objects and return a new vector (with
		 * common type) containing the result of the element-wise operation.
		 *
		 * Vectors must have same dimensions. To cast, use Vec.as<TYPE,
		 * DIMS>()
		 */
		template<typename T>
		Vec<Common<T>, 3> operator+(const Vec<T, 3> &other) const {
			return Vec<Common<T>, 3>(x + other.x, y + other.y, z + other.z);
		}

		template<typename T>
		Vec<Common<T>, 3> operator-(const Vec<T, 3> &other) const {
			return Vec<Common<T>, 3>(x - other.x, y - other.y, z - other.z);
		}

		template<typename T>
		Vec<Common<T>, 3> operator*(const Vec<T, 3> &other) const {
			return Vec<Common<T>, 3>(x * other.x, y * other.y, z * other.z);
		}

		template<typename T>
		Vec<Common<T>, 3> operator/(const Vec<T, 3> &other) const {
			return Vec<Common<T>, 3>(x / other.x, y / other.y, z / other.z);
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> operator+(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(3, tmpDims)); ++i) {
				res[i] = ((i < 3) ? (&x)[i] : 0) + ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> operator-(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(3, tmpDims)); ++i) {
				res[i] = ((i < 3) ? (&x)[i] : 0) - ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> operator*(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(3, tmpDims)); ++i) {
				res[i] = ((i < 3) ? (&x)[i] : 0) * ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		template<typename T, int64_t tmpDims>
		Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> operator/(const Vec<T, tmpDims> &other) const {
			Vec<Common<T>, MAX_DIM_CLAMP(3, tmpDims)> res;
			for (int64_t i = 0; i < (MAX_DIM_CLAMP(3, tmpDims)); ++i) {
				res[i] = ((i < 3) ? (&x)[i] : 0) / ((i < tmpDims) ? other[i] : 0);
			}
			return res;
		}

		/**
		 * Implement simple arithmetic operators + - * /
		 *
		 * Operations take a vector and a scalar, and return a new vector
		 * (with common type) containing the result of the element-wise
		 * operation.
		 */

		template<typename T>
		Vec<Common<T>, 3> operator+(const T &other) const {
			return Vec<Common<T>, 3>(x + other, y + other, z + other);
		}

		template<typename T>
		Vec<Common<T>, 3> operator-(const T &other) const {
			return Vec<Common<T>, 3>(x - other, y - other, z - other);
		}

		template<typename T>
		Vec<Common<T>, 3> operator*(const T &other) const {
			return Vec<Common<T>, 3>(x * other, y * other, z * other);
		}

		template<typename T>
		Vec<Common<T>, 3> operator/(const T &other) const {
			return Vec<Common<T>, 3>(x / other, y / other, z / other);
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, 3> &operator+=(const Vec<T, tmpDims> &other) {
			x += other.x;
			y += other.y;
			z += other.z;
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, 3> &operator-=(const Vec<T, tmpDims> &other) {
			x -= other.x;
			y -= other.y;
			z -= other.z;
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, 3> &operator*=(const Vec<T, tmpDims> &other) {
			x *= other.x;
			y *= other.y;
			z *= other.z;
			return *this;
		}

		template<typename T, int64_t tmpDims>
		Vec<DTYPE, 3> &operator/=(const Vec<T, tmpDims> &other) {
			x /= other.x;
			y /= other.y;
			z /= other.z;
			return *this;
		}

		template<typename T>
		Vec<DTYPE, 3> &operator+=(const T &other) {
			x += other;
			y += other;
			z += other;
			return *this;
		}

		template<typename T>
		Vec<DTYPE, 3> &operator-=(const T &other) {
			x -= other;
			y -= other;
			z -= other;
			return *this;
		}

		template<typename T>
		Vec<DTYPE, 3> &operator*=(const T &other) {
			x *= other;
			y *= other;
			z *= other;
			return *this;
		}

		template<typename T>
		Vec<DTYPE, 3> &operator/=(const T &other) {
			x /= other;
			y /= other;
			z /= other;
			return *this;
		}

		/**
		 * Return the magnitude squared of a vector
		 */
		DTYPE mag2() const { return x * x + y * y + z * z; }

		/**
		 * Return the magnitude of a vector
		 */
		DTYPE mag() const { return sqrt(x * x + y * y + z * z); }

		DTYPE invMag() const {
			DTYPE mag = x * x + y * y + z * z;
			return DTYPE(1) / mag;
		}

		template<typename T, int64_t tmpDims>
		typename std::common_type<DTYPE, T>::type dist2(const Vec<T, tmpDims> &other) const {
			// Specific case for a 2D vector
			if constexpr (tmpDims == 2) {
				return ((x - other.x) * (x - other.x)) + ((y - other.y) * (y - other.y)) + (z * z);
			}

			// Specific case for a 3D vector
			if constexpr (tmpDims == 3) {
				return ((x - other.x) * (x - other.x)) + ((y - other.y) * (y - other.y)) +
					   ((z - other.z) * (z - other.z));
			}

			// Specific case for a 4D vector
			if constexpr (tmpDims == 4) {
				return ((x - other.x) * (x - other.x)) + ((y - other.y) * (y - other.y)) +
					   ((z - other.z) * (z - other.z)) + (other.w * other.w);
			}

			// General case for 1, 5, 6, 7, ... dimensional vectors
			using RET	= typename std::common_type<DTYPE, T>::type;
			RET squared = 0;
			int64_t i	= 0;

			// Compute the squares of the differences for the matching
			// components
			for (; i < MIN_DIM_CLAMP(3, tmpDims); ++i) {
				squared += ((&x)[i] - other[i]) * ((&x)[i] - other[i]);
			}

			// Compute the squares of the values for the remaining values.
			// This just enables calculating the distance between two vectors
			// with different dimensions
			for (; i < MAX_DIM_CLAMP(3, tmpDims); ++i) {
				if (i < 3)
					squared += (&x)[i] * (&x)[i];
				else
					squared += other[i] * other[i];
			}
		}

		template<typename T, int64_t tmpDims>
		typename std::common_type<DTYPE, T>::type dist(const Vec<T, tmpDims> &other) const {
			return sqrt(dist2(other));
		}

		/**
		 * Compute the vector dot product
		 * AxBx + AyBy + AzCz + ...
		 */
		template<typename T>
		Common<T> dot(const Vec<T, 3> &other) const {
			return x * other.x + y * other.y + z * other.z;
		}

		/**
		 * Compute the vector cross product
		 */
		template<typename T>
		Vec<Common<T>, 3> cross(const Vec<T, 3> &other) const {
			return Vec<Common<T>, 3>(
			  y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x);
		}

		// Swizzle Operations

		inline Vec<DTYPE, 2> xy() const { return {x, y}; }

		inline Vec<DTYPE, 2> yx() const { return {y, x}; }

		inline Vec<DTYPE, 3> xyz() const { return {x, y, z}; }

		inline Vec<DTYPE, 3> xzy() const { return {x, z, y}; }

		inline Vec<DTYPE, 3> yxz() const { return {y, x, z}; }

		inline Vec<DTYPE, 3> yzx() const { return {y, z, x}; }

		inline Vec<DTYPE, 3> zxy() const { return {z, x, y}; }

		inline Vec<DTYPE, 3> zyx() const { return {z, y, x}; }

		inline Vec<DTYPE, 4> xyzw() const { return {x, y, z, w}; }

		inline Vec<DTYPE, 4> xywz() const { return {x, y, w, z}; }

		inline Vec<DTYPE, 4> xzyw() const { return {x, z, y, w}; }

		inline Vec<DTYPE, 4> xzwy() const { return {x, z, w, y}; }

		inline Vec<DTYPE, 4> xwyz() const { return {x, w, y, z}; }

		inline Vec<DTYPE, 4> xwzy() const { return {x, w, z, y}; }

		inline Vec<DTYPE, 4> yxzw() const { return {y, x, z, w}; }

		inline Vec<DTYPE, 4> yxwz() const { return {y, x, w, z}; }

		inline Vec<DTYPE, 4> yzxw() const { return {y, z, x, w}; }

		inline Vec<DTYPE, 4> yzwx() const { return {y, z, w, x}; }

		inline Vec<DTYPE, 4> ywxz() const { return {y, w, x, z}; }

		inline Vec<DTYPE, 4> ywzx() const { return {y, w, z, x}; }

		inline Vec<DTYPE, 4> zxyw() const { return {z, x, y, w}; }

		inline Vec<DTYPE, 4> zxwy() const { return {z, x, w, y}; }

		inline Vec<DTYPE, 4> zyxw() const { return {z, y, x, w}; }

		inline Vec<DTYPE, 4> zywx() const { return {z, y, w, x}; }

		inline Vec<DTYPE, 4> zwxy() const { return {z, w, x, y}; }

		inline Vec<DTYPE, 4> zwyx() const { return {z, w, y, x}; }

		inline Vec<DTYPE, 4> wxyz() const { return {w, x, y, z}; }

		inline Vec<DTYPE, 4> wxzy() const { return {w, x, z, y}; }

		inline Vec<DTYPE, 4> wyxz() const { return {w, y, x, z}; }

		inline Vec<DTYPE, 4> wyzx() const { return {w, y, z, x}; }

		inline Vec<DTYPE, 4> wzxy() const { return {w, z, x, y}; }

		inline Vec<DTYPE, 4> wzyx() const { return {w, z, y, x}; }

		[[nodiscard]] std::string str() const {
			return std::string("(") + std::to_string(x) + ", " + std::to_string(y) + ", " +
				   std::to_string(z) + ")";
		}

		void setX(DTYPE val) { x = val; }

		void setY(DTYPE val) { y = val; }

		void setZ(DTYPE val) { z = val; }

		void setW(DTYPE val) { w = val; }

		DTYPE getX() { return x; }

		DTYPE getY() { return y; }

		DTYPE getZ() { return z; }

		DTYPE getW() { return w; }

		DTYPE x = 0;
		DTYPE y = 0;
		DTYPE z = 0;
		DTYPE w = 0;
	};

	/**
	 * Implement simple arithmetic operators + - * /
	 *
	 * Operations take a scalar and a vector and return a new vector (with
	 * common type) containing the result of the element-wise operation.
	 */

	template<typename T, typename DTYPE,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, 3> operator+(const T &value,
																const Vec<DTYPE, 3> &vec) {
		return Vec<typename std::common_type<T, DTYPE>::type, 3>(
		  value + vec.x, value + vec.y, value + vec.z);
	}

	template<typename T, typename DTYPE,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, 3> operator-(const T &value,
																const Vec<DTYPE, 3> &vec) {
		return Vec<typename std::common_type<T, DTYPE>::type, 3>(
		  value - vec.x, value - vec.y, value - vec.z);
	}

	template<typename T, typename DTYPE,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, 3> operator*(const T &value,
																const Vec<DTYPE, 3> &vec) {
		return Vec<typename std::common_type<T, DTYPE>::type, 3>(
		  value * vec.x, value * vec.y, value * vec.z);
	}

	template<typename T, typename DTYPE,
			 typename std::enable_if<std::is_scalar<T>::value, int>::type = 0>
	Vec<typename std::common_type<T, DTYPE>::type, 3> operator/(const T &value,
																const Vec<DTYPE, 3> &vec) {
		return Vec<typename std::common_type<T, DTYPE>::type, 3>(
		  value / vec.x, value / vec.y, value / vec.z);
	}

	using Vec2i = Vec<int64_t, 2>;
	using Vec2f = Vec<float, 2>;
	using Vec2d = Vec<double, 2>;

	using Vec3i = Vec<int64_t, 3>;
	using Vec3f = Vec<float, 3>;
	using Vec3d = Vec<double, 3>;

	using Vec4i = Vec<int64_t, 4>;
	using Vec4f = Vec<float, 4>;
	using Vec4d = Vec<double, 4>;

	template<typename T, int64_t dims>
	std::ostream &operator<<(std::ostream &os, const Vec<T, dims> &vec) {
		return os << vec.str();
	}
} // namespace librapid

#ifdef FMT_API
template<typename T, int64_t D>
struct fmt::formatter<librapid::Vec<T, D>> {
	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		return ctx.begin();
	}

	template<typename FormatContext>
	auto format(const librapid::Vec<T, D> &arr, FormatContext &ctx) {
		return fmt::format_to(ctx.out(), arr.str());
	}
};
#endif // FMT_API

#endif // LIBRAPID_VECTOR

#if defined(LIBRAPID_OS_WINDOWS)
#	define WIN32_LEAN_AND_MEAN
#	include <Windows.h>
#elif defined(LIBRAPID_OS_UNIX)
#	include <sys/ioctl.h>
#	include <unistd.h>
#endif

namespace librapid {
	struct ConsoleSize {
		int rows, cols;
	};

#if defined(LIBRAPID_OS_WINDOWS)
	ConsoleSize getConsoleSize() {
		static CONSOLE_SCREEN_BUFFER_INFO bufferInfo;
		int cols, rows;
		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &bufferInfo);
		cols = bufferInfo.srWindow.Right - bufferInfo.srWindow.Left + 1;
		rows = bufferInfo.srWindow.Bottom - bufferInfo.srWindow.Top + 1;
		return {rows, cols};
	}
#elif defined(LIBRAPID_OS_UNIX)
	ConsoleSize getConsoleSize() {
		static struct winsize w;
		ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
		return {w.ws_row, w.ws_col};
	}
#else
	ConsoleSize getConsoleSize() {
		// Not a clue what this would run on, or how it would be done
		// correctly, so just return some arbitrary values...
		return {30, 120};
	}
#endif
} // namespace librapid

#include <cfloat>

namespace librapid {
#define CTYPED static const double

	// 32bit float minimum value
	CTYPED EPSILON32 = FLT_MIN;
	// 64bit float minimum value
	CTYPED EPSILON64 = DBL_MIN;
	// π² / 6.
	CTYPED PISQRDIV6 =
	  1.6449340668482264364724151666460251892189499012067984377355582293;
	// 180 / π.
	CTYPED RADTODEG =
	  57.295779513082320876798154814105170332405472466564321549160243861;
	// π / 180.
	CTYPED DEGTORAD =
	  0.0174532925199432957692369076848861271344287188854172545609719144;
	// π.
	CTYPED PI =
	  3.1415926535897932384626433832795028841971693993751058209749445923;
	// √π.
	CTYPED SQRTPI =
	  1.7724538509055160272981674833411451827975494561223871282138077898;
	// τ.
	CTYPED TAU =
	  6.2831853071795864769252867665590057683943387987502116419498891846;
	// π / 2.
	CTYPED HALFPI =
	  1.5707963267948966192313216916397514420985846996875529104874722961;
	// π x 2.
	CTYPED TWOPI =
	  6.2831853071795864769252867665590057683943387987502116419498891846156;
	// Eulers number.
	CTYPED E =
	  2.7182818284590452353602874713526624977572470936999595749669676277;
	// The square root of Eulers number.
	CTYPED SQRTE =
	  1.6487212707001281468486507878141635716537761007101480115750793116;
	// √2.
	CTYPED SQRT2 =
	  1.4142135623730950488016887242096980785696718753769480731766797379;
	// √3.
	CTYPED SQRT3 =
	  1.7320508075688772935274463415058723669428052538103806280558069794;
	// √5.
	CTYPED SQRT5 =
	  2.2360679774997896964091736687312762354406183596115257242708972454;
	// The golden ratio φ.
	CTYPED GOLDENRATIO =
	  1.6180339887498948482045868343656381177203091798057628621354486227;
	// The Euler-Mascheroni constant, γ.
	CTYPED EULERMASCHERONI =
	  0.5772156649015328606065120900824024310421593359399235988057672348;
	// The Twin Primes Constant.
	CTYPED TWINPRIMES =
	  0.6601618158468695739278121100145557784326233602847334133194484233;
	// ln(2).
	CTYPED LN2 =
	  0.6931471805599453094172321214581765680755001343602552541206800094;
	// ln(3).
	CTYPED LN3 =
	  1.0986122886681096913952452369225257046474905578227494517346943336;
	// ln(5).
	CTYPED LN5 =
	  1.6094379124341003746007593332261876395256013542685177219126478914;
	// ζ(3).
	CTYPED ZETA3 =
	  1.2020569031595942853997381615114499907649862923404988817922715553;
	// ∛2.
	CTYPED CUBEROOT2 =
	  1.2599210498948731647672106072782283505702514647015079800819751121;
	// ∛3.
	CTYPED CUBEROOT3 =
	  1.4422495703074083823216383107801095883918692534993505775464161945;
	// the speed of light, c.
	CTYPED LIGHTSPEED = 299792458.0;
	// Gravity of Earth, denoted as g
	CTYPED EARTHGRAVITY = 9.80665;
	// Wallis Constant.
	CTYPED WALLISCONST =
	  2.0945514815423265914823865405793029638573061056282391803041285290;
	// The Laplace limit, λ.
	CTYPED LAPLACELIMIT =
	  0.6627434193491815809747420971092529070562335491150224175203925349;
	// Gauss's constant.
	CTYPED GAUSSCONST =
	  0.8346268416740731862814297327990468089939930134903470024498273701;
	// Cahen's constant, C.
	CTYPED CAHENSCONST =
	  0.6434105462883380261822543077575647632865878602682395059870309203;
	// The parabolic constant, P₂.
	CTYPED P2 =
	  2.2955871493926380740342980491894903875978322036385834839299753466;
	// The Dottie number.
	CTYPED DOTTIENUMBER =
	  0.7390851332151606416553120876738734040134117589007574649656806357;
	// The Meissel-Mertens constant.
	CTYPED MEISSELMERTENS =
	  0.2614972128476427837554268386086958590515666482611992061920642139;
	// Gelfond's constant e^π.
	CTYPED ETOPI =
	  23.140692632779269005729086367948547380266106242600211993445046409;
	// The golden angle.
	CTYPED GOLDENANGLE =
	  2.3999632297286533222315555066336138531249990110581150429351127507;
	// The area of the Mandelbrot fractal.
	CTYPED MANDELBROTAREA = 1.5065918849;
	// The Gieseking constant.
	CTYPED GIESEKINGCONST =
	  1.0149416064096536250212025542745202859416893075302997920174891067;
	// The Bloch-Landau constant.
	CTYPED BLOCHLANDAU =
	  0.5432589653429767069527282953006132311388632937583569889557325691;
	// The Golomb-Dickman constant.
	CTYPED GOLOMBDICKMAN =
	  0.6243299885435508709929363831008372441796426201805292869735519024;
	// The Feller-Tornier constant.
	CTYPED FELLERTORNIER = 0.6613170494696223352897658462741185332854752898329;
	// The 2^(√2).
	CTYPED TWOTOSQRT2 =
	  2.6651441426902251886502972498731398482742113137146594928359795933;
	// The Khinchin's constant.
	CTYPED KHINCHINSCONST =
	  2.6854520010653064453097148354817956938203822939944629530511523455;
	// Mill's constant.
	CTYPED MILLSCONST =
	  1.3063778838630806904686144926026057129167845851567136443680537599;
	// π / ln(2).
	CTYPED PIOVERLN2 =
	  4.5323601418271938096276829457166668101718614677237955841860165479;
	// Loch's constant.
	CTYPED LOCHSCONST =
	  0.9702701143920339257402560192100108337812847047851612866103505299;
	// Niven's constant.
	CTYPED NIVENSCONST =
	  1.7052111401053677642885514534345081607620276516534690999942849065;
	// Reciprocal Fibonacci constant.
	CTYPED RECIPFIBCONST =
	  3.3598856662431775531720113029189271796889051337319684864955538153;
	// Backhouse's constant.
	CTYPED BACKHOUSECONST =
	  1.4560749485826896713995953511165435576531783748471315402707024374;
	// The MRB constant.
	CTYPED MRBCONST =
	  0.1878596424620671202485179340542732300559030949001387861720046840;
	// Somos' quadratic recurrence constant.
	CTYPED QUADRECURR =
	  1.6616879496335941212958189227499507499644186350250682081897111680;
	// The plastic number.
	CTYPED PLASTICNUMBER =
	  1.3247179572447460259609088544780973407344040569017333645340150503;
} // namespace librapid

#ifndef LIBRAPID_CORE_MATH
#define LIBRAPID_CORE_MATH

namespace librapid {
	int64_t product(const std::vector<int64_t> &vals) {
		int64_t res = 1;
		for (const auto &val : vals) res *= val;
		return res;
	}

	double product(const std::vector<double> &vals) {
		double res = 1;
		for (const auto &val : vals) res *= val;
		return res;
	}

	template<typename T>
	T min(const std::vector<T> &vals) {
		T min_found = 0;
		for (const auto &val : vals)
			if (val < min_found) min_found = val;
		return min_found;
	}

	template<typename T>
	T &&min(T &&val) {
		return std::forward<T>(val);
	}

	template<typename T0, typename T1, typename... Ts>
	inline auto min(T0 &&val1, T1 &&val2, Ts &&...vs) {
		return (val1 < val2) ? min(val1, std::forward<Ts>(vs)...)
							 : min(val2, std::forward<Ts>(vs)...);
	}

	template<typename T>
	T max(const std::vector<T> &vals) {
		T min_found = 0;
		for (const auto &val : vals)
			if (val > min_found) min_found = val;
		return min_found;
	}

	template<typename T>
	inline T &&max(T &&val) {
		return std::forward<T>(val);
	}

	template<typename T0, typename T1, typename... Ts>
	inline auto max(T0 &&val1, T1 &&val2, Ts &&...vs) {
		return (val1 > val2) ? max(val1, std::forward<Ts>(vs)...)
							 : max(val2, std::forward<Ts>(vs)...);
	}

	template<typename T>
	inline T abs(T a) {
		return std::abs(a);
	}

	template<typename A, typename B,
			 typename std::enable_if_t<std::is_fundamental_v<A> && std::is_fundamental_v<B>> = 0>
	inline A pow(A a, B exp) {
		return std::pow(a, exp);
	}

	template<typename T>
	inline T sqrt(T a) {
		return std::sqrt(a);
	}

	template<typename T>
	inline T exp(T a) {
		return std::exp(a);
	}

	template<typename T>
	inline T pow(T a, T power) {
		return std::pow(a, power);
	}

	template<typename T>
	inline T ln(T a) {
		return std::log(a);
	}

	template<typename T>
	inline T log2(T a) {
		return std::log2(a);
	}

	template<typename T>
	inline T log10(T a) {
		return std::log10(a);
	}

	template<typename T>
	inline T log(T a, T base) {
		return ln(a) / ln(base);
	}

	template<typename T, typename std::enable_if_t<std::is_fundamental_v<T>> = 0>
	inline T exp(T a) {
		return std::exp(a);
	}

	template<typename T>
	inline T sin(T a) {
		return std::sin(a);
	}

	template<typename T>
	inline T cos(T a) {
		return std::cos(a);
	}

	template<typename T>
	inline T tan(T a) {
		return std::tan(a);
	}

	template<typename T>
	inline T asin(T a) {
		return std::asin(a);
	}

	template<typename T>
	inline T acos(T a) {
		return std::acos(a);
	}

	template<typename T>
	inline T atan(T a) {
		return std::atan(a);
	}

	template<typename T>
	inline T sinh(T a) {
		return std::sinh(a);
	}

	template<typename T>
	inline T cosh(T a) {
		return std::cosh(a);
	}

	template<typename T>
	inline T tanh(T a) {
		return std::tanh(a);
	}

	template<typename T>
	inline T asinh(T a) {
		return std::asinh(a);
	}

	template<typename T>
	inline T acosh(T a) {
		return std::acosh(a);
	}

	template<typename T>
	inline T atanh(T a) {
		return std::atanh(a);
	}

	template<typename T>
	T map(T val, T start1, T stop1, T start2, T stop2) {
		return start2 + (stop2 - start2) * ((val - start1) / (stop1 - start1));
	}

	inline double random(double lower = 0, double upper = 1, uint64_t seed = -1) {
		// Random floating point value in range [lower, upper)
		static std::uniform_real_distribution<double> distribution(0., 1.);
		static std::mt19937 generator(seed == (uint64_t)-1 ? (unsigned int)(now() * 10) : seed);
		return lower + (upper - lower) * distribution(generator);
	}

	template<typename T, typename std::enable_if_t<std::is_integral_v<T>, int> = 0>
	inline T randint(T lower, T upper, uint64_t seed = -1) {
		// Random integral value in range [lower, upper]
		return (int64_t)random((double)(lower - (lower < 0 ? 1 : 0)), (double)upper + 1, seed);
	}

	inline double trueRandomEntropy() {
		static std::random_device rd;
		return rd.entropy();
	}

	template<typename T = double>
	inline double trueRandom(T lower = 0, T upper = 1) {
		// Truly random value in range [lower, upper)
		static std::random_device rd;
		std::uniform_real_distribution<double> dist((double)lower, (double)upper);
		return dist(rd);
	}

	inline int64_t trueRandint(int64_t lower, int64_t upper) {
		// Truly random value in range [lower, upper)
		return (int64_t)trueRandom((double)(lower - (lower < 0 ? 1 : 0)), (double)upper + 1);
	}

	/**
	 * Adapted from
	 * https://docs.oracle.com/javase/6/docs/api/java/util/Random.html#nextGaussian()
	 */
	inline double randomGaussian() {
		static double nextGaussian;
		static bool hasNextGaussian = false;

		double res;
		if (hasNextGaussian) {
			hasNextGaussian = false;
			res				= nextGaussian;
		} else {
			double v1, v2, s;
			do {
				v1 = 2 * random() - 1; // between -1.0 and 1.0
				v2 = 2 * random() - 1; // between -1.0 and 1.0
				s  = v1 * v1 + v2 * v2;
			} while (s >= 1 || s == 0);
			double multiplier = sqrt(-2 * ln(s) / s);
			nextGaussian	  = v2 * multiplier;
			hasNextGaussian	  = true;
			res				  = v1 * multiplier;
		}

		return res;
	}

	template<typename T = double>
	auto pow10(int64_t exponent) {
		using Scalar = typename internal::traits<T>::Scalar;

		const static Scalar pows[] = {
		  0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000};
		if (exponent >= -5 && exponent <= 5) return pows[exponent + 5];

		Scalar res = 1;

		if (exponent > 0)
			for (int64_t i = 0; i < exponent; i++) res *= 10;
		else
			for (int64_t i = 0; i > exponent; i--) res *= 0.1;

		return res;
	}

	template<typename T1, typename T2,
			 typename std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>, int> = 0>
	auto mod(T1 val, T2 divisor) {
		return val % divisor;
	}

	template<typename T1, typename T2,
			 typename std::enable_if_t<std::is_floating_point_v<T1> || std::is_floating_point_v<T2>,
									   int> = 0>
	auto mod(T1 val, T2 divisor) {
		return std::fmod(val, divisor);
	}

	namespace roundMode {
		// Rounding Mode Information:
		// Bit mask:
		// [0] -> Round up if difference >= 0.5
		// [1] -> Round up if difference < 0.5
		// [2] -> Round to nearest even
		// [3] -> Round to nearest odd
		// [4] -> Round only if difference == 0.5

		static constexpr int8_t UP		  = 0b00000011;
		static constexpr int8_t DOWN	  = 0b00000000;
		static constexpr int8_t TRUNC	  = 0b00000000;
		static constexpr int8_t HALF_EVEN = 0b00010100;
		static constexpr int8_t MATH	  = 0b00000001;
	} // namespace roundMode

	template<typename T = double>
	auto round(T num, int64_t dp, int8_t mode = roundMode::MATH) {
		using Scalar = typename internal::traits<T>::Scalar;

		const Scalar alpha	= pow10<T>(dp);
		const Scalar beta	= pow10<T>(-dp);
		const Scalar absNum = abs(num * alpha);
		Scalar y			= floor(absNum);
		Scalar diff			= absNum - y;
		if (mode & (1 << 0) && diff >= 0.5) y += 1;
		if (mode & (1 << 2)) {
			auto integer	 = (uint64_t)y;
			auto nearestEven = (integer & 1) ? (y + 1) : (Scalar)integer;
			if (mode & (1 << 4) && diff == 0.5) y = nearestEven;
		}
		return (num >= 0 ? y : -y) * beta;
	}

	template<typename T1 = double, typename T2 = double>
	typename std::common_type_t<T1, T2> roundTo(T1 num, T2 val) {
		auto rem = mod(num, val);
		if (rem >= val / 2) return (num + val) - rem;
		return num - rem;
	}

	template<typename T1 = double, typename T2 = double>
	typename std::common_type_t<T1, T2> roundUpTo(T1 num, T2 val) {
		auto rem = mod(num, val);
		if (rem == 0) return num;
		return (num + val) - rem;
	}

	template<typename T>
	T roundSigFig(T num, int64_t figs) {
		LR_ASSERT(figs > 0,
				  "Cannot round to {} significant figures. Value must be greater than zero",
				  figs);

		T tmp	  = num > 0 ? num : -num;
		int64_t n = 0;

		while (tmp > 10) {
			tmp /= 10;
			++n;
		}

		while (tmp < 1) {
			tmp *= 10;
			--n;
		}

		return (tmp > 0 ? 1 : -1) * (round(tmp, figs - 1) * pow10<T>(n));
	}
} // namespace librapid

#endif // LIBRAPID_CORE_MATH

namespace librapid {
	template<typename LAMBDA>
	LR_NODISCARD("")
	double differentiate(const LAMBDA &fx, double x, double h = 1e-5) {
		double t1 = fx(x - 2 * h) / 12;
		double t2 = 2 * fx(x - h) / 3;
		double t3 = 2 * fx(x + h) / 3;
		double t4 = fx(x + 2 * h) / 12;
		return (1 / h) * (t1 - t2 + t3 - t4);
	}

	template<typename LAMBDA>
	LR_NODISCARD("")
	double integrate(const LAMBDA &fx, double lower, double upper, double inc = 1e-6) {
		double sum	= inc * inc; // Small error correction
		auto blocks = (int64_t)((upper - lower) / inc);
		for (int64_t i = 0; i < blocks; ++i) {
			double tmp = fx(inc * (double)i) * inc;
			if (std::isinf(tmp)) {
				sum += inc; // Big number?
			} else {
				sum += tmp;
			}
		}
		return sum;
	}

	namespace gammaImpl {
		static int64_t elemsP				   = 8;
		static std::complex<double> p[] = {676.5203681218851,
												  -1259.1392167224028,
												  771.32342877765313,
												  -176.61502916214059,
												  12.507343278686905,
												  -0.13857109526572012,
												  9.9843695780195716e-6,
												  1.5056327351493116e-7};

		static double epsilon = 1e-7;
		LR_NODISCARD("") auto dropImag(const std::complex<double> &z) {
			if (abs(z.imag()) < epsilon) std::complex<double>(z.real());
			return z;
		}

		template<typename T>
		LR_NODISCARD("")
		double gamma(T z_) {
			auto z = std::complex<double>(z_);
			std::complex<double> y;
			if (z.real() < 0.5) {
				y = PI / (sin(PI * z) * gamma(std::complex<double>(1) - z));
			} else {
				z -= 1;
				std::complex<double> x = 0.99999999999980993;
				for (int64_t i = 0; i < elemsP; ++i) {
					auto pVal = p[i];
					x += std::complex<double>(pVal) /
						 (z + std::complex<double>(i) + std::complex<double>(1));
				}
				auto t = z + std::complex<double>(elemsP) - std::complex<double>(0.5);
				y	   = sqrt(2 * PI) * pow(t, z + 0.5) * exp(-t) * x;
			}

			return dropImag(y).real();
		}
	} // namespace gammaImpl

	LR_NODISCARD("") double gamma(double x) {
		LR_ASSERT(x < 143, "Gamma(x = {}) exceeds 64bit floating point range when x >= 143", x);
		return gammaImpl::gamma(x);
	}

	LR_NODISCARD("") double digamma(double z) {
		double sum = 0;
		for (int64_t k = 0; k < 7500; ++k) { sum += (z - 1) / ((double)(k + 1) * ((double)k + z)); }
		return -EULERMASCHERONI + sum;
	}

	LR_NODISCARD("") double polygamma(int64_t n, double z, int64_t lim = 100) {
		if (n == 0) return digamma(z);

		double t1	= n & 1 ? 1 : -1;
		double fact = gamma(n - 1);
		double sum	= 0;
		for (int64_t k = 0; k < lim; ++k) { sum += 1 / pow<double>(z + k, n + 1); }
		return t1 * fact * sum;
	}

	LR_NODISCARD("") double lambertW(double z) {
		/*
		 * Lambert W function, principal branch.
		 * See http://en.wikipedia.org/wiki/Lambert_W_function
		 * Code taken from http://keithbriggs.info/software.html
		 */

		double eps = 4.0e-16;
		double em1 = 0.3678794411714423215955237701614608;
		LR_ASSERT(z >= -em1, "Invalid argument to Lambert W function");

		if (z == 0) return 0;

		if (z < -em1 + 1e-4) {
			double q  = z + em1;
			double r  = sqrt(q);
			double q2 = q * q;
			double q3 = q2 * q;

			// clang-format off
			return -1.0 +
				   2.331643981597124203363536062168 * r -
				   1.812187885639363490240191647568 * q +
				   1.936631114492359755363277457668 * r * q -
				   2.353551201881614516821543561516 * q2 +
				   3.066858901050631912893148922704 * r * q2 -
				   4.175335600258177138854984177460 * q3 +
				   5.858023729874774148815053846119 * r * q3 -
				   8.401032217523977370984161688514 * q3 * q;
			// clang-format on
		}

		double p, w;

		if (z < 1) {
			p = sqrt(2.0 * (2.7182818284590452353602874713526625 * z + 1.0));
			w = -1.0 + p * (1.0 + p * (-0.333333333333333333333 + p * 0.152777777777777777777777));
		} else {
			w = ln(z);
		}

		if (z > 3) w -= ln(w);

		for (int64_t i = 0; i < 10; ++i) {
			double e = exp(w);
			double t = w * e - z;
			p		 = w + 1;
			t /= e * p - 0.5 * (p + 1.0) * t / p;
			w -= t;
			if (abs(t) < eps * (1 + abs(w))) return w;
		}

		LR_ASSERT(z >= -em1, "Invalid argument to Lambert W function");
		return 0;
	}

	double invGamma(double x, int64_t prec = 5) {
		// Run a very coarse calculation to get a guess for the guess
		double guess = 2;
		// double tmp	 = gamma(guess);
		// while (abs(gamma(guess) / x) > 0.5) guess += (x < tmp) ? 1 : -1;

		double dx = DBL_MAX;
		while (abs(dx) > pow10(-prec - 1)) {
			double gammaGuess = gamma(guess);
			double num		  = gammaGuess - x;
			double den		  = gammaGuess * polygamma(0, guess);
			double frac		  = num / den;
			double newGuess	  = guess - frac;
			dx				  = guess - newGuess;

			// Avoid nan problems
			if (newGuess > 142) {
				if (newGuess > guess)
					guess *= 2;
				else
					guess /= 2;

				if (guess > 142) guess = 140;
			} else {
				guess = newGuess;
			}
		}
		return round(guess, prec);
	}
} // namespace librapid

namespace librapid {
	float sqrtApprox(float z) {
		union {
			float f;
			uint32_t i;
		} val = {z};
		val.i -= 1 << 23;
		val.i >>= 1;
		val.i += 1 << 29;
		return val.f;
	}

	float invSqrtApprox(float x) {
		float halfX = 0.5f * x;
		union {
			float x;
			uint32_t i;
		} u;
		u.x = x;
		u.i = 0x5f375a86 - (u.i >> 1);
		u.x = u.x * (1.5f - halfX * (u.x * u.x)); // Newtonian iteration
		return u.x;
	}
} // namespace librapid

/*
 * Sources:
 *
 * sqrtApprox,
 * invSqrtApprox => https://en.wikipedia.org/wiki/Methods_of_computing_square_roots
 *
 */

#if defined(LIBRAPID_HAS_CUDA)

// Memory alignment adapted from
// https://gist.github.com/dblalock/255e76195676daa5cbc57b9b36d1c99a

namespace librapid { namespace memory {
	static bool streamCreated = false;
	static cudaStream_t cudaStream;

	LR_INLINE void initializeCudaStream() {
#	ifdef LIBRAPID_HAS_CUDA
		if (!streamCreated) LIBRAPID_UNLIKELY {
				checkCudaErrors(cudaStreamCreateWithFlags(&cudaStream, cudaStreamNonBlocking));
				streamCreated = true;
			}
#	endif // LIBRAPID_HAS_CUDA
	}

	template<typename T, typename d,
			 typename std::enable_if_t<std::is_same_v<d, device::GPU>, int> = 0>
	LR_NODISCARD("Do not leave a dangling pointer")
	LR_FORCE_INLINE T *malloc(size_t num, size_t alignment = memAlign, bool zero = false) {
		// Ignore memory alignment
		T *buf;
		cudaSafeCall(cudaMallocAsync(&buf, sizeof(T) * num, cudaStream));

// Slightly altered traceback call to log u_chars being allocated
#	ifdef LIBRAPID_TRACEBACK
		LR_STATUS("LIBRAPID TRACEBACK -- MALLOC {} u_charS -> {}", size, (void *)buf);
#	endif

		return buf;
	}

	template<typename T, typename d,
			 typename std::enable_if_t<std::is_same_v<d, device::GPU>, int> = 0>
	LR_FORCE_INLINE void free(T *ptr) {
#	ifdef LIBRAPID_TRACEBACK
		LR_STATUS("LIBRAPID TRACEBACK -- FREE {}", (void *)alignedPtr);
#	endif

		cudaSafeCall(cudaFreeAsync(ptr, cudaStream));
	}

	template<typename T, typename d, typename T_, typename d_,
			 typename std::enable_if_t<
			   !(std::is_same_v<d, device::CPU> && std::is_same_v<d_, device::CPU>), int> = 0>
	LR_FORCE_INLINE void memcpy(T *dst, T_ *src, int64_t size) {
		if constexpr (std::is_same_v<T, T_>) {
			if constexpr (std::is_same_v<d, device::CPU> && std::is_same_v<d_, device::GPU>) {
				// Device to Host
				cudaSafeCall(
				  cudaMemcpyAsync(dst, src, sizeof(T) * size, cudaMemcpyDeviceToHost, cudaStream));
			} else if constexpr (std::is_same_v<d, device::GPU> &&
								 std::is_same_v<d_, device::CPU>) {
				// Host to Device
				cudaSafeCall(
				  cudaMemcpyAsync(dst, src, sizeof(T) * size, cudaMemcpyHostToDevice, cudaStream));
			} else if constexpr (std::is_same_v<d, device::GPU> &&
								 std::is_same_v<d_, device::GPU>) {
				// Host to Device
				cudaSafeCall(cudaMemcpyAsync(
				  dst, src, sizeof(T) * size, cudaMemcpyDeviceToDevice, cudaStream));
			}
		} else {
			// TODO: Optimise this

			if constexpr (std::is_same_v<d_, device::CPU>) {
				// Source device is CPU
				for (int64_t i = 0; i < size; ++i) {
					T tmp = src[i]; // Required to cast value

					// Copy from host to device
					cudaSafeCall(cudaMemcpyAsync(
					  dst + i, &tmp, sizeof(T), cudaMemcpyHostToDevice, cudaStream));
				}
			} else if constexpr (std::is_same_v<d, device::CPU>) {
				// Destination device is CPU
				for (int64_t i = 0; i < size; ++i) {
					T_ tmp; // Required to cast value

					// Copy from device to host
					cudaSafeCall(cudaMemcpyAsync(
					  &tmp, src + i, sizeof(T_), cudaMemcpyDeviceToHost, cudaStream));
					dst[i] = tmp; // Write final result
				}
			} else {
				const char *kernel = R"V0G0N(memcpyKernel
					#include <stdint.h>
					template<typename DST, typename SRC>
					__global__
					void memcpyKernel(DST *dst, SRC *src, int64_t size) {
						uint64_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
						if (kernelIndex < size) dst[kernelIndex] = src[kernelIndex];
					}
				)V0G0N";

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel);
				unsigned int threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (size < 512) {
					threadsPerBlock = (unsigned int)size;
					blocksPerGrid	= 1;
				} else {
					threadsPerBlock = 512;
					blocksPerGrid	= ceil(double(size) / double(threadsPerBlock));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				using jitify::reflection::Type;
				jitifyCall(program.kernel("memcpyKernel")
							 .instantiate(Type<T>(), Type<T_>())
							 .configure(grid, block, 0, cudaStream)
							 .launch(dst, src, size));
			}
		}
	}
} } // namespace librapid::memory
#endif // LIBRAPID_HAS_CUDA

namespace librapid {
	template<typename T, int64_t maxDims, int64_t align_ = 4>
	class ExtentType {
	public:
		static constexpr int64_t Align = align_;

		ExtentType() = default;

		template<typename... T_>
		explicit ExtentType(T_... args) : m_dims(sizeof...(T_)), m_data {args...} {}

		template<typename T_>
		ExtentType(const std::initializer_list<T_> &args) : m_dims(args.size()) {
			LR_ASSERT(args.size() <= maxDims,
					  "A maximum of {} dimensions are allowed in an Extent object",
					  maxDims);
			int64_t i = 0;
			for (const auto &val : args) m_data[i++] = val;
		}

		template<typename T_>
		ExtentType(const std::vector<T_> &args) : m_dims(args.size()) {
			LR_ASSERT(args.size() <= maxDims,
					  "A maximum of {} dimensions are allowed in an Extent object",
					  maxDims);
			for (int64_t i = 0; i < m_dims; ++i) m_data[i] = args[i];
		}

		template<typename T_, int64_t d_, int64_t a_>
		ExtentType(const ExtentType<T_, d_, a_> &e) {
			LR_ASSERT(e.dims() < maxDims,
					  "Extent with {} dimensions cannot be stored in an extent with a maximum of "
					  "{} dimensions",
					  d_,
					  maxDims);
			m_dims = e.dims();
			for (int64_t i = 0; i < m_dims; ++i) { m_data[i] = e[i]; }
		}

		ExtentType &operator=(const ExtentType &other) {
			if (this == &other) return *this;
			m_dims = other.dims();
			for (int64_t i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
			return *this;
		}

		template<typename T_, int64_t d_, int64_t a_>
		ExtentType &operator=(const ExtentType<T_, d_, a_> &other) {
			LR_ASSERT(other.dims() < maxDims,
					  "Extent with {} dimensions cannot be stored in an extent with a maximum of "
					  "{} dimensions",
					  d_,
					  maxDims);
			m_dims = other.dims();
			for (int64_t i = 0; i < m_dims; ++i) { m_data[i] = other[i]; }
			return *this;
		}

		static ExtentType zero(int64_t dims) {
			// Data is already zeroed
			ExtentType res;
			res.m_dims = dims;
			return res;
		}

		ExtentType stride() const {
			ExtentType res = zero(m_dims);
			int64_t prod   = 1;
			for (int64_t i = m_dims - 1; i >= 0; --i) {
				res[i] = prod;
				prod *= m_data[i];
			}
			return res;
		}

		ExtentType strideAdjusted() const {
			ExtentType res = zero(m_dims);
			int64_t prod   = 1;
			for (int64_t i = m_dims - 1; i >= 0; --i) {
				res[i] = prod;
				prod *= adjusted(i);
			}
			return res;
		}

		template<typename First, typename... Other>
		T index(First index, Other... others) const {
			return indexImpl(0, index, others...);
		}

		T index(const ExtentType &index) const {
			LR_ASSERT(
			  index.dims() == m_dims,
			  "Cannot get index of Extent with {} dimensions using Extent with {} dimensions",
			  m_dims,
			  index.dims());

			T res			   = 0;
			ExtentType strides = stride();
			for (int64_t i = 0; i < index.dims(); ++i) {
				LR_ASSERT(index.m_data[i] >= 0 && index.m_data[i] <= m_data[i],
						  "Index {} is out of range for Extent with dimension {}",
						  index.m_data[i],
						  m_data[i]);
				res += strides[i] * index[i];
			}
			return res;
		}

		template<typename First, typename... Other>
		T indexAdjusted(First index, Other... others) const {
			return indexImplAdjusted(0, index, others...);
		}

		T indexAdjusted(const ExtentType &index) const {
			LR_ASSERT(
			  index.dims() == m_dims,
			  "Cannot get index of Extent with {} dimensions using Extent with {} dimensions",
			  m_dims,
			  index.dims());

			T res			   = 0;
			ExtentType strides = strideAdjusted();
			for (int64_t i = 0; i < index.dims(); ++i) {
				LR_ASSERT(index.m_data[i] >= 0 && index[i] <= adjusted(i),
						  "Index {} is out of range for Extent with adjusted dimension {}",
						  index[i],
						  adjusted(i));
				res += strides[i] * index[i];
			}
			return res;
		}

		ExtentType reverseIndex(int64_t index) const {
			ExtentType res	   = zero(m_dims);
			ExtentType strides = stride();
			for (int64_t i = 0; i < m_dims; ++i) {
				res[i] = index / strides[i];
				index -= strides[i] * res[i];
			}
			return res;
		}

		ExtentType reverseIndexAdjusted(int64_t index) const {
			ExtentType res	   = zero(m_dims);
			ExtentType strides = strideAdjusted();
			for (int64_t i = 0; i < m_dims; ++i) {
				res[i] = index / strides[i];
				index -= strides[i] * res[i];
			}
			return res;
		}

		ExtentType partial(int64_t start = 0, int64_t end = -1) const {
			if (end == -1) end = m_dims;
			ExtentType res;
			res.m_dims = m_dims - 1;
			for (int64_t i = start; i < end; ++i) { res[i - start] = m_data[i]; }
			return res;
		}

		template<typename T_ = T, int64_t d = maxDims>
		LR_NODISCARD("")
		ExtentType swivel(const ExtentType<T_, d> &order) const {
			LR_ASSERT(
			  order.dims() == m_dims,
			  "Swivel order must contain the same number of dimensions as the Extent to swivel");

#if defined(LIBRAPID_DEBUG)
			// Check the order contains only valid numbers
			for (int64_t i = 0; i < order.dims(); ++i) {
				bool found = false;
				for (int64_t j = 0; j < order.dims(); ++j) {
					if (order[j] == i) {
						found = true;
						break;
					}
				}
				LR_ASSERT(found, "Swivel missing index {}", i);
			}
#endif

			ExtentType res = zero(m_dims);
			for (int64_t i = 0; i < order.dims(); ++i) { res[order[i]] = m_data[i]; }
			return res;
		}

		template<typename T_ = T, int64_t d = maxDims>
		void swivelInplace(const ExtentType<T_, d> &order) {
			*this = swivel(order);
		}

		LR_NODISCARD("") LR_FORCE_INLINE int64_t size() const {
			int64_t res = 1;
			for (int64_t i = 0; i < m_dims; ++i) res *= m_data[i];
			return res;
		}

		LR_NODISCARD("") LR_FORCE_INLINE int64_t sizeAdjusted() const {
			int64_t res = 1;
			for (int64_t i = 0; i < m_dims; ++i) res *= adjusted(i);
			return res;
		}

		LR_NODISCARD("") LR_FORCE_INLINE int64_t dims() const { return m_dims; }

		const T &operator[](int64_t index) const {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return m_data[index];
		}

		T &operator[](int64_t index) {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return m_data[index];
		}

		T adjusted(int64_t index) const {
			LR_ASSERT(index >= 0 && index < m_dims,
					  "Index {} is out of range for Extent with {} dimensions",
					  index,
					  m_dims);
			return roundUpTo(m_data[index], Align);
		}

		template<typename T_, int64_t d_, int64_t a_>
		LR_NODISCARD("")
		bool operator==(const ExtentType<T_, d_, a_> &other) const {
			if (m_dims != other.m_dims) return false;
			for (int64_t i = 0; i < m_dims; ++i)
				if (m_data[i] != other.m_data[i]) return false;
			return true;
		}

		LR_NODISCARD("") std::string str() const {
			std::string res = "Extent(";
			for (int64_t i = 0; i < m_dims - 1; ++i) res += fmt::format("{}, ", m_data[i]);
			return res + fmt::format("{})", m_data[m_dims - 1]);
		}

	private:
		template<typename First>
		T indexImpl(T index, First first) const {
			LR_ASSERT(first >= 0 && first < m_data[index],
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			int64_t extentProd = 1;
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first;
		}

		template<typename First, typename... Other>
		T indexImpl(T index, First first, Other... others) const {
			LR_ASSERT(first >= 0 && first < m_data[index],
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			int64_t extentProd = 1;
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first + indexImpl(index + 1, others...);
		}

		template<typename First>
		T indexImplAdjusted(T index, First first) const {
			LR_ASSERT(first >= 0 && first < adjusted(index),
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			int64_t extentProd = 1;
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first;
		}

		template<typename First, typename... Other>
		T indexImplAdjusted(T index, First first, Other... others) const {
			LR_ASSERT(first >= 0 && first < adjusted(index),
					  "Index {} is out of range for dimension {} of an Array with {}",
					  first,
					  index,
					  str());

			int64_t extentProd = 1;
			for (int64_t i = index + 1; i < m_dims; ++i) extentProd *= adjusted(i);
			return extentProd * first + indexImpl(index + 1, others...);
		}

	private:
		T m_dims = -1;
		T m_data[maxDims] {};
	};

	using Extent = ExtentType<int64_t, 32>;
} // namespace librapid

namespace librapid { namespace functors { namespace binary {
	template<typename LHS, typename RHS>
	class ScalarOp {
	public:

		using LhsType				   = LHS;
		using RhsType				   = RHS;
		using RetType				   = std::false_type;
		static constexpr int64_t Flags = 0;

		template<typename T, int64_t d, int64_t a>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d, a> &lhs, const ExtentType<T, d, a> &rhs) const {
			return lhs;
		}

		// For a scalar operation
		template<typename T, int64_t d, int64_t a>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d, a> &generic) const {
			return generic;
		}
	};
} } } // namespace librapid::functors::binary

namespace librapid { namespace functors { namespace binary {
	template<typename LHS, typename RHS>
	class BitwiseOr : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Bitwise |
										 internal::flags::PacketBitwise |
										 internal::flags::ScalarBitwise;

		BitwiseOr() = default;

		BitwiseOr(const BitwiseOr<LHS, RHS> &other) = default;

		BitwiseOr<LHS, RHS> &operator=(const BitwiseOr<LHS, RHS> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left | right;
		}

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &left, const PacketType &right) const {
			return left | right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "|"; }

	private:
	};

	template<typename LHS, typename RHS>
	class BitwiseAnd : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Bitwise |
										 internal::flags::PacketBitwise |
										 internal::flags::ScalarBitwise;

		BitwiseAnd() = default;

		BitwiseAnd(const BitwiseAnd<LHS, RHS> &other) = default;

		BitwiseAnd<LHS, RHS> &operator=(const BitwiseAnd<LHS, RHS> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left & right;
		}

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &left, const PacketType &right) const {
			return left & right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "&"; }

	private:
	};

	template<typename LHS, typename RHS>
	class BitwiseXor : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Bitwise |
										 internal::flags::PacketBitwise |
										 internal::flags::ScalarBitwise;

		BitwiseXor() = default;

		BitwiseXor(const BitwiseXor<LHS, RHS> &other) = default;

		BitwiseXor<LHS, RHS> &operator=(const BitwiseXor<LHS, RHS> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left ^ right;
		}

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &left, const PacketType &right) const {
			return left ^ right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "^"; }

	private:
	};
} } } // namespace librapid::functors::binary

namespace librapid { namespace functors { namespace binary {
	template<typename LHS, typename RHS>
	class ScalarSum : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		ScalarSum() = default;

		ScalarSum(const ScalarSum<LHS, RHS> &other) = default;

		ScalarSum<LHS, RHS> &operator=(const ScalarSum<LHS, RHS> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left + right;
		}

		template<typename PacketTypeLHS, typename PacketTypeRHS>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet
		  packetOp(const PacketTypeLHS &left, const PacketTypeRHS &right) const {
			return left + right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "+"; }

	private:
	};

	template<typename LHS, typename RHS>
	class ScalarDiff : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		ScalarDiff() = default;

		ScalarDiff(const ScalarDiff<LHS, RHS> &other) = default;

		ScalarDiff<LHS, RHS> &operator=(const ScalarDiff<LHS, RHS> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left - right;
		}

		template<typename PacketTypeLHS, typename PacketTypeRHS>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet
		  packetOp(const PacketTypeLHS &left, const PacketTypeRHS &right) const {
			return left - right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "-"; }

	private:
	};

	template<typename LHS, typename RHS>
	class ScalarProd : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		ScalarProd() = default;

		ScalarProd(const ScalarProd<LHS, RHS> &other) = default;

		ScalarProd<LHS, RHS> &operator=(const ScalarProd<LHS, RHS> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left * right;
		}

		template<typename PacketTypeLHS, typename PacketTypeRHS>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet
		  packetOp(const PacketTypeLHS &left, const PacketTypeRHS &right) const {
			return left * right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "*"; }

	private:
	};

	template<typename LHS, typename RHS>
	class ScalarDiv : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		ScalarDiv() = default;

		ScalarDiv(const ScalarDiv<LHS, RHS> &other) = default;

		ScalarDiv<LHS, RHS> &operator=(const ScalarDiv<LHS, RHS> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left / right;
		}

		template<typename PacketTypeLHS, typename PacketTypeRHS>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet
		  packetOp(const PacketTypeLHS &left, const PacketTypeRHS &right) const {
			return left / right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "/"; }

	private:
	};
} } } // namespace librapid::functors::binary

namespace librapid { namespace functors { namespace unop {
	template<typename TYPE>
	class UnaryOp {
	public:
		using Type					   = TYPE;
		using RetType				   = std::false_type;
		static constexpr int64_t Flags = 0;

		template<typename T, int64_t d, int64_t a>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d, a> &extent) const {
			return extent;
		}
	};
}}} // namespace librapid::functors::unop

namespace librapid { namespace functors { namespace unop {
	template<typename Type_>
	class UnaryMinus : public UnaryOp<Type_> {
	public:
		using Type					   = Type_;
		using Scalar				   = typename internal::traits<Type_>::Scalar;
		using RetType				   = Scalar;
		using Packet				   = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags = internal::flags::Unary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		UnaryMinus() = default;

		UnaryMinus(const UnaryMinus<Type> &other) = default;

		UnaryMinus<Type> &operator=(const UnaryMinus<Type> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return -val; }

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {
			return -val;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "-"; }

	private:
	};

	template<typename Type_>
	class UnaryNot : public UnaryOp<Type_> {
	public:
		using Type					   = Type_;
		using Scalar				   = typename internal::traits<Type_>::Scalar;
		using RetType				   = Scalar;
		using Packet				   = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags = internal::flags::Unary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		UnaryNot() = default;

		UnaryNot(const UnaryNot<Type> &other) = default;

		UnaryNot<Type> &operator=(const UnaryNot<Type> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return !val; }

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {
			return !val;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "-"; }

	private:
	};

	template<typename Type_>
	class BitwiseNot : public UnaryOp<Type_> {
	public:
		using Type					   = Type_;
		using Scalar				   = typename internal::traits<Type_>::Scalar;
		using RetType				   = Scalar;
		using Packet				   = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Bitwise |
										 internal::flags::PacketBitwise |
										 internal::flags::ScalarBitwise;

		BitwiseNot() = default;

		BitwiseNot(const BitwiseNot<Type> &other) = default;

		BitwiseNot<Type> &operator=(const BitwiseNot<Type> &other) { return *this; }

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return ~val; }

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {
			return ~val;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "~"; }

	private:
	};
} } } // namespace librapid::functors::unop

namespace librapid::functors::matrix {
	template<typename Type_>
	class Transpose {
	public:
		using Type	  = Type_;
		using Scalar  = typename internal::traits<Type_>::Scalar;
		using RetType = Scalar;
		using Packet  = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags =
		  internal::flags::Matrix | internal::flags::Unary | internal::flags::RequireInput;

		Transpose() = default;

		template<typename T, int64_t d>
		explicit Transpose(const ExtentType<T, d> &order) : m_order(order) {};

		Transpose(const Transpose<Type> &other) = default;

		Transpose<Type> &operator=(const Transpose<Type> &other) {
			m_order = other.m_order;
			return *this;
		}

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return 0; }

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {
			return 1;
		}

		template<typename Derived>
		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOpInput(const Derived &other, int64_t index) const {
			auto extent	   = other.extent();
			auto swivelled = extent.reverseIndex(index).swivel(m_order);
			auto first	   = extent.index(swivelled);
			return other.scalar(first);
		}

		template<typename Derived>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOpInput(const Derived &other, int64_t index) const {
			using BaseScalar = typename internal::traits<Scalar>::BaseScalar;
			BaseScalar buffer[internal::traits<BaseScalar>::PacketWidth];
			auto extent	   = other.extent();
			auto swivelled = extent.reverseIndexAdjusted(index).swivel(m_order);
			auto first	   = extent.indexAdjusted(swivelled);
			auto stride	   = extent.strideAdjusted();

			if constexpr (std::is_same_v<Scalar, bool>) {
				LR_ASSERT(false, "Boolean Arrays do not currently support Matrix operations");
			} else {
				for (int64_t i = 0; i < internal::traits<BaseScalar>::PacketWidth; ++i) {
					buffer[i] = other.scalar(first);
					first += stride[m_order[extent.dims() - 1]];
				}
			}

			Packet res(&(buffer[0]));
			return res;
		}

		template<typename T, int64_t d, int64_t a>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d, a> &extent) const {
			return extent.swivel(m_order);
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const {
			LR_ASSERT(false, "Array transpose has no dedicated GPU kernel");
			return "ERROR";
		}

	private:
		ExtentType<int64_t, 32> m_order;
	};
} // namespace librapid::functors::matrix

namespace librapid::functors {
	template<typename Derived, typename OtherDerived, bool evalBeforeAssign>
	struct AssignSelector;

	template<typename Derived, typename OtherDerived>
	struct AssignSelector<Derived, OtherDerived, false> {
		static Derived &run(Derived &left, const OtherDerived &right) {
			return left.assignLazy(right);
		}
	};

	template<typename Derived, typename OtherDerived>
	struct AssignOp {
		LR_FORCE_INLINE static void run(Derived &dst, const OtherDerived &src) {
			constexpr bool dstIsHost =
			  is_same_v<typename internal::traits<Derived>::Device, device::CPU>;
			constexpr bool srcIsHost =
			  is_same_v<typename internal::traits<OtherDerived>::Device, device::CPU>;
			using Scalar				   = typename internal::traits<Derived>::Scalar;
			using BaseScalar			   = typename internal::traits<Derived>::BaseScalar;
			using Packet				   = typename internal::traits<Scalar>::Packet;
			static constexpr int64_t Flags = internal::traits<OtherDerived>::Flags;
			constexpr bool isMatrixOp	   = (bool) (Flags & internal::flags::Matrix);

#if !defined(LIBRAPID_HAS_CUDA)
			static_assert(dstIsHost && srcIsHost, "CUDA support was not enabled");
#endif

			if constexpr (dstIsHost && srcIsHost) {
				int64_t packetWidth = internal::traits<Scalar>::PacketWidth;
				int64_t len			= dst.extent().sizeAdjusted();
				int64_t alignedLen	= len - (len % packetWidth);
				if (alignedLen < 0) alignedLen = 0;
				int64_t processThreads = isMatrixOp ? matrixThreads : numThreads;

				if constexpr (is_same_v<Scalar, bool>) {
					len += internal::traits<Scalar>::PacketWidth;
					len /= internal::traits<Scalar>::PacketWidth;
					packetWidth = 1;
					alignedLen	= len;
				}

				// Only use a Packet type if possible
				if constexpr (!is_same_v<Packet, std::false_type>) {
					// Use the entire packet width where possible
					if (LIBRAPID_OMP_VAL && (processThreads < 2 || len < 500)) {
						for (int64_t i = 0; i < alignedLen; i += packetWidth) {
							dst.loadFrom(i, src);
						}
					}
#if defined(LIBRAPID_HAS_OMP)
					else {
						// Multi-threaded approach
#	pragma omp parallel for shared(dst, src, alignedLen, packetWidth) default(none)               \
	  num_threads(processThreads)
						for (int64_t i = 0; i < alignedLen; i += packetWidth) {
							dst.loadFrom(i, src);
						}
					}
#endif
				}

				// Ensure the remaining values are filled
				int64_t start = alignedLen;
				for (int64_t i = start < 0 ? 0 : start; i < len; ++i) {
					dst.loadFromScalar(i, src);
				}
			} else {
#if defined(LIBRAPID_HAS_CUDA)
				// LR_LOG_STATUS("Size of Type: {}", sizeof(OtherDerived));
				static_assert(sizeof(OtherDerived) < (1 << 15), // Defines the max op size
							  "Calculation is too large to be run in a single call. Please call "
							  "eval() somewhere earlier");

				int64_t elems = src.extent().size();

				if constexpr (is_same_v<Scalar, bool>) {
					elems += 512;
					elems /= 512;
				}

				std::vector<BaseScalar *> arrays = {dst.storage().heap()};
				std::string scalarName			 = internal::traits<BaseScalar>::Name;
				int64_t index					 = 0;
				std::string microKernel			 = src.genKernel(arrays, index);

				std::string mainArgs;
				for (int64_t i = 0; i < index; ++i) {
					mainArgs += fmt::format("{} *{}{}", scalarName, "arg", i);
					if (i + 1 < index) mainArgs += ", ";
				}

				std::string functionArgs;
				for (int64_t i = 0; i < index; ++i) {
					functionArgs += fmt::format("{} arg{}", scalarName, i);
					if (i + 1 < index) functionArgs += ", ";
				}

				std::string indArgs;
				for (int64_t i = 0; i < index; ++i) {
					indArgs += fmt::format("arg{}[kernelIndex]", i);
					if (i + 1 < index) indArgs += ", ";
				}

				std::string varExtractor;
				for (int64_t i = 0; i < index; ++i)
					varExtractor +=
					  fmt::format("{} *arg{} = pointers[{}];\n\t", scalarName, i, i + 1);

				std::string varArgs;
				for (int64_t i = 0; i < index; ++i) {
					varArgs += fmt::format("src{}", i);
					if (i + 1 < index) varArgs += ", ";
				}

				std::string kernel = fmt::format(R"V0G0N(kernelOp
#include<stdint.h>

__device__
{4} function({0}) {{
	return {3};
}}

__global__
void applyOp({4} **pointers, int64_t size) {{
	const int64_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
	{4} *dst = pointers[0];
	{5}

	if (kernelIndex < size) {{
		dst[kernelIndex] = function({2});
	}}
}}
				)V0G0N",
												 functionArgs,
												 mainArgs,
												 indArgs,
												 microKernel,
												 scalarName,
												 varExtractor,
												 varArgs);

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel, cudaHeaders, nvccOptions);

				int64_t threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (elems < 512) {
					threadsPerBlock = elems;
					blocksPerGrid	= 1;
				} else {
					threadsPerBlock = 512;
					blocksPerGrid	= ceil(double(elems) / double(threadsPerBlock));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				// Copy the pointer array to the device
				BaseScalar **devicePointers = memory::malloc<BaseScalar *, device::GPU>(index + 1);
				memory::memcpy<BaseScalar *, device::GPU, BaseScalar *, device::CPU>(
				  devicePointers, &arrays[0], index + 1);

				jitifyCall(program.kernel("applyOp")
							 .instantiate()
							 .configure(grid, block, 0, memory::cudaStream)
							 .launch(devicePointers, elems));

				// Free device pointers
				memory::free<BaseScalar *, device::GPU>(devicePointers);
#endif
			}
		}
	};
} // namespace librapid::functors

namespace librapid {
	namespace internal {
		template<typename DST, typename OtherDerived>
		struct traits<unary::Cast<DST, OtherDerived>> {
			static constexpr bool IsScalar = false;
			using Valid					   = std::true_type;
			using Type					   = unary::Cast<DST, OtherDerived>;
			using Scalar				   = DST;
			using Packet				   = typename traits<Scalar>::Packet;
			using Device				   = typename internal::traits<OtherDerived>::Device;
			using StorageType			   = memory::DenseStorage<Scalar, Device>;
			static const uint64_t Flags	   = 0;
		};
	} // namespace internal

	namespace unary {
		template<typename DST, typename OtherDerived>
		class Cast : public ArrayBase<Cast<DST, OtherDerived>,
									  typename internal::traits<OtherDerived>::Device> {
		public:
			using Scalar	  = DST;
			using Packet	  = typename internal::traits<Scalar>::Packet;
			using Device	  = typename internal::traits<OtherDerived>::Device;
			using InputType	  = OtherDerived;
			using InputScalar = typename internal::traits<InputType>::Scalar;
			using Type		  = Cast<DST, OtherDerived>;
			using Base		  = ArrayBase<Cast<DST, OtherDerived>, Device>;

			Cast() = delete;

			Cast(const InputType &toCast) : Base(toCast.extent()), m_toCast(toCast) {}

			Cast(const Type &caster) : Base(caster.extent()), m_toCast(caster.m_toCast) {}

			Cast &operator=(const Type &caster) {
				if (this == &caster) return *this;
				Base::m_extent = caster.m_extent;
				Base::m_data   = caster.m_data;
				m_toCast	   = caster.m_toCast;
				return *this;
			}

			LR_NODISCARD("Do not ignore the result of an evaluated calculation")
			Array<Scalar, Device> eval() const {
				Array<Scalar, Device> res(Base::extent());
				res.assign(*this);
				return res;
			}

			LR_FORCE_INLINE Packet packet(int64_t index) const {
				// Quick return if possible
				if constexpr (is_same_v<Scalar, InputScalar>) return m_toCast.packet(index);
				static Scalar buffer[Packet::size()];
				for (int64_t i = 0; i < Packet::size(); ++i) buffer[i] = m_toCast.scalar(index + i);
				return Packet(&(buffer[0]));
			}

			LR_FORCE_INLINE Scalar scalar(int64_t index) const {
				return Scalar(m_toCast.scalar(index));
			}

			LR_NODISCARD("")
			std::string str(std::string format = "", const std::string &delim = " ",
							int64_t stripWidth = -1, int64_t beforePoint = -1,
							int64_t afterPoint = -1, int64_t depth = 0) const {
				return eval().str(format, delim, stripWidth, beforePoint, afterPoint, depth);
			}

		private:
			InputType m_toCast;
		};
	} // namespace unary
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<typename DST, typename OtherDerived>
struct fmt::formatter<librapid::unary::Cast<DST, OtherDerived>> {
	std::string formatStr = "{}";

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		formatStr = "{:";
		auto it = ctx.begin();
		for (; it != ctx.end(); ++it) {
			if (*it == '}') break;
			formatStr += *it;
		}
		formatStr += "}";
		return it;
	}

	template<typename FormatContext>
	auto format(const librapid::unary::Cast<DST, OtherDerived> &arr, FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), arr.str(formatStr));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API

#define IMPL_BINOP(NAME, TYPE)                                                                     \
	template<typename OtherDerived, typename OtherDevice>                                          \
	LR_NODISCARD("")                                                                               \
	auto NAME(const ArrayBase<OtherDerived, OtherDevice> &other) const {                           \
		using ScalarOther = typename internal::traits<OtherDerived>::Scalar;                       \
		using ResDevice	  = typename memory::PromoteDevice<Device, OtherDevice>::type;             \
		using RetType =                                                                            \
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, ScalarOther>, Derived, OtherDerived>;   \
		static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;                      \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;      \
                                                                                                   \
		static_assert(                                                                             \
		  is_same_v<Scalar, ScalarOther>,                                                          \
		  "Cannot operate on Arrays with different types. Please use Array::cast<T>()");           \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		LR_ASSERT(extent() == other.extent(),                                                      \
				  "Arrays must have equal extents. Cannot operate on Arrays with {} and {}",       \
				  extent().str(),                                                                  \
				  other.extent().str());                                                           \
                                                                                                   \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))             \
			return RetType(derived(), other.derived()).eval();                                     \
		else                                                                                       \
			return RetType(derived(), other.derived());                                            \
	}

#define IMPL_BINOP_SCALAR(NAME, TYPE)                                                              \
	template<typename OtherScalar,                                                                 \
			 typename std::enable_if_t<internal::traits<OtherScalar>::IsScalar, int> = 0>          \
	LR_NODISCARD("")                                                                               \
	auto NAME(const OtherScalar &other) const {                                                    \
		using ResDevice = Device;                                                                  \
		using RetType =                                                                            \
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, OtherScalar>, Derived, OtherScalar>;    \
		static constexpr uint64_t Flags	   = internal::traits<OtherScalar>::Flags;                 \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;      \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))             \
			return RetType(derived(), other).eval();                                               \
		else                                                                                       \
			return RetType(derived(), other);                                                      \
	}

#define IMPL_BINOP_SCALAR_EXTERNAL(NAME, TYPE)                                                     \
	template<typename OtherScalar,                                                                 \
			 typename Derived,                                                                     \
			 typename Device,                                                                      \
			 typename std::enable_if_t<internal::traits<OtherScalar>::IsScalar, int> = 0>          \
	LR_NODISCARD("")                                                                               \
	auto NAME(const OtherScalar &other, const ArrayBase<Derived, Device> &arr) {                   \
		using Scalar	= typename internal::traits<Derived>::Scalar;                              \
		using ResDevice = Device;                                                                  \
		using RetType =                                                                            \
		  binop::CWiseBinop<functors::binary::TYPE<Scalar, OtherScalar>, OtherScalar, Derived>;    \
		static constexpr uint64_t Flags	   = internal::traits<OtherScalar>::Flags;                 \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;      \
                                                                                                   \
		static_assert(!(Required & ~(Flags & Required)),                                           \
					  "Scalar type is incompatible with Functor");                                 \
                                                                                                   \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))             \
			return RetType(other, arr.derived()).eval();                                           \
		else                                                                                       \
			return RetType(other, arr.derived());                                                  \
	}

#define IMPL_UNOP(NAME, TYPE)                                                                        \
	LR_NODISCARD("")                                                                                 \
	auto NAME() const {                                                                              \
		using RetType					   = unop::CWiseUnop<functors::unop::TYPE<Scalar>, Derived>; \
		static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;                        \
		static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;        \
                                                                                                     \
		static_assert(!(Required & ~(Flags & Required)),                                             \
					  "Scalar type is incompatible with Functor");                                   \
                                                                                                     \
		if constexpr ((bool)((Flags | RetType::Flags) & internal::flags::RequireEval))               \
			return RetType(derived()).eval();                                                        \
		else                                                                                         \
			return RetType(derived());                                                               \
	}

namespace librapid {
	namespace internal {
		template<typename Derived>
		struct traits<ArrayBase<Derived, device::CPU>> {
			static constexpr bool IsScalar	= false;
			using Valid						= std::true_type;
			using Scalar					= typename traits<Derived>::Scalar;
			using BaseScalar				= typename traits<Scalar>::BaseScalar;
			using Device					= device::CPU;
			using StorageType				= memory::DenseStorage<Scalar, device::CPU>;
			static constexpr uint64_t Flags = traits<Derived>::Flags;
		};

		template<typename Derived>
		struct traits<ArrayBase<Derived, device::GPU>> {
			using Scalar					= typename traits<Derived>::Scalar;
			using Device					= device::GPU;
			using StorageType				= memory::DenseStorage<Scalar, device::CPU>;
			static constexpr uint64_t Flags = traits<Derived>::Flags;
		};
	} // namespace internal

	template<typename Derived, typename Device>
	class ArrayBase {
	public:
		using Scalar	  = typename internal::traits<Derived>::Scalar;
		using This		  = ArrayBase<Derived, Device>;
		using Packet	  = typename internal::traits<Derived>::Packet;
		using StorageType = typename internal::traits<Derived>::StorageType;
		using ArrayExtent = ExtentType<int64_t, 32, internal::traits<Scalar>::PacketWidth>;
		static constexpr uint64_t Flags = internal::traits<This>::Flags;

		friend Derived;

		ArrayBase() = default;

		template<typename T_, int64_t d_, int64_t a_>
		explicit ArrayBase(const ExtentType<T_, d_, a_> &extent) :
				m_isScalar(extent.size() == 0), m_extent(extent), m_storage(extent.sizeAdjusted()) {
		}

		template<typename T_, int64_t d_, int64_t a_>
		explicit ArrayBase(const ExtentType<T_, d_, a_> &extent, int) :
				m_isScalar(extent.size() == 0), m_extent(extent) {}

		template<typename OtherDerived>
		ArrayBase(const OtherDerived &other) {
			assign(other);
		}

		template<typename OtherDerived>
		LR_INLINE Derived &operator=(const OtherDerived &other) {
			return assign(other);
		}

		template<typename T>
		LR_NODISCARD("")
		LR_FORCE_INLINE auto cast() const {
			using ScalarType = typename internal::traits<T>::Scalar;
			using RetType	 = unary::Cast<ScalarType, Derived>;
			return RetType(derived());
		}

		IMPL_BINOP(operator+, ScalarSum)
		IMPL_BINOP(operator-, ScalarDiff)
		IMPL_BINOP(operator*, ScalarProd)
		IMPL_BINOP(operator/, ScalarDiv)

		IMPL_BINOP_SCALAR(operator+, ScalarSum)
		IMPL_BINOP_SCALAR(operator-, ScalarDiff)
		IMPL_BINOP_SCALAR(operator*, ScalarProd)
		IMPL_BINOP_SCALAR(operator/, ScalarDiv)

		IMPL_BINOP(operator|, BitwiseOr)
		IMPL_BINOP(operator&, BitwiseAnd)
		IMPL_BINOP(operator^, BitwiseXor)

		IMPL_UNOP(operator-, UnaryMinus)
		IMPL_UNOP(operator~, BitwiseNot)
		IMPL_UNOP(operator!, UnaryNot)

		template<typename T_ = int64_t, int64_t d_ = 32, int64_t a_ = 1>
		auto transposed(const ExtentType<T_, d_, a_> &order_ = {}) const {
			using RetType = unop::CWiseUnop<functors::matrix::Transpose<Derived>, Derived>;
			static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;
			static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;

			static_assert(!(Required & ~(Flags & Required)),
						  "Scalar type is incompatible with Functor");

			ArrayExtent order;
			if (order_.dims() == -1) {
				// Default order is to reverse all indices
				order = ArrayExtent::zero(m_extent.dims());
				for (int64_t i = 0; i < m_extent.dims(); ++i) {
					order[m_extent.dims() - i - 1] = i;
				}
			} else {
				order = order_;
			}

			return RetType(derived(), order);
		}

		LR_NODISCARD("Do not ignore the result of an evaluated calculation")
		auto eval() const { return derived(); }

		template<typename OtherDerived>
		LR_FORCE_INLINE void loadFrom(int64_t index, const OtherDerived &other) {
			LR_ASSERT(index >= 0 && index < m_extent.sizeAdjusted(),
					  "Index {} is out of range for Array with extent {}",
					  index,
					  m_extent.str());
			derived().writePacket(index, other.packet(index));
		}

		template<typename ScalarType>
		LR_FORCE_INLINE void loadFromScalar(int64_t index, const ScalarType &other) {
			LR_ASSERT(index >= 0 && index < m_extent.sizeAdjusted(),
					  "Index {} is out of range for Array with extent {}",
					  index,
					  m_extent.str());
			derived().writeScalar(index, other.scalar(index));
		}

		LR_FORCE_INLINE Derived &assign(const Scalar &other) {
			// Construct if necessary
			if (!m_storage) {
				m_extent   = ArrayExtent(1);
				m_storage  = StorageType(m_extent.sizeAdjusted());
				m_isScalar = true;
			}

			LR_ASSERT(m_isScalar, "Cannot assign Scalar to non-scalar Array");
			m_storage[0] = other;
			return derived();
		}

		template<typename OtherDerived>
		LR_FORCE_INLINE Derived &assign(const OtherDerived &other) {
			// Construct if necessary
			if (!m_storage) {
				m_extent  = other.extent();
				m_storage = StorageType(m_extent.sizeAdjusted());
			}

			LR_ASSERT(m_extent == other.extent(),
					  "Cannot perform operation on Arrays with {} and {}. Extents must be equal",
					  m_extent.str(),
					  other.extent().str());

			m_isScalar = other.isScalar();

			using Selector = functors::AssignSelector<Derived, OtherDerived, false>;
			return Selector::run(derived(), other.derived());
		}

		template<typename OtherDerived>
		LR_FORCE_INLINE Derived &assignLazy(const OtherDerived &other) {
			LR_ASSERT(m_extent == other.extent(),
					  "Cannot perform operation on Arrays with {} and {}. Extents must be equal",
					  m_extent.str(),
					  other.extent().str());

			using Selector = functors::AssignOp<Derived, OtherDerived>;
			Selector::run(derived(), other.derived());
			return derived();
		}

		LR_NODISCARD("") LR_FORCE_INLINE const Derived &derived() const {
			return *static_cast<const Derived *>(this);
		}

		LR_FORCE_INLINE Packet packet(int64_t index) const {
			Packet p;
			if constexpr (is_same_v<Scalar, bool>)
				p.load(m_storage.heap() + (index / 64));
			else
				p.load(m_storage.heap() + index);
			return p;
		}

		LR_FORCE_INLINE Scalar scalar(int64_t index) const { return m_storage[index]; }

		template<typename T>
		std::string genKernel(std::vector<T> &vec, int64_t &index) const {
			vec.emplace_back(m_storage.heap());
			return fmt::format("arg{}", index++);
		}

		LR_NODISCARD("") LR_FORCE_INLINE Derived &derived() {
			return *static_cast<Derived *>(this);
		}

		LR_NODISCARD("") bool isScalar() const { return m_isScalar; }
		LR_NODISCARD("") const StorageType &storage() const { return m_storage; }
		LR_NODISCARD("") StorageType &storage() { return m_storage; }
		LR_NODISCARD("") ArrayExtent extent() const { return m_extent; }
		LR_NODISCARD("") ArrayExtent &extent() { return m_extent; }

	private:
		bool m_isScalar = false;
		ArrayExtent m_extent;
		StorageType m_storage;
	};

	IMPL_BINOP_SCALAR_EXTERNAL(operator+, ScalarSum)
	IMPL_BINOP_SCALAR_EXTERNAL(operator-, ScalarDiff)
	IMPL_BINOP_SCALAR_EXTERNAL(operator*, ScalarProd)
	IMPL_BINOP_SCALAR_EXTERNAL(operator/, ScalarDiv)
} // namespace librapid

#undef IMPL_BINOP
#undef IMPL_BINOP_SCALAR
#undef IMPL_BINOP_SCALAR_EXTERNAL
#undef IMPL_UNOP

// TODO: Optimise this for GPU accesses
#define IMPL_BINOP(NAME, ASSIGN, OP)                                                               \
	template<typename Other>                                                                       \
	auto NAME(const Other &other) const {                                                          \
		return get() OP((T)other);                                                                 \
	}                                                                                              \
                                                                                                   \
	template<typename Other>                                                                       \
	void ASSIGN(const Other &other) {                                                              \
		set(get() OP(T) other);                                                                    \
	}

#define IMPL_BINOP_EXTERN(NAME, ASSIGN, OP)                                                        \
	template<typename Other,                                                                       \
			 typename T,                                                                           \
			 typename d,                                                                           \
			 typename std::enable_if_t<!is_same_v<Other, ValueReference<T, d>>, int> = 0>               \
	auto NAME(const Other &other, const ValueReference<T, d> &val) {                               \
		return other OP((T)val);                                                                   \
	}                                                                                              \
                                                                                                   \
	template<typename Other,                                                                       \
			 typename T,                                                                           \
			 typename d,                                                                           \
			 typename std::enable_if_t<!is_same_v<Other, ValueReference<T, d>>, int> = 0>               \
	void ASSIGN(Other &other, const ValueReference<T, d> &val) {                                   \
		other = other OP((T)val);                                                                  \
	}

// TODO: Optimise this for GPU accesses
#define IMPL_BINOP2(NAME, OP)                                                                      \
	template<typename Other>                                                                       \
	auto NAME(const Other &other) const {                                                          \
		return get() OP((T)other);                                                                 \
	}

#define IMPL_BINOP2_EXTERN(NAME, OP)                                                               \
	template<typename Other,                                                                       \
			 typename T,                                                                           \
			 typename d,                                                                           \
			 typename std::enable_if_t<!is_same_v<Other, ValueReference<T, d>>, int> = 0>               \
	auto NAME(const Other &other, const ValueReference<T, d> &val) {                               \
		return other OP((T)val);                                                                   \
	}

#define IMPL_UNOP(NAME, OP)                                                                        \
	template<typename Other>                                                                       \
	auto NAME() const {                                                                            \
		return OP get();                                                                           \
	}

namespace librapid {
	namespace internal {
		template<typename T, typename d>
		struct traits<memory::ValueReference<T, d>> {
			using Scalar = T;
			using Device = d;
			using Packet = typename traits<Scalar>::Packet;
		};
	} // namespace internal

	namespace memory {
		template<typename T, typename d>
		class ValueReference {
		public:
			ValueReference() = delete;

			explicit ValueReference(T *val) : m_value(val) {}

			explicit ValueReference(T &val) : m_value(&val) {
				static_assert(std::is_same<d, device::CPU>::value,
							  "Cannot construct Device ValueReference from Host scalar");
			}

			ValueReference(const ValueReference<T, d> &other) : m_value(other.m_value) {}

			ValueReference &operator=(const ValueReference<T, d> &other) {
				if (&other == this) return *this;
				m_value = other.m_value;
				return *this;
			}

			template<typename Type, typename Device>
			ValueReference &operator=(const ValueReference<Type, Device> &other) {
				if constexpr (std::is_same<d, device::CPU>::value)
					*m_value = *other.get_();
				else
					memcpy<T, d, Type, Device>(m_value, other.get_(), 1);
				return *this;
			}

			ValueReference &operator=(const T &val) {
				if constexpr (std::is_same<d, device::CPU>::value) {
					*m_value = val;
				} else {
					T tmp = val;
					memcpy<T, d, T, device::CPU>(m_value, &tmp, 1);
				}
				return *this;
			}

			template<typename Type>
			LR_NODISCARD("")
			LR_INLINE operator Type() const {
				if constexpr (std::is_same<d, device::CPU>::value) {
					return *m_value;
				} else {
					T res;
					memcpy<T, device::CPU, T, d>(&res, m_value, 1);
					return res;
				}
			}

			LR_NODISCARD("") T get() const { return *m_value; }
			void set(T value) { *m_value = value; }

			IMPL_BINOP2(operator==, ==);
			IMPL_BINOP2(operator!=, !=);
			IMPL_BINOP2(operator>, >);
			IMPL_BINOP2(operator>=, >=);
			IMPL_BINOP2(operator<, <);
			IMPL_BINOP2(operator<=, <=);

			IMPL_BINOP(operator+, operator+=, +);
			IMPL_BINOP(operator-, operator-=, -);
			IMPL_BINOP(operator*, operator*=, *);
			IMPL_BINOP(operator/, operator/=, /);

			IMPL_BINOP(operator|, operator|=, |);
			IMPL_BINOP(operator&, operator&=, &);
			IMPL_BINOP(operator^, operator^=, ^);

			LR_NODISCARD("") T *get_() const { return m_value; }

		protected:
			T *m_value = nullptr;
		};

		template<typename d>
		class ValueReference<bool, d>
				: public ValueReference<typename internal::traits<bool>::BaseScalar, d> {
		public:
			using BaseScalar = typename internal::traits<bool>::BaseScalar;
			using Base		 = ValueReference<BaseScalar, d>;
			using T			 = bool;

			ValueReference() : Base() {}

			explicit ValueReference(BaseScalar *val, uint16_t bit) : Base(val), m_bit(bit) {
				LR_ASSERT(
				  bit >= 0 && bit < (sizeof(BaseScalar) * 8),
				  "Bit {} is out of range for ValueReference<bool>. Bit index must be in the "
				  "range [0, {})",
				  bit,
				  sizeof(BaseScalar) * 8);
			}

			ValueReference(const ValueReference<bool, d> &other) :
					Base(other), m_bit(other.m_bit) {}

			template<typename Type, typename Device>
			ValueReference &operator=(const ValueReference<Type, Device> &other) {
				if (&other == this) return *this;
				set((bool)other);
				return *this;
			}

			ValueReference &operator=(const bool &val) {
				set(val);
				return *this;
			}

			LR_NODISCARD("") LR_FORCE_INLINE bool get() const {
				if constexpr (is_same_v<d, device::CPU>) return *(this->m_value) & (1ull << m_bit);

				BaseScalar tmp;
				memcpy<BaseScalar, device::CPU, BaseScalar, d>(&tmp, this->m_value, 1);
				return tmp & (1ull << m_bit);
			}

			LR_FORCE_INLINE void set(bool value) {
				if constexpr (is_same_v<d, device::CPU>) {
					if (value)
						*(this->m_value) |= (1ull << m_bit);
					else
						*(this->m_value) &= ~(1ull << m_bit);
				} else {
#if defined(LIBRAPID_HAS_CUDA)
					std::string kernel = fmt::format(R"V0G0N(bitSetKernel
					#include <stdint.h>
					__global__
					void bitSetKernel({} *block, uint16_t bit, bool value) {{
						if (value)
							*block |= (1ull << bit);
						else
							*block &= ~(1ull << bit);
					}}
				)V0G0N",
													 internal::traits<BaseScalar>::Name);

					static jitify::JitCache kernelCache;
					jitify::Program program = kernelCache.program(kernel);

					dim3 grid(1);
					dim3 block(1);

					using jitify::reflection::Type;
					jitifyCall(program.kernel("bitSetKernel")
								 .instantiate()
								 .configure(grid, block, 0, cudaStream)
								 .launch(this->m_value, m_bit, value));
#else
					LR_ASSERT(false, "CUDA support was not enabled");
#endif
				}
			}

			IMPL_BINOP2(operator==, ==);
			IMPL_BINOP2(operator!=, !=);
			IMPL_BINOP2(operator>, >);
			IMPL_BINOP2(operator>=, >=);
			IMPL_BINOP2(operator<, <);
			IMPL_BINOP2(operator<=, <=);

			IMPL_BINOP(operator+, operator+=, +);
			IMPL_BINOP(operator-, operator-=, -);
			IMPL_BINOP(operator*, operator*=, *);
			IMPL_BINOP(operator/, operator/=, /);

			IMPL_BINOP(operator|, operator|=, |);
			IMPL_BINOP(operator&, operator&=, &);
			IMPL_BINOP(operator^, operator^=, ^);

			IMPL_UNOP(operator!, !);
			IMPL_UNOP(operator~, ~);

			template<typename Type>
			LR_NODISCARD("")
			LR_INLINE operator Type() const {
				return (Type)get();
			}

		private:
			uint16_t m_bit; // Bit masked value
		};

		IMPL_BINOP2_EXTERN(operator==, ==)
		IMPL_BINOP2_EXTERN(operator!=, !=)
		IMPL_BINOP2_EXTERN(operator>, >)
		IMPL_BINOP2_EXTERN(operator>=, >=)
		IMPL_BINOP2_EXTERN(operator<, <)
		IMPL_BINOP2_EXTERN(operator<=, <=)

		IMPL_BINOP_EXTERN(operator+, operator+=, +)
		IMPL_BINOP_EXTERN(operator-, operator-=, -)
		IMPL_BINOP_EXTERN(operator*, operator*=, *)
		IMPL_BINOP_EXTERN(operator/, operator/=, /)

		IMPL_BINOP_EXTERN(operator|, operator|=, |)
		IMPL_BINOP_EXTERN(operator&, operator&=, &)
		IMPL_BINOP_EXTERN(operator^, operator^=, ^)
	} // namespace memory
} // namespace librapid

#ifdef FMT_API
template<typename T, typename d>
struct fmt::formatter<librapid::memory::ValueReference<T, d>> {
	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		return ctx.begin();
	}

	template<typename FormatContext>
	auto format(const librapid::memory::ValueReference<T, d> &val, FormatContext &ctx) {
		return fmt::format_to(ctx.out(), fmt::format("{}", (T)val));
	}
};
#endif // FMT_API

#undef IMPL_BINOP
#undef IMPL_BINOP2
#undef IMPL_BINOP_EXTERN
#undef IMPL_BINOP2_EXTERN
#undef IMPL_UNOP

namespace librapid::memory {
	template<typename T, typename d>
	class DenseStorage {
	public:
		using Type = T;
		friend DenseStorage<bool, d>;

		DenseStorage() = default;

		explicit DenseStorage(size_t size) :
				m_size(roundUpTo(size, internal::traits<T>::PacketWidth)),
				m_heap(memory::malloc<T, d>(m_size)), m_refCount(new std::atomic<int64_t>(1)) {
#if defined(LIBRAPID_HAS_CUDA)
			if constexpr (is_same_v<d, device::GPU>) initializeCudaStream();
#endif
		}

		DenseStorage(const DenseStorage<T, d> &other) {
			m_refCount	= other.m_refCount;
			m_size		= other.m_size;
			m_heap		= other.m_heap;
			m_memOffset = other.m_memOffset;
			increment();
		}

		DenseStorage &operator=(const DenseStorage<T, d> &other) {
			if (this == &other) return *this;

			decrement();
			m_size		= other.m_size;
			m_heap		= other.m_heap;
			m_refCount	= other.m_refCount;
			m_memOffset = other.m_memOffset;
			increment();

			return *this;
		}

		~DenseStorage() { decrement(); }

		operator bool() const { return (bool)m_refCount; }

		ValueReference<T, d> operator[](int64_t index) const {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);
			return ValueReference<T, d>(m_heap + m_memOffset + index);
		}

		ValueReference<T, d> operator[](int64_t index) {
			LR_ASSERT(index >= 0 && index < m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  m_size);
			return ValueReference<T, d>(m_heap + m_memOffset + index);
		}

		void offsetMemory(int64_t off) { m_memOffset += off; }

		LR_NODISCARD("") LR_FORCE_INLINE T *__restrict heap() const { return m_heap + m_memOffset; }

		LR_NODISCARD("") int64_t size() const { return m_size; }

		LR_NODISCARD("") int64_t bytes() const { return sizeof(T) * m_size; }

		void increment() const { (*m_refCount)++; }

		void decrement() {
			if (!m_refCount) return;
			(*m_refCount)--;
			if (*m_refCount == 0) {
				delete m_refCount;
				memory::free<T, d>(m_heap);
			}
		}

	protected:
		int64_t m_size					 = 0;
		T *m_heap						 = nullptr;
		std::atomic<int64_t> *m_refCount = nullptr;
		int64_t m_memOffset				 = 0;
	};

	template<typename d>
	class DenseStorage<bool, d>
			: public DenseStorage<typename internal::traits<bool>::BaseScalar, d> {
	public:
		using Type		 = bool;
		using BaseScalar = typename internal::traits<bool>::BaseScalar;
		using Base		 = DenseStorage<BaseScalar, d>;

		DenseStorage() : Base() {};

		explicit DenseStorage(int64_t size) : Base((size + 512) / (sizeof(BaseScalar) * 8)) {
			this->m_size = size;
#if defined(LIBRAPID_HAS_CUDA)
			if constexpr (is_same_v<d, device::GPU>) initializeCudaStream();
#endif
		}

		ValueReference<bool, d> operator[](int64_t index) const {
			LR_ASSERT(index >= 0 && index < this->m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  this->m_size);
			index += this->m_memOffset;
			uint64_t block = index / (sizeof(BaseScalar) * 8);
			uint16_t bit   = mod<BaseScalar>(index, sizeof(BaseScalar) * 8);
			return ValueReference<bool, d>(this->m_heap + block, bit);
		}

		ValueReference<bool, d> operator[](int64_t index) {
			LR_ASSERT(index >= 0 && index < this->m_size,
					  "Index {} is out of range for DenseStorage object with size {}",
					  index,
					  this->m_size);
			index += this->m_memOffset;
			uint64_t block = index / (sizeof(BaseScalar) * 8);
			uint16_t bit   = mod<BaseScalar>(index, sizeof(BaseScalar) * 8);
			return ValueReference<bool, d>(this->m_heap + block, bit);
		}

		LR_NODISCARD("") BaseScalar *heap() const {
			return this->m_heap + (this->m_memOffset / (sizeof(BaseScalar) * 8));
		}
	};

	template<typename T, typename d, typename T_, typename d_>
	LR_INLINE void memcpy(DenseStorage<T, d> &dst, const DenseStorage<T_, d_> &src) {
		LR_ASSERT(dst.size() == src.size(),
				  "Cannot copy data between DenseStorage objects with different sizes");
		memcpy<T, d, T_, d_>(dst.heap(), src.heap(), dst.size());
	}
} // namespace librapid::memory

namespace librapid { namespace detail {
	template<typename T>
	std::string kernelFormat(const T &val) {
		return fmt::format("{}", val);
	}
} }

namespace librapid {
	namespace internal {
		template<typename Binop, typename LHS, typename RHS>
		struct traits<binop::CWiseBinop<Binop, LHS, RHS>> {
			static constexpr bool IsScalar = false;
			using Valid					   = std::true_type;
			using Type					   = binop::CWiseBinop<Binop, LHS, RHS>;
			using Scalar				   = typename Binop::RetType;
			using BaseScalar			   = typename traits<Scalar>::BaseScalar;
			using Packet				   = typename traits<Scalar>::Packet;
			using DeviceLHS				   = typename traits<LHS>::Device;
			using DeviceRHS				   = typename traits<RHS>::Device;
			using Device	  = typename memory::PromoteDevice<DeviceLHS, DeviceRHS>::type;
			using StorageType = memory::DenseStorage<Scalar, Device>;
			static constexpr uint64_t Flags =
			  Binop::Flags | traits<LHS>::Flags | traits<RHS>::Flags;
		};
	} // namespace internal

	namespace binop {
		template<typename Binop, typename LHS, typename RHS>
		class CWiseBinop
				: public ArrayBase<CWiseBinop<Binop, LHS, RHS>,
								   typename internal::PropagateDeviceType<LHS, RHS>::Device> {
		public:
			using Operation = Binop;
			using Scalar	= typename Binop::RetType;
			using Packet	= typename internal::traits<Scalar>::Packet;
			using LeftType	= typename internal::StripQualifiers<LHS>;
			using RightType = typename internal::StripQualifiers<RHS>;
			using DeviceLHS = typename internal::traits<LHS>::Device;
			using DeviceRHS = typename internal::traits<RHS>::Device;
			using Device	= typename memory::PromoteDevice<DeviceRHS, DeviceLHS>::type;
			using Type		= CWiseBinop<Binop, LHS, RHS>;
			using Base		= ArrayBase<Type, Device>;
			static constexpr bool LhsIsScalar = internal::traits<LeftType>::IsScalar;
			static constexpr bool RhsIsScalar = internal::traits<RightType>::IsScalar;
			static constexpr uint64_t Flags	  = internal::traits<Type>::Flags;

			CWiseBinop() = delete;

			template<typename... Args>
			CWiseBinop(const LeftType &lhs, const RightType &rhs, Args... opArgs) :
					Base(
					  [&]() {
						  if constexpr (LhsIsScalar)
							  return rhs.extent();
						  else
							  return lhs.extent();
					  }(),
					  0),
					m_lhs(lhs), m_rhs(rhs), m_operation(opArgs...) {}

			CWiseBinop(const Type &op) :
					Base(op.extent(), 0), m_lhs(op.m_lhs), m_rhs(op.m_rhs),
					m_operation(op.m_operation) {}

			CWiseBinop &operator=(const Type &op) {
				if (this == &op) return *this;

				Base::m_extent = op.m_extent;

				m_lhs		= op.m_lhs;
				m_rhs		= op.m_rhs;
				m_operation = op.m_operation;

				return *this;
			}

			LR_NODISCARD("") Array<Scalar, Device> operator[](int64_t index) const {
				LR_WARN_ONCE(
				  "Calling operator[] on a lazy-evaluation object forces evaluation every time. "
				  "Consider using operator() instead");

				auto res = eval();
				return res[index];
			}

			template<typename... T>
			LR_NODISCARD("")
			auto operator()(T... indices) const {
				LR_ASSERT((this->m_isScalar && sizeof...(T) == 1) ||
							sizeof...(T) == Base::extent().dims(),
						  "Array with {0} dimensions requires {0} access indices. Received {1}",
						  Base::extent().dims(),
						  sizeof...(indices));

				int64_t index = Base::isScalar() ? 0 : Base::extent().index(indices...);
				return scalar(index);
			}

			LR_NODISCARD("Do not ignore the result of an evaluated calculation")
			Array<Scalar, Device> eval() const {
				ExtentType<int64_t, 32> resExtent;
				if constexpr (LhsIsScalar && RhsIsScalar) {
					LR_ASSERT(false, "This should never happen");
				} else if constexpr (LhsIsScalar && !RhsIsScalar) {
					resExtent = m_operation.genExtent(m_rhs.extent());
				} else if constexpr (!LhsIsScalar && RhsIsScalar) {
					resExtent = m_operation.genExtent(m_lhs.extent());
				} else {
					resExtent = m_operation.genExtent(m_lhs.extent(), m_rhs.extent());
				}

				Array<Scalar, Device> res(resExtent);

				if constexpr ((bool)(Flags & internal::flags::HasCustomEval)) {
					m_operation.customEval(m_lhs, m_rhs, res);
					return res;
				}

				res.assign(*this);
				return res;
			}

			LR_FORCE_INLINE Packet packet(int64_t index) const {
				if constexpr (LhsIsScalar && RhsIsScalar)
					return m_operation.packetOp(m_lhs, m_rhs);
				else if constexpr (LhsIsScalar && !RhsIsScalar)
					return m_operation.packetOp(m_lhs, m_rhs.packet(index));
				else if constexpr (!LhsIsScalar && RhsIsScalar)
					return m_operation.packetOp(m_lhs.packet(index), m_rhs);
				else
					return m_operation.packetOp(m_lhs.packet(index), m_rhs.packet(index));
			}

			LR_FORCE_INLINE Scalar scalar(int64_t index) const {
				if constexpr (LhsIsScalar && RhsIsScalar)
					return m_operation.scalarOp(m_lhs, m_rhs);
				else if constexpr (LhsIsScalar && !RhsIsScalar)
					return m_operation.scalarOp(m_lhs, m_rhs.scalar(index));
				else if constexpr (!LhsIsScalar && RhsIsScalar)
					return m_operation.scalarOp(m_lhs.scalar(index), m_rhs);
				else
					return m_operation.scalarOp(m_lhs.scalar(index), m_rhs.scalar(index));
			}

			template<typename T>
			std::string genKernel(std::vector<T> &vec, int64_t &index) const {
				// std::string leftKernel	= m_lhs.genKernel(vec, index);
				// std::string rightKernel = m_rhs.genKernel(vec, index);

				std::string leftKernel, rightKernel;

				if constexpr (LhsIsScalar && RhsIsScalar) {
					leftKernel	= detail::kernelFormat(m_lhs);
					rightKernel = detail::kernelFormat(m_rhs);
				} else if constexpr (LhsIsScalar && !RhsIsScalar) {
					leftKernel	= detail::kernelFormat(m_lhs);
					rightKernel = m_rhs.genKernel(vec, index);
				} else if constexpr (!LhsIsScalar && RhsIsScalar) {
					leftKernel	= m_lhs.genKernel(vec, index);
					rightKernel = detail::kernelFormat(m_rhs);
				} else {
					leftKernel	= m_lhs.genKernel(vec, index);
					rightKernel = m_rhs.genKernel(vec, index);
				}

				std::string op = m_operation.genKernel();
				return fmt::format("({} {} {})", leftKernel, op, rightKernel);
			}

			LR_NODISCARD("")
			std::string str(std::string format = "", const std::string &delim = " ",
							int64_t stripWidth = -1, int64_t beforePoint = -1,
							int64_t afterPoint = -1, int64_t depth = 0) const {
				return eval().str(format, delim, stripWidth, beforePoint, afterPoint, depth);
			}

		private:
			LeftType m_lhs;
			RightType m_rhs;
			Binop m_operation {};
		};
	} // namespace binop
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<typename Binop, typename LHS, typename RHS>
struct fmt::formatter<librapid::binop::CWiseBinop<Binop, LHS, RHS>> {
	std::string formatStr = "{}";

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		formatStr = "{:";
		auto it	  = ctx.begin();
		for (; it != ctx.end(); ++it) {
			if (*it == '}') break;
			formatStr += *it;
		}
		formatStr += "}";
		return it;
	}

	template<typename FormatContext>
	auto format(const librapid::binop::CWiseBinop<Binop, LHS, RHS> &arr, FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), arr.str(formatStr));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API

namespace librapid {
	namespace internal {
		template<typename Unop, typename TYPE>
		struct traits<unop::CWiseUnop<Unop, TYPE>> {
			static constexpr bool IsScalar	= false;
			using Valid						= std::true_type;
			using Type						= unop::CWiseUnop<Unop, TYPE>;
			using Scalar					= typename Unop::RetType;
			using BaseScalar				= typename traits<Scalar>::BaseScalar;
			using Packet					= typename traits<Scalar>::Packet;
			using Device					= typename traits<TYPE>::Device;
			using StorageType				= memory::DenseStorage<Scalar, Device>;
			static constexpr uint64_t Flags = Unop::Flags | traits<TYPE>::Flags;
		};
	} // namespace internal

	namespace unop {
		template<typename Unop, typename TYPE>
		class CWiseUnop
				: public ArrayBase<CWiseUnop<Unop, TYPE>, typename internal::traits<TYPE>::Device> {
		public:
			using Operation					= Unop;
			using Scalar					= typename Unop::RetType;
			using Packet					= typename internal::traits<Scalar>::Packet;
			using ValType					= typename internal::StripQualifiers<TYPE>;
			using Device					= typename internal::traits<TYPE>::Device;
			using Type						= CWiseUnop<Unop, TYPE>;
			using Base						= ArrayBase<Type, Device>;
			static constexpr uint64_t Flags = internal::traits<Type>::Flags;

			CWiseUnop() = delete;

			template<typename... Args>
			explicit CWiseUnop(const ValType &value, Args... opArgs) :
					Base(value.extent(), 0), m_value(value), m_operation(opArgs...) {}

			CWiseUnop(const Type &op) :
					Base(op.extent(), 0), m_value(op.m_value), m_operation(op.m_operation) {}

			template<typename T>
			CWiseUnop &operator=(const T &op) {
				static_assert(
				  is_same_v<T, Type>,
				  "Lazy-evaluated result cannot be assigned a different type. Please either "
				  "evaluate the result (using 'eval()') or create a new variable");

				if (this == &op) return *this;

				Base::m_extent = op.m_extent;

				m_value		= op.m_value;
				m_operation = op.m_operation;

				return *this;
			}

			LR_NODISCARD("") Array<Scalar, Device> operator[](int64_t index) const {
				LR_WARN_ONCE(
				  "Calling operator[] on a lazy-evaluation object forces evaluation every time. "
				  "Consider using operator() instead");

				auto res = eval();
				return res[index];
			}

			template<typename... T>
			LR_NODISCARD("")
			auto operator()(T... indices) const {
				LR_ASSERT((this->m_isScalar && sizeof...(T) == 1) ||
							sizeof...(T) == Base::extent().dims(),
						  "Array with {0} dimensions requires {0} access indices. Received {1}",
						  Base::extent().dims(),
						  sizeof...(indices));

				int64_t index = Base::isScalar() ? 0 : Base::extent().index(indices...);
				return scalar(index);
			}

			LR_NODISCARD("Do not ignore the result of an evaluated calculation")
			Array<Scalar, Device> eval() const {
				Array<Scalar, Device> res(m_operation.genExtent(m_value.extent()));

				if constexpr ((bool)(Flags & internal::flags::HasCustomEval)) {
					m_operation.customEval(m_value, res);
					return res;
				}

				res.assign(*this);
				return res;
			}

			template<typename Input, typename Output>
			void customEval(const Input &input, Output &output) {
				if constexpr ((bool)(Flags & internal::flags::HasCustomEval)) {
					m_operation.customEval(input, output);
				} else {
					LR_ASSERT(false, "Type does not support custom eval");
				}
			}

			LR_FORCE_INLINE Packet packet(int64_t index) const {
				if constexpr ((bool) (Flags & internal::flags::RequireInput)) {
					return m_operation.packetOpInput(m_value, index);
				} else {
					return m_operation.packetOp(m_value.packet(index));
				}
			}

			LR_FORCE_INLINE Scalar scalar(int64_t index) const {
				if constexpr ((bool) (Flags & internal::flags::RequireInput)) {
					return m_operation.scalarOpInput(m_value, index);
				} else {
					return m_operation.scalarOp(m_value.scalar(index));
				}
			}

			template<typename T>
			LR_NODISCARD("")
			std::string genKernel(std::vector<T> &vec, int64_t &index) const {
				std::string kernel = m_value.genKernel(vec, index);
				std::string op	   = m_operation.genKernel();
				return fmt::format("({}{})", op, kernel);
			}

			LR_NODISCARD("")
			std::string str(std::string format = "", const std::string &delim = " ",
							int64_t stripWidth = -1, int64_t beforePoint = -1,
							int64_t afterPoint = -1, int64_t depth = 0) const {
				return eval().str(format, delim, stripWidth, beforePoint, afterPoint, depth);
			}

		protected:
			ValType m_value;
			Operation m_operation {};
		};
	} // namespace unop
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<typename Unop, typename TYPE>
struct fmt::formatter<librapid::unop::CWiseUnop<Unop, TYPE>> {
	std::string formatStr = "{}";

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		formatStr = "{:";
		auto it	  = ctx.begin();
		for (; it != ctx.end(); ++it) {
			if (*it == '}') break;
			formatStr += *it;
		}
		formatStr += "}";
		return it;
	}

	template<typename FormatContext>
	auto format(const librapid::unop::CWiseUnop<Unop, TYPE> &arr, FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), arr.str(formatStr));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API

namespace librapid { namespace internal {
	template<typename ArrT>
	class CommaInitializer {
	public:
		using Scalar = typename traits<ArrT>::Scalar;

		CommaInitializer() = delete;
		explicit CommaInitializer(ArrT &dst, const Scalar &val) : m_array(dst) {
			next(val);
		}

		CommaInitializer &operator,(const Scalar &other) {
			next(other);
			return *this;
		}

		void next(const Scalar &other) {
			m_array.storage()[m_index] = other;
			++m_index;
		}

	private:
		ArrT &m_array;
		int64_t m_index = 0;
	};
} } // namespace librapid::internal

namespace librapid {
	namespace internal {
		template<typename Scalar_, typename Device_>
		struct traits<Array<Scalar_, Device_>> {
			static constexpr bool IsScalar = false;
			using Valid					   = std::true_type;
			using Scalar				   = Scalar_;
			using BaseScalar			   = typename traits<Scalar>::BaseScalar;
			using Device				   = Device_;
			using Packet				   = typename traits<Scalar>::Packet;
			using StorageType			   = memory::DenseStorage<Scalar, Device>;
			static constexpr int64_t Flags = 0;
		};
	} // namespace internal

	template<typename Scalar_, typename Device_ = device::CPU>
	class Array : public ArrayBase<Array<Scalar_, Device_>, Device_> {
	public:
#if !defined(LIBRAPID_HAS_CUDA)
		static_assert(is_same_v<Device_, device::CPU>, "CUDA support was not enabled");
#endif

		using Scalar					= Scalar_;
		using Device					= Device_;
		using Packet					= typename internal::traits<Scalar>::Packet;
		using Type						= Array<Scalar, Device>;
		using Base						= ArrayBase<Type, Device>;
		using StorageType				= typename internal::traits<Type>::StorageType;
		static constexpr uint64_t Flags = internal::flags::Evaluated;

		Array() = default;

		template<typename T, int64_t d, int64_t a_>
		explicit Array(const ExtentType<T, d, a_> &extent) : Base(extent) {}

		template<typename OtherDerived>
		Array(const OtherDerived &other) : Base(other.extent()) {
			Base::assign(other);
		}

		Array &operator=(const Scalar &other) { return Base::assign(other); }

		Array &operator=(const Array<Scalar, Device> &other) { return Base::assign(other); }

		template<typename OtherDerived,
				 typename std::enable_if_t<!internal::traits<OtherDerived>::IsScalar, int> = 0>
		Array &operator=(const OtherDerived &other) {
			using ScalarOther = typename internal::traits<OtherDerived>::Scalar;
			static_assert(is_same_v<Scalar, ScalarOther>,
						  "Cannot assign Arrays with different types. Please use Array::cast<T>()");

			return Base::assign(other);
		}

		internal::CommaInitializer<Type> operator<<(const Scalar &value) {
			return internal::CommaInitializer<Type>(*this, value);
		}

		Array copy() const {
			Array res(Base::extent());
			res = *this * 1;
			return res;
		}

		auto copyLazy() const { return *this * 1; }

		LR_NODISCARD("") Array<Scalar, Device> operator[](int64_t index) const {
			int64_t memIndex = this->m_isScalar ? 0 : Base::extent().indexAdjusted(index);
			Array<Scalar, Device> res;
			res.m_extent   = Base::extent().partial(1);
			res.m_isScalar = Base::extent().dims() == 1;
			res.m_storage  = Base::storage();
			res.m_storage.offsetMemory(memIndex);

			return res;
		}

		LR_NODISCARD("") Array<Scalar, Device> operator[](int64_t index) {
			int64_t memIndex = this->m_isScalar ? 0 : Base::extent().indexAdjusted(index);
			Array<Scalar, Device> res;
			res.m_extent   = Base::extent().partial(1);
			res.m_isScalar = Base::extent().dims() == 1;
			res.m_storage  = Base::storage();
			res.m_storage.offsetMemory(memIndex);

			return res;
		}

		template<typename... T>
		LR_NODISCARD("")
		auto operator()(T... indices) const {
			LR_ASSERT((this->m_isScalar && sizeof...(T) == 1) ||
						sizeof...(T) == Base::extent().dims(),
					  "Array with {0} dimensions requires {0} access indices. Received {1}",
					  Base::extent().dims(),
					  sizeof...(indices));

			int64_t index = Base::isScalar() ? 0 : Base::extent().index(indices...);
			return Base::storage()[index];
		}

		template<typename... T>
		LR_NODISCARD("")
		auto operator()(T... indices) {
			LR_ASSERT((this->m_isScalar && sizeof...(T) == 1) ||
						sizeof...(T) == Base::extent().dims(),
					  "Array with {0} dimensions requires {0} access indices. Received {1}",
					  Base::extent().dims(),
					  sizeof...(indices));

			int64_t index = Base::isScalar() ? 0 : Base::extent().index(indices...);
			return Base::storage()[index];
		}

		template<typename T = int64_t, int64_t d = 32>
		void transpose(const ExtentType<T, d> &order_ = {}) {
			// Transpose inplace
			auto &extent = Base::extent();
			ExtentType<int64_t, 32> order;
			if (order_.dims() == -1) {
				// Default order is to reverse all indices
				order = ExtentType<int64_t, 32>::zero(extent.dims());
				for (int64_t i = 0; i < extent.dims(); ++i) { order[extent.dims() - i - 1] = i; }
			} else {
				order = order_;
			}

			if constexpr (is_same_v<Device, device::CPU>) {
				Scalar *buffer = memory::malloc<Scalar, Device>(extent.size());
				detail::transpose(true, Base::storage().heap(), extent[0], extent[1], buffer);
				memory::free<Scalar, Device>(buffer);
			} else {
				LR_ASSERT(false, "CUDA support was not enabled");
			}

			extent.swivelInplace(order);
		}

		LR_FORCE_INLINE void writePacket(int64_t index, const Packet &p) {
			LR_ASSERT(index >= 0 && index < Base::extent().sizeAdjusted(),
					  "Index {} is out of range",
					  index);
			p.store(Base::storage().heap() + index);
		}

		LR_FORCE_INLINE void writeScalar(int64_t index, const Scalar &s) {
			Base::storage()[index] = s;
		}

		template<typename T>
		LR_FORCE_INLINE operator T() const {
			LR_ASSERT(Base::isScalar(), "Cannot cast non-scalar Array to scalar value");
			return operator()(0);
		}

		void findLongest(const std::string &format, bool strip, int64_t stripWidth,
						 int64_t &longestInteger, int64_t &longestFloating) const {
			int64_t dims	= Base::extent().dims();
			int64_t zeroDim = Base::extent()[0];
			if (dims > 1) {
				for (int64_t i = 0; i < zeroDim; ++i) {
					if (stripWidth != 0 && strip && i == stripWidth && zeroDim > stripWidth * 2)
						i = zeroDim - stripWidth;

					this->operator[](i).findLongest(
					  format, strip, stripWidth, longestInteger, longestFloating);
				}
			} else {
				// Stringify vector
				for (int64_t i = 0; i < zeroDim; ++i) {
					if (stripWidth != 0 && strip && i == stripWidth && zeroDim > stripWidth * 2)
						i = zeroDim - stripWidth;

					Scalar val			  = this->operator()(i);
					std::string formatted = fmt::format(format, val);
					auto findIter		  = std::find(formatted.begin(), formatted.end(), '.');
					int64_t pointPos	  = findIter - formatted.begin();
					if (findIter == formatted.end()) {
						// No decimal point present
						if (formatted.length() > longestInteger)
							longestInteger = formatted.length();
					} else {
						// Decimal point present
						auto integer  = formatted.substr(0, pointPos);
						auto floating = formatted.substr(pointPos);
						if (integer.length() > longestInteger) longestInteger = integer.length();
						if (floating.length() - 1 > longestFloating)
							longestFloating = floating.length() - 1;
					}
				}
			}
		}

		// Strip modes:
		// stripWidth == -1 => Default
		// stripWidth == 0  => Never strip
		// stripWidth >= 1  => n values are shown
		LR_NODISCARD("")
		std::string str(std::string format = "", const std::string &delim = " ",
						int64_t stripWidth = -1, int64_t beforePoint = -1, int64_t afterPoint = -1,
						int64_t depth = 0) const {
			bool strip = stripWidth > 0;
			if (depth == 0) {
				strip = false;
				// Configure the strip width

				// Always print the full vector if the array has all dimensions as 1 except a single
				// axis, unless specified otherwise
				int64_t nonOneDims = 0;
				for (int64_t i = 0; i < Base::extent().dims(); ++i)
					if (Base::extent()[i] != 1) ++nonOneDims;

				if (nonOneDims == 1 && stripWidth == -1) {
					strip	   = false;
					stripWidth = 0;
				}

				if (stripWidth == -1) {
					if (Base::extent().size() >= 1000) {
						// Strip the middle values
						strip	   = true;
						stripWidth = 3;
					}
				} else if (stripWidth > 0) {
					strip = true;
				}

				if (format.empty()) {
					if constexpr (std::is_floating_point_v<Scalar>)
						format = "{:.6f}";
					else
						format = "{}";
				}

				// Scalars
				if (Base::isScalar()) return fmt::format(format, this->operator()(0));

				int64_t tmpBeforePoint = 0, tmpAfterPoint = 0;
				findLongest(format, strip, stripWidth, tmpBeforePoint, tmpAfterPoint);
				if (beforePoint == -1) beforePoint = tmpBeforePoint;
				if (afterPoint == -1) afterPoint = tmpAfterPoint;
			}

			std::string res = "[";
			int64_t dims	= Base::extent().dims();
			int64_t zeroDim = Base::extent()[0];
			if (dims > 1) {
				for (int64_t i = 0; i < zeroDim; ++i) {
					if (stripWidth != 0 && strip && i == stripWidth && zeroDim > stripWidth * 2) {
						i = zeroDim - stripWidth;
						res += std::string(depth + 1, ' ') + "...\n";
						if (dims > 2) res += "\n";
					}

					if (i > 0) res += std::string(depth + 1, ' ');
					res += this->operator[](i).str(
					  format, delim, stripWidth, beforePoint, afterPoint, depth + 1);
					if (i + 1 < zeroDim) res += "\n";
					if (i + 1 < zeroDim && dims > 2) res += "\n";
				}
			} else {
				// Stringify vector
				for (int64_t i = 0; i < zeroDim; ++i) {
					if (stripWidth != 0 && strip && i == stripWidth && zeroDim > stripWidth * 2) {
						i = zeroDim - stripWidth;
						res += "... ";
					}

					Scalar val			  = this->operator()(i);
					std::string formatted = fmt::format(format, val);
					auto findIter		  = std::find(formatted.begin(), formatted.end(), '.');
					int64_t pointPos	  = findIter - formatted.begin();
					if (findIter == formatted.end()) {
						// No decimal point present
						res += fmt::format("{:>{}}", formatted, beforePoint);
					} else {
						// Decimal point present
						auto integer  = formatted.substr(0, pointPos);
						auto floating = formatted.substr(pointPos);
						res +=
						  fmt::format("{:>{}}{:<{}}", integer, beforePoint, floating, afterPoint);
					}
					if (i + 1 < zeroDim) res += delim;
				}
			}
			return res + "]";
		}
	};
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<typename Scalar, typename Device>
struct fmt::formatter<librapid::Array<Scalar, Device>> {
	std::string formatStr = "{}";

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		formatStr = "{:";
		auto it	  = ctx.begin();
		for (; it != ctx.end(); ++it) {
			if (*it == '}') break;
			formatStr += *it;
		}
		formatStr += "}";
		return it;
	}

	template<typename FormatContext>
	auto format(const librapid::Array<Scalar, Device> &arr, FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), arr.str(formatStr));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API

#pragma warning(pop)

#endif // LIBRAPID_INCLUDE