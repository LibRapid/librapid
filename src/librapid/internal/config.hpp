#ifndef LIBRAPID_CONFIG_HPP
#define LIBRAPID_CONFIG_HPP

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
#include <future>
#include <random>
#include <fstream>
#include <streambuf>
#include <utility>

#if defined(__unix__)
#	include <unistd.h>
#endif

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
#if defined(LIBRAPID_USE_VC)
#	include <Vc/Vc>
#	include <Vc/algorithm>
#	include <Vc/cpuid.h>
#endif

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

#if !defined(LIBRAPID_MAX_ALLOWED_THREADS)
// Maximum number of threads LibRapid can reasonably support by default
#	define LIBRAPID_MAX_ALLOWED_THREADS 256
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

// If we don't know, defualt to 64 bit
#if !defined(LIBRAPID_32BIT) && !defined(LIBRAPID_64BIT)
#	define LIBRAPID_64BIT
#endif

// Check C++ Version
#if defined(LIBRAPID_MSVC_CXX)
// https://developercommunity.visualstudio.com/t/msvc-incorrectly-defines-cplusplus/139261
// MSVC incorrectly defines the C++ version, so we cannot reliably detect it.
#	if defined(_HAS_CXX20) && _HAS_CXX20
#		define LIBRAPID_CXX_20
#	else
#		define LIBRAPID_CXX_17
#	endif
#else
#	if __cplusplus >= 199711L
#		define LIBRAPID_CXX_98
#	elif __cplusplus >= 201103L
#		define LIBRAPID_CXX_11
#	elif __cplusplus >= 201402L
#		define LIBRAPID_CXX_14
#	elif __cplusplus >= 201703L
#		define LIBRAPID_CXX_17
#	elif __cplusplus >= 202002L
#		define LIBRAPID_CXX_20
#	else
#		define LIBRAPID_CXX_11 // Assume we're using C++ 11???
#	endif
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

#define STRINGIFY(a) STR_IMPL_(a)
#define STR_IMPL_(a) #a

// Settings and macros for inline functions
#if defined(LIBRAPID_NO_INLINE)
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
namespace librapid::internal {
	class ForceANSI {
	public:
		ForceANSI() { system(("chcp " + std::to_string(CP_UTF8)).c_str()); }
	};

	inline const auto ansiForcer = ForceANSI();
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
#	define LR_WARN_ONCE(msg, ...)                                                                 \
		do {                                                                                       \
		} while (0)
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

// BLAS enabled LibRapid
#ifdef LIBRAPID_HAS_BLAS
#	include <cblas.h>
// TODO: LAPACK?
// #	include <lapacke.h> // Currently errors
#endif

#if defined(OPENBLAS_OPENMP) || defined(OPENBLAS_THREAD) || defined(OPENBLAS_SEQUENTIAL)
#	define LIBRAPID_HAS_OPENBLAS
#endif

// CUDA enabled LibRapid
#ifdef LIBRAPID_HAS_CUDA

#	ifdef _MSC_VER
// Disable warnings about unsafe classes
#		pragma warning(disable : 4996)

// Disable zero division errors
#		pragma warning(disable : 4723)
#	endif

#	define CUDA_NO_HALF
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

#	include "../cuda/helper_cuda.h"
#	include "../cuda/helper_functions.h"

#else

#	define CUDA_INCLUDE_DIRS ""

#endif // LIBRAPID_HAS_CUDA

namespace librapid::device {
	struct CPU {};
	struct GPU {};
} // namespace librapid::device

// User Config Variables

namespace librapid {
#ifdef LIBRAPID_HAS_OMP
	static inline unsigned int numThreads	   = 8;
	static inline unsigned int matrixThreads   = 8;
	static inline unsigned int threadThreshold = 2500;
#else
	static unsigned int numThreads		= 1;
	static unsigned int matrixThreads	= 1;
	static unsigned int threadThreshold = 0;
#endif
	static inline bool throwOnAssert					 = false;
	static inline std::vector<std::string> cudaHeaders	 = {};
	static inline std::vector<std::string> nvccOptions	 = {};
	static inline std::vector<std::string> customHeaders = {};
	static inline std::string customCudaCode;
	static inline bool checkComplex = true; // Use faster, less safe methods in the Complex type

	void prec(int64_t dig10);

	namespace internal {
		class PreOptimize {
		public:
			PreOptimize() {
				numThreads	  = (int64_t)((double)std::thread::hardware_concurrency() * (3. / 4.));
				matrixThreads = std::thread::hardware_concurrency();
				nvccOptions.emplace_back("--device-int128");
				prec(25);
			}
		};

		inline const auto optimizer = PreOptimize();
	} // namespace internal
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

#include "../math/mpfr.hpp"
#endif // LIBRAPID_CONFIG_HPP