#ifndef LIBRAPID_CORE_LIBRAPID_PCH_HPP
#define LIBRAPID_CORE_LIBRAPID_PCH_HPP

/*
 * Include standard library headers and precompile them as part of
 * librapid. This reduces compile times dramatically.
 *
 * Additionally, include the header files of dependencies.
 */

// Standard Library
#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <utility>

#if defined(LIBRAPID_HAS_OMP)
#	include <omp.h>
#endif // LIBRAPID_HAS_OMP

#if (defined(_WIN32) || defined(_WIN64)) && !defined(LIBRAPID_NO_WINDOWS_H)
#	define WIN32_LEAN_AND_MEAN
#	include <Windows.h>
#endif

// Remove a few macros
#undef min
#undef max

// fmtlib
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

// scnlib
#include <scn/scn.h>
#include <scn/tuple_return/tuple_return.h>

// Vc -- SIMD instructions
#if defined(_MSC_VER)
// For Vc, we need to disable the following warnings
#	pragma warning(push)
#	pragma warning(disable : 4244) // conversion from 'int' to 'float', possible loss of data
#	pragma warning(disable : 4324) // structure was padded due to alignment specifier
#	pragma warning(disable : 4127) // conditional expression is constant
#endif

#include <Vc/Vc>
#include <Vc/algorithm>
#include <Vc/cpuid.h>
#include <Vc/common/simdize.h>

#if defined(_MSC_VER)
#	pragma warning(pop)
#endif

// MPFR (modified) -- arbitrary precision floating point numbers
#if defined(LIBRAPID_USE_MULTIPREC)
#	include <mpirxx.h>
#	include <mpreal.h>
#endif // LIBRAPID_USE_MULTIPREC

#endif // LIBRAPID_CORE_LIBRAPID_PCH_HPP