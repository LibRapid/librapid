#ifndef RAPID_MATH_HPP
#define RAPID_MATH_HPP

#include <librapid/config.hpp>
#include <librapid/math/constants.hpp>
#include <librapid/math/core_math.hpp>

// Disable zero-division warnings for the vector library
#pragma warning(push)
#pragma warning(disable : 4723)
#include <librapid/math/vector.hpp>
#pragma warning(pop)

#endif // RAPID_MATH_HPP