#ifndef LIBRAPID_HPP
#define LIBRAPID_HPP

#include "core/core.hpp"
#include "utils/utils.hpp"
#include "math/math.hpp"
#include "simd/simd.hpp"
#include "datastructures/datastructures.hpp"
#include "array/array.hpp"
#include "autodiff/autodiff.hpp"
#include "core/literals.hpp"
#include "ml/ml.hpp"

namespace librapid {
	LIBRAPID_ALWAYS_INLINE void anotherTestFunction(int x) {
		if (x % 3 == 0) {
			throw std::runtime_error("Divisible by 3");
		} else {
			fmt::print("Not divisible by 3\n");
		}
	}

	LIBRAPID_ALWAYS_INLINE void testFunction(int x) {
		if (x % 5 == 0) {
			anotherTestFunction(x);
		} else {
			fmt::print("Not divisible by 5\n");
		}
	}
}

#endif // LIBRAPID_HPP