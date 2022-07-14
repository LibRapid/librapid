#include <librapid/internal/config.hpp>

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
}

/*
 * Sources:
 *
 * sqrtApprox,
 * invSqrtApprox => https://en.wikipedia.org/wiki/Methods_of_computing_square_roots
 *
 */
