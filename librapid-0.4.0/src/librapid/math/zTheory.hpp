#pragma once

#include "../internal/config.hpp"

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