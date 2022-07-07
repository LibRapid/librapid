#pragma once

// Dynamically strength-reduced div and mod
//
// Ideas taken from Sean Baxter's MGPU library.
// These classes provide for reduced complexity division and modulus
// on integers, for the case where the same divisor or modulus will
// be used repeatedly.

#include "../../internal/config.hpp"

#define IS_POW_2(x) (0 == ((x) & ((x)-1)))

namespace librapid { namespace detail {
	LR_INLINE void findDivisor(unsigned int denom, unsigned int &mulCoeff,
							   unsigned int &shiftCoeff);

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

	LR_INLINE int intLog2(int x, bool round_up = false) {
		int a = 31 - clz(x);
		if (round_up) a += !IS_POW_2(x);
		return a;
	}

	LR_INLINE int intLog2(long long x, bool round_up = false) {
		int a = 63 - clz(x);
		if (round_up) a += !IS_POW_2(x);
		return a;
	}

	LR_INLINE void findDivisor(unsigned int denom, unsigned int &mulCoeff,
							   unsigned int &shiftCoeff) {
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
