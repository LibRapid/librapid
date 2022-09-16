#pragma once

namespace librapid::suffix {
	LR_INLINE auto operator""_h(long double val) { return half((float)val); }
	LR_INLINE auto operator""_mp(const char *val, std::size_t) { return mpfr(val); }
} // namespace librapid::suffix
