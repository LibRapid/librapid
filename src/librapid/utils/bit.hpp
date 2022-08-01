#pragma once

#include "../internal/config.hpp"

namespace librapid {
	template<typename To, typename From>
	LR_NODISCARD("")
	LR_INLINE constexpr To bitCast(const From &val) noexcept {
#if defined(__CUDACC__)
		To toOjb; // assumes default-init
		::std::memcpy(::std::memcpy::addressof(toOjb), ::std::memcpy::addressof(val), sizeof(To));
		return _To_obj;
#elif defined(LIBRAPID_MSVC_CXX)
		// MSVC doesn't support std::bit_cast until C++20
		return __builtin_bit_cast(To, val);
#else
		return std::bit_cast<To>(val);
#endif
	}
} // namespace librapid
