#pragma once

#include "../../../internal/config.hpp"

namespace librapid::functors::unop {
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
} // namespace librapid::functors::unop
