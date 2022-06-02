#pragma once

#include "../../../internal/config.hpp"
#include "../../traits.hpp"

namespace librapid::functors::unop {
	template<typename TYPE>
	class UnaryOp {
	public:
		using Type					   = TYPE;
		using RetType				   = std::false_type;
		static constexpr int64_t Flags = 0;

		template<typename T, int64_t d>
		LR_NODISCARD("")
		Extent<T, d> genExtent(const Extent<T, d> &extent) const {
			return extent;
		}
	};
} // namespace librapid::functors::unop
