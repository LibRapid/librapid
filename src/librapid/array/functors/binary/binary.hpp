#pragma once

#include "../../../internal/config.hpp"
#include "../../traits.hpp"

namespace librapid::functors::binary {
	template<typename LHS, typename RHS>
	class ScalarOp {
		using LhsType = LHS;
		using RhsType = RHS;
		using RetType = std::false_type;
		static constexpr int64_t Flags = 0;
	};
}