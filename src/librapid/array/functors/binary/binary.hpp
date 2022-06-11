#pragma once

#include "../../../internal/config.hpp"
#include "../../traits.hpp"

namespace librapid { namespace functors { namespace binary {
	template<typename LHS, typename RHS>
	class ScalarOp {
	public:

		using LhsType				   = LHS;
		using RhsType				   = RHS;
		using RetType				   = std::false_type;
		static constexpr int64_t Flags = 0;

		template<typename T, int64_t d>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d> &lhs, const ExtentType<T, d> &rhs) const {
			return lhs;
		}

		// For a scalar operation
		template<typename T, int64_t d>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d> &generic) const {
			return generic;
		}
	};
} } } // namespace librapid::functors::binary