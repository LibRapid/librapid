#pragma once

namespace librapid::functors::binary {
	template<typename LHS, typename RHS>
	class ScalarOp {
	public:
		using LhsType				   = LHS;
		using RhsType				   = RHS;
		using RetType				   = std::false_type;
		static constexpr i64 Flags = 0;

		template<typename T, i64 d, i64 a>
		LR_NODISCARD("")
		LR_FORCE_INLINE ExtentType<T, d> genExtent(const ExtentType<T, d, a> &lhs,
								   const ExtentType<T, d, a> &rhs) const {
			return lhs;
		}

		// For a scalar operation
		template<typename T, i64 d, i64 a>
		LR_NODISCARD("")
		LR_FORCE_INLINE ExtentType<T, d> genExtent(const ExtentType<T, d, a> &generic) const {
			return generic;
		}
	};
} // namespace librapid::functors::binary