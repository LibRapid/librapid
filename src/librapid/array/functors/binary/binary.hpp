#pragma once

namespace librapid::functors::binary {
	template<typename LHS, typename RHS>
	class ScalarOp {
	public:
		using LhsType				   = LHS;
		using RhsType				   = RHS;
		using RetType				   = std::false_type;
		static constexpr int64_t Flags = 0;

		template<typename T, i32 d, i32 a>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d, a> &lhs,
								   const ExtentType<T, d, a> &rhs) const {
			return lhs;
		}

		// For a scalar operation
		template<typename T, i32 d, i32 a>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d, a> &generic) const {
			return generic;
		}
	};
} // namespace librapid::functors::binary