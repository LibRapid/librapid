#pragma once

namespace librapid::functors::unop {
	template<typename TYPE>
	class UnaryOp {
	public:
		using Type					   = TYPE;
		using RetType				   = std::false_type;
		static constexpr int64_t Flags = 0;

		template<typename T, i32 d, i32 a>
		LR_NODISCARD("")
		ExtentType<T, d, a> genExtent(const ExtentType<T, d, a> &extent) const {
			return extent;
		}
	};
} // namespace librapid::functors::unop
