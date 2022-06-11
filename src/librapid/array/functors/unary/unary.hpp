#pragma once

#include "../../../internal/config.hpp"
#include "../../traits.hpp"

namespace librapid { namespace functors { namespace unop {
	template<typename TYPE>
	class UnaryOp {
	public:
		using Type					   = TYPE;
		using RetType				   = std::false_type;
		static constexpr int64_t Flags = 0;

		template<typename T, int64_t d>
		LR_NODISCARD("")
		ExtentType<T, d> genExtent(const ExtentType<T, d> &extent) const {
			return extent;
		}
	};
} } } // namespace librapid::functors::unop
