#pragma once

#include "../../../internal/config.hpp"
#include "../../traits.hpp"

namespace librapid::functors::unop {
	template<typename TYPE>
	class UnaryOp {
		using Type = TYPE;
		using RetType = std::false_type;
		static constexpr int64_t Flags = 0;
	};
}
