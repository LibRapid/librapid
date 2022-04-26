#pragma once

#include "../../../internal/config.hpp"
#include "../../traits.hpp"

namespace librapid::functors::binary {
	template<typename LHS, typename RHS>
	class ScalarOp {
		using LhsType = LHS;
		using RhsType = RHS;
		using RetType = std::false_type;
	};

	template<typename LHS, typename RHS>
	class ScalarSum : public ScalarOp<LHS, RHS> {
	public:
		using LhsType = typename internal::traits<LHS>::Scalar;
		using RhsType = typename internal::traits<RHS>::Scalar;
		using RetType = typename std::common_type<LhsType, RhsType>::type;
		using Packet = typename internal::traits<RetType>::Packet;

		ScalarSum() = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left + right;
		}

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &left, const PacketType &right) const {
			return left + right;
		}

	private:
	};
} // namespace librapid::binary