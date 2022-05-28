#pragma once

#include "unary.hpp"

namespace librapid::functors::unop {
	template<typename Type_>
	class UnaryMinus : public UnaryOp<Type_> {
	public:
		using Scalar				   = typename internal::traits<Type_>::Scalar;
		using RetType				   = Scalar;
		using Packet				   = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		UnaryMinus() = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return -val; }

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {
			return -val;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "-"; }

	private:
	};
} // namespace librapid::functors::unop