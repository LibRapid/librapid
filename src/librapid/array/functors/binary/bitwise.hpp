#pragma once

#include "binary.hpp"

namespace librapid::functors::binary {
#define DEF_BITWISE_FUNCTOR(NAME_, OP_)                                                            \
	template<typename LHS, typename RHS>                                                           \
	class Bitwise##NAME_ : public ScalarOp<LHS, RHS> {                                             \
	public:                                                                                        \
		using LhsType				   = typename internal::traits<LHS>::Scalar;                   \
		using RhsType				   = typename internal::traits<RHS>::Scalar;                   \
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;        \
		using Packet				   = typename internal::traits<RetType>::Packet;               \
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Bitwise |      \
										 internal::flags::PacketBitwise |                          \
										 internal::flags::ScalarBitwise;                           \
                                                                                                   \
		Bitwise##NAME_() = default;                                                                \
                                                                                                   \
		Bitwise##NAME_(const Bitwise##NAME_<LHS, RHS> &other) = default;                           \
                                                                                                   \
		Bitwise##NAME_<LHS, RHS> &operator=(const Bitwise##NAME_<LHS, RHS> &other) = default;      \
                                                                                                   \
		LR_NODISCARD("")                                                                           \
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {        \
			return left OP_ right;                                                                 \
		}                                                                                          \
                                                                                                   \
		template<typename PacketType>                                                              \
		LR_NODISCARD("")                                                                           \
		LR_FORCE_INLINE Packet packetOp(const PacketType &left, const PacketType &right) const {   \
			return left OP_ right;                                                                 \
		}                                                                                          \
                                                                                                   \
		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return STRINGIFY(OP_); }  \
                                                                                                   \
	private:                                                                                       \
	}

	DEF_BITWISE_FUNCTOR(Or, |);
	DEF_BITWISE_FUNCTOR(And, &);
	DEF_BITWISE_FUNCTOR(Xor, ^);

#undef DEF_BITWISE_FUNCTOR
} // namespace librapid::functors::binary