#pragma once

#include "binary.hpp"

namespace librapid::functors::binary {
#define DEF_SCALAR_FUNCTOR(NAME_, OP_)                                                             \
	template<typename LHS, typename RHS>                                                           \
	class Scalar##NAME_ : public ScalarOp<LHS, RHS> {                                              \
	public:                                                                                        \
		using LhsType				   = typename internal::traits<LHS>::Scalar;                   \
		using RhsType				   = typename internal::traits<RHS>::Scalar;                   \
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;        \
		using Packet				   = typename internal::traits<RetType>::Packet;               \
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Arithmetic |   \
										 internal::flags::PacketArithmetic |                       \
										 internal::flags::ScalarArithmetic;                        \
                                                                                                   \
		Scalar##NAME_() = default;                                                                 \
                                                                                                   \
		Scalar##NAME_(const Scalar##NAME_<LHS, RHS> &other) = default;                             \
                                                                                                   \
		Scalar##NAME_<LHS, RHS> &operator=(const Scalar##NAME_<LHS, RHS> &other) = default;        \
                                                                                                   \
		LR_NODISCARD("")                                                                           \
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {        \
			return left OP_ right;                                                                 \
		}                                                                                          \
                                                                                                   \
		template<typename PacketTypeLHS, typename PacketTypeRHS>                                   \
		LR_NODISCARD("")                                                                           \
		LR_FORCE_INLINE Packet                                                                     \
		  packetOp(const PacketTypeLHS &left, const PacketTypeRHS &right) const {                  \
			return left OP_ right;                                                                 \
		}                                                                                          \
                                                                                                   \
		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return STRINGIFY(OP_); }  \
                                                                                                   \
	private:                                                                                       \
	}

	DEF_SCALAR_FUNCTOR(Sum, +);
	DEF_SCALAR_FUNCTOR(Diff, -);
	DEF_SCALAR_FUNCTOR(Prod, *);
	DEF_SCALAR_FUNCTOR(Div, /);

#undef DEF_SCALAR_FUNCTOR
} // namespace librapid::functors::binary