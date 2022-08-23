#pragma once

#include "unary.hpp"

namespace librapid::functors::unop {
#define DEF_NEGATE_FUNCTOR(NAME_, OP_)                                                             \
	template<typename Type_>                                                                       \
	class Unary##NAME_ : public UnaryOp<Type_> {                                                   \
	public:                                                                                        \
		using Type					   = Type_;                                                    \
		using Scalar				   = typename internal::traits<Type_>::Scalar;                 \
		using RetType				   = Scalar;                                                   \
		using Packet				   = typename internal::traits<Scalar>::Packet;                \
		static constexpr int64_t Flags = internal::flags::Unary | internal::flags::Arithmetic |    \
										                        \
										 internal::flags::ScalarArithmetic;                        \
                                                                                                   \
		Unary##NAME_() = default;                                                                  \
                                                                                                   \
		Unary##NAME_(const Unary##NAME_<Type> &other) = default;                                   \
                                                                                                   \
		Unary##NAME_<Type> &operator=(const Unary##NAME_<Type> &other) = default;                  \
                                                                                                   \
		LR_NODISCARD("")                                                                           \
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return OP_ val; }              \
                                                                                                   \
		template<typename PacketType>                                                              \
		LR_NODISCARD("")                                                                           \
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {                             \
			return OP_ val;                                                                        \
		}                                                                                          \
                                                                                                   \
		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return STRINGIFY(OP_); }  \
                                                                                                   \
	private:                                                                                       \
	}

	DEF_NEGATE_FUNCTOR(Minus, -);
	DEF_NEGATE_FUNCTOR(Not, !);
	DEF_NEGATE_FUNCTOR(Invert, ~);

#undef DEF_NEGATE_FUNCTOR
} // namespace librapid::functors::unop