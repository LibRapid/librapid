#pragma once

#include "unary.hpp"

namespace librapid::functors::unop {
#define DEF_UNARY_FUNCTOR_GENERIC(NAME_, OP_)                                                      \
	template<typename Type_>                                                                       \
	class NAME_ : public UnaryOp<Type_> {                                                          \
	public:                                                                                        \
		using Type					   = Type_;                                                    \
		using Scalar				   = typename internal::traits<Type_>::Scalar;                 \
		using RetType				   = Scalar;                                                   \
		using Packet				   = typename internal::traits<Scalar>::Packet;                \
		static constexpr int64_t Flags = internal::flags::Unary | internal::flags::Arithmetic |    \
                                                                                                   \
										 internal::flags::ScalarArithmetic;                        \
                                                                                                   \
		NAME_() = default;                                                                         \
                                                                                                   \
		NAME_(const NAME_ &other) = default;                                                       \
                                                                                                   \
		NAME_<Type> &operator=(const NAME_ &other) = default;                                      \
                                                                                                   \
		LR_NODISCARD("")                                                                           \
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return ::librapid::OP_(val); } \
                                                                                                   \
		template<typename PacketType>                                                              \
		LR_NODISCARD("")                                                                           \
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {                             \
			return ::librapid::OP_(val);                                                           \
		}                                                                                          \
                                                                                                   \
		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return STRINGIFY(OP_); }  \
                                                                                                   \
	private:                                                                                       \
	}

	DEF_UNARY_FUNCTOR_GENERIC(Sin, sin);
	DEF_UNARY_FUNCTOR_GENERIC(Cos, cos);
	DEF_UNARY_FUNCTOR_GENERIC(Tan, tan);
	DEF_UNARY_FUNCTOR_GENERIC(Asin, asin);
	DEF_UNARY_FUNCTOR_GENERIC(Acos, acos);
	DEF_UNARY_FUNCTOR_GENERIC(Atan, atan);
	DEF_UNARY_FUNCTOR_GENERIC(Sinh, sinh);
	DEF_UNARY_FUNCTOR_GENERIC(Cosh, cosh);
	DEF_UNARY_FUNCTOR_GENERIC(Tanh, tanh);
	DEF_UNARY_FUNCTOR_GENERIC(Asinh, asinh);
	DEF_UNARY_FUNCTOR_GENERIC(Acosh, acosh);
	DEF_UNARY_FUNCTOR_GENERIC(Atanh, atanh);
	DEF_UNARY_FUNCTOR_GENERIC(Exp, exp);
	DEF_UNARY_FUNCTOR_GENERIC(Exp2, exp2);
	DEF_UNARY_FUNCTOR_GENERIC(Exp10, exp10);
	DEF_UNARY_FUNCTOR_GENERIC(Log, log);
	DEF_UNARY_FUNCTOR_GENERIC(Log2, log2);
	DEF_UNARY_FUNCTOR_GENERIC(Log10, log10);
	DEF_UNARY_FUNCTOR_GENERIC(Sqrt, sqrt);
	DEF_UNARY_FUNCTOR_GENERIC(Abs, abs);
	DEF_UNARY_FUNCTOR_GENERIC(Floor, floor);
	DEF_UNARY_FUNCTOR_GENERIC(Ceil, ceil);

#undef DEF_UNARY_FUNCTOR_GENERIC
} // namespace librapid::functors::unop