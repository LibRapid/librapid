#pragma once

#include "unary.hpp"

namespace librapid { namespace functors { namespace unop {
	template<typename Type_>
	class UnaryMinus : public UnaryOp<Type_> {
	public:
		using Type					   = Type_;
		using Scalar				   = typename internal::traits<Type_>::Scalar;
		using RetType				   = Scalar;
		using Packet				   = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags = internal::flags::Unary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		UnaryMinus() = default;

		UnaryMinus(const UnaryMinus<Type> &other) = default;

		UnaryMinus<Type> &operator=(const UnaryMinus<Type> &other) = default;

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

	template<typename Type_>
	class UnaryNot : public UnaryOp<Type_> {
	public:
		using Type					   = Type_;
		using Scalar				   = typename internal::traits<Type_>::Scalar;
		using RetType				   = Scalar;
		using Packet				   = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags = internal::flags::Unary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		UnaryNot() = default;

		UnaryNot(const UnaryNot<Type> &other) = default;

		UnaryNot<Type> &operator=(const UnaryNot<Type> &other) = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return !val; }

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {
			return !val;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "-"; }

	private:
	};

	template<typename Type_>
	class BitwiseNot : public UnaryOp<Type_> {
	public:
		using Type					   = Type_;
		using Scalar				   = typename internal::traits<Type_>::Scalar;
		using RetType				   = Scalar;
		using Packet				   = typename internal::traits<Scalar>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Bitwise |
										 internal::flags::PacketBitwise |
										 internal::flags::ScalarBitwise;

		BitwiseNot() = default;

		BitwiseNot(const BitwiseNot<Type> &other) = default;

		BitwiseNot<Type> &operator=(const BitwiseNot<Type> &other) = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const Scalar &val) const { return ~val; }

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &val) const {
			return ~val;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "~"; }

	private:
	};
}}} // namespace librapid::functors::unop