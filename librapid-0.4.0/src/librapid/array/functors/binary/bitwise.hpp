#pragma once

#include "binary.hpp"

namespace librapid { namespace functors { namespace binary {
	template<typename LHS, typename RHS>
	class BitwiseOr : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Bitwise |
										 internal::flags::PacketBitwise |
										 internal::flags::ScalarBitwise;

		BitwiseOr() = default;

		BitwiseOr(const BitwiseOr<LHS, RHS> &other) = default;

		BitwiseOr<LHS, RHS> &operator=(const BitwiseOr<LHS, RHS> &other) = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left | right;
		}

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &left, const PacketType &right) const {
			return left | right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "|"; }

	private:
	};

	template<typename LHS, typename RHS>
	class BitwiseAnd : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Bitwise |
										 internal::flags::PacketBitwise |
										 internal::flags::ScalarBitwise;

		BitwiseAnd() = default;

		BitwiseAnd(const BitwiseAnd<LHS, RHS> &other) = default;

		BitwiseAnd<LHS, RHS> &operator=(const BitwiseAnd<LHS, RHS> &other) = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left & right;
		}

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &left, const PacketType &right) const {
			return left & right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "&"; }

	private:
	};

	template<typename LHS, typename RHS>
	class BitwiseXor : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Bitwise |
										 internal::flags::PacketBitwise |
										 internal::flags::ScalarBitwise;

		BitwiseXor() = default;

		BitwiseXor(const BitwiseXor<LHS, RHS> &other) = default;

		BitwiseXor<LHS, RHS> &operator=(const BitwiseXor<LHS, RHS> &other) = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left ^ right;
		}

		template<typename PacketType>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet packetOp(const PacketType &left, const PacketType &right) const {
			return left ^ right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "^"; }

	private:
	};
}}} // namespace librapid::functors::binary