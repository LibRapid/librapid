#pragma once

#include "binary.hpp"

namespace librapid::functors::binary {
	template<typename LHS, typename RHS>
	class ScalarSum : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		ScalarSum() = default;

		ScalarSum(const ScalarSum<LHS, RHS> &other) = default;

		ScalarSum<LHS, RHS> &operator=(const ScalarSum<LHS, RHS> &other) = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left + right;
		}

		template<typename PacketTypeLHS, typename PacketTypeRHS>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet
		  packetOp(const PacketTypeLHS &left, const PacketTypeRHS &right) const {
			return left + right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "+"; }

	private:
	};

	template<typename LHS, typename RHS>
	class ScalarDiff : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		ScalarDiff() = default;

		ScalarDiff(const ScalarDiff<LHS, RHS> &other) = default;

		ScalarDiff<LHS, RHS> &operator=(const ScalarDiff<LHS, RHS> &other) = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left - right;
		}

		template<typename PacketTypeLHS, typename PacketTypeRHS>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet
		  packetOp(const PacketTypeLHS &left, const PacketTypeRHS &right) const {
			return left - right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "-"; }

	private:
	};

	template<typename LHS, typename RHS>
	class ScalarProd : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		ScalarProd() = default;

		ScalarProd(const ScalarProd<LHS, RHS> &other) = default;

		ScalarProd<LHS, RHS> &operator=(const ScalarProd<LHS, RHS> &other) = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left * right;
		}

		template<typename PacketTypeLHS, typename PacketTypeRHS>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet
		  packetOp(const PacketTypeLHS &left, const PacketTypeRHS &right) const {
			return left * right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "*"; }

	private:
	};

	template<typename LHS, typename RHS>
	class ScalarDiv : public ScalarOp<LHS, RHS> {
	public:
		using LhsType				   = typename internal::traits<LHS>::Scalar;
		using RhsType				   = typename internal::traits<RHS>::Scalar;
		using RetType				   = typename std::common_type<LhsType, RhsType>::type;
		using Packet				   = typename internal::traits<RetType>::Packet;
		static constexpr int64_t Flags = internal::flags::Binary | internal::flags::Arithmetic |
										 internal::flags::PacketArithmetic |
										 internal::flags::ScalarArithmetic;

		ScalarDiv() = default;

		ScalarDiv(const ScalarDiv<LHS, RHS> &other) = default;

		ScalarDiv<LHS, RHS> &operator=(const ScalarDiv<LHS, RHS> &other) = default;

		LR_NODISCARD("")
		LR_FORCE_INLINE RetType scalarOp(const LhsType &left, const RhsType &right) const {
			return left / right;
		}

		template<typename PacketTypeLHS, typename PacketTypeRHS>
		LR_NODISCARD("")
		LR_FORCE_INLINE Packet
		  packetOp(const PacketTypeLHS &left, const PacketTypeRHS &right) const {
			return left / right;
		}

		LR_NODISCARD("") LR_FORCE_INLINE std::string genKernel() const { return "/"; }

	private:
	};
} // namespace librapid::functors::binary