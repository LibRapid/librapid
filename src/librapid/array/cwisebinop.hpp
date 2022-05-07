#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "arrayBase.hpp"

namespace librapid {
	namespace internal {
		template<typename Binop, typename LHS, typename RHS>
		struct traits<binop::CWiseBinop<Binop, LHS, RHS>> {
			using Type		  = binop::CWiseBinop<Binop, LHS, RHS>;
			using Scalar	  = typename Binop::RetType;
			using Packet	  = typename traits<Scalar>::Packet;
			using DeviceLHS	  = typename traits<LHS>::Device;
			using DeviceRHS	  = typename traits<RHS>::Device;
			using Device	  = typename memory::PromoteDevice<DeviceLHS, DeviceRHS>::type;
			using StorageType = memory::DenseStorage<Scalar, Device>;
			static constexpr uint64_t Flags		 = Binop::Flags;
		};
	} // namespace internal

	namespace binop {
		template<typename Binop, typename LHS, typename RHS>
		class CWiseBinop
				: public ArrayBase<CWiseBinop<Binop, LHS, RHS>,
								   typename internal::PropagateDeviceType<LHS, RHS>::Device> {
		public:
			using Operation = Binop;
			using Scalar	= typename Binop::RetType;
			using Packet	= typename internal::traits<Scalar>::Packet;
			using LeftType	= typename internal::StripQualifiers<LHS>;
			using RightType = typename internal::StripQualifiers<RHS>;
			using DeviceLHS = typename internal::traits<LHS>::Device;
			using DeviceRHS = typename internal::traits<RHS>::Device;
			using Device	= typename memory::PromoteDevice<DeviceRHS, DeviceLHS>::type;
			using Type		= CWiseBinop<Binop, LHS, RHS>;
			using Base		= ArrayBase<Type, Device>;
			static constexpr uint64_t Flags = internal::traits<Type>::Flags;

			CWiseBinop() = delete;

			CWiseBinop(const LeftType &lhs, const RightType &rhs) :
					Base(lhs.extent(), 0), m_lhs(lhs), m_rhs(rhs) {}

			CWiseBinop(const Type &op) :
					Base(op.extent(), 0), m_lhs(op.m_lhs), m_rhs(op.m_rhs),
					m_operation(op.m_operation) {}

			CWiseBinop &operator=(const Type &op) {
				if (this == &op) return *this;

				Base::m_extent = op.m_extent;
				Base::m_data   = op.m_data;

				m_lhs		= op.m_lhs;
				m_rhs		= op.m_rhs;
				m_operation = op.m_operation;

				return *this;
			}

			LR_NODISCARD("Do not ignore the result of an evaluated calculation")
			Array<Scalar, Device> eval() const {
				Array<Scalar, Device> res(Base::extent());
				res.assign(*this);
				return res;
			}

			LR_FORCE_INLINE Packet packet(int64_t index) const {
				return m_operation.packetOp(m_lhs.packet(index), m_rhs.packet(index));
			}

			LR_FORCE_INLINE Scalar scalar(int64_t index) const {
				return m_operation.scalarOp(m_lhs.scalar(index), m_rhs.scalar(index));
			}

			template<typename T>
			std::string genKernel(std::vector<T> &vec, int64_t &index) const {
				std::string leftKernel	= m_lhs.genKernel(vec, index);
				std::string rightKernel = m_rhs.genKernel(vec, index);
				std::string op			= m_operation.genKernel();
				return fmt::format("({} {} {})", leftKernel, op, rightKernel);
			}

		private:
			// const LeftType &m_lhs;
			// const RightType &m_rhs;
			LeftType m_lhs;
			RightType m_rhs;
			const Binop m_operation {};
		};
	} // namespace binop
} // namespace librapid
