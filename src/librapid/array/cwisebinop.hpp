#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "arrayBase.hpp"

namespace librapid {
	namespace internal {
		template<typename Binop, typename LHS, typename RHS>
		struct traits<binop::CWiseBinop<Binop, LHS, RHS>> {
			using Scalar	  = typename Binop::RetType;
			using Packet	  = typename traits<Scalar>::Packet;
			using DeviceLHS	  = typename traits<LHS>::Device;
			using DeviceRHS	  = typename traits<RHS>::Device;
			using Device	  = typename memory::PromoteDevice<DeviceLHS, DeviceRHS>::type;
			using StorageType = memory::DenseStorage<Scalar, Device>;
			static const uint64_t Flags = 0;
		};
	} // namespace internal

	namespace binop {
		template<typename Binop, typename Derived, typename OtherDerived>
		class CWiseBinop
				: public ArrayBase<
					CWiseBinop<Binop, Derived, OtherDerived>,
					typename internal::PropagateDeviceType<Derived, OtherDerived>::Device> {
		public:
			using Operation = Binop;
			using Scalar	= typename Binop::RetType;
			using DeviceLHS = typename internal::traits<Derived>::Device;
			using DeviceRHS = typename internal::traits<OtherDerived>::Device;
			using Device	= typename memory::PromoteDevice<DeviceRHS, DeviceLHS>::type;
			using Base		= ArrayBase<CWiseBinop<Binop, Derived, OtherDerived>, Device>;

			CWiseBinop() = default;

		private:
		};
	} // namespace binop
} // namespace librapid
