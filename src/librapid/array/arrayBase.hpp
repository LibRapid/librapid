#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "helpers/extent.hpp"
#include "traits.hpp"

namespace librapid {
	namespace internal {
		template<typename Derived>
		struct traits<ArrayBase<Derived, device::CPU>> {
			using Scalar				= typename traits<Derived>::Scalar;
			using Device				= device::CPU;
			using StorageType			= memory::DenseStorage<Scalar, device::CPU>;
			static const uint64_t Flags = 0;
		};

		template<typename Derived>
		struct traits<ArrayBase<Derived, device::GPU>> {
			using Scalar				= typename traits<Derived>::Scalar;
			using Device				= device::GPU;
			using StorageType			= memory::DenseStorage<Scalar, device::CPU>;
			static const uint64_t Flags = 0;
		};
	} // namespace internal

	template<typename Derived, typename Device>
	class ArrayBase {
	public:
		using Scalar				= typename internal::traits<Derived>::Scalar;
		using This					= ArrayBase<Derived, Device>;
		using Packet				= typename internal::traits<Derived>::Packet;
		using StorageType			= typename internal::traits<This>::StorageType;
		static const uint64_t Flags = internal::traits<This>::Flags;

		ArrayBase() = default;

		template<typename T_, int64_t d_>
		explicit ArrayBase(const Extent<T_, d_> &extent) :
				m_extent(extent), m_storage(extent.size()) {}

		template<typename OtherDerived, typename OtherDevice>
		LR_NODISCARD("Do not ignore the result of an arithmetic operation")
		auto operator+(const ArrayBase<OtherDerived, OtherDevice> &) const {
			using ScalarOther = typename internal::traits<OtherDerived>::Scalar;
			using ResDevice	  = typename memory::PromoteDevice<Device, OtherDevice>::type;
			using RetType	  = binop::
			  CWiseBinop<functors::binary::ScalarSum<Scalar, OtherScalar>, Derived, OtherDerived>;
			return RetType(derived(), other.derived());
		}

		LR_NODISCARD("") LR_FORCE_INLINE Derived derived() const {
			return *static_cast<const Derived *>(this);
		}

		LR_NODISCARD("") LR_FORCE_INLINE Derived derived() { return *static_cast<Derived *>(this); }

	private:
		Extent<int64_t, 32> m_extent;
		StorageType m_storage;
	};
} // namespace librapid