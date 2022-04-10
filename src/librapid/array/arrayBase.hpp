#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "helpers/extent.hpp"
#include "packet.hpp"

namespace librapid {
	namespace packet {
		template<typename Derived>
		struct traits<ArrayBase<Derived, device::CPU>> {
			using Scalar				= typename traits<Derived>::Scalar;
			using StorageType			= memory::DenseStorage<Scalar, device::CPU>;
			static const uint64_t Flags = 0;
		};
	} // namespace packet

	template<typename Derived, typename device>
	class ArrayBase {
		using Scalar				= typename packet::traits<Derived>::Scalar;
		using This					= ArrayBase<Derived, device>;
		using Packet				= typename packet::traits<Derived>::Packet;
		using StorageType			= typename packet::traits<This>::StorageType;
		static const uint64_t Flags = packet::traits<This>::Flags;

	public:
		ArrayBase() = default;

		template<typename T_, int64_t d_>
		explicit ArrayBase(const Extent<T_, d_> &extent) :
				m_extent(extent), m_storage(extent.size()) {}

	private:
		Extent<int64_t, 32> m_extent;
		StorageType m_storage;
	};
} // namespace librapid