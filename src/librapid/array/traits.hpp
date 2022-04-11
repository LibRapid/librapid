#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"

namespace librapid::internal {
	//------- 8bit Signed Integer ---------------------------------------------
	template<>
	struct traits<int8_t> {
		using Scalar						 = int8_t;
		using StorageType					 = memory::DenseStorage<int8_t, device::CPU>;
		using Packet						 = vcl::Vec64c;
		static constexpr int64_t PacketWidth = 64;
	};

	//------- 8bit Unsigned Integer -------------------------------------------
	template<>
	struct traits<uint8_t> {
		using Scalar						 = uint8_t;
		using StorageType					 = memory::DenseStorage<uint8_t>;
		using Packet						 = vcl::Vec64uc;
		static constexpr int64_t PacketWidth = 64;
	};

	//------- 16bit Signed Integer --------------------------------------------
	template<>
	struct traits<int16_t> {
		using Scalar						 = int16_t;
		using StorageType					 = memory::DenseStorage<int16_t>;
		using Packet						 = vcl::Vec32s;
		static constexpr int64_t PacketWidth = 32;
	};

	//------- 16bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint16_t> {
		using Scalar						 = uint16_t;
		using StorageType					 = memory::DenseStorage<uint16_t>;
		using Packet						 = vcl::Vec32us;
		static constexpr int64_t PacketWidth = 32;
	};

	//------- 32bit Signed Integer --------------------------------------------
	template<>
	struct traits<int32_t> {
		using Scalar						 = int32_t;
		using StorageType					 = memory::DenseStorage<int32_t>;
		using Packet						 = vcl::Vec16s;
		static constexpr int64_t PacketWidth = 16;
	};

	//------- 32bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint32_t> {
		using Scalar						 = uint32_t;
		using StorageType					 = memory::DenseStorage<uint32_t>;
		using Packet						 = vcl::Vec16us;
		static constexpr int64_t PacketWidth = 16;
	};

	//------- 64bit Signed Integer --------------------------------------------
	template<>
	struct traits<int64_t> {
		using Scalar						 = int64_t;
		using StorageType					 = memory::DenseStorage<int64_t>;
		using Packet						 = vcl::Vec8q;
		static constexpr int64_t PacketWidth = 8;
	};

	//------- 64bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint64_t> {
		using Scalar						 = uint64_t;
		using StorageType					 = memory::DenseStorage<uint64_t>;
		using Packet						 = vcl::Vec8uq;
		static constexpr int64_t PacketWidth = 8;
	};

	//------- 32bit Floating Point --------------------------------------------
	template<>
	struct traits<float> {
		using Scalar						 = float;
		using StorageType					 = memory::DenseStorage<float>;
		using Packet						 = vcl::Vec16f;
		static constexpr int64_t PacketWidth = 16;
	};

	//------- 64bit Floating Point --------------------------------------------
	template<>
	struct traits<double> {
		using Scalar						 = double;
		using StorageType					 = memory::DenseStorage<double>;
		using Packet						 = vcl::Vec8d;
		static constexpr int64_t PacketWidth = 8;
	};

	template<typename LHS, typename RHS>
	struct PropagateDeviceType {
		using DeviceLHS = typename traits<LHS>::Device;
		using DeviceRHS = typename traits<RHS>::Device;
		using Device = typename memory::PromoteDevice<DeviceLHS, DeviceRHS>::type;
	};
} // namespace librapid::packet