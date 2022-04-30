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
		static constexpr char * Name = "int8_t";
	};

	//------- 8bit Unsigned Integer -------------------------------------------
	template<>
	struct traits<uint8_t> {
		using Scalar						 = uint8_t;
		using StorageType					 = memory::DenseStorage<uint8_t>;
		using Packet						 = vcl::Vec64uc;
		static constexpr int64_t PacketWidth = 64;
		static constexpr char * Name = "uint8_t";
	};

	//------- 16bit Signed Integer --------------------------------------------
	template<>
	struct traits<int16_t> {
		using Scalar						 = int16_t;
		using StorageType					 = memory::DenseStorage<int16_t>;
		using Packet						 = vcl::Vec32s;
		static constexpr int64_t PacketWidth = 32;
		static constexpr char * Name = "int16_t";
	};

	//------- 16bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint16_t> {
		using Scalar						 = uint16_t;
		using StorageType					 = memory::DenseStorage<uint16_t>;
		using Packet						 = vcl::Vec32us;
		static constexpr int64_t PacketWidth = 32;
		static constexpr char * Name = "uint16_t";
	};

	//------- 32bit Signed Integer --------------------------------------------
	template<>
	struct traits<int32_t> {
		using Scalar						 = int32_t;
		using StorageType					 = memory::DenseStorage<int32_t>;
		using Packet						 = vcl::Vec8i;
		static constexpr int64_t PacketWidth = 8;
		static constexpr char * Name = "int32_t";
	};

	//------- 32bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint32_t> {
		using Scalar						 = uint32_t;
		using StorageType					 = memory::DenseStorage<uint32_t>;
		using Packet						 = vcl::Vec8ui;
		static constexpr int64_t PacketWidth = 4;
		static constexpr char * Name = "uint32_t";
	};

	//------- 64bit Signed Integer --------------------------------------------
	template<>
	struct traits<int64_t> {
		using Scalar						 = int64_t;
		using StorageType					 = memory::DenseStorage<int64_t>;
		using Packet						 = vcl::Vec8q;
		static constexpr int64_t PacketWidth = 8;
		static constexpr char * Name = "int64_t";
	};

	//------- 64bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint64_t> {
		using Scalar						 = uint64_t;
		using StorageType					 = memory::DenseStorage<uint64_t>;
		using Packet						 = vcl::Vec8uq;
		static constexpr int64_t PacketWidth = 8;
		static constexpr char * Name = "uint64_t";
	};

	//------- 32bit Floating Point --------------------------------------------
	template<>
	struct traits<float> {
		using Scalar						 = float;
		using StorageType					 = memory::DenseStorage<float>;
		using Packet						 = vcl::Vec16f;
		static constexpr int64_t PacketWidth = 16;
		static constexpr char * Name = "float";
	};

	//------- 64bit Floating Point --------------------------------------------
	template<>
	struct traits<double> {
		using Scalar						 = double;
		using StorageType					 = memory::DenseStorage<double>;
		using Packet						 = vcl::Vec8d;
		static constexpr int64_t PacketWidth = 8;
		static constexpr char * Name = "double";
	};

	template<typename LHS, typename RHS>
	struct PropagateDeviceType {
		using DeviceLHS = typename traits<LHS>::Device;
		using DeviceRHS = typename traits<RHS>::Device;
		using Device = typename memory::PromoteDevice<DeviceLHS, DeviceRHS>::type;
	};

	template<typename LHS, typename RHS>
	struct ReturnType {
		using LhsType = LHS;
		using RhsType = RHS;
		using RetType = typename std::common_type<LhsType, RhsType>::type;
	};

	template<typename T>
	using StripQualifiers = typename std::remove_cv_t<typename std::remove_reference_t<T>>;
} // namespace librapid::internal