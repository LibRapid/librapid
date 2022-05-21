#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"

namespace librapid::internal {
	namespace flags {
		/**
		 * Flag Configuration:
		 *
		 * [0, 9]     -> Requirement flags
		 * [10, 31]   -> Operation type flags
		 * [32]       -> Unary operation
		 * [33]       -> Binary operation
		 * [34]       -> Matrix operation
		 */
		inline constexpr uint64_t RequireEval = 1 << 0;

		inline constexpr uint64_t Bitwise	 = 1 < 10;
		inline constexpr uint64_t Arithmetic = 1 < 11;
		inline constexpr uint64_t Logical	 = 1 < 12;

		inline constexpr uint64_t OperationMask = 0b111111111111111111110000000000;

		inline constexpr uint64_t PacketBitwise	   = 1 << 13;
		inline constexpr uint64_t PacketArithmetic = 1 << 14;
		inline constexpr uint64_t PacketLogical	   = 1 << 15;

		inline constexpr uint64_t ScalarBitwise	   = 1 << 16;
		inline constexpr uint64_t ScalarArithmetic = 1 << 17;
		inline constexpr uint64_t ScalarLogical	   = 1 << 18;

		inline constexpr uint64_t Unary	 = 1 < 32;
		inline constexpr uint64_t Binary = 1 < 33;
		inline constexpr uint64_t Matrix = 1 < 34;
	} // namespace flags

	//------- Just a  Character -----------------------------------------------
	template<>
	struct traits<char> {
		using Valid							 = std::true_type;
		using Scalar						 = char;
		using BaseScalar					 = char;
		using StorageType					 = memory::DenseStorage<char, device::CPU>;
		using Packet						 = std::false_type;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "char";
		static constexpr uint64_t Flags =
		  flags::ScalarBitwise | flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- Boolean ---------------------------------------------------------
	template<>
	struct traits<bool> {
		using Valid							 = std::true_type;
		using Scalar						 = bool;
		using BaseScalar					 = uint64_t;
		using StorageType					 = memory::DenseStorage<bool, device::CPU>;
		using Packet						 = vcl::Vec512b;
		static constexpr int64_t PacketWidth = 512;
		static constexpr char Name[]		 = "bool";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 8bit Signed Integer ---------------------------------------------
	template<>
	struct traits<int8_t> {
		using Valid							 = std::true_type;
		using Scalar						 = int8_t;
		using BaseScalar					 = int8_t;
		using StorageType					 = memory::DenseStorage<int8_t, device::CPU>;
		using Packet						 = vcl::Vec64c;
		static constexpr int64_t PacketWidth = 64;
		static constexpr char Name[]		 = "int8_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::PacketArithmetic |
										  flags::PacketLogical | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 8bit Unsigned Integer -------------------------------------------
	template<>
	struct traits<uint8_t> {
		using Valid							 = std::true_type;
		using Scalar						 = uint8_t;
		using BaseScalar					 = uint8_t;
		using StorageType					 = memory::DenseStorage<uint8_t>;
		using Packet						 = vcl::Vec64uc;
		static constexpr int64_t PacketWidth = 64;
		static constexpr char Name[]		 = "uint8_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::PacketArithmetic |
										  flags::PacketLogical | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 16bit Signed Integer --------------------------------------------
	template<>
	struct traits<int16_t> {
		using Valid							 = std::true_type;
		using Scalar						 = int16_t;
		using BaseScalar					 = int16_t;
		using StorageType					 = memory::DenseStorage<int16_t>;
		using Packet						 = vcl::Vec32s;
		static constexpr int64_t PacketWidth = 32;
		static constexpr char Name[]		 = "int16_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::PacketArithmetic |
										  flags::PacketLogical | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 16bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint16_t> {
		using Valid							 = std::true_type;
		using Scalar						 = uint16_t;
		using BaseScalar					 = uint16_t;
		using StorageType					 = memory::DenseStorage<uint16_t>;
		using Packet						 = vcl::Vec32us;
		static constexpr int64_t PacketWidth = 32;
		static constexpr char Name[]		 = "uint16_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::PacketArithmetic |
										  flags::PacketLogical | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 32bit Signed Integer --------------------------------------------
	template<>
	struct traits<int32_t> {
		using Valid							 = std::true_type;
		using Scalar						 = int32_t;
		using BaseScalar					 = int32_t;
		using StorageType					 = memory::DenseStorage<int32_t>;
		using Packet						 = vcl::Vec8i;
		static constexpr int64_t PacketWidth = 8;
		static constexpr char Name[]		 = "int32_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::PacketArithmetic |
										  flags::PacketLogical | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 32bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint32_t> {
		using Valid							 = std::true_type;
		using Scalar						 = uint32_t;
		using BaseScalar					 = uint32_t;
		using StorageType					 = memory::DenseStorage<uint32_t>;
		using Packet						 = vcl::Vec8ui;
		static constexpr int64_t PacketWidth = 4;
		static constexpr char Name[]		 = "uint32_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::PacketArithmetic |
										  flags::PacketLogical | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 64bit Signed Integer --------------------------------------------
	template<>
	struct traits<int64_t> {
		using Valid							 = std::true_type;
		using Scalar						 = int64_t;
		using BaseScalar					 = int64_t;
		using StorageType					 = memory::DenseStorage<int64_t>;
		using Packet						 = vcl::Vec8q;
		static constexpr int64_t PacketWidth = 8;
		static constexpr char Name[]		 = "int64_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::PacketArithmetic |
										  flags::PacketLogical | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 64bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint64_t> {
		using Valid							 = std::true_type;
		using Scalar						 = uint64_t;
		using BaseScalar					 = uint64_t;
		using StorageType					 = memory::DenseStorage<uint64_t>;
		using Packet						 = vcl::Vec8uq;
		static constexpr int64_t PacketWidth = 8;
		static constexpr char Name[]		 = "uint64_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::PacketArithmetic |
										  flags::PacketLogical | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 32bit Floating Point --------------------------------------------
	template<>
	struct traits<float> {
		using Valid							 = std::true_type;
		using Scalar						 = float;
		using BaseScalar					 = float;
		using StorageType					 = memory::DenseStorage<float>;
		using Packet						 = vcl::Vec16f;
		static constexpr int64_t PacketWidth = 16;
		static constexpr char Name[]		 = "float";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::PacketLogical |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 64bit Floating Point --------------------------------------------
	template<>
	struct traits<double> {
		using Valid							 = std::true_type;
		using Scalar						 = double;
		using BaseScalar					 = double;
		using StorageType					 = memory::DenseStorage<double>;
		using Packet						 = vcl::Vec8d;
		static constexpr int64_t PacketWidth = 8;
		static constexpr char Name[]		 = "double";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::PacketLogical |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	template<typename LHS, typename RHS>
	struct PropagateDeviceType {
		using DeviceLHS = typename traits<LHS>::Device;
		using DeviceRHS = typename traits<RHS>::Device;
		using Device	= typename memory::PromoteDevice<DeviceLHS, DeviceRHS>::type;
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