#pragma once

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "../internal/memUtils.hpp"

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
		 * [35]       -> Packet operation is illegal
		 */

		constexpr uint64_t Evaluated	 = 1ll << 0; // Result is already evaluated
		constexpr uint64_t RequireEval	 = 1ll << 1; // Result must be evaluated
		constexpr uint64_t RequireInput	 = 1ll << 2; // Requires the entire array (not scalar)
		constexpr uint64_t HasCustomEval = 1ll << 3; // Has a custom eval function

		constexpr uint64_t Bitwise	  = 1ll << 10; // Bitwise functions
		constexpr uint64_t Arithmetic = 1ll << 11; // Arithmetic functions
		constexpr uint64_t Logical	  = 1ll << 12; // Logical functions
		constexpr uint64_t Matrix	  = 1ll << 13; // Matrix operation

		// Extract only operation information
		constexpr uint64_t OperationMask = 0b1111111111111111100000000000000;

		constexpr uint64_t PacketBitwise	= 1ll << 14; // Packet needs bitwise
		constexpr uint64_t PacketArithmetic = 1ll << 15; // Packet needs arithmetic
		constexpr uint64_t PacketLogical	= 1ll << 16; // Packet needs logical

		constexpr uint64_t ScalarBitwise	= 1ll << 17; // Scalar needs bitwise
		constexpr uint64_t ScalarArithmetic = 1ll << 18; // Scalar needs arithmetic
		constexpr uint64_t ScalarLogical	= 1ll << 19; // Scalar needs logical

		constexpr uint64_t Unary  = 1ll << 32; // Operation takes one argument
		constexpr uint64_t Binary = 1ll << 33; // Operation takes two arguments

		constexpr uint64_t NoPacketOp = 1ll << 34; // Supports packet operations

#if defined(LIBRAPID_PYTHON)
		constexpr uint64_t PythonFlags = RequireEval;
#else
		constexpr uint64_t PythonFlags = 0;
#endif
	} // namespace flags

	template<typename T>
	struct traits {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = T;
		using BaseScalar					 = T;
		using StorageType					 = memory::DenseStorage<T, device::CPU>;
		using Packet						 = std::false_type;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "[NO DEFINED TYPE]";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- Just a  Character -----------------------------------------------
	template<>
	struct traits<char> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = char;
		using BaseScalar					 = char;
		using StorageType					 = memory::DenseStorage<char, device::CPU>;
		using Packet						 = std::false_type;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "char";
		// Packet ops here are a hack -- Packet = std::false_type means the packet ops will never
		// be called
		static constexpr uint64_t Flags		 = flags::ScalarBitwise | flags::ScalarArithmetic |
										  flags::ScalarLogical | flags::PacketArithmetic |
										  flags::PacketLogical | flags::PacketBitwise;
	};

	//------- Boolean ---------------------------------------------------------
	template<>
	struct traits<bool> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = bool;
		using BaseScalar					 = uint32_t;
		using StorageType					 = memory::DenseStorage<bool, device::CPU>;
		using Packet						 = Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "bool";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;
	};

	//------- 8bit Signed Integer ---------------------------------------------
	template<>
	struct traits<int8_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int8_t;
		using BaseScalar					 = int8_t;
		using StorageType					 = memory::DenseStorage<int8_t, device::CPU>;
		using Packet						 = Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "int8_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 8bit Unsigned Integer -------------------------------------------
	template<>
	struct traits<uint8_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint8_t;
		using BaseScalar					 = uint8_t;
		using StorageType					 = memory::DenseStorage<uint8_t>;
		using Packet						 = Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "uint8_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 16bit Signed Integer --------------------------------------------
	template<>
	struct traits<int16_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int16_t;
		using BaseScalar					 = int16_t;
		using StorageType					 = memory::DenseStorage<int16_t>;
		using Packet						 = Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "int16_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 16bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint16_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint16_t;
		using BaseScalar					 = uint16_t;
		using StorageType					 = memory::DenseStorage<uint16_t>;
		using Packet						 = Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "uint16_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 32bit Signed Integer --------------------------------------------
	template<>
	struct traits<int32_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int32_t;
		using BaseScalar					 = int32_t;
		using StorageType					 = memory::DenseStorage<int32_t>;
		using Packet						 = Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "int32_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 32bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint32_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint32_t;
		using BaseScalar					 = uint32_t;
		using StorageType					 = memory::DenseStorage<uint32_t>;
		using Packet						 = Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "uint32_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 64bit Signed Integer --------------------------------------------
	template<>
	struct traits<int64_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int64_t;
		using BaseScalar					 = int64_t;
		using StorageType					 = memory::DenseStorage<int64_t>;
		using Packet						 = std::false_type; // Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1; // Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "int64_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 64bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint64_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint64_t;
		using BaseScalar					 = uint64_t;
		using StorageType					 = memory::DenseStorage<uint64_t>;
		using Packet						 = std::false_type; // Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1; // Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "uint64_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 32bit Floating Point --------------------------------------------
	template<>
	struct traits<float> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = float;
		using BaseScalar					 = float;
		using StorageType					 = memory::DenseStorage<float>;
		using Packet						 = Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "float";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
	};

	//------- 64bit Floating Point --------------------------------------------
	template<>
	struct traits<double> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = double;
		using BaseScalar					 = double;
		using StorageType					 = memory::DenseStorage<double>;
		using Packet						 = Vc::Vector<BaseScalar>;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = Vc::Vector<BaseScalar>::size();
		static constexpr char Name[]		 = "double";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;
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