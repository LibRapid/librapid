#pragma once

/*
 * Provide traits for specific scalar types used by LibRapid. Correctly overloading a similar
 * traits object for a user-defined datatype will allow efficient interoperability between that
 * type and LibRapid. A default implementation is provided, though may not work for all types.
 */

#include "../internal/config.hpp"
#include "../internal/forward.hpp"
#include "../internal/memUtils.hpp"

#if defined(LIBRAPID_USE_VC)
#	define LR_VC_TYPE(X) Vc::Vector<X>
#	define LR_VC_SIZE(X) Vc::Vector<X>::size()
#else
#	define LR_VC_TYPE(X) std::false_type
#	define LR_VC_SIZE(X) 1
#endif

#define LIMIT_IMPL_CONSTEXPR(NAME_) static constexpr auto NAME_() noexcept
#define LIMIT_IMPL(NAME_)			static auto NAME_() noexcept
#define NUM_LIM(NAME_)				std::numeric_limits<Scalar>::NAME_()

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

		static constexpr uint64_t Size	= sizeof(T);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const CAST &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
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
		static constexpr uint64_t Flags = flags::ScalarBitwise | flags::ScalarArithmetic |
										  flags::ScalarLogical | flags::PacketArithmetic |
										  flags::PacketLogical | flags::PacketBitwise;

		static constexpr uint64_t Size	= sizeof(char);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const char &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- Boolean ---------------------------------------------------------
	template<>
	struct traits<bool> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = bool;
		using BaseScalar					 = uint32_t;
		using StorageType					 = memory::DenseStorage<bool, device::CPU>;
		using Packet						 = LR_VC_TYPE(BaseScalar);
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "bool";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::ScalarArithmetic | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(uint32_t);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const bool &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- 8bit Signed Integer ---------------------------------------------
	template<>
	struct traits<int8_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int8_t;
		using BaseScalar					 = int8_t;
		using StorageType					 = memory::DenseStorage<int8_t, device::CPU>;
		using Packet						 = LR_VC_TYPE(BaseScalar);
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "int8_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(int8_t);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const int8_t &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- 8bit Unsigned Integer -------------------------------------------
	template<>
	struct traits<uint8_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint8_t;
		using BaseScalar					 = uint8_t;
		using StorageType					 = memory::DenseStorage<uint8_t>;
		using Packet						 = LR_VC_TYPE(BaseScalar);
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "uint8_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(uint8_t);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const uint8_t &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- 16bit Signed Integer --------------------------------------------
	template<>
	struct traits<int16_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int16_t;
		using BaseScalar					 = int16_t;
		using StorageType					 = memory::DenseStorage<int16_t>;
		using Packet						 = LR_VC_TYPE(BaseScalar);
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "int16_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(int16_t);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const int16_t &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- 16bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint16_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint16_t;
		using BaseScalar					 = uint16_t;
		using StorageType					 = memory::DenseStorage<uint16_t>;
		using Packet						 = LR_VC_TYPE(BaseScalar);
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "uint16_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(uint16_t);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const uint16_t &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- 32bit Signed Integer --------------------------------------------
	template<>
	struct traits<int32_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = int32_t;
		using BaseScalar					 = int32_t;
		using StorageType					 = memory::DenseStorage<int32_t>;
		using Packet						 = LR_VC_TYPE(BaseScalar);
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "int32_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(int32_t);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const int32_t &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- 32bit Unsigned Integer ------------------------------------------
	template<>
	struct traits<uint32_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = uint32_t;
		using BaseScalar					 = uint32_t;
		using StorageType					 = memory::DenseStorage<uint32_t>;
		using Packet						 = LR_VC_TYPE(BaseScalar);
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "uint32_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(uint32_t);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const uint32_t &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
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
		static constexpr int64_t PacketWidth = 1; // LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "int64_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(int64_t);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const int64_t &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
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
		static constexpr int64_t PacketWidth = 1; // LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "uint64_t";
		static constexpr uint64_t Flags		 = flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(uint64_t);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const uint64_t &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	// 16bit Floating Point implementation is in "librapid/modified/float16/float16.hpp"

	//------- 32bit Floating Point --------------------------------------------
	template<>
	struct traits<float> {
		static constexpr bool IsScalar = true;
		using Valid					   = std::true_type;
		using Scalar				   = float;
		using BaseScalar			   = float;
		using StorageType			   = memory::DenseStorage<float>;
		using Packet				   = LR_VC_TYPE(BaseScalar);
		;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "float";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(float);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const float &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- 64bit Floating Point --------------------------------------------
	template<>
	struct traits<double> {
		static constexpr bool IsScalar = true;
		using Valid					   = std::true_type;
		using Scalar				   = double;
		using BaseScalar			   = double;
		using StorageType			   = memory::DenseStorage<double>;
		using Packet				   = LR_VC_TYPE(BaseScalar);
		;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]		 = "double";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(double);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const double &val) {
			return (CAST)val;
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- Complex Number --------------------------------------------
	template<typename T>
	struct traits<Complex<T>> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = Complex<T>;
		using BaseScalar					 = Complex<T>;
		using StorageType					 = memory::DenseStorage<Complex<T>>;
		using Packet						 = std::false_type;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "NO_MAPPED_TYPE";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(Complex<T>);
		static constexpr bool CanAlign	= traits<T>::CanAlign;
		static constexpr bool CanMemcpy = traits<T>::CanMemcpy;

		template<typename S>
		struct IsComplex : public std::false_type {};

		template<typename S>
		struct IsComplex<Complex<S>> : public std::true_type {};

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const Complex<T> &val) {
			if constexpr (IsComplex<T>::value) {
				return {traits<T>::template cast<CAST>(real(val)),
						traits<T>::template cast<CAST>(imag(val))};
			} else {
				return traits<T>::template cast<CAST>(real(val));
			}
		}

		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(min) { return traits<T>::min(); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(max) { return traits<T>::min(); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(epsilon) { return traits<T>::min(); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(roundError) { return traits<T>::round_error(); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(denormMin) { return traits<T>::denorm_min(); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(infinity) { return traits<T>::infinity(); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(quietNaN) { return traits<T>::quiet_NaN(); }
		LR_FORCE_INLINE LIMIT_IMPL_CONSTEXPR(signalingNaN) { return traits<T>::signaling_NaN(); }
	};

#if defined(LIBRAPID_USE_MULTIPREC)

	//------- Multiprecision Integer (MPZ) ------------------------------------
	template<>
	struct traits<mpz> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = mpz;
		using BaseScalar					 = mpz;
		using StorageType					 = memory::DenseStorage<mpz>;
		using Packet						 = std::false_type;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "NO_VALID_CONVERSION";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketBitwise | flags::ScalarBitwise |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(mpz);
		static constexpr bool CanAlign	= false;
		static constexpr bool CanMemcpy = false;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const mpz &val) {
			if constexpr (std::is_fundamental_v<CAST>) {
				if constexpr (std::is_floating_point_v<CAST>) return (CAST)val.get_d();
				if constexpr (std::is_unsigned_v<CAST>) return (CAST)val.get_ui();
				if constexpr (std::is_signed_v<CAST>) return (CAST)val.get_si();
			}
			if constexpr (std::is_same_v<CAST, mpz>) return toMpz(val);
			if constexpr (std::is_same_v<CAST, mpq>) return toMpq(val);
			if constexpr (std::is_same_v<CAST, mpfr>) return toMpfr(val);
			return CAST(val.get_d());
		}

		LR_FORCE_INLINE LIMIT_IMPL(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- Multiprecision Rational (MPQ) ---------------------------------
	template<>
	struct traits<mpq> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = mpq;
		using BaseScalar					 = mpq;
		using StorageType					 = memory::DenseStorage<mpq>;
		using Packet						 = std::false_type;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "NO_VALID_CONVERSION";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(mpq);
		static constexpr bool CanAlign	= false;
		static constexpr bool CanMemcpy = false;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const mpq &val) {
			if constexpr (std::is_fundamental_v<CAST> && std::is_floating_point_v<CAST>)
				return (CAST)val.get_d();

			if constexpr (std::is_same_v<CAST, mpz>) return toMpz(val);
			if constexpr (std::is_same_v<CAST, mpq>) return toMpq(val);
			if constexpr (std::is_same_v<CAST, mpfr>) return toMpfr(val);
			return CAST(val.get_d());
		}

		LR_FORCE_INLINE LIMIT_IMPL(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

	//------- Multiprecision Rational (MPFR) ---------------------------------
	template<>
	struct traits<mpfr> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = mpfr;
		using BaseScalar					 = mpfr;
		using StorageType					 = memory::DenseStorage<mpfr>;
		using Packet						 = std::false_type;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "NO_VALID_CONVERSION";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= sizeof(mpfr);
		static constexpr bool CanAlign	= false;
		static constexpr bool CanMemcpy = false;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const mpfr &val) {
			if constexpr (std::is_same_v<CAST, bool>) return val != 0;
			if constexpr (std::is_integral_v<CAST> && std::is_signed_v<CAST>)
				return (CAST)val.toLLong();
			if constexpr (std::is_integral_v<CAST> && std::is_unsigned_v<CAST>)
				return (CAST)val.toULLong();
			if constexpr (std::is_same_v<CAST, float>) return (CAST)val.toFloat();
			if constexpr (std::is_same_v<CAST, double>) return (CAST)val.toDouble();
			if constexpr (std::is_same_v<CAST, long double>) return (CAST)val.toLDouble();
			if constexpr (std::is_same_v<CAST, mpz>) return toMpz(val);
			if constexpr (std::is_same_v<CAST, mpq>) return toMpq(val);
			if constexpr (std::is_same_v<CAST, mpfr>) return toMpfr(val);
			return (CAST)val.toDouble();
		}

		LR_FORCE_INLINE LIMIT_IMPL(min) { return NUM_LIM(min); }
		LR_FORCE_INLINE LIMIT_IMPL(max) { return NUM_LIM(max); }
		LR_FORCE_INLINE LIMIT_IMPL(epsilon) { return NUM_LIM(epsilon); }
		LR_FORCE_INLINE LIMIT_IMPL(roundError) { return NUM_LIM(round_error); }
		LR_FORCE_INLINE LIMIT_IMPL(denormMin) { return NUM_LIM(denorm_min); }
		LR_FORCE_INLINE LIMIT_IMPL(infinity) { return NUM_LIM(infinity); }
		LR_FORCE_INLINE LIMIT_IMPL(quietNaN) { return NUM_LIM(quiet_NaN); }
		LR_FORCE_INLINE LIMIT_IMPL(signalingNaN) { return NUM_LIM(signaling_NaN); }
	};

#endif // LIBRAPID_USE_MPIR

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

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE bool isNaN(const T &val) noexcept {
		return std::isnan(val);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE bool isFinite(const T &val) noexcept {
		return std::isfinite(val);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE bool isInf(const T &val) noexcept {
		return std::isinf(val);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE T copySign(const T &mag, const T &sign) noexcept {
		return std::copysign(mag, sign);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE bool signBit(const T &val) noexcept {
		return signBit((double)val);
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE bool signBit(const long double &val) noexcept {
		return std::signbit(val);
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE bool signBit(const double &val) noexcept {
		return std::signbit(val);
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE bool signBit(const float &val) noexcept {
		return std::signbit(val);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE T ldexp(const T &x, const int64_t exp) noexcept {
		return std::ldexp(x, (int)exp);
	}

#if defined(LIBRAPID_USE_MULTIPREC)
	// MPIR does not support NaN, so chances are it'll have errored already...
	template<typename A, typename B>
	LR_NODISCARD("")
	LR_INLINE bool isNaN(const __gmp_expr<A, B> &val) noexcept {
		return false;
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE bool isNaN(const mpfr &val) noexcept {
		return ::mpfr::isnan(val);
	}

	// MPIR does not support Inf, so we can probably just return true
	template<typename A, typename B>
	LR_NODISCARD("")
	LR_INLINE bool isFinite(const __gmp_expr<A, B> &val) noexcept {
		return true;
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE bool isFinite(const mpfr &val) noexcept {
		return ::mpfr::isfinite(val);
	}

	// MPIR does not support Inf, so chances are it'll have errored already...
	template<typename A, typename B>
	LR_NODISCARD("")
	LR_INLINE bool isInf(const __gmp_expr<A, B> &val) noexcept {
		return false;
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE bool isInf(const mpfr &val) noexcept {
		return ::mpfr::isinf(val);
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE mpfr copySign(const mpfr &mag, const mpfr &sign) noexcept {
		return ::mpfr::copysign(mag, sign);
	}

	template<typename A, typename B>
	LR_NODISCARD("")
	LR_INLINE __gmp_expr<A, B> copySign(const __gmp_expr<A, B> &mag,
										const __gmp_expr<A, B> &sign) noexcept {
		if (sign >= 0 && mag >= 0) return mag;
		if (sign >= 0 && mag < 0) return -mag;
		if (sign < 0 && mag >= 0) return -mag;
		if (sign < 0 && mag < 0) return mag;
		return 0; // Should never get here
	}

	template<typename A, typename B>
	LR_NODISCARD("")
	LR_INLINE bool signBit(const __gmp_expr<A, B> &val) noexcept {
		return val < 0 || val == -0.0; // I have no idea if this works
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE bool signBit(const mpfr &val) noexcept {
		return ::mpfr::signbit(val);
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE mpfr ldexp(const mpfr &x, const int64_t exp) noexcept {
		return ::mpfr::ldexp(x, exp);
	}

	template<typename A, typename B>
	LR_NODISCARD("")
	LR_INLINE __gmp_expr<A, B> ldexp(const __gmp_expr<A, B> &x, const int64_t exp) noexcept {
		return x << exp;
	}
#endif // LIBRAPID_USE_MULTIPREC
} // namespace librapid::internal

#undef LR_VC_TYPE
#undef LR_VC_SIZE
#undef LIMIT_IMPL_CONSTEXPR
#undef LIMIT_IMPL
#undef NUM_LIM