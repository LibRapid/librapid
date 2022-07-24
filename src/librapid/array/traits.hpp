#pragma once

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

namespace librapid::extended {
	struct float16_t;
}

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
	};

	//------- 16bit Floating Point --------------------------------------------
	template<>
	struct traits<extended::float16_t> {
		static constexpr bool IsScalar		 = true;
		using Valid							 = std::true_type;
		using Scalar						 = extended::float16_t;
		using BaseScalar					 = extended::float16_t;
		using StorageType					 = memory::DenseStorage<extended::float16_t>;
		using Packet						 = std::false_type;
		using Device						 = device::CPU;
		static constexpr int64_t PacketWidth = 1;
		static constexpr char Name[]		 = "__half";
		static constexpr uint64_t Flags		 = flags::PacketArithmetic | flags::ScalarArithmetic |
										  flags::PacketLogical | flags::ScalarLogical;

		static constexpr uint64_t Size	= 2;
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const extended::float16_t &val) {
			return (CAST)val;
		}
	};

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
			return (CAST) val.toDouble();
		}
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
} // namespace librapid::internal