#pragma once

/*
 * Provide traits for specific scalar types used by LibRapid. Correctly overloading a similar
 * traits object for a user-defined datatype will allow efficient interoperability between that
 * type and LibRapid. A default implementation is provided, though may not work for all types.
 */

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

		constexpr ui64 Evaluated	 = 1ll << 0; // Result is already evaluated
		constexpr ui64 RequireEval	 = 1ll << 1; // Result must be evaluated
		constexpr ui64 RequireInput	 = 1ll << 2; // Requires the entire array (not scalar)
		constexpr ui64 HasCustomEval = 1ll << 3; // Has a custom eval function

		constexpr ui64 Bitwise	  = 1ll << 10; // Bitwise functions
		constexpr ui64 Arithmetic = 1ll << 11; // Arithmetic functions
		constexpr ui64 Logical	  = 1ll << 12; // Logical functions
		constexpr ui64 Matrix	  = 1ll << 13; // Matrix operation

		constexpr ui64 Unary  = 1ll << 14; // Operation takes one argument
		constexpr ui64 Binary = 1ll << 15; // Operation takes two arguments

		// Extract only operation information
		constexpr ui64 OperationMask = 0b1111111111111110000000000000000;

		constexpr ui64 PacketBitwise	= 1ll << 16; // Packet needs bitwise
		constexpr ui64 PacketArithmetic = 1ll << 17; // Packet needs arithmetic
		constexpr ui64 PacketLogical	= 1ll << 18; // Packet needs logical

		constexpr ui64 ScalarBitwise	= 1ll << 19; // Scalar needs bitwise
		constexpr ui64 ScalarArithmetic = 1ll << 20; // Scalar needs arithmetic
		constexpr ui64 ScalarLogical	= 1ll << 21; // Scalar needs logical

		constexpr ui64 NoPacketOp		 = 1ll << 34; // Supports packet operations
		constexpr ui64 CustomFunctionGen = 1ll << 35; // Needs a custom function to be generated
		constexpr ui64 MatrixTranspose	 = 1ll << 36; // Some functions need this information

#if defined(LIBRAPID_PYTHON)
		constexpr ui64 PythonFlags = RequireEval;
#else
		constexpr ui64 PythonFlags = 0;
#endif
	} // namespace flags

	template<typename T>
	struct traits {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = T;
		using BaseScalar				  = T;
		using StorageType				  = memory::DenseStorage<T, device::CPU>;
		using Packet					  = std::false_type;
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = 1;
		static constexpr char Name[]	  = "[NO DEFINED TYPE]";
		static constexpr ui64 Flags		  = flags::PacketBitwise | flags::ScalarBitwise |
									  flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#endif

		static constexpr ui64 Size		= sizeof(T);
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
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = char;
		using BaseScalar				  = char;
		using StorageType				  = memory::DenseStorage<char, device::CPU>;
		using Packet					  = std::false_type;
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = 1;
		static constexpr char Name[]	  = "char";
		// Packet ops here are a hack -- Packet = std::false_type means the packet ops will never
		// be called
		static constexpr ui64 Flags = flags::ScalarBitwise | flags::ScalarArithmetic |
									  flags::ScalarLogical | flags::PacketArithmetic |
									  flags::PacketLogical | flags::PacketBitwise;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_8I;
#endif

		static constexpr ui64 Size		= sizeof(char);
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
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = bool;
		using BaseScalar				  = bool;
		using StorageType				  = memory::DenseStorage<bool, device::CPU>;
		using Packet					  = std::false_type; // LR_VC_TYPE(BaseScalar);
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "bool";
		static constexpr ui64 Flags		  = flags::PacketBitwise | flags::ScalarBitwise |
									  flags::ScalarArithmetic | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#endif

		static constexpr ui64 Size		= sizeof(ui32);
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
	struct traits<i8> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = i8;
		using BaseScalar				  = i8;
		using StorageType				  = memory::DenseStorage<i8, device::CPU>;
		using Packet					  = LR_VC_TYPE(BaseScalar);
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "int8_t";
		static constexpr ui64 Flags		  = flags::PacketBitwise | flags::ScalarBitwise |
									  flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_8I;
#endif

		static constexpr ui64 Size		= sizeof(i8);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const i8 &val) {
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
	struct traits<ui8> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = ui8;
		using BaseScalar				  = ui8;
		using StorageType				  = memory::DenseStorage<ui8, device::CPU>;
		using Packet					  = LR_VC_TYPE(BaseScalar);
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "uint8_t";
		static constexpr ui64 Flags		  = flags::PacketBitwise | flags::ScalarBitwise |
									  flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_8U;
#endif

		static constexpr ui64 Size		= sizeof(ui8);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const ui8 &val) {
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
	struct traits<i16> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = i16;
		using BaseScalar				  = i16;
		using StorageType				  = memory::DenseStorage<i16, device::CPU>;
		using Packet					  = LR_VC_TYPE(BaseScalar);
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "int16_t";
		static constexpr ui64 Flags		  = flags::PacketBitwise | flags::ScalarBitwise |
									  flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_16I;
#endif

		static constexpr ui64 Size		= sizeof(i16);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const i16 &val) {
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
	struct traits<ui16> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = ui16;
		using BaseScalar				  = ui16;
		using StorageType				  = memory::DenseStorage<ui16, device::CPU>;
		using Packet					  = LR_VC_TYPE(BaseScalar);
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "uint16_t";
		static constexpr ui64 Flags		  = flags::PacketBitwise | flags::ScalarBitwise |
									  flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_16U;
#endif

		static constexpr ui64 Size		= sizeof(ui16);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const ui16 &val) {
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
	struct traits<i32> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = i32;
		using BaseScalar				  = i32;
		using StorageType				  = memory::DenseStorage<i32, device::CPU>;
		using Packet					  = LR_VC_TYPE(BaseScalar);
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "int32_t";
		static constexpr ui64 Flags		  = flags::PacketBitwise | flags::ScalarBitwise |
									  flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_32I;
#endif

		static constexpr ui64 Size		= sizeof(i32);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const i32 &val) {
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
	struct traits<ui32> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = ui32;
		using BaseScalar				  = ui32;
		using StorageType				  = memory::DenseStorage<ui32, device::CPU>;
		using Packet					  = LR_VC_TYPE(BaseScalar);
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "uint32_t";
		static constexpr ui64 Flags		  = flags::PacketBitwise | flags::ScalarBitwise |
									  flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_32U;
#endif

		static constexpr ui64 Size		= sizeof(ui32);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const ui32 &val) {
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
	struct traits<i64> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = i64;
		using BaseScalar				  = i64;
		using StorageType				  = memory::DenseStorage<i64, device::CPU>;
		using Packet					  = std::false_type; // Vc::Vector<BaseScalar>;
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = 1; // LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "int64_t";
		static constexpr ui64 Flags		  = flags::PacketBitwise | flags::ScalarBitwise |
									  flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64I;
#endif

		static constexpr ui64 Size		= sizeof(i64);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const i64 &val) {
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
	struct traits<ui64> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = ui64;
		using BaseScalar				  = ui64;
		using StorageType				  = memory::DenseStorage<ui64, device::CPU>;
		using Packet					  = std::false_type; // Vc::Vector<BaseScalar>;
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = 1; // LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "uint64_t";
		static constexpr ui64 Flags		  = flags::PacketBitwise | flags::ScalarBitwise |
									  flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64U;
#endif

		static constexpr ui64 Size		= sizeof(ui64);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const ui64 &val) {
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

	// 16bit f32ing Point implementation is in "librapid/modified/f3216/f3216.hpp"

	//------- 32bit f32ing Point --------------------------------------------
	template<>
	struct traits<f32> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = f32;
		using BaseScalar				  = f32;
		using StorageType				  = memory::DenseStorage<f32, device::CPU>;
		using Packet					  = LR_VC_TYPE(BaseScalar);
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "float";
		static constexpr ui64 Flags		  = flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_32F;
#endif

		static constexpr ui64 Size		= sizeof(f32);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const f32 &val) {
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

	//------- 64bit f32ing Point --------------------------------------------
	template<>
	struct traits<f64> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = f64;
		using BaseScalar				  = f64;
		using StorageType				  = memory::DenseStorage<f64, device::CPU>;
		using Packet					  = LR_VC_TYPE(BaseScalar);
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = LR_VC_SIZE(BaseScalar);
		static constexpr char Name[]	  = "double";
		static constexpr ui64 Flags		  = flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#endif

		static constexpr ui64 Size		= sizeof(f64);
		static constexpr bool CanAlign	= true;
		static constexpr bool CanMemcpy = true;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const f64 &val) {
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
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = Complex<T>;
		using BaseScalar				  = Complex<T>;
		using StorageType				  = memory::DenseStorage<Complex<T>, device::CPU>;
		using Packet					  = std::false_type;
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = 1;
		static constexpr char Name[]	  = "NO_MAPPED_TYPE";
		static constexpr ui64 Flags		  = flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_C_64F;
#endif

		static constexpr ui64 Size		= sizeof(Complex<T>);
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
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = mpz;
		using BaseScalar				  = mpz;
		using StorageType				  = memory::DenseStorage<mpz, device::CPU>;
		using Packet					  = std::false_type;
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = 1;
		static constexpr char Name[]	  = "NO_VALID_CONVERSION";
		static constexpr ui64 Flags		  = flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketBitwise | flags::ScalarBitwise |
									  flags::PacketLogical | flags::ScalarLogical;

#	if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#	endif

		static constexpr ui64 Size		= sizeof(mpz);
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
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = mpq;
		using BaseScalar				  = mpq;
		using StorageType				  = memory::DenseStorage<mpq, device::CPU>;
		using Packet					  = std::false_type;
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = 1;
		static constexpr char Name[]	  = "NO_VALID_CONVERSION";
		static constexpr ui64 Flags		  = flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#	if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#	endif

		static constexpr ui64 Size		= sizeof(mpq);
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

	//------- Multiprecision f32 (MPF) ---------------------------------
	template<>
	struct traits<mpf> {
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = mpf;
		using BaseScalar				  = mpf;
		using StorageType				  = memory::DenseStorage<mpf, device::CPU>;
		using Packet					  = std::false_type;
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = 1;
		static constexpr char Name[]	  = "NO_VALID_CONVERSION";
		static constexpr ui64 Flags		  = flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#	if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#	endif

		static constexpr ui64 Size		= sizeof(mpf);
		static constexpr bool CanAlign	= false;
		static constexpr bool CanMemcpy = false;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const mpf &val) {
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
		static constexpr bool IsScalar	  = true;
		static constexpr bool IsEvaluated = true;
		using Valid						  = std::true_type;
		using Scalar					  = mpfr;
		using BaseScalar				  = mpfr;
		using StorageType				  = memory::DenseStorage<mpfr, device::CPU>;
		using Packet					  = std::false_type;
		using Device					  = device::CPU;
		static constexpr i64 PacketWidth  = 1;
		static constexpr char Name[]	  = "NO_VALID_CONVERSION";
		static constexpr ui64 Flags		  = flags::PacketArithmetic | flags::ScalarArithmetic |
									  flags::PacketLogical | flags::ScalarLogical;

#	if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#	endif

		static constexpr ui64 Size		= sizeof(mpfr);
		static constexpr bool CanAlign	= false;
		static constexpr bool CanMemcpy = false;

		template<typename CAST>
		LR_FORCE_INLINE static CAST cast(const mpfr &val) {
			if constexpr (std::is_same_v<CAST, bool>) return val != 0;
			if constexpr (std::is_integral_v<CAST> && std::is_signed_v<CAST>)
				return (CAST)val.toLLong();
			if constexpr (std::is_integral_v<CAST> && std::is_unsigned_v<CAST>)
				return (CAST)val.toULLong();
			if constexpr (std::is_same_v<CAST, f32>) return (CAST)val.toFloat();
			if constexpr (std::is_same_v<CAST, f64>) return (CAST)val.toDouble();
			if constexpr (std::is_same_v<CAST, long long>) return (CAST)val.toLLong();
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
	struct ReturnType {
		using LhsType = LHS;
		using RhsType = RHS;
		using RetType = typename std::common_type<LhsType, RhsType>::type;
	};

	template<typename T>
	using StripQualifiers = typename std::remove_cv_t<typename std::remove_reference_t<T>>;

	template<typename T>
	constexpr bool isVector(const T &) {
		return false;
	}

	template<typename T, typename ABI>
	constexpr bool isVector(const Vc::Vector<T, ABI> &) {
		return true;
	}

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

	template<typename T, typename M>
	LR_NODISCARD("")
	LR_INLINE T copySign(const T &mag, const M &sign) noexcept {
#if defined(LIBRAPID_MSVC_CXX)
		return std::copysign(mag, static_cast<T>(sign));
#else
		if constexpr (std::is_fundamental_v<T> && std::is_fundamental_v<M>) {
			return std::copysign(mag, static_cast<T>(sign));
		} else {
			if (sign < 0) return -mag;
			return mag;
		}
#endif
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE bool signBit(const T &val) noexcept {
		return signBit((f64)val);
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE bool signBit(const long double &val) noexcept {
		return std::signbit(val);
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE bool signBit(const f64 &val) noexcept {
		return std::signbit(val);
	}

	template<>
	LR_NODISCARD("")
	LR_INLINE bool signBit(const f32 &val) noexcept {
		return std::signbit(val);
	}

	template<typename T>
	LR_NODISCARD("")
	LR_INLINE T ldexp(const T &x, const i64 exp) noexcept {
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
	LR_INLINE mpfr ldexp(const mpfr &x, const i64 exp) noexcept {
		return ::mpfr::ldexp(x, exp);
	}

	template<typename A, typename B>
	LR_NODISCARD("")
	LR_INLINE __gmp_expr<A, B> ldexp(const __gmp_expr<A, B> &x, const i64 exp) noexcept {
		return x << exp;
	}
#endif // LIBRAPID_USE_MULTIPREC
} // namespace librapid::internal

#undef LR_VC_TYPE
#undef LR_VC_SIZE
#undef LIMIT_IMPL_CONSTEXPR
#undef LIMIT_IMPL
#undef NUM_LIM