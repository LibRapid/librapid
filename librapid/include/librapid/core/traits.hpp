#ifndef LIBRAPID_CORE_TRAITS_HPP
#define LIBRAPID_CORE_TRAITS_HPP

/*
 * The TypeInfo struct provides compile-time about types (templated types, in particular).
 * This allows for easier SFINAE implementations and simpler function dispatching.
 * Furthermore, the TypeInfo struct defines some useful conversion functions to cast between
 * types without raising compiler warnings, or worse, errors.
 *
 * A TypeInfo struct should be defined for every class defined by LibRapid.
 */

namespace librapid::typetraits {
	/// Provides compile-time information about a data type, allowing for easier function
	/// switching and compile-time evaluation
	/// \tparam T The type to get information about
	template<typename T>
	struct TypeInfo {
		static constexpr bool isScalar			 = true;
		using Scalar							 = T;
		using Packet							 = std::false_type;
		static constexpr int64_t packetWidth	 = 1;
		static constexpr char name[]			 = "[NO DEFINED TYPE]";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#endif

		static constexpr uint64_t Size	= sizeof(T);
		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<bool> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = bool;
		using Packet							 = std::false_type;
		static constexpr int64_t packetWidth	 = 1;
		static constexpr char name[]			 = "char";
		static constexpr bool supportsArithmetic = false;
		static constexpr bool supportsLogical	 = false;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_8I;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<char> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = char;
		using Packet							 = std::false_type;
		static constexpr int64_t packetWidth	 = 1;
		static constexpr char name[]			 = "bool";
		static constexpr bool supportsArithmetic = false;
		static constexpr bool supportsLogical	 = false;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_8I;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<int8_t> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = int8_t;
		using Packet							 = Vc::Vector<int8_t>;
		static constexpr int64_t packetWidth	 = Packet::size();
		static constexpr char name[]			 = "int8_t";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_8I;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<uint8_t> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = uint8_t;
		using Packet							 = Vc::Vector<uint8_t>;
		static constexpr int64_t packetWidth	 = Packet::size();
		static constexpr char name[]			 = "uint8_t";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_8U;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<int16_t> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = int16_t;
		using Packet							 = Vc::Vector<int16_t>;
		static constexpr int64_t packetWidth	 = Packet::size();
		static constexpr char name[]			 = "int16_t";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_16I;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<uint16_t> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = uint16_t;
		using Packet							 = Vc::Vector<uint16_t>;
		static constexpr int64_t packetWidth	 = Packet::size();
		static constexpr char name[]			 = "uint16_t";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_16U;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<int32_t> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = int32_t;
		using Packet							 = Vc::Vector<int32_t>;
		static constexpr int64_t packetWidth	 = Packet::size();
		static constexpr char name[]			 = "int32_t";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_32I;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<uint32_t> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = uint32_t;
		using Packet							 = Vc::Vector<uint32_t>;
		static constexpr int64_t packetWidth	 = Packet::size();
		static constexpr char name[]			 = "uint32_t";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_32U;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<int64_t> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = int64_t;
		using Packet							 = Vc::Vector<int64_t>;
		static constexpr int64_t packetWidth	 = Packet::size();
		static constexpr char name[]			 = "int64_t";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64I;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<uint64_t> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = uint64_t;
		using Packet							 = Vc::Vector<uint64_t>;
		static constexpr int64_t packetWidth	 = Packet::size();
		static constexpr char name[]			 = "uint64_t";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = true;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64U;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<float> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = float;
		using Packet							 = Vc::Vector<float>;
		static constexpr int64_t packetWidth	 = Packet::size();
		static constexpr char name[]			 = "float";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = false;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_32F;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};

	template<>
	struct TypeInfo<double> {
		static constexpr bool isScalar			 = true;
		using Scalar							 = double;
		using Packet							 = Vc::Vector<double>;
		static constexpr int64_t packetWidth	 = Packet::size();
		static constexpr char name[]			 = "double";
		static constexpr bool supportsArithmetic = true;
		static constexpr bool supportsLogical	 = true;
		static constexpr bool supportsBinary	 = false;

#if defined(LIBRAPID_HAS_CUDA)
		static constexpr cudaDataType_t CudaType = cudaDataType_t::CUDA_R_64F;
#endif

		static constexpr bool canAlign	= true;
		static constexpr bool canMemcpy = true;
	};
} // namespace librapid::typetraits

#endif // LIBRAPID_CORE_TRAITS_HPP