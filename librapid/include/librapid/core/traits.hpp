#ifndef LIBRAPID_CORE_TRAITS_HPP
#define LIBRAPID_CORE_TRAITS_HPP

/*
 * The TypeInfo struct provides compile-time about types (templated types, in particular).
 * This allows for easier SFINAE implementations and simpler function dispatching.
 * Furthermore, the TypeInfo struct provides a simple way to access a type-independent information
 * about the limits of a type, as well as various infinities and rounding errors.
 *
 * A TypeInfo struct should be defined for every class defined by LibRapid, though for compound
 * types, you may need to access the Scalar member of type to get the relevant information you'd
 * like.
 *
 * This file only implements the TypeInfo struct for primitive types. TypeInfo details for other
 * types are defined in their respective headers.
 */

#define LIMIT_IMPL_CONSTEXPR(NAME_) LIBRAPID_ALWAYS_INLINE static constexpr auto NAME_() noexcept
#define LIMIT_IMPL(NAME_)			LIBRAPID_ALWAYS_INLINE static auto NAME_() noexcept
#define NUM_LIM(NAME_)				std::numeric_limits<Scalar>::NAME_()

namespace librapid {
	namespace detail {
		/// An enum class representing different types within LibRapid. Intended maily for
		/// internal use
		enum class LibRapidType { Scalar, ArrayContainer, ArrayFunction, ArrayView };

		/*
		 * Pretty string representations of data types at compile time. This is adapted from
		 * https://bitwizeshift.github.io/posts/2021/03/09/getting-an-unmangled-type-name-at-compile-time/
		 * and I have simply adapted it to work with LibRapid.
		 */

		template<std::size_t... Idxs>
		constexpr auto substringAsArray(std::string_view str, std::index_sequence<Idxs...>) {
			return std::array {str[Idxs]...};
		}

		template<typename T>
		constexpr auto typeNameArray() {
#if defined(__clang__)
			constexpr auto prefix	= std::string_view {"[T = "};
			constexpr auto suffix	= std::string_view {"]"};
			constexpr auto function = std::string_view {__PRETTY_FUNCTION__};
#elif defined(__GNUC__)
			constexpr auto prefix	= std::string_view {"with T = "};
			constexpr auto suffix	= std::string_view {"]"};
			constexpr auto function = std::string_view {__PRETTY_FUNCTION__};
#elif defined(_MSC_VER)
			constexpr auto prefix	= std::string_view {"type_name_array<"};
			constexpr auto suffix	= std::string_view {">(void)"};
			constexpr auto function = std::string_view {__FUNCSIG__};
#else
#	define LIBRAPID_NO_TYPE_TO_STRING
#endif

#if !defined(LIBRAPID_NO_TYPE_TO_STRING)
			constexpr auto start = function.find(prefix) + prefix.size();
			constexpr auto end	 = function.rfind(suffix);

			static_assert(start < end);

			constexpr auto name = function.substr(start, (end - start));
			return substringAsArray(name, std::make_index_sequence<name.size()> {});
#else
			return std::array<char, 0> {};
#endif
		}

		template<typename T>
		struct TypeNameHolder {
			static inline constexpr auto value = typeNameArray<T>();
		};
	} // namespace detail

	namespace typetraits {
		template<typename T>
		constexpr auto typeName() -> std::string_view {
			constexpr auto &value = detail::TypeNameHolder<T>::value;
			return std::string_view {value.data(), value.size()};
		}

		/// Provides compile-time information about a data type, allowing for easier function
		/// switching and compile-time evaluation
		/// \tparam T The type to get information about
		template<typename T>
		struct TypeInfo {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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
			static constexpr bool canAlign	= true;
			static constexpr bool canMemcpy = true;

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<bool> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<int8_t> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<uint8_t> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<int16_t> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<uint16_t> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<int32_t> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<uint32_t> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<int64_t> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<uint64_t> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<float> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};

		template<>
		struct TypeInfo<double> {
			detail::LibRapidType type				 = detail::LibRapidType::Scalar;
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

			LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
			LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
			LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
			LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
			LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
			LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
			LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
			LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
		};
	} // namespace typetraits
} // namespace librapid

#endif // LIBRAPID_CORE_TRAITS_HPP