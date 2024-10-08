#ifndef LIBRAPID_ARRAY_OPERATIONS_HPP
#define LIBRAPID_ARRAY_OPERATIONS_HPP

#define LIBRAPID_BINARY_FUNCTOR(NAME_, OP_)                                                        \
	struct NAME_ {                                                                                 \
		template<typename T, typename V>                                                           \
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator()(const T &lhs,                    \
																  const V &rhs) const {            \
			return lhs OP_ rhs;                                                                    \
		}                                                                                          \
                                                                                                   \
		template<typename Packet>                                                                  \
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto packet(const Packet &lhs,                   \
															  const Packet &rhs) const {           \
			return lhs OP_ rhs;                                                                    \
		}                                                                                          \
	}

#define LIBRAPID_BINARY_COMPARISON_FUNCTOR(NAME_, OP_)                                             \
	struct NAME_ {                                                                                 \
		template<typename T, typename V>                                                           \
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator()(const T &lhs,                    \
																  const V &rhs) const {            \
			return (typename std::common_type_t<T, V>)(lhs OP_ rhs);                               \
		}                                                                                          \
                                                                                                   \
		template<typename Packet>                                                                  \
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto packet(const Packet &lhs,                   \
															  const Packet &rhs) const {           \
			return Packet(lhs OP_ rhs);                                                            \
		}                                                                                          \
	}

#define LIBRAPID_UNARY_KERNEL_GETTER                                                               \
	template<typename... Args>                                                                     \
	static constexpr const char *getKernelName(std::tuple<Args...> args) {                         \
		static_assert(sizeof...(Args) == 1, "Invalid number of arguments for unary operation");    \
		return kernelName;                                                                         \
	}

#define LIBRAPID_BINARY_KERNEL_GETTER                                                              \
	template<typename T1, typename T2>                                                             \
	static constexpr const char *getKernelNameImpl(std::tuple<T1, T2> args) {                      \
		if constexpr (TypeInfo<std::decay_t<T1>>::type != detail::LibRapidType::Scalar &&          \
					  TypeInfo<std::decay_t<T2>>::type != detail::LibRapidType::Scalar) {          \
			return kernelName;                                                                     \
		} else if constexpr (TypeInfo<std::decay_t<T1>>::type == detail::LibRapidType::Scalar) {   \
			return kernelNameScalarLhs;                                                            \
		} else if constexpr (TypeInfo<std::decay_t<T2>>::type == detail::LibRapidType::Scalar) {   \
			return kernelNameScalarRhs;                                                            \
		} else {                                                                                   \
			return kernelName;                                                                     \
		}                                                                                          \
	}                                                                                              \
                                                                                                   \
	template<typename... Args>                                                                     \
	static constexpr const char *getKernelName(std::tuple<Args...> args) {                         \
		static_assert(sizeof...(Args) == 2, "Invalid number of arguments for binary operation");   \
		return getKernelNameImpl(args);                                                            \
	}

#define LIBRAPID_UNARY_SHAPE_EXTRACTOR                                                             \
	template<typename... Args>                                                                     \
	LIBRAPID_NODISCARD static LIBRAPID_ALWAYS_INLINE auto getShape(                                \
	  const std::tuple<Args...> &args) {                                                           \
		static_assert(sizeof...(Args) == 1, "Invalid number of arguments for unary operation");    \
		return std::get<0>(args).shape();                                                          \
	}

//#define LIBRAPID_BINARY_SHAPE_EXTRACTOR                                                            \
//	template<typename First, typename Second>                                                      \
//	LIBRAPID_NODISCARD static LIBRAPID_ALWAYS_INLINE auto getShapeImpl(                            \
//	  const std::tuple<First, Second> &tup) {                                                      \
//		if constexpr (TypeInfo<std::decay_t<First>>::type != detail::LibRapidType::Scalar &&       \
//					  TypeInfo<std::decay_t<Second>>::type != detail::LibRapidType::Scalar) {      \
//			LIBRAPID_ASSERT(std::get<0>(tup).shape() == std::get<1>(tup).shape(),                  \
//							"Shapes must match for binary operations");                            \
//			return std::get<0>(tup).shape();                                                       \
//		} else if constexpr (TypeInfo<std::decay_t<First>>::type ==                                \
//							 detail::LibRapidType::Scalar) {                                       \
//			return std::get<1>(tup).shape();                                                       \
//		} else {                                                                                   \
//			return std::get<0>(tup).shape();                                                       \
//		}                                                                                          \
//	}                                                                                                 \


#define LIBRAPID_BINARY_SHAPE_EXTRACTOR                                                            \
	template<typename First, typename Second>                                                      \
	LIBRAPID_NODISCARD static LIBRAPID_ALWAYS_INLINE auto getShapeImpl(                            \
	  const std::tuple<First, Second> &tup) {                                                      \
		if constexpr (IsArrayType<std::decay_t<First>>::value) {                                   \
			if constexpr (IsArrayType<std::decay_t<Second>>::value) {                              \
				LIBRAPID_ASSERT_WITH_EXCEPTION(                                                    \
				  std::range_error,                                                                \
				  std::get<0>(tup).shape() == std::get<1>(tup).shape(),                            \
				  "Shapes must match for binary operations. {} vs {}",                             \
				  std::get<0>(tup).shape(),                                                        \
				  std::get<1>(tup).shape());                                                       \
				return std::get<0>(tup).shape();                                                   \
			}                                                                                      \
			return std::get<0>(tup).shape();                                                       \
		} else if constexpr (IsArrayType<std::decay_t<Second>>::value) {                           \
			return std::get<1>(tup).shape();                                                       \
		}                                                                                          \
	}                                                                                              \
                                                                                                   \
	template<typename... Args>                                                                     \
	LIBRAPID_NODISCARD static LIBRAPID_ALWAYS_INLINE auto getShape(                                \
	  const std::tuple<Args...> &args) {                                                           \
		static_assert(sizeof...(Args) == 2, "Invalid number of arguments for binary operation");   \
		return getShapeImpl(args);                                                                 \
	}

#define LIBRAPID_UNARY_FUNCTOR(NAME, OP)                                                           \
	struct NAME {                                                                                  \
		template<typename T>                                                                       \
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator()(const T &arg) const {            \
			return (T)(OP(arg));                                                                   \
		}                                                                                          \
                                                                                                   \
		template<typename Packet>                                                                  \
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto packet(const Packet &arg) const {           \
			return OP(arg);                                                                        \
		}                                                                                          \
	};

namespace librapid {
	namespace detail {
		/// Construct a new function object with the given functor type and arguments.
		/// \tparam desc Functor descriptor
		/// \tparam Functor Function type
		/// \tparam Args Argument types
		/// \param args Arguments passed to the function
		/// \return A new Function instance
		template<typename desc, typename Functor, typename... Args>
		auto makeFunction(Args &&...args) {
			using OperationType = Function<desc, Functor, Args...>;
			return OperationType(Functor(), std::forward<Args>(args)...);
		}

		LIBRAPID_BINARY_FUNCTOR(Plus, +);	  // a + b
		LIBRAPID_BINARY_FUNCTOR(Minus, -);	  // a - b
		LIBRAPID_BINARY_FUNCTOR(Multiply, *); // a * b
		LIBRAPID_BINARY_FUNCTOR(Divide, /);	  // a / b

		LIBRAPID_UNARY_FUNCTOR(Neg, -);

		LIBRAPID_BINARY_COMPARISON_FUNCTOR(LessThan, <);			 // a < b
		LIBRAPID_BINARY_COMPARISON_FUNCTOR(GreaterThan, >);			 // a > b
		LIBRAPID_BINARY_COMPARISON_FUNCTOR(LessThanEqual, <=);		 // a <= b
		LIBRAPID_BINARY_COMPARISON_FUNCTOR(GreaterThanEqual, >=);	 // a >= b
		LIBRAPID_BINARY_COMPARISON_FUNCTOR(ElementWiseEqual, ==);	 // a == b
		LIBRAPID_BINARY_COMPARISON_FUNCTOR(ElementWiseNotEqual, !=); // a != b

		LIBRAPID_UNARY_FUNCTOR(Sin, ::librapid::sin);	 // sin(a)
		LIBRAPID_UNARY_FUNCTOR(Cos, ::librapid::cos);	 // cos(a)
		LIBRAPID_UNARY_FUNCTOR(Tan, ::librapid::tan);	 // tan(a)
		LIBRAPID_UNARY_FUNCTOR(Asin, ::librapid::asin);	 // asin(a)
		LIBRAPID_UNARY_FUNCTOR(Acos, ::librapid::acos);	 // acos(a)
		LIBRAPID_UNARY_FUNCTOR(Atan, ::librapid::atan);	 // atan(a)
		LIBRAPID_UNARY_FUNCTOR(Sinh, ::librapid::sinh);	 // sinh(a)
		LIBRAPID_UNARY_FUNCTOR(Cosh, ::librapid::cosh);	 // cosh(a)
		LIBRAPID_UNARY_FUNCTOR(Tanh, ::librapid::tanh);	 // tanh(a)
		LIBRAPID_UNARY_FUNCTOR(Asinh, ::librapid::sinh); // asinh(a)
		LIBRAPID_UNARY_FUNCTOR(Acosh, ::librapid::cosh); // acosh(a)
		LIBRAPID_UNARY_FUNCTOR(Atanh, ::librapid::tanh); // atanh(a)

		LIBRAPID_UNARY_FUNCTOR(Exp, ::librapid::exp);	  // exp(a)
		LIBRAPID_UNARY_FUNCTOR(Exp2, ::librapid::exp);	  // exp2(a)
		LIBRAPID_UNARY_FUNCTOR(Exp10, ::librapid::exp);	  // exp10(a)
		LIBRAPID_UNARY_FUNCTOR(Log, ::librapid::log);	  // log(a)
		LIBRAPID_UNARY_FUNCTOR(Log2, ::librapid::log2);	  // log2(a)
		LIBRAPID_UNARY_FUNCTOR(Log10, ::librapid::log10); // log10(a)
		LIBRAPID_UNARY_FUNCTOR(Sqrt, ::librapid::sqrt);	  // sqrt(a)
		LIBRAPID_UNARY_FUNCTOR(Cbrt, ::librapid::cbrt);	  // cbrt(a)
		LIBRAPID_UNARY_FUNCTOR(Abs, ::librapid::abs);	  // abs(a)
		LIBRAPID_UNARY_FUNCTOR(Floor, ::librapid::floor); // floor(a)
		LIBRAPID_UNARY_FUNCTOR(Ceil, ::librapid::ceil);	  // ceil(a)

	} // namespace detail

	namespace typetraits {
		/// Merge together two Descriptor types. Two trivial operations will result in
		/// another trivial operation, while any other combination will result in a Combined
		/// operation. \tparam Descriptor1 The first descriptor \tparam Descriptor2 The
		/// second descriptor
		template<typename Descriptor1, typename Descriptor2>
		struct DescriptorMerger {
			using Type = ::librapid::detail::descriptor::Combined;
		};

		template<typename Descriptor1>
		struct DescriptorMerger<Descriptor1, Descriptor1> {
			using Type = Descriptor1;
		};

		/// Extracts the Descriptor type of the provided type.
		/// \tparam T The type to extract the descriptor from
		template<typename T>
		struct DescriptorExtractor {
			using Type = ::librapid::detail::descriptor::Trivial;
		};

		/// Extracts the Descriptor type of an ArrayContainer object. In this case, the
		/// Descriptor is Trivial \tparam ShapeType The shape type of the ArrayContainer
		/// \tparam StorageType The storage type of the ArrayContainer
		template<typename ShapeType, typename StorageType>
		struct DescriptorExtractor<array::ArrayContainer<ShapeType, StorageType>> {
			using Type = ::librapid::detail::descriptor::Trivial;
		};

		/// Extracts the Descriptor type of an ArrayView object
		/// \tparam T The Array type of the ArrayView
		template<typename T, typename S>
		struct DescriptorExtractor<array::GeneralArrayView<T, S>> {
			using Type = ::librapid::detail::descriptor::Trivial;
		};

		/// Extracts the Descriptor type of a Function object
		/// \tparam Descriptor The descriptor of the Function
		/// \tparam Functor The functor type of the Function
		/// \tparam Args The argument types of the Function
		template<typename Descriptor, typename Functor, typename... Args>
		struct DescriptorExtractor<::librapid::detail::Function<Descriptor, Functor, Args...>> {
			using Type = Descriptor;
		};

		/// Return the combined Descriptor type of the provided types
		/// \tparam First The first type to merge
		/// \tparam Rest The remaining types
		template<typename First, typename... Rest>
		struct DescriptorType;

		namespace impl {
			/// A `constexpr` function which supports the DescriptorType for multi-type
			/// inputs \tparam Rest \return
			template<typename... Rest>
			constexpr auto descriptorExtractor() {
				if constexpr (sizeof...(Rest) > 0) {
					using ReturnType = typename DescriptorType<Rest...>::Type;
					return ReturnType {};
				} else {
					using ReturnType = ::librapid::detail::descriptor::Trivial;
					return ReturnType {};
				}
			}
		} // namespace impl

		/// Allows a number of Descriptor types to be merged together into a single
		/// Descriptor type. The Descriptors used are extracted from the ***typenames*** of
		/// the provided types. \tparam First The first type to merge \tparam Rest The
		/// remaining types
		template<typename First, typename... Rest>
		struct DescriptorType {
			using FirstType		  = std::decay_t<First>;
			using FirstDescriptor = typename DescriptorExtractor<FirstType>::Type;
			using RestDescriptor  = decltype(impl::descriptorExtractor<Rest...>());

			using Type = typename DescriptorMerger<FirstDescriptor, RestDescriptor>::Type;
		};

		/// A simplification of the DescriptorType to reduce code size
		/// \tparam Args Input types
		/// \see DescriptorType
		template<typename... Args>
		using DescriptorType_t = typename DescriptorType<Args...>::Type;

		template<>
		struct TypeInfo<::librapid::detail::Plus> {
			static constexpr const char *name				 = "plus";
			static constexpr const char *filename			 = "arithmetic";
			static constexpr const char *kernelName			 = "addArrays";
			static constexpr const char *kernelNameScalarRhs = "addArraysScalarRhs";
			static constexpr const char *kernelNameScalarLhs = "addArraysScalarLhs";
			LIBRAPID_BINARY_KERNEL_GETTER
			LIBRAPID_BINARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Minus> {
			static constexpr const char *name				 = "minus";
			static constexpr const char *filename			 = "arithmetic";
			static constexpr const char *kernelName			 = "subArrays";
			static constexpr const char *kernelNameScalarRhs = "subArraysScalarRhs";
			static constexpr const char *kernelNameScalarLhs = "subArraysScalarLhs";
			LIBRAPID_BINARY_KERNEL_GETTER
			LIBRAPID_BINARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Multiply> {
			static constexpr const char *name				 = "multiply";
			static constexpr const char *filename			 = "arithmetic";
			static constexpr const char *kernelName			 = "mulArrays";
			static constexpr const char *kernelNameScalarRhs = "mulArraysScalarRhs";
			static constexpr const char *kernelNameScalarLhs = "mulArraysScalarLhs";
			LIBRAPID_BINARY_KERNEL_GETTER
			LIBRAPID_BINARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Divide> {
			static constexpr const char *name				 = "divide";
			static constexpr const char *filename			 = "arithmetic";
			static constexpr const char *kernelName			 = "divArrays";
			static constexpr const char *kernelNameScalarRhs = "divArraysScalarRhs";
			static constexpr const char *kernelNameScalarLhs = "divArraysScalarLhs";
			LIBRAPID_BINARY_KERNEL_GETTER
			LIBRAPID_BINARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::LessThan> {
			static constexpr const char *name				 = "less than";
			static constexpr const char *filename			 = "arithmetic";
			static constexpr const char *kernelName			 = "lessThanArrays";
			static constexpr const char *kernelNameScalarRhs = "lessThanArraysScalarRhs";
			static constexpr const char *kernelNameScalarLhs = "lessThanArraysScalarLhs";
			LIBRAPID_BINARY_KERNEL_GETTER
			LIBRAPID_BINARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::GreaterThan> {
			static constexpr const char *name				 = "greater than";
			static constexpr const char *filename			 = "arithmetic";
			static constexpr const char *kernelName			 = "greaterThanArrays";
			static constexpr const char *kernelNameScalarRhs = "greaterThanArraysScalarRhs";
			static constexpr const char *kernelNameScalarLhs = "greaterThanArraysScalarLhs";
			LIBRAPID_BINARY_KERNEL_GETTER
			LIBRAPID_BINARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::LessThanEqual> {
			static constexpr const char *name				 = "less than or equal";
			static constexpr const char *filename			 = "arithmetic";
			static constexpr const char *kernelName			 = "lessThanEqualArrays";
			static constexpr const char *kernelNameScalarRhs = "lessThanEqualArraysScalarRhs";
			static constexpr const char *kernelNameScalarLhs = "lessThanEqualArraysScalarLhs";
			LIBRAPID_BINARY_KERNEL_GETTER
			LIBRAPID_BINARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::GreaterThanEqual> {
			static constexpr const char *name				 = "greater than or equal";
			static constexpr const char *filename			 = "arithmetic";
			static constexpr const char *kernelName			 = "greaterThanEqualArrays";
			static constexpr const char *kernelNameScalarRhs = "greaterThanEqualArraysScalarRhs";
			static constexpr const char *kernelNameScalarLhs = "greaterThanEqualArraysScalarLhs";
			LIBRAPID_BINARY_KERNEL_GETTER
			LIBRAPID_BINARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::ElementWiseEqual> {
			static constexpr const char *name				 = "element wise equal";
			static constexpr const char *filename			 = "arithmetic";
			static constexpr const char *kernelName			 = "elementWiseEqualArrays";
			static constexpr const char *kernelNameScalarRhs = "elementWiseEqualArraysScalarRhs";
			static constexpr const char *kernelNameScalarLhs = "elementWiseEqualArraysScalarLhs";
			LIBRAPID_BINARY_KERNEL_GETTER
			LIBRAPID_BINARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::ElementWiseNotEqual> {
			static constexpr const char *name				 = "element wise not equal";
			static constexpr const char *filename			 = "arithmetic";
			static constexpr const char *kernelName			 = "elementWiseNotEqualArrays";
			static constexpr const char *kernelNameScalarRhs = "elementWiseNotEqualArraysScalarRhs";
			static constexpr const char *kernelNameScalarLhs = "elementWiseNotEqualArraysScalarLhs";
			LIBRAPID_BINARY_KERNEL_GETTER
			LIBRAPID_BINARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Neg> {
			static constexpr const char *name		= "negate";
			static constexpr const char *filename	= "negate";
			static constexpr const char *kernelName = "negateArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Sin> {
			static constexpr const char *name		= "sin";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "sinArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Cos> {
			static constexpr const char *name		= "cos";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "cosArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Tan> {
			static constexpr const char *name		= "tan";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "tanArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Asin> {
			static constexpr const char *name		= "arcsin";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "asinArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Acos> {
			static constexpr const char *name		= "arcos";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "acosArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Atan> {
			static constexpr const char *name		= "arctan";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "atanArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Sinh> {
			static constexpr const char *name		= "hyperbolic sine";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "sinhArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Cosh> {
			static constexpr const char *name		= "hyperbolic cosine";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "coshArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Tanh> {
			static constexpr const char *name		= "hyperbolic tangent";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "tanhArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Asinh> {
			static constexpr const char *name		= "hyperbolic arcsine";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "asinhArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Acosh> {
			static constexpr const char *name		= "hyperbolic arccosine";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "acoshArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Atanh> {
			static constexpr const char *name		= "hyperbolic arctangent";
			static constexpr const char *filename	= "trigonometry";
			static constexpr const char *kernelName = "atanhArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Exp> {
			static constexpr const char *name		= "exponent";
			static constexpr const char *filename	= "expLogPow";
			static constexpr const char *kernelName = "expArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Exp2> {
			static constexpr const char *name		= "exponent base 2";
			static constexpr const char *filename	= "expLogPow";
			static constexpr const char *kernelName = "exp2Arrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Exp10> {
			static constexpr const char *name		= "exponent base 10";
			static constexpr const char *filename	= "expLogPow";
			static constexpr const char *kernelName = "exp10Arrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Log> {
			static constexpr const char *name		= "logarithm";
			static constexpr const char *filename	= "expLogPow";
			static constexpr const char *kernelName = "logArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Log2> {
			static constexpr const char *name		= "logarithm base 2";
			static constexpr const char *filename	= "expLogPow";
			static constexpr const char *kernelName = "log2Arrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Log10> {
			static constexpr const char *name		= "logarithm base 10";
			static constexpr const char *filename	= "expLogPow";
			static constexpr const char *kernelName = "log10Arrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Sqrt> {
			static constexpr const char *name		= "square root";
			static constexpr const char *filename	= "expLogPow";
			static constexpr const char *kernelName = "sqrtArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Cbrt> {
			static constexpr const char *name		= "cube root";
			static constexpr const char *filename	= "expLogPow";
			static constexpr const char *kernelName = "cbrtArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Abs> {
			static constexpr const char *name		= "absolute value";
			static constexpr const char *filename	= "abs";
			static constexpr const char *kernelName = "absArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Floor> {
			static constexpr const char *name		= "floor";
			static constexpr const char *filename	= "floorCeilRound";
			static constexpr const char *kernelName = "floorArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};

		template<>
		struct TypeInfo<::librapid::detail::Ceil> {
			static constexpr const char *name		= "ceiling";
			static constexpr const char *filename	= "floorCeilRound";
			static constexpr const char *kernelName = "ceilArrays";
			LIBRAPID_UNARY_KERNEL_GETTER
			LIBRAPID_UNARY_SHAPE_EXTRACTOR
		};
	} // namespace typetraits

	namespace detail {
		template<typename T, LibRapidType... ValidTypes>
		constexpr bool isType() {
			constexpr LibRapidType t = typetraits::TypeInfo<std::decay_t<T>>::type;
			return ((t == ValidTypes) || ...);
		}

		template<typename VAL>
		constexpr bool isArrayOp() {
			return (typetraits::IsArrayContainer<std::decay_t<VAL>>::value ||
					typetraits::IsLibRapidType<std::decay_t<VAL>>::value);
		}

		template<typename LHS, typename RHS>
		constexpr bool isArrayOpArray() {
			constexpr bool lhsIsValid = isType<LHS,
											   LibRapidType::ArrayContainer,
											   LibRapidType::ArrayFunction,
											   LibRapidType::GeneralArrayView>();

			constexpr bool rhsIsValid = isType<RHS,
											   LibRapidType::ArrayContainer,
											   LibRapidType::ArrayFunction,
											   LibRapidType::GeneralArrayView>();

			return lhsIsValid && rhsIsValid;
		}

		template<typename LHS, typename RHS>
		constexpr bool isArrayOpWithScalar() {
			constexpr bool lhsIsArray = isType<LHS,
											   LibRapidType::ArrayContainer,
											   LibRapidType::ArrayFunction,
											   LibRapidType::GeneralArrayView>();

			constexpr bool rhsIsArray = isType<RHS,
											   LibRapidType::ArrayContainer,
											   LibRapidType::ArrayFunction,
											   LibRapidType::GeneralArrayView>();

			constexpr bool lhsIsScalar = isType<LHS, LibRapidType::Scalar, LibRapidType::Vector>();

			constexpr bool rhsIsScalar = isType<RHS, LibRapidType::Scalar, LibRapidType::Vector>();

			return (lhsIsArray ^ rhsIsArray) && (lhsIsScalar ^ rhsIsScalar);
		}

		template<typename VAL>
		concept IsArrayOp = isArrayOp<VAL>();

		template<typename LHS, typename RHS>
		concept IsArrayOpArray = isArrayOpArray<LHS, RHS>();

		template<typename LHS, typename RHS>
		concept IsArrayOpWithScalar = isArrayOpWithScalar<LHS, RHS>();
	} // namespace detail

	namespace array {
#define IS_ARRAY_OP				detail::isArrayOp<VAL>()
#define IS_ARRAY_OP_ARRAY		detail::isArrayOpArray<LHS, RHS>()
#define IS_ARRAY_OP_WITH_SCALAR detail::isArrayOpWithScalar<LHS, RHS>()

		/// \brief Element-wise array addition
		///
		/// Performs element-wise addition on two arrays. They must both be the same size
		/// and of the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise sum of the two arrays
		template<class LHS, class RHS>
			requires(detail::IsArrayOpArray<LHS, RHS> || detail::IsArrayOpWithScalar<LHS, RHS>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator+(LHS &&lhs, RHS &&rhs) {
			if constexpr (IS_ARRAY_OP_ARRAY) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(std::range_error,
											   lhs.shape().operator==(rhs.shape()),
											   "Shapes must be equal. {} vs {}",
											   lhs.shape(),
											   rhs.shape());
			}

			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::Plus>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array subtraction
		///
		/// Performs element-wise subtraction on two arrays. They must both be the same size
		/// and of the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise difference of the two arrays
		template<class LHS, class RHS>
			requires(detail::IsArrayOpArray<LHS, RHS> || detail::IsArrayOpWithScalar<LHS, RHS>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator-(LHS &&lhs, RHS &&rhs) {
			if constexpr (IS_ARRAY_OP_ARRAY) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(std::range_error,
											   lhs.shape().operator==(rhs.shape()),
											   "Shapes must be equal. {} vs {}",
											   lhs.shape(),
											   rhs.shape());
			}

			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::Minus>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array multiplication
		///
		/// Performs element-wise multiplication on two arrays. They must both be the same
		/// size and of the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise product of the two arrays
		template<class LHS, class RHS>
			requires(detail::IsArrayOpArray<LHS, RHS> || detail::IsArrayOpWithScalar<LHS, RHS>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator*(LHS &&lhs, RHS &&rhs) {
			if constexpr (IS_ARRAY_OP_ARRAY) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(std::range_error,
											   lhs.shape().operator==(rhs.shape()),
											   "Shapes must be equal. {} vs {}",
											   lhs.shape(),
											   rhs.shape());
			}

			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::Multiply>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array division
		///
		/// Performs element-wise division on two arrays. They must both be the same size
		/// and of the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise division of the two arrays
		template<class LHS, class RHS>
			requires(detail::IsArrayOpArray<LHS, RHS> || detail::IsArrayOpWithScalar<LHS, RHS>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator/(LHS &&lhs, RHS &&rhs) {
			if constexpr (IS_ARRAY_OP_ARRAY) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(std::range_error,
											   lhs.shape().operator==(rhs.shape()),
											   "Shapes must be equal. {} vs {}",
											   lhs.shape(),
											   rhs.shape());
			}

			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::Divide>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a < b for all a, b in
		/// input arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value
		/// is less than the second. They must both be the same size and of the same data
		/// type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
			requires(detail::IsArrayOpArray<LHS, RHS> || detail::IsArrayOpWithScalar<LHS, RHS>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator<(LHS &&lhs, RHS &&rhs) {
			if constexpr (IS_ARRAY_OP_ARRAY) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(std::range_error,
											   lhs.shape().operator==(rhs.shape()),
											   "Shapes must be equal. {} vs {}",
											   lhs.shape(),
											   rhs.shape());
			}

			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::LessThan>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a > b for all a, b in
		/// input arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value
		/// is greater than the second. They must both be the same size and of the same data
		/// type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
			requires(detail::IsArrayOpArray<LHS, RHS> || detail::IsArrayOpWithScalar<LHS, RHS>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator>(LHS &&lhs, RHS &&rhs) {
			if constexpr (IS_ARRAY_OP_ARRAY) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(std::range_error,
											   lhs.shape().operator==(rhs.shape()),
											   "Shapes must be equal. {} vs {}",
											   lhs.shape(),
											   rhs.shape());
			}

			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>,
										detail::GreaterThan>(std::forward<LHS>(lhs),
															 std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a <= b for all a, b in
		/// input arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value
		/// is less than or equal to the second. They must both be the same size and of the
		/// same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
			requires(detail::IsArrayOpArray<LHS, RHS> || detail::IsArrayOpWithScalar<LHS, RHS>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator<=(LHS &&lhs, RHS &&rhs) {
			if constexpr (IS_ARRAY_OP_ARRAY) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(std::range_error,
											   lhs.shape().operator==(rhs.shape()),
											   "Shapes must be equal. {} vs {}",
											   lhs.shape(),
											   rhs.shape());
			}

			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>,
										detail::LessThanEqual>(std::forward<LHS>(lhs),
															   std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a >= b for all a, b in
		/// input arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value
		/// is greater than or equal to the second. They must both be the same size and of
		/// the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
			requires(detail::IsArrayOpArray<LHS, RHS> || detail::IsArrayOpWithScalar<LHS, RHS>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator>=(LHS &&lhs, RHS &&rhs) {
			if constexpr (IS_ARRAY_OP_ARRAY) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(std::range_error,
											   lhs.shape().operator==(rhs.shape()),
											   "Shapes must be equal. {} vs {}",
											   lhs.shape(),
											   rhs.shape());
			}

			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>,
										detail::GreaterThanEqual>(std::forward<LHS>(lhs),
																  std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a == b for all a, b in
		/// input arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value
		/// is equal to the second. They must both be the same size and of the same data
		/// type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
			requires(detail::IsArrayOpArray<LHS, RHS> || detail::IsArrayOpWithScalar<LHS, RHS>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator==(LHS &&lhs, RHS &&rhs) {
			if constexpr (IS_ARRAY_OP_ARRAY) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(std::range_error,
											   lhs.shape().operator==(rhs.shape()),
											   "Shapes must be equal. {} vs {}",
											   lhs.shape(),
											   rhs.shape());
			}

			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>,
										detail::ElementWiseEqual>(std::forward<LHS>(lhs),
																  std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a != b for all a, b in
		/// input arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value
		/// is not equal to the second. They must both be the same size and of the same data
		/// type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
			requires(detail::IsArrayOpArray<LHS, RHS> || detail::IsArrayOpWithScalar<LHS, RHS>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator!=(LHS &&lhs, RHS &&rhs) {
			if constexpr (IS_ARRAY_OP_ARRAY) {
				LIBRAPID_ASSERT_WITH_EXCEPTION(std::range_error,
											   lhs.shape().operator==(rhs.shape()),
											   "Shapes must be equal. {} vs {}",
											   lhs.shape(),
											   rhs.shape());
			}

			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>,
										detail::ElementWiseNotEqual>(std::forward<LHS>(lhs),
																	 std::forward<RHS>(rhs));
		}

		/// \brief Negate each element in the array
		/// \tparam VAL Type to negate
		/// \param val The input array or function
		/// \return Negation function object
		template<class VAL>
			requires(detail::IsArrayOp<VAL>)
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator-(VAL &&val) {
			return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Neg>(
			  std::forward<VAL>(val));
		}
	} // namespace array

	/// \brief Calculate the sine of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \sin(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Sine function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	sin(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Sin, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Sin>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the cosine of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \cos(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Cosine function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	cos(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Cos, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Cos>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the tangent of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \tan(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Tangent function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	tan(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Tan, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Tan>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the arcsine of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \sin^{-1}(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Arcsine function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	asin(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Asin, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Asin>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the arccosine of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \cos^{-1}(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Arccosine function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	acos(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Acos, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Acos>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the arctangent of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \tan^{-1}(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Arctangent function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	atan(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Atan, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Atan>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the hyperbolic sine of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \sinh(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Hyperbolic sine function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	sinh(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Sinh, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Sinh>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the hyperbolic cosine of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \cosh(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Hyperbolic cosine function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	cosh(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Cosh, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Cosh>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the hyperbolic tangent of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \tanh(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Hyperbolic tangent function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	tanh(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Tanh, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Tanh>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the hyperbolic arcsine of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \sinh^{-1}(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Hyperbolic sine function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	asinh(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Asinh, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Asinh>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the hyperbolic arccosine of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \cosh^{-1}(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Hyperbolic cosine function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	acosh(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Acosh, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Acosh>(
		  std::forward<VAL>(val));
	}

	/// \brief Calculate the hyperbolic arctangent of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \tanh^{-1}(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Hyperbolic tangent function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	atanh(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Atanh, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Atanh>(
		  std::forward<VAL>(val));
	}

	/// \brief Raise e to the power of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = e^{A_i}\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Exponential function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	exp(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Exp, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Exp>(
		  std::forward<VAL>(val));
	}

	/// \brief Raise 2 to the power of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = 2^{A_i}\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Exponential function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	exp2(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Exp2, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Exp2>(
		  std::forward<VAL>(val));
	}

	/// \brief Raise 10 to the power of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = 10^{A_i}\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Exponential function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	exp10(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Exp10, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Exp10>(
		  std::forward<VAL>(val));
	}

	// \brief Compute the natural logarithm of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \ln(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Natural logarithm function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	log(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Log, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Log>(
		  std::forward<VAL>(val));
	}

	/// \brief Compute the base 10 logarithm of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \log_{10}(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Base 10 logarithm function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	log10(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Log10, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Log10>(
		  std::forward<VAL>(val));
	}

	/// \brief Compute the base 2 logarithm of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \log_{2}(A_i)\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Base 2 logarithm function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	log2(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Log2, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Log2>(
		  std::forward<VAL>(val));
	}

	/// \brief Compute the square root of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \sqrt{A_i}\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Square root function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	sqrt(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Sqrt, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Sqrt>(
		  std::forward<VAL>(val));
	}

	/// \brief Compute the cube root of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \sqrt[3]{A_i}\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Cube root function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	cbrt(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Cbrt, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Cbrt>(
		  std::forward<VAL>(val));
	}

	/// \brief Compute the absolute value of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = |A_i|\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Absolute value function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	abs(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Abs, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Abs>(
		  std::forward<VAL>(val));
	}

	/// \brief Compute the floor of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \lfloor A_i \rfloor\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Floor function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	floor(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Floor, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Floor>(
		  std::forward<VAL>(val));
	}

	/// \brief Compute the ceiling of each element in the array
	///
	/// \f$R = \{ R_0, R_1, R_2, ... \} \f$ \text{ where } \f$R_i = \lceil A_i \rceil\f$
	///
	/// \tparam VAL Type of the input
	/// \param val The input array or function
	/// \return Ceiling function object
	template<class VAL>
		requires(detail::IsArrayOp<VAL>)
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
	ceil(VAL &&val) -> detail::Function<typetraits::DescriptorType_t<VAL>, detail::Ceil, VAL> {
		return detail::makeFunction<typetraits::DescriptorType_t<VAL>, detail::Ceil>(
		  std::forward<VAL>(val));
	}
} // namespace librapid

#endif // LIBRAPID_ARRAY_OPERATIONS_HPP
