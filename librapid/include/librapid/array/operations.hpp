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
			auto mask = lhs OP_ rhs;                                                               \
			Packet res(1);                                                                         \
			res.setZero(!mask);                                                                    \
			return res;                                                                            \
		}                                                                                          \
	}

namespace librapid {
	namespace detail {
		/// Construct a new function object with the given functor type and arguments.
		/// \tparam desc Functor descriptor
		/// \tparam Functor Function type
		/// \tparam Args Argument types
		/// \param args Arguments passed to the function (forwarded)
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

		LIBRAPID_BINARY_COMPARISON_FUNCTOR(LessThan, <);			 // a < b
		LIBRAPID_BINARY_COMPARISON_FUNCTOR(GreaterThan, >);			 // a > b
		LIBRAPID_BINARY_COMPARISON_FUNCTOR(LessThanEqual, <=);		 // a <= b
		LIBRAPID_BINARY_COMPARISON_FUNCTOR(GreaterThanEqual, >=);	 // a >= b
		LIBRAPID_BINARY_COMPARISON_FUNCTOR(ElementWiseEqual, ==);	 // a == b
		LIBRAPID_BINARY_COMPARISON_FUNCTOR(ElementWiseNotEqual, !=); // a != b
	}																 // namespace detail

	namespace typetraits {
		/// Merge together two Descriptor types. Two trivial operations will result in another
		/// trivial operation, while any other combination will result in a Combined operation.
		/// \tparam Descriptor1 The first descriptor
		/// \tparam Descriptor2 The second descriptor
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

		/// Extracts the Descriptor type of an ArrayContainer object. In this case, the Descriptor
		/// is Trivial
		/// \tparam ShapeType The shape type of the ArrayContainer
		/// \tparam StorageType The storage type of the ArrayContainer
		template<typename ShapeType, typename StorageType>
		struct DescriptorExtractor<array::ArrayContainer<ShapeType, StorageType>> {
			using Type = ::librapid::detail::descriptor::Trivial;
		};

		/// Extracts the Descriptor type of an ArrayView object
		/// \tparam T The Array type of the ArrayView
		template<typename T>
		struct DescriptorExtractor<array::ArrayView<T>> {
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
			/// A `constexpr` function which supports the DescriptorType for multi-type inputs
			/// \tparam Rest
			/// \return
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

		/// Allows a number of Descriptor types to be merged together into a single Descriptor type.
		/// The Descriptors used are extracted from the ***typenames*** of the provided types.
		/// \tparam First The first type to merge
		/// \tparam Rest The remaining types
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

			template<typename... Args>
			static constexpr const char *getKernelName(std::tuple<Args...> args) {
				static_assert(sizeof...(Args) == 2, "Invalid number of arguments for addition");
				return getKernelNameImpl(args);
			}

		private:
			template<typename T1, typename T2>
			static constexpr const char *getKernelNameImpl(std::tuple<T1, T2> args) {
				if constexpr (TypeInfo<std::decay_t<T1>>::type != detail::LibRapidType::Scalar &&
							  TypeInfo<std::decay_t<T2>>::type != detail::LibRapidType::Scalar) {
					return kernelName;
				} else if constexpr (TypeInfo<std::decay_t<T1>>::type ==
									 detail::LibRapidType::Scalar) {
					return kernelNameScalarLhs;
				} else if constexpr (TypeInfo<std::decay_t<T2>>::type ==
									 detail::LibRapidType::Scalar) {
					return kernelNameScalarRhs;
				} else {
					LIBRAPID_ASSERT(false, "Invalid type combination");
				}
			}
		};

		template<>
		struct TypeInfo<::librapid::detail::Minus> {
			static constexpr const char *name		= "minus";
			static constexpr const char *filename	= "arithmetic";
			static constexpr const char *kernelName = "subArrays";
		};

		template<>
		struct TypeInfo<::librapid::detail::Multiply> {
			static constexpr const char *name		= "multiply";
			static constexpr const char *filename	= "arithmetic";
			static constexpr const char *kernelName = "mulArrays";
		};

		template<>
		struct TypeInfo<::librapid::detail::Divide> {
			static constexpr const char *name		= "divide";
			static constexpr const char *filename	= "arithmetic";
			static constexpr const char *kernelName = "divArrays";
		};

		template<>
		struct TypeInfo<::librapid::detail::LessThan> {
			static constexpr const char *name		= "less than";
			static constexpr const char *filename	= "arithmetic";
			static constexpr const char *kernelName = "lessThanArrays";
		};

		template<>
		struct TypeInfo<::librapid::detail::GreaterThan> {
			static constexpr const char *name		= "greater than";
			static constexpr const char *filename	= "arithmetic";
			static constexpr const char *kernelName = "greaterThanArrays";
		};

		template<>
		struct TypeInfo<::librapid::detail::LessThanEqual> {
			static constexpr const char *name		= "less than or equal";
			static constexpr const char *filename	= "arithmetic";
			static constexpr const char *kernelName = "lessThanEqualArrays";
		};

		template<>
		struct TypeInfo<::librapid::detail::GreaterThanEqual> {
			static constexpr const char *name		= "greater than or equal";
			static constexpr const char *filename	= "arithmetic";
			static constexpr const char *kernelName = "greaterThanEqualArrays";
		};

		template<>
		struct TypeInfo<::librapid::detail::ElementWiseEqual> {
			static constexpr const char *name		= "element wise equal";
			static constexpr const char *filename	= "arithmetic";
			static constexpr const char *kernelName = "elementWiseEqualArrays";
		};

		template<>
		struct TypeInfo<::librapid::detail::ElementWiseNotEqual> {
			static constexpr const char *name		= "element wise not equal";
			static constexpr const char *filename	= "arithmetic";
			static constexpr const char *kernelName = "elementWiseNotEqualArrays";
		};
	} // namespace typetraits

	namespace array {
		/// \brief Element-wise array addition
		///
		/// Performs element-wise addition on two arrays. They must both be the same size and of
		/// the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise sum of the two arrays
		template<class LHS, class RHS,
				 typename std::enable_if_t<typetraits::TypeInfo<std::decay_t<RHS>>::type !=
											 ::librapid::detail::LibRapidType::Scalar,
										   int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		operator+(LHS &&lhs, RHS &&rhs) LIBRAPID_RELEASE_NOEXCEPT
		  ->detail::Function<typetraits::DescriptorType_t<LHS, RHS>, detail::Plus, LHS, RHS> {
			static_assert(
			  typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
								 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
			  "Operands must have the same data type");
			LIBRAPID_ASSERT(lhs.shape().operator==(rhs.shape()), "Shapes must be equal");
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::Plus>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		template<class LHS, class RHS,
				 typename std::enable_if_t<typetraits::TypeInfo<std::decay_t<RHS>>::type ==
											 ::librapid::detail::LibRapidType::Scalar,
										   int> = 0>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		operator+(LHS &&lhs, RHS &&rhs) LIBRAPID_RELEASE_NOEXCEPT
		  ->detail::Function<typetraits::DescriptorType_t<LHS, RHS>, detail::Plus, LHS, RHS> {
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::Plus>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array subtraction
		///
		/// Performs element-wise subtraction on two arrays. They must both be the same size and
		/// of the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise difference of the two arrays
		template<class LHS, class RHS>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		operator-(LHS &&lhs, RHS &&rhs) LIBRAPID_RELEASE_NOEXCEPT
		  ->detail::Function<typetraits::DescriptorType_t<LHS, RHS>, detail::Minus, LHS, RHS> {
			static_assert(
			  typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
								 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
			  "Operands must have the same data type");
			LIBRAPID_ASSERT(lhs.shape().operator==(rhs.shape()), "Shapes must be equal");
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::Minus>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array multiplication
		///
		/// Performs element-wise multiplication on two arrays. They must both be the same size
		/// and of the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise product of the two arrays
		template<class LHS, class RHS>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		operator*(LHS &&lhs, RHS &&rhs) LIBRAPID_RELEASE_NOEXCEPT
		  ->detail::Function<typetraits::DescriptorType_t<LHS, RHS>, detail::Multiply, LHS, RHS> {
			static_assert(
			  typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
								 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
			  "Operands must have the same data type");
			LIBRAPID_ASSERT(lhs.shape().operator==(rhs.shape()), "Shapes must be equal");
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::Multiply>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array division
		///
		/// Performs element-wise division on two arrays. They must both be the same size and of
		/// the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise division of the two arrays
		template<class LHS, class RHS>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		operator/(LHS &&lhs, RHS &&rhs) LIBRAPID_RELEASE_NOEXCEPT
		  ->detail::Function<typetraits::DescriptorType_t<LHS, RHS>, detail::Divide, LHS, RHS> {
			static_assert(
			  typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
								 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
			  "Operands must have the same data type");
			LIBRAPID_ASSERT(lhs.shape().operator==(rhs.shape()), "Shapes must be equal");
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::Divide>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a < b for all a, b in input
		/// arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value is
		/// less than the second. They must both be the same size and of the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto
		operator<(LHS &&lhs, RHS &&rhs) LIBRAPID_RELEASE_NOEXCEPT
		  ->detail::Function<typetraits::DescriptorType_t<LHS, RHS>, detail::LessThan, LHS, RHS> {
			static_assert(
			  typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
								 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
			  "Operands must have the same data type");
			LIBRAPID_ASSERT(lhs.shape().operator==(rhs.shape()), "Shapes must be equal");
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>, detail::LessThan>(
			  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a > b for all a, b in input
		/// arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value is
		/// greater than the second. They must both be the same size and of the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator>(LHS &&lhs, RHS &&rhs)
		  LIBRAPID_RELEASE_NOEXCEPT->detail::Function<typetraits::DescriptorType_t<LHS, RHS>,
													  detail::GreaterThan, LHS, RHS> {
			static_assert(
			  typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
								 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
			  "Operands must have the same data type");
			LIBRAPID_ASSERT(lhs.shape().operator==(rhs.shape()), "Shapes must be equal");
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>,
										detail::GreaterThan>(std::forward<LHS>(lhs),
															 std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a <= b for all a, b in input
		/// arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value is
		/// less than or equal to the second. They must both be the same size and of the same
		/// data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator<=(LHS &&lhs, RHS &&rhs)
		  LIBRAPID_RELEASE_NOEXCEPT->detail::Function<typetraits::DescriptorType_t<LHS, RHS>,
													  detail::LessThanEqual, LHS, RHS> {
			static_assert(
			  typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
								 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
			  "Operands must have the same data type");
			LIBRAPID_ASSERT(lhs.shape().operator==(rhs.shape()), "Shapes must be equal");
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>,
										detail::LessThanEqual>(std::forward<LHS>(lhs),
															   std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a >= b for all a, b in input
		/// arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value is
		/// greater than or equal to the second. They must both be the same size and of the same
		/// data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator>=(LHS &&lhs, RHS &&rhs)
		  LIBRAPID_RELEASE_NOEXCEPT->detail::Function<typetraits::DescriptorType_t<LHS, RHS>,
													  detail::GreaterThanEqual, LHS, RHS> {
			static_assert(
			  typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
								 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
			  "Operands must have the same data type");
			LIBRAPID_ASSERT(lhs.shape().operator==(rhs.shape()), "Shapes must be equal");
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>,
										detail::GreaterThanEqual>(std::forward<LHS>(lhs),
																  std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a == b for all a, b in input
		/// arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value is
		/// equal to the second. They must both be the same size and of the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator==(LHS &&lhs, RHS &&rhs)
		  LIBRAPID_RELEASE_NOEXCEPT->detail::Function<typetraits::DescriptorType_t<LHS, RHS>,
													  detail::ElementWiseEqual, LHS, RHS> {
			static_assert(
			  typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
								 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
			  "Operands must have the same data type");
			LIBRAPID_ASSERT(lhs.shape().operator==(rhs.shape()), "Shapes must be equal");
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>,
										detail::ElementWiseEqual>(std::forward<LHS>(lhs),
																  std::forward<RHS>(rhs));
		}

		/// \brief Element-wise array comparison, checking whether a != b for all a, b in input
		/// arrays
		///
		/// Performs an element-wise comparison on two arrays, checking if the first value is
		/// not equal to the second. They must both be the same size and of the same data type.
		///
		/// \tparam LHS Type of the LHS element
		/// \tparam RHS Type of the RHS element
		/// \param lhs The first array
		/// \param rhs The second array
		/// \return The element-wise comparison of the two arrays
		template<class LHS, class RHS>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator!=(LHS &&lhs, RHS &&rhs)
		  LIBRAPID_RELEASE_NOEXCEPT->detail::Function<typetraits::DescriptorType_t<LHS, RHS>,
													  detail::ElementWiseNotEqual, LHS, RHS> {
			static_assert(
			  typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
								 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
			  "Operands must have the same data type");
			LIBRAPID_ASSERT(lhs.shape().operator==(rhs.shape()), "Shapes must be equal");
			return detail::makeFunction<typetraits::DescriptorType_t<LHS, RHS>,
										detail::ElementWiseNotEqual>(std::forward<LHS>(lhs),
																	 std::forward<RHS>(rhs));
		}
	} // namespace array
} // namespace librapid

#endif // LIBRAPID_ARRAY_OPERATIONS_HPP