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

namespace librapid {
	namespace detail {
		template<Descriptor desc, typename Functor, typename... Args>
		auto makeFunction(Args &&...args) {
			using OperationType = Function<desc, Functor, Args...>;
			return OperationType(Functor(), std::forward<Args>(args)...);
		}

		LIBRAPID_BINARY_FUNCTOR(Plus, +);	  // a + b
		LIBRAPID_BINARY_FUNCTOR(Minus, -);	  // a - b
		LIBRAPID_BINARY_FUNCTOR(Multiply, *); // a * b
		LIBRAPID_BINARY_FUNCTOR(Divide, /);	  // a / b

	} // namespace detail

	/// \brief Element-wise array addition
	///
	/// Performs element-wise addition on two arrays. They must both be the same size and of the
	/// same data type.
	///
	/// \tparam LHS Type of the LHS element
	/// \tparam RHS Type of the RHS element
	/// \param lhs The first array
	/// \param rhs The second array
	/// \return The element-wise sum of the two arrays
	template<class LHS, class RHS>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator+(LHS &&lhs,
															 RHS &&rhs) LIBRAPID_RELEASE_NOEXCEPT
	  ->detail::Function<detail::Descriptor::Trivial, detail::Plus, LHS, RHS> {
		static_assert(typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
										 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
					  "Operands must have the same data type");
		LIBRAPID_ASSERT(lhs.shape() == rhs.shape(), "Shapes must be equal");
		return detail::makeFunction<detail::Descriptor::Trivial, detail::Plus>(
		  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
	}

	/// \brief Element-wise array subtraction
	///
	/// Performs element-wise subtraction on two arrays. They must both be the same size and of the
	/// same data type.
	///
	/// \tparam LHS Type of the LHS element
	/// \tparam RHS Type of the RHS element
	/// \param lhs The first array
	/// \param rhs The second array
	/// \return The element-wise difference of the two arrays
	template<class LHS, class RHS>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator-(LHS &&lhs,
															 RHS &&rhs) LIBRAPID_RELEASE_NOEXCEPT
	  ->detail::Function<detail::Descriptor::Trivial, detail::Minus, LHS, RHS> {
		static_assert(typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
										 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
					  "Operands must have the same data type");
		LIBRAPID_ASSERT(lhs.shape() == rhs.shape(), "Shapes must be equal");
		return detail::makeFunction<detail::Descriptor::Trivial, detail::Minus>(
		  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
	}

	/// \brief Element-wise array multiplication
	///
	/// Performs element-wise multiplication on two arrays. They must both be the same size and of
	/// the same data type.
	///
	/// \tparam LHS Type of the LHS element
	/// \tparam RHS Type of the RHS element
	/// \param lhs The first array
	/// \param rhs The second array
	/// \return The element-wise product of the two arrays
	template<class LHS, class RHS>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator*(LHS &&lhs,
															 RHS &&rhs) LIBRAPID_RELEASE_NOEXCEPT
	  ->detail::Function<detail::Descriptor::Trivial, detail::Multiply, LHS, RHS> {
		static_assert(typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
										 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
					  "Operands must have the same data type");
		LIBRAPID_ASSERT(lhs.shape() == rhs.shape(), "Shapes must be equal");
		return detail::makeFunction<detail::Descriptor::Trivial, detail::Multiply>(
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
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator/(LHS &&lhs,
															 RHS &&rhs) LIBRAPID_RELEASE_NOEXCEPT
	  ->detail::Function<detail::Descriptor::Trivial, detail::Divide, LHS, RHS> {
		static_assert(typetraits::IsSame<typename typetraits::TypeInfo<std::decay_t<LHS>>::Scalar,
										 typename typetraits::TypeInfo<std::decay_t<RHS>>::Scalar>,
					  "Operands must have the same data type");
		LIBRAPID_ASSERT(lhs.shape() == rhs.shape(), "Shapes must be equal");
		return detail::makeFunction<detail::Descriptor::Trivial, detail::Divide>(
		  std::forward<LHS>(lhs), std::forward<RHS>(rhs));
	}
} // namespace librapid

#endif // LIBRAPID_ARRAY_OPERATIONS_HPP