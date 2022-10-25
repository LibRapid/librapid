#ifndef LIBRAPID_ARRAY_OPERATIONS_HPP
#define LIBRAPID_ARRAY_OPERATIONS_HPP

#define LIBRAPID_BINARY_OPERATION(NAME_, OP_)                                                      \
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
		template<typename Functor, typename... Args>
		auto makeFunction(Args &&...args) {
			using OperationType = Function<Functor, Args...>;
			return OperationType(Functor(), std::forward<Args>(args)...);
		}

		LIBRAPID_BINARY_OPERATION(Plus, +);
		LIBRAPID_BINARY_OPERATION(Minus, -);
		LIBRAPID_BINARY_OPERATION(Multiply, *);
		LIBRAPID_BINARY_OPERATION(Divide, /);
	} // namespace detail

	template<class LHS, class RHS>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator+(LHS &&lhs, RHS &&rhs)
	  LIBRAPID_RELEASE_NOEXCEPT->detail::Function<detail::Plus, LHS, RHS> {
		LIBRAPID_ASSERT(lhs.shape() == rhs.shape(), "Shapes must be equal");
		return detail::makeFunction<detail::Plus>(std::forward<LHS>(lhs), std::forward<RHS>(rhs));
	}
} // namespace librapid

#endif // LIBRAPID_ARRAY_OPERATIONS_HPP