#ifndef LIBRAPID_AUTODIFF_ARRAYAUTODIFF_HPP
#define LIBRAPID_AUTODIFF_ARRAYAUTODIFF_HPP

namespace librapid::autodiff {
	template<typename T>
	struct IsDiffType : std::false_type {};

	struct Variable {};
	template<>
	struct IsDiffType<Variable> : std::true_type {};

	struct WithRespectTo {};
	template<>
	struct IsDiffType<WithRespectTo> : std::true_type {};

	template<typename LHS, typename RHS>
	struct Add {};
	template<typename LHS, typename RHS>
	struct IsDiffType<Add<LHS, RHS>> : std::true_type {};

	template<typename LHS, typename RHS>
	struct Sub {};
	template<typename LHS, typename RHS>
	struct IsDiffType<Sub<LHS, RHS>> : std::true_type {};

	template<
	  typename LHS, typename RHS,
	  typename std::enable_if_t<
		IsDiffType<std::decay_t<LHS>>::value && IsDiffType<std::decay_t<RHS>>::value, int> = 0>
	constexpr auto operator+(LHS &&lhs, RHS &&rhs) {
		return Add<std::decay_t<LHS>, std::decay_t<RHS>> {};
	}

	template<
	  typename LHS, typename RHS,
	  typename std::enable_if_t<
		IsDiffType<std::decay_t<LHS>>::value && IsDiffType<std::decay_t<RHS>>::value, int> = 0>
	constexpr auto operator-(LHS &&lhs, RHS &&rhs) {
		return Sub<std::decay_t<LHS>, std::decay_t<RHS>> {};
	}

	template<typename T>
	LIBRAPID_INLINE auto differentiate(const T &) {
		LIBRAPID_NOT_IMPLEMENTED;
		return 0;
	}

	template<>
	LIBRAPID_INLINE auto differentiate(const Variable &) {
		return 0;
	}

	template<>
	LIBRAPID_INLINE auto differentiate(const WithRespectTo &) {
		return 1;
	}
} // namespace librapid::autodiff

#endif // LIBRAPID_AUTODIFF_ARRAYAUTODIFF_HPP