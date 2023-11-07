#ifndef LIBRAPID_CORE_TYPETRAITS_HPP
#define LIBRAPID_CORE_TYPETRAITS_HPP

/*
 * Defines a range of helper template types to increase code clarity
 * while simultaneously reducing code verbosity.
 */

namespace librapid::typetraits {
	template<typename A, typename B>
	constexpr bool IsSame = std::is_same<A, B>::value;

	namespace impl {
		/*
		 * These functions test for the presence of certain features of a type
		 * by providing two valid function overloads, but the preferred one
		 * (the one taking an integer) is only valid if the requested feature
		 * exists. The return type of both functions differ, and can be evaluated
		 * as "true" and "false" depending on the presence of the feature.
		 *
		 * This is really cool :)
		 */

		template<typename T, typename Index,
				 typename = decltype(std::declval<T &>()[std::declval<Index>()])>
		std::true_type testSubscript(int);
		template<typename T, typename Index>
		std::false_type testSubscript(float);

		template<typename T, typename V,
				 typename = decltype(std::declval<T &>() + std::declval<V &>())>
		std::true_type testAddition(int);
		template<typename T, typename V>
		std::false_type testAddition(float);

		template<typename T, typename V,
				 typename = decltype(std::declval<T &>() * std::declval<V &>())>
		std::true_type testMultiplication(int);
		template<typename T, typename V>
		std::false_type testMultiplication(float);

		template<typename From, typename To, typename = decltype((From)std::declval<From &>())>
		std::true_type testCast(int);
		template<typename From, typename To>
		std::false_type testCast(float);

		// Test for T::allowVectorisation (static constexpr bool)
		template<typename T, typename = decltype(T::allowVectorisation)>
		std::true_type testAllowVectorisation(int);
		template<typename T>
		std::false_type testAllowVectorisation(float);
	} // namespace impl

	template<typename T, typename Index = int64_t>
	struct HasSubscript : public decltype(impl::testSubscript<T, Index>(1)) {};

	template<typename T, typename V = T>
	struct HasAddition : public decltype(impl::testAddition<T, V>(1)) {};

	template<typename T, typename V = T>
	struct HasMultiplication : public decltype(impl::testMultiplication<T, V>(1)) {};

	template<typename From, typename To>
	struct CanCast : public decltype(impl::testCast<From, To>(1)) {};

	// Detect whether a class can be default constructed
	template<class T>
	using TriviallyDefaultConstructible = std::is_trivially_default_constructible<T>;

	// Detect whether a class has a static constexpr bool member called allowVectorization
	template<typename T>
	struct HasAllowVectorisation : public decltype(impl::testAllowVectorisation<T>(1)) {};
} // namespace librapid::typetraits

#endif // LIBRAPID_CORE_TYPETRAITS_HPP