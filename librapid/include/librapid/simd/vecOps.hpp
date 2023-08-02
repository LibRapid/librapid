#ifndef LIBRAPID_SIMD_TRIGONOMETRY
#define LIBRAPID_SIMD_TRIGONOMETRY

namespace librapid {
	namespace typetraits {
		template<typename T>
		struct IsVector : std::false_type {};

		// template<typename T>
		// struct IsVector<Vc::Vector<T>> : std::true_type {};

		template<typename T, typename U>
		struct IsVector<Vc_1::Vector<T, U>> : std::true_type {};
	} // namespace typetraits

#define REQUIRE_VECTOR(TYPE) typename std::enable_if_t<typetraits::IsVector<TYPE>::value, int> = 0
#define IF_FLOATING(TYPE)	 if constexpr (std::is_floating_point_v<typename TYPE::value_type>)

	template<typename T, REQUIRE_VECTOR(T)>
	auto sin(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::sin(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = sin(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto cos(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::cos(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = cos(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto tan(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::sin(x) / Vc::cos(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = tan(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto asin(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::asin(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = asin(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto acos(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) {
			static const auto asin1 = Vc::asin(T(1));
			return asin1 - Vc::asin(x);
		}
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = acos(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto atan(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::atan(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = atan(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto sinh(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return (Vc::exp(x) - Vc::exp(-x)) * T(0.5); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = sinh(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto cosh(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return (Vc::exp(x) + Vc::exp(-x)) * T(0.5); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = cosh(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto tanh(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return (Vc::exp(2 * x) - 1) / (Vc::exp(2 * x) + 1); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = tanh(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto exp(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::exp(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = exp(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto log(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::log(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = log(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto log2(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::log2(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = log2(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto log10(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::log10(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = log10(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto sqrt(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::sqrt(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = sqrt(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto cbrt(const T &x) -> T {
		using Scalar = typename T::value_type;
		T result;
		for (int i = 0; i < x.size(); ++i) { result[i] = cbrt(static_cast<Scalar>(x[i])); }
		return result;
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto abs(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::abs(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = abs(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto floor(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::floor(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = floor(static_cast<Scalar>(x[i])); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	auto ceil(const T &x) -> T {
		using Scalar = typename T::value_type;
		IF_FLOATING(T) { return Vc::ceil(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = ceil(static_cast<Scalar>(x[i])); }
			return result;
		}
	}
} // namespace librapid

#endif // LIBRAPID_SIMD_TRIGONOMETRY