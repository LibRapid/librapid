#ifndef LIBRAPID_SIMD_TRIGONOMETRY
#define LIBRAPID_SIMD_TRIGONOMETRY

namespace librapid {
	namespace typetraits {
		template<typename T>
		struct IsVector : std::false_type {};

		template<typename T>
		struct IsVector<Vc::Vector<T>> : std::true_type {};
	} // namespace typetraits

#define REQUIRE_VECTOR(TYPE) typename std::enable_if_t<typetraits::IsVector<TYPE>::value, int> = 0
#define IF_FLOATING(TYPE)	 if constexpr (std::is_floating_point_v<typename TYPE::value_type>)

	template<typename T, REQUIRE_VECTOR(T)>
	T sin(const T &x) {
		IF_FLOATING(T) { return Vc::sin(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = sin(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T cos(const T &x) {
		IF_FLOATING(T) { return Vc::cos(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = cos(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T tan(const T &x) {
		IF_FLOATING(T) { return Vc::sin(x) / Vc::cos(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = tan(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T asin(const T &x) {
		IF_FLOATING(T) { return Vc::asin(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = asin(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T acos(const T &x) {
		IF_FLOATING(T) {
			static const auto asin1 = Vc::asin(T(1));
			return asin1 - Vc::asin(x);
		}
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = acos(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T atan(const T &x) {
		IF_FLOATING(T) { return Vc::atan(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = atan(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T sinh(const T &x) {
		IF_FLOATING(T) { return (Vc::exp(x) - Vc::exp(-x)) * T(0.5); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = sinh(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T cosh(const T &x) {
		IF_FLOATING(T) { return (Vc::exp(x) + Vc::exp(-x)) * T(0.5); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = cosh(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T tanh(const T &x) {
		IF_FLOATING(T) { return (Vc::exp(2 * x) - 1) / (Vc::exp(2 * x) + 1); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = tanh(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T exp(const T &x) {
		IF_FLOATING(T) { return Vc::exp(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = exp(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T log(const T &x) {
		IF_FLOATING(T) { return Vc::log(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = log(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T log2(const T &x) {
		IF_FLOATING(T) { return Vc::log2(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = log2(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T log10(const T &x) {
		IF_FLOATING(T) { return Vc::log10(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = log10(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T sqrt(const T &x) {
		IF_FLOATING(T) { return Vc::sqrt(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = sqrt(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T cbrt(const T &x) {
		T result;
		for (int i = 0; i < x.size(); ++i) { result[i] = cbrt(x[i]); }
		return result;
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T abs(const T &x) {
		IF_FLOATING(T) { return Vc::abs(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = abs(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T floor(const T &x) {
		IF_FLOATING(T) { return Vc::floor(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = floor(x[i]); }
			return result;
		}
	}

	template<typename T, REQUIRE_VECTOR(T)>
	T ceil(const T &x) {
		IF_FLOATING(T) { return Vc::ceil(x); }
		else {
			T result;
			for (int i = 0; i < x.size(); ++i) { result[i] = ceil(x[i]); }
			return result;
		}
	}
} // namespace librapid

#endif // LIBRAPID_SIMD_TRIGONOMETRY