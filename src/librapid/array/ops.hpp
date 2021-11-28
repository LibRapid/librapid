#ifndef LIBRAPID_OPS
#define LIBRAPID_OPS

namespace librapid::ops {
	struct Copy {
		std::string name = "copy";
		std::string kernel = R"V0G0N(
				return a;
			)V0G0N";

		template<typename A>
		auto operator()(A a, int64_t) const {
			return a;
		}
	};

	struct Fill {
		std::string name = "fill";
		std::string kernel = R"V0G0N(
				return b;
			)V0G0N";

		template<typename A, typename B>
		auto operator()(A, B b, int64_t, int64_t) const {
			return b;
		}
	};

	template<typename T = double>
	struct FillRandom {
		FillRandom(T minVal = 0, T maxVal = 1, uint64_t rngSeed = -1) : min(minVal), max(maxVal),
																		seed(seed == -1 ? (int64_t) (seconds() * 10)
																						: rngSeed) {
			kernel = "if constexpr (std::is_same<A, double>::value) {\n";
			kernel += "double randNum = curand_uniform_double(_curandState) * (";
			kernel += std::to_string(max - min - std::numeric_limits<T>::epsilon()) +
					  " + std::is_integral<T_DST>::value) + ";
			kernel += std::to_string(min) + ";\n";

			kernel += "return randNum;\n}\n";

			kernel += "else {\n";
			kernel += "float randNum = curand_uniform(_curandState) * (";
			kernel += std::to_string(max - min - std::numeric_limits<T>::epsilon()) +
					  " + std::is_integral<T_DST>::value) + ";
			kernel += std::to_string(min) + ";\n";

			kernel += "return randNum;\n}\n";
		}

		std::string name = "fillRandom";
		std::string kernel = R"V0G0N(
				return 0;
			)V0G0N";

		template<typename A>
		auto operator()(A, int64_t) const {
			return random((A) min, (A) max, seed);
		}

		T min;
		T max;
		uint64_t seed;
	};

	template<>
	struct FillRandom<Complex < double>> {
	FillRandom(const Complex<double> &min = 0, const Complex<double> &max = 1, uint64_t seed = -1) : min(min),
																									 max(max),
																									 seed(seed) {
		kernel = "double randNumReal = curand_uniform_double(_curandState) * (";

		kernel += std::to_string(max.real() - min.real() - std::numeric_limits<double>::epsilon()) +
				  " + std::is_integral<T_DST>::value) + ";
		kernel += std::to_string(min.real()) + ";";

		kernel += "\ndouble randNumImag = curand_uniform_double(&state[indexA / 64]) * ";

		kernel += std::to_string(max.imag() - min.imag() - std::numeric_limits<double>::epsilon()) + " + ";
		kernel += std::to_string(min.imag()) + ";";

		kernel += R"V0G0N(
                        return librapid::Complex<double>(randNumReal, randNumImag);
                    )V0G0N";
	}

	std::string name = "fillRandomComplex";
	std::string kernel = R"V0G0N(
                        return 0;
                    )V0G0N";

	template<typename A>
	auto operator()(A, int64_t) const {
		return random((A) min, (A) max, seed);
	}

	Complex<double> min;
	Complex<double> max;
	uint64_t seed;
};

struct Add {
	std::string name = "add";
	std::string kernel = R"V0G0N(
					return a + b;
				)V0G0N";

	template<typename A, typename B>
	auto operator()(A a, B b, int64_t, int64_t) const {
		return a + b;
	}
};

struct Sub {
	std::string name = "sub";
	std::string kernel = R"V0G0N(
					return a - b;
				)V0G0N";

	template<typename A, typename B>
	auto operator()(A a, B b, int64_t, int64_t) const {
		return a - b;
	}
};

struct Mul {
	std::string name = "mul";
	std::string kernel = R"V0G0N(
					return a * b;
				)V0G0N";

	template<typename A, typename B>
	auto operator()(A a, B b, int64_t, int64_t) const {
		return a * b;
	}
};

struct Div {
	std::string name = "div";
	std::string kernel = R"V0G0N(
					return a / b;
				)V0G0N";

	template<typename A, typename B>
	auto operator()(A a, B b, int64_t, int64_t) const {
		return a / b;
	}
};

template<typename T>
uint64_t getSeed(T x) {
	return 0;
}

template<typename T>
uint64_t getSeed(const FillRandom <T> &x) {
	return x.seed;
}

}

#endif // LIBRAPID_OPS