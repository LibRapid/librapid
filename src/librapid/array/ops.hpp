#ifndef LIBRAPID_OPS
#define LIBRAPID_OPS

#ifndef DOXYGEN_BUILD

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

	template<typename T = double> struct FillRandom {
		FillRandom(T minVal = 0, T maxVal = 1, uint64_t rngSeed = -1)
				: min(minVal),
				  max(maxVal),
				  seed(seed == -1 ? (int64_t) (seconds() * 10) : rngSeed) {
			// No format   => 0.081 us
			// std::string => 1.965 us
			// fmt::format => 0.680 us

//            kernel = "if constexpr (std::is_same<A, double>::value) {\n";
//            kernel += "double randNum = curand_uniform_double(_curandState) * (";
//            kernel += std::to_string(max - min - std::numeric_limits<T>::epsilon()) +
//                      " + std::is_integral<T_DST>::value) + ";
//            kernel += std::to_string(min) + ";\n";
//
//            kernel += "return randNum;\n}\n";
//
//            kernel += "else {\n";
//            kernel += "float randNum = curand_uniform(_curandState) * (";
//            kernel += std::to_string(max - min - std::numeric_limits<T>::epsilon()) +
//                      " + std::is_integral<T_DST>::value) + ";
//            kernel += std::to_string(min) + ";\n";
//
//            kernel += "return randNum;\n}\n";

			kernel = fmt::format(R"V0G0N(
									if constexpr (std::is_same<A, double>::value) {{
										double randNum = curand_uniform_double(_curandState) * {0}
														 + int(std::is_integral<T_DST>::value) + {1};
										return randNum;
									}} else {{
										float randNum = curand_uniform(_curandState) * {0}
														+ int(std::is_integral<T_DST>::value) + {1};
										return randNum;
									}}
								 )V0G0N", max - min - std::numeric_limits<T>::epsilon(), min);
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

	template<> struct FillRandom<Complex < double>> {
	FillRandom(const Complex<double> &min = 0, const Complex<double> &max = 1, uint64_t seed = -1)
			: min(min), max(max), seed(seed) {
//		kernel = "double randNumReal = curand_uniform_double(_curandState) * (";
//
//		kernel
//				+=
//				std::to_string(max.real() - min.real() - std::numeric_limits<double>::epsilon()) +
//				" + std::is_integral<T_DST>::value) + ";
//		kernel += std::to_string(min.real()) + ";";
//
//		kernel += "\ndouble randNumImag = curand_uniform_double(&state[indexA / 64]) * ";
//
//		kernel
//				+=
//				std::to_string(max.imag() - min.imag() - std::numeric_limits<double>::epsilon()) +
//				" + ";
//		kernel += std::to_string(min.imag()) + ";";
//
//		kernel += R"V0G0N(
//                        return librapid::Complex<double>(randNumReal, randNumImag);
//                    )V0G0N";

		kernel = fmt::format(R"V0G0N(
		 						  double randNumReal = curand_uniform_double(_curandState) * {0}
		 						  				 + int(std::is_integral<T_DST>::value) + {1};
		 						  double randNumImag = curand_uniform_double(_curandState) * {2}
		 						  				 + int(std::is_integral<T_DST>::value) + {3};
		 						  return librapid::Complex<double>(randNumReal, randNumImag);
		 						 )V0G0N",
							 max.real() - min.real() - std::numeric_limits<double>::epsilon(),
							 min.real(),
							 max.imag() - min.imag() - std::numeric_limits<double>::epsilon(),
							 min.imag());
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

struct Negate {
	std::string name = "negate";
	std::string kernel = R"V0G0N(
					return -a;
				)V0G0N";

	template<typename A>
	auto operator()(A a, int64_t) const {
		return -a;
	}
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

#endif // DOXYGEN_BUILD

namespace librapid {

// A lightweight wrapper for a GPU kernel
	class GPUKernel {
	public:
		static uint64_t kernelsCreated;

		GPUKernel() = default;

		explicit GPUKernel(const std::string &kernel_) {
			name = fmt::format("gpuKernel{}", kernelsCreated++);
			kernel = kernel_;
		}

		GPUKernel(const std::string &name_, const std::string &kernel_) {
			name = name_;
			kernel = kernel_;
		}

		template<typename... Pack>
		double operator()(Pack...) const {
			throw std::runtime_error(fmt::format(
					"Cannot apply GPUKernel '{}' operation to a CPU-based array",
					kernel));
		}

		[[nodiscard]] std::string str() const {
			return fmt::format("Name => {}\n{}", name, kernel);
		}

		[[nodiscard]] inline const std::string &getName() const {
			return name;
		}

		[[nodiscard]] inline const std::string &getKernel() const {
			return kernel;
		}

		std::string name;
		std::string kernel;
	};

	inline uint64_t GPUKernel::kernelsCreated = 0;
}

#endif // LIBRAPID_OPS