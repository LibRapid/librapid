#include <librapid/array/multiarray.hpp>
#include <librapid/math/rapid_math.hpp>
#include <librapid/array/math_utils.hpp>
#include <librapid/patches/math.hpp>

namespace librapid {
	namespace kernels {
		struct pow {
			pow(double pow) : m_pow(pow) {
				kernel = fmt::format("return pow(a, {});", m_pow);
			}

			template<typename A, typename B>
			A operator()(A val, B pow, int64_t, int64_t) const {
				return librapid::pow_numeric_only(val, pow);
			}

			std::string name   = "powKernel";
			std::string kernel = "return 0;";
			double m_pow	   = 0;
		};

		struct sqrt {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::sqrt(val);
			}

			std::string name   = "sqrtKernel";
			std::string kernel = "return sqrt(a);";
		};

		struct exp {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::exp(val);
			}

			std::string name   = "expKernel";
			std::string kernel = "return exp(a);";
		};
	} // namespace kernels

	Array pow(const Array &arr, double power) {
		return Array::applyBinaryOp<false, false>(arr, Array(power), kernels::pow(power));
	}

	Array sqrt(const Array &arr) { return arr.mapped(kernels::sqrt()); }
	Array exp(const Array &arr) { return arr.mapped(kernels::exp()); }
} // namespace librapid