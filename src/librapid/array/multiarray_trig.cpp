#include <librapid/array/multiarray.hpp>
#include <librapid/array/math_utils.hpp>
#include <librapid/math/rapid_math.hpp>

namespace librapid {
	// A collection of trigonometric kernels for the Array library
	namespace kernels {
		struct sin {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::sin(val);
			}

			std::string name   = "sinKernel";
			std::string kernel = "return sin(a);";
		};

		struct cos {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::cos(val);
			}

			std::string name   = "cosKernel";
			std::string kernel = "return cos(a);";
		};

		struct tan {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::tan(val);
			}

			std::string name   = "tanKernel";
			std::string kernel = "return tan(a);";
		};

		struct asin {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::asin(val);
			}

			std::string name   = "asinKernel";
			std::string kernel = "return asin(a);";
		};

		struct acos {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::acos(val);
			}

			std::string name   = "acosKernel";
			std::string kernel = "return acos(a);";
		};

		struct atan {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::atan(val);
			}

			std::string name   = "atanKernel";
			std::string kernel = "return atan(a);";
		};

		struct sinh {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::sinh(val);
			}

			std::string name   = "sinhKernel";
			std::string kernel = "return sinh(a);";
		};

		struct cosh {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::cosh(val);
			}

			std::string name   = "coshKernel";
			std::string kernel = "return cosh(a);";
		};

		struct tanh {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::tanh(val);
			}

			std::string name   = "tanhKernel";
			std::string kernel = "return tanh(a);";
		};

		struct asinh {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::asinh(val);
			}

			std::string name   = "asinhKernel";
			std::string kernel = "return asinh(a);";
		};

		struct acosh {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::acosh(val);
			}

			std::string name   = "acoshKernel";
			std::string kernel = "return acosh(a);";
		};

		struct atanh {
			template<typename T>
			T operator()(T val, int64_t) const {
				return librapid::atanh(val);
			}

			std::string name   = "atanhKernel";
			std::string kernel = "return atanh(a);";
		};
	} // namespace kernels

	Array sin(const Array &arr) { return arr.mapped(kernels::sin()); }
	Array cos(const Array &arr) { return arr.mapped(kernels::cos()); }
	Array tan(const Array &arr) { return arr.mapped(kernels::tan()); }

	Array asin(const Array &arr) { return arr.mapped(kernels::asin()); }
	Array acos(const Array &arr) { return arr.mapped(kernels::acos()); }
	Array atan(const Array &arr) { return arr.mapped(kernels::atan()); }

	Array sinh(const Array &arr) { return arr.mapped(kernels::sinh()); }
	Array cosh(const Array &arr) { return arr.mapped(kernels::cosh()); }
	Array tanh(const Array &arr) { return arr.mapped(kernels::tanh()); }

	Array asinh(const Array &arr) { return arr.mapped(kernels::asinh()); }
	Array acosh(const Array &arr) { return arr.mapped(kernels::acosh()); }
	Array atanh(const Array &arr) { return arr.mapped(kernels::atanh()); }

} // namespace librapid