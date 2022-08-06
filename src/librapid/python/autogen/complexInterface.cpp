
#include <librapid/librapid.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <functional>
#include <string>

// Just remove these. They're pointless
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

namespace lrc = librapid;
namespace py = pybind11;

void init_complex(py::module &module) {
py::class_<librapid::Complex<float>>(module, "ComplexF32")
	.def(py::init<>())
	.def(py::init<int64_t>())
	.def(py::init<int64_t, int64_t>())
	.def(py::init<double>())
	.def(py::init<double, double>())
	.def(py::init<const lrc::mpfr &>())
	.def(py::init<const lrc::mpfr &, const lrc::mpfr &>())
	.def(py::init<const librapid::Complex<float> &>())
	.def("__add__", [](const librapid::Complex<float> & this_, const librapid::Complex<float> & other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::Complex<float> & this_, const librapid::Complex<float> & other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::Complex<float> & this_, const librapid::Complex<float> & other) { return this_ * other; }, py::arg("other"))
	.def("__truediv__", [](const librapid::Complex<float> & this_, const librapid::Complex<float> & other) { return this_ / other; }, py::arg("other"))
	.def("__radd__", [](const librapid::Complex<float> & this_, const librapid::Complex<float> & other) { return other + this_; }, py::arg("other"))
	.def("__rsub__", [](const librapid::Complex<float> & this_, const librapid::Complex<float> & other) { return other - this_; }, py::arg("other"))
	.def("__rmul__", [](const librapid::Complex<float> & this_, const librapid::Complex<float> & other) { return other * this_; }, py::arg("other"))
	.def("__rtruediv__", [](const librapid::Complex<float> & this_, const librapid::Complex<float> & other) { return other / this_; }, py::arg("other"))
	.def("__iadd__", [](librapid::Complex<float> & this_, const librapid::Complex<float> & other) { this_ += other; return this_; }, py::arg("other"))
	.def("__isub__", [](librapid::Complex<float> & this_, const librapid::Complex<float> & other) { this_ -= other; return this_; }, py::arg("other"))
	.def("__imul__", [](librapid::Complex<float> & this_, const librapid::Complex<float> & other) { this_ *= other; return this_; }, py::arg("other"))
	.def("__itruediv__", [](librapid::Complex<float> & this_, const librapid::Complex<float> & other) { this_ /= other; return this_; }, py::arg("other"))
	.def("str", [](const librapid::Complex<float> & this_, int64_t base) { return librapid::str(this_, {-1, base, false}); }, py::arg("base") = int64_t(10))
	.def("__str__", [](const librapid::Complex<float> & this_) { return librapid::str(this_, {-1, 10, false}); })
	.def("__repr__", [](const librapid::Complex<float> & this_) { return "librapid::ComplexF32(\"" + librapid::str(this_, {-1, 10, false}) + "\")"; });

py::class_<librapid::Complex<double>>(module, "ComplexF64")
	.def(py::init<>())
	.def(py::init<int64_t>())
	.def(py::init<int64_t, int64_t>())
	.def(py::init<double>())
	.def(py::init<double, double>())
	.def(py::init<const lrc::mpfr &>())
	.def(py::init<const lrc::mpfr &, const lrc::mpfr &>())
	.def(py::init<const librapid::Complex<double> &>())
	.def("__add__", [](const librapid::Complex<double> & this_, const librapid::Complex<double> & other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::Complex<double> & this_, const librapid::Complex<double> & other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::Complex<double> & this_, const librapid::Complex<double> & other) { return this_ * other; }, py::arg("other"))
	.def("__truediv__", [](const librapid::Complex<double> & this_, const librapid::Complex<double> & other) { return this_ / other; }, py::arg("other"))
	.def("__radd__", [](const librapid::Complex<double> & this_, const librapid::Complex<double> & other) { return other + this_; }, py::arg("other"))
	.def("__rsub__", [](const librapid::Complex<double> & this_, const librapid::Complex<double> & other) { return other - this_; }, py::arg("other"))
	.def("__rmul__", [](const librapid::Complex<double> & this_, const librapid::Complex<double> & other) { return other * this_; }, py::arg("other"))
	.def("__rtruediv__", [](const librapid::Complex<double> & this_, const librapid::Complex<double> & other) { return other / this_; }, py::arg("other"))
	.def("__iadd__", [](librapid::Complex<double> & this_, const librapid::Complex<double> & other) { this_ += other; return this_; }, py::arg("other"))
	.def("__isub__", [](librapid::Complex<double> & this_, const librapid::Complex<double> & other) { this_ -= other; return this_; }, py::arg("other"))
	.def("__imul__", [](librapid::Complex<double> & this_, const librapid::Complex<double> & other) { this_ *= other; return this_; }, py::arg("other"))
	.def("__itruediv__", [](librapid::Complex<double> & this_, const librapid::Complex<double> & other) { this_ /= other; return this_; }, py::arg("other"))
	.def("str", [](const librapid::Complex<double> & this_, int64_t base) { return librapid::str(this_, {-1, base, false}); }, py::arg("base") = int64_t(10))
	.def("__str__", [](const librapid::Complex<double> & this_) { return librapid::str(this_, {-1, 10, false}); })
	.def("__repr__", [](const librapid::Complex<double> & this_) { return "librapid::ComplexF64(\"" + librapid::str(this_, {-1, 10, false}) + "\")"; });

py::class_<librapid::Complex<librapid::mpfr>>(module, "ComplexMPFR")
	.def(py::init<>())
	.def(py::init<int64_t>())
	.def(py::init<int64_t, int64_t>())
	.def(py::init<double>())
	.def(py::init<double, double>())
	.def(py::init<const lrc::mpfr &>())
	.def(py::init<const lrc::mpfr &, const lrc::mpfr &>())
	.def(py::init<const librapid::Complex<librapid::mpfr> &>())
	.def("__add__", [](const librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { return this_ * other; }, py::arg("other"))
	.def("__truediv__", [](const librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { return this_ / other; }, py::arg("other"))
	.def("__radd__", [](const librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { return other + this_; }, py::arg("other"))
	.def("__rsub__", [](const librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { return other - this_; }, py::arg("other"))
	.def("__rmul__", [](const librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { return other * this_; }, py::arg("other"))
	.def("__rtruediv__", [](const librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { return other / this_; }, py::arg("other"))
	.def("__iadd__", [](librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { this_ += other; return this_; }, py::arg("other"))
	.def("__isub__", [](librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { this_ -= other; return this_; }, py::arg("other"))
	.def("__imul__", [](librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { this_ *= other; return this_; }, py::arg("other"))
	.def("__itruediv__", [](librapid::Complex<librapid::mpfr> & this_, const librapid::Complex<librapid::mpfr> & other) { this_ /= other; return this_; }, py::arg("other"))
	.def("str", [](const librapid::Complex<librapid::mpfr> & this_, int64_t base) { return librapid::str(this_, {-1, base, false}); }, py::arg("base") = int64_t(10))
	.def("__str__", [](const librapid::Complex<librapid::mpfr> & this_) { return librapid::str(this_, {-1, 10, false}); })
	.def("__repr__", [](const librapid::Complex<librapid::mpfr> & this_) { return "librapid::ComplexMPFR(\"" + librapid::str(this_, {-1, 10, false}) + "\")"; });



	module.def("abs", [](const librapid::Complex<float> & val) { return librapid::abs(val); }, py::arg("val"));
	module.def("pow", [](float base, const librapid::Complex<float> & power) { return librapid::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("pow", [](const librapid::Complex<float> & base, float power) { return librapid::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("pow", [](const librapid::Complex<float> & base, const librapid::Complex<float> & power) { return librapid::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("sqrt", [](const librapid::Complex<float> & val) { return librapid::sqrt(val); }, py::arg("val"));
	module.def("exp", [](const librapid::Complex<float> & val) { return librapid::exp(val); }, py::arg("val"));
	module.def("exp2", [](const librapid::Complex<float> & val) { return librapid::exp2(val); }, py::arg("val"));
	module.def("exp10", [](const librapid::Complex<float> & val) { return librapid::exp10(val); }, py::arg("val"));
	module.def("log", [](const librapid::Complex<float> & val) { return librapid::log(val); }, py::arg("val"));
	module.def("log2", [](const librapid::Complex<float> & val) { return librapid::log2(val); }, py::arg("val"));
	module.def("log10", [](const librapid::Complex<float> & val) { return librapid::log10(val); }, py::arg("val"));
	module.def("log", [](const librapid::Complex<float> & val, const librapid::Complex<float> & base) { return librapid::log(val, base); }, py::arg("val"), py::arg("base"));
	module.def("log", [](const librapid::Complex<float> & val, float base) { return librapid::log(val, base); }, py::arg("val"), py::arg("base"));
	module.def("sin", [](const librapid::Complex<float> & val) { return librapid::sin(val); }, py::arg("val"));
	module.def("cos", [](const librapid::Complex<float> & val) { return librapid::cos(val); }, py::arg("val"));
	module.def("tan", [](const librapid::Complex<float> & val) { return librapid::tan(val); }, py::arg("val"));
	module.def("asin", [](const librapid::Complex<float> & val) { return librapid::asin(val); }, py::arg("val"));
	module.def("acos", [](const librapid::Complex<float> & val) { return librapid::acos(val); }, py::arg("val"));
	module.def("atan", [](const librapid::Complex<float> & val) { return librapid::atan(val); }, py::arg("val"));
	module.def("atan2", [](const librapid::Complex<float> & a, const librapid::Complex<float> & b) { return librapid::atan2(a, b); }, py::arg("a"), py::arg("b"));
	module.def("csc", [](const librapid::Complex<float> & val) { return librapid::csc(val); }, py::arg("val"));
	module.def("sec", [](const librapid::Complex<float> & val) { return librapid::sec(val); }, py::arg("val"));
	module.def("cot", [](const librapid::Complex<float> & val) { return librapid::cot(val); }, py::arg("val"));
	module.def("acsc", [](const librapid::Complex<float> & val) { return librapid::acsc(val); }, py::arg("val"));
	module.def("asec", [](const librapid::Complex<float> & val) { return librapid::asec(val); }, py::arg("val"));
	module.def("acot", [](const librapid::Complex<float> & val) { return librapid::acot(val); }, py::arg("val"));
	module.def("sinh", [](const librapid::Complex<float> & val) { return librapid::sinh(val); }, py::arg("val"));
	module.def("cosh", [](const librapid::Complex<float> & val) { return librapid::cosh(val); }, py::arg("val"));
	module.def("tanh", [](const librapid::Complex<float> & val) { return librapid::tanh(val); }, py::arg("val"));
	module.def("asinh", [](const librapid::Complex<float> & val) { return librapid::asinh(val); }, py::arg("val"));
	module.def("acosh", [](const librapid::Complex<float> & val) { return librapid::acosh(val); }, py::arg("val"));
	module.def("atanh", [](const librapid::Complex<float> & val) { return librapid::atanh(val); }, py::arg("val"));
	module.def("arg", [](const librapid::Complex<float> & val) { return librapid::arg(val); }, py::arg("val"));
	module.def("proj", [](const librapid::Complex<float> & val) { return librapid::proj(val); }, py::arg("val"));
	module.def("norm", [](const librapid::Complex<float> & val) { return librapid::norm(val); }, py::arg("val"));
	module.def("polar", [](float rho, float theta) { return librapid::polar(rho, theta); }, py::arg("rho"), py::arg("theta"));
	module.def("abs", [](const librapid::Complex<double> & val) { return librapid::abs(val); }, py::arg("val"));
	module.def("pow", [](double base, const librapid::Complex<double> & power) { return librapid::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("pow", [](const librapid::Complex<double> & base, double power) { return librapid::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("pow", [](const librapid::Complex<double> & base, const librapid::Complex<double> & power) { return librapid::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("sqrt", [](const librapid::Complex<double> & val) { return librapid::sqrt(val); }, py::arg("val"));
	module.def("exp", [](const librapid::Complex<double> & val) { return librapid::exp(val); }, py::arg("val"));
	module.def("exp2", [](const librapid::Complex<double> & val) { return librapid::exp2(val); }, py::arg("val"));
	module.def("exp10", [](const librapid::Complex<double> & val) { return librapid::exp10(val); }, py::arg("val"));
	module.def("log", [](const librapid::Complex<double> & val) { return librapid::log(val); }, py::arg("val"));
	module.def("log2", [](const librapid::Complex<double> & val) { return librapid::log2(val); }, py::arg("val"));
	module.def("log10", [](const librapid::Complex<double> & val) { return librapid::log10(val); }, py::arg("val"));
	module.def("log", [](const librapid::Complex<double> & val, const librapid::Complex<double> & base) { return librapid::log(val, base); }, py::arg("val"), py::arg("base"));
	module.def("log", [](const librapid::Complex<double> & val, double base) { return librapid::log(val, base); }, py::arg("val"), py::arg("base"));
	module.def("sin", [](const librapid::Complex<double> & val) { return librapid::sin(val); }, py::arg("val"));
	module.def("cos", [](const librapid::Complex<double> & val) { return librapid::cos(val); }, py::arg("val"));
	module.def("tan", [](const librapid::Complex<double> & val) { return librapid::tan(val); }, py::arg("val"));
	module.def("asin", [](const librapid::Complex<double> & val) { return librapid::asin(val); }, py::arg("val"));
	module.def("acos", [](const librapid::Complex<double> & val) { return librapid::acos(val); }, py::arg("val"));
	module.def("atan", [](const librapid::Complex<double> & val) { return librapid::atan(val); }, py::arg("val"));
	module.def("atan2", [](const librapid::Complex<double> & a, const librapid::Complex<double> & b) { return librapid::atan2(a, b); }, py::arg("a"), py::arg("b"));
	module.def("csc", [](const librapid::Complex<double> & val) { return librapid::csc(val); }, py::arg("val"));
	module.def("sec", [](const librapid::Complex<double> & val) { return librapid::sec(val); }, py::arg("val"));
	module.def("cot", [](const librapid::Complex<double> & val) { return librapid::cot(val); }, py::arg("val"));
	module.def("acsc", [](const librapid::Complex<double> & val) { return librapid::acsc(val); }, py::arg("val"));
	module.def("asec", [](const librapid::Complex<double> & val) { return librapid::asec(val); }, py::arg("val"));
	module.def("acot", [](const librapid::Complex<double> & val) { return librapid::acot(val); }, py::arg("val"));
	module.def("sinh", [](const librapid::Complex<double> & val) { return librapid::sinh(val); }, py::arg("val"));
	module.def("cosh", [](const librapid::Complex<double> & val) { return librapid::cosh(val); }, py::arg("val"));
	module.def("tanh", [](const librapid::Complex<double> & val) { return librapid::tanh(val); }, py::arg("val"));
	module.def("asinh", [](const librapid::Complex<double> & val) { return librapid::asinh(val); }, py::arg("val"));
	module.def("acosh", [](const librapid::Complex<double> & val) { return librapid::acosh(val); }, py::arg("val"));
	module.def("atanh", [](const librapid::Complex<double> & val) { return librapid::atanh(val); }, py::arg("val"));
	module.def("arg", [](const librapid::Complex<double> & val) { return librapid::arg(val); }, py::arg("val"));
	module.def("proj", [](const librapid::Complex<double> & val) { return librapid::proj(val); }, py::arg("val"));
	module.def("norm", [](const librapid::Complex<double> & val) { return librapid::norm(val); }, py::arg("val"));
	module.def("polar", [](double rho, double theta) { return librapid::polar(rho, theta); }, py::arg("rho"), py::arg("theta"));
	module.def("abs", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::abs(val); }, py::arg("val"));
	module.def("pow", [](const librapid::mpfr & base, const librapid::Complex<librapid::mpfr> & power) { return librapid::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("pow", [](const librapid::Complex<librapid::mpfr> & base, const librapid::mpfr & power) { return librapid::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("pow", [](const librapid::Complex<librapid::mpfr> & base, const librapid::Complex<librapid::mpfr> & power) { return librapid::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("sqrt", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::sqrt(val); }, py::arg("val"));
	module.def("exp", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::exp(val); }, py::arg("val"));
	module.def("exp2", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::exp2(val); }, py::arg("val"));
	module.def("exp10", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::exp10(val); }, py::arg("val"));
	module.def("log", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::log(val); }, py::arg("val"));
	module.def("log2", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::log2(val); }, py::arg("val"));
	module.def("log10", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::log10(val); }, py::arg("val"));
	module.def("log", [](const librapid::Complex<librapid::mpfr> & val, const librapid::Complex<librapid::mpfr> & base) { return librapid::log(val, base); }, py::arg("val"), py::arg("base"));
	module.def("log", [](const librapid::Complex<librapid::mpfr> & val, const librapid::mpfr & base) { return librapid::log(val, base); }, py::arg("val"), py::arg("base"));
	module.def("sin", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::sin(val); }, py::arg("val"));
	module.def("cos", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::cos(val); }, py::arg("val"));
	module.def("tan", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::tan(val); }, py::arg("val"));
	module.def("asin", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::asin(val); }, py::arg("val"));
	module.def("acos", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::acos(val); }, py::arg("val"));
	module.def("atan", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::atan(val); }, py::arg("val"));
	module.def("atan2", [](const librapid::Complex<librapid::mpfr> & a, const librapid::Complex<librapid::mpfr> & b) { return librapid::atan2(a, b); }, py::arg("a"), py::arg("b"));
	module.def("csc", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::csc(val); }, py::arg("val"));
	module.def("sec", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::sec(val); }, py::arg("val"));
	module.def("cot", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::cot(val); }, py::arg("val"));
	module.def("acsc", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::acsc(val); }, py::arg("val"));
	module.def("asec", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::asec(val); }, py::arg("val"));
	module.def("acot", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::acot(val); }, py::arg("val"));
	module.def("sinh", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::sinh(val); }, py::arg("val"));
	module.def("cosh", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::cosh(val); }, py::arg("val"));
	module.def("tanh", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::tanh(val); }, py::arg("val"));
	module.def("asinh", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::asinh(val); }, py::arg("val"));
	module.def("acosh", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::acosh(val); }, py::arg("val"));
	module.def("atanh", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::atanh(val); }, py::arg("val"));
	module.def("arg", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::arg(val); }, py::arg("val"));
	module.def("proj", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::proj(val); }, py::arg("val"));
	module.def("norm", [](const librapid::Complex<librapid::mpfr> & val) { return librapid::norm(val); }, py::arg("val"));
	module.def("polar", [](const librapid::mpfr & rho, const librapid::mpfr & theta) { return librapid::polar(rho, theta); }, py::arg("rho"), py::arg("theta"));

}