
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

void init_math(py::module &module) {
	module.def("abs", [](const int64_t & val) { return lrc::abs(val); }, py::arg("val"));
	module.def("floor", [](const int64_t & val) { return lrc::floor(val); }, py::arg("val"));
	module.def("ceil", [](const int64_t & val) { return lrc::ceil(val); }, py::arg("val"));
	module.def("pow", [](const int64_t & base, const int64_t & power) { return lrc::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("sqrt", [](const int64_t & val) { return lrc::sqrt(val); }, py::arg("val"));
	module.def("exp", [](const int64_t & val) { return lrc::exp(val); }, py::arg("val"));
	module.def("exp2", [](const int64_t & val) { return lrc::exp2(val); }, py::arg("val"));
	module.def("exp10", [](const int64_t & val) { return lrc::exp10(val); }, py::arg("val"));
	module.def("ln", [](const int64_t & val) { return lrc::ln(val); }, py::arg("val"));
	module.def("log2", [](const int64_t & val) { return lrc::log2(val); }, py::arg("val"));
	module.def("log10", [](const int64_t & val) { return lrc::log10(val); }, py::arg("val"));
	module.def("log", [](const int64_t & val, const int64_t & base) { return lrc::log(val, base); }, py::arg("val"), py::arg("base"));
	module.def("sin", [](const int64_t & val) { return lrc::sin(val); }, py::arg("val"));
	module.def("cos", [](const int64_t & val) { return lrc::cos(val); }, py::arg("val"));
	module.def("tan", [](const int64_t & val) { return lrc::tan(val); }, py::arg("val"));
	module.def("asin", [](const int64_t & val) { return lrc::asin(val); }, py::arg("val"));
	module.def("acos", [](const int64_t & val) { return lrc::acos(val); }, py::arg("val"));
	module.def("atan", [](const int64_t & val) { return lrc::atan(val); }, py::arg("val"));
	module.def("csc", [](const int64_t & val) { return lrc::csc(val); }, py::arg("val"));
	module.def("sec", [](const int64_t & val) { return lrc::sec(val); }, py::arg("val"));
	module.def("cot", [](const int64_t & val) { return lrc::cot(val); }, py::arg("val"));
	module.def("acsc", [](const int64_t & val) { return lrc::acsc(val); }, py::arg("val"));
	module.def("asec", [](const int64_t & val) { return lrc::asec(val); }, py::arg("val"));
	module.def("acot", [](const int64_t & val) { return lrc::acot(val); }, py::arg("val"));
	module.def("sinh", [](const int64_t & val) { return lrc::sinh(val); }, py::arg("val"));
	module.def("cosh", [](const int64_t & val) { return lrc::cosh(val); }, py::arg("val"));
	module.def("tanh", [](const int64_t & val) { return lrc::tanh(val); }, py::arg("val"));
	module.def("asinh", [](const int64_t & val) { return lrc::asinh(val); }, py::arg("val"));
	module.def("acosh", [](const int64_t & val) { return lrc::acosh(val); }, py::arg("val"));
	module.def("atanh", [](const int64_t & val) { return lrc::atanh(val); }, py::arg("val"));
	module.def("mod", [](const int64_t & val, const int64_t & divisor) { return lrc::mod(val, divisor); }, py::arg("val"), py::arg("divisor"));
	module.def("round", [](const int64_t & val, int64_t dp) { return lrc::round(val, dp); }, py::arg("val"), py::arg("dp") = int64_t(0));
	module.def("roundSigFig", [](const int64_t & val, int64_t dp) { return lrc::roundSigFig(val, dp); }, py::arg("val"), py::arg("dp") = int64_t(3));
	module.def("roundTo", [](const int64_t & val, const int64_t & num) { return lrc::roundTo(val, num); }, py::arg("val"), py::arg("num") = int64_t(0));
	module.def("roundUpTo", [](const int64_t & val, const int64_t & num) { return lrc::roundUpTo(val, num); }, py::arg("val"), py::arg("num") = int64_t(0));
	module.def("abs", [](const double & val) { return lrc::abs(val); }, py::arg("val"));
	module.def("floor", [](const double & val) { return lrc::floor(val); }, py::arg("val"));
	module.def("ceil", [](const double & val) { return lrc::ceil(val); }, py::arg("val"));
	module.def("pow", [](const double & base, const double & power) { return lrc::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("sqrt", [](const double & val) { return lrc::sqrt(val); }, py::arg("val"));
	module.def("exp", [](const double & val) { return lrc::exp(val); }, py::arg("val"));
	module.def("exp2", [](const double & val) { return lrc::exp2(val); }, py::arg("val"));
	module.def("exp10", [](const double & val) { return lrc::exp10(val); }, py::arg("val"));
	module.def("ln", [](const double & val) { return lrc::ln(val); }, py::arg("val"));
	module.def("log2", [](const double & val) { return lrc::log2(val); }, py::arg("val"));
	module.def("log10", [](const double & val) { return lrc::log10(val); }, py::arg("val"));
	module.def("log", [](const double & val, const double & base) { return lrc::log(val, base); }, py::arg("val"), py::arg("base"));
	module.def("sin", [](const double & val) { return lrc::sin(val); }, py::arg("val"));
	module.def("cos", [](const double & val) { return lrc::cos(val); }, py::arg("val"));
	module.def("tan", [](const double & val) { return lrc::tan(val); }, py::arg("val"));
	module.def("asin", [](const double & val) { return lrc::asin(val); }, py::arg("val"));
	module.def("acos", [](const double & val) { return lrc::acos(val); }, py::arg("val"));
	module.def("atan", [](const double & val) { return lrc::atan(val); }, py::arg("val"));
	module.def("csc", [](const double & val) { return lrc::csc(val); }, py::arg("val"));
	module.def("sec", [](const double & val) { return lrc::sec(val); }, py::arg("val"));
	module.def("cot", [](const double & val) { return lrc::cot(val); }, py::arg("val"));
	module.def("acsc", [](const double & val) { return lrc::acsc(val); }, py::arg("val"));
	module.def("asec", [](const double & val) { return lrc::asec(val); }, py::arg("val"));
	module.def("acot", [](const double & val) { return lrc::acot(val); }, py::arg("val"));
	module.def("sinh", [](const double & val) { return lrc::sinh(val); }, py::arg("val"));
	module.def("cosh", [](const double & val) { return lrc::cosh(val); }, py::arg("val"));
	module.def("tanh", [](const double & val) { return lrc::tanh(val); }, py::arg("val"));
	module.def("asinh", [](const double & val) { return lrc::asinh(val); }, py::arg("val"));
	module.def("acosh", [](const double & val) { return lrc::acosh(val); }, py::arg("val"));
	module.def("atanh", [](const double & val) { return lrc::atanh(val); }, py::arg("val"));
	module.def("mod", [](const double & val, const double & divisor) { return lrc::mod(val, divisor); }, py::arg("val"), py::arg("divisor"));
	module.def("round", [](const double & val, int64_t dp) { return lrc::round(val, dp); }, py::arg("val"), py::arg("dp") = int64_t(0));
	module.def("roundSigFig", [](const double & val, int64_t dp) { return lrc::roundSigFig(val, dp); }, py::arg("val"), py::arg("dp") = int64_t(3));
	module.def("roundTo", [](const double & val, const double & num) { return lrc::roundTo(val, num); }, py::arg("val"), py::arg("num") = double(0));
	module.def("roundUpTo", [](const double & val, const double & num) { return lrc::roundUpTo(val, num); }, py::arg("val"), py::arg("num") = double(0));
	module.def("abs", [](const librapid::mpfr & val) { return lrc::abs(val); }, py::arg("val"));
	module.def("floor", [](const librapid::mpfr & val) { return lrc::floor(val); }, py::arg("val"));
	module.def("ceil", [](const librapid::mpfr & val) { return lrc::ceil(val); }, py::arg("val"));
	module.def("pow", [](const librapid::mpfr & base, const librapid::mpfr & power) { return lrc::pow(base, power); }, py::arg("base"), py::arg("power"));
	module.def("sqrt", [](const librapid::mpfr & val) { return lrc::sqrt(val); }, py::arg("val"));
	module.def("exp", [](const librapid::mpfr & val) { return lrc::exp(val); }, py::arg("val"));
	module.def("exp2", [](const librapid::mpfr & val) { return lrc::exp2(val); }, py::arg("val"));
	module.def("exp10", [](const librapid::mpfr & val) { return lrc::exp10(val); }, py::arg("val"));
	module.def("ln", [](const librapid::mpfr & val) { return lrc::ln(val); }, py::arg("val"));
	module.def("log2", [](const librapid::mpfr & val) { return lrc::log2(val); }, py::arg("val"));
	module.def("log10", [](const librapid::mpfr & val) { return lrc::log10(val); }, py::arg("val"));
	module.def("log", [](const librapid::mpfr & val, const librapid::mpfr & base) { return lrc::log(val, base); }, py::arg("val"), py::arg("base"));
	module.def("sin", [](const librapid::mpfr & val) { return lrc::sin(val); }, py::arg("val"));
	module.def("cos", [](const librapid::mpfr & val) { return lrc::cos(val); }, py::arg("val"));
	module.def("tan", [](const librapid::mpfr & val) { return lrc::tan(val); }, py::arg("val"));
	module.def("asin", [](const librapid::mpfr & val) { return lrc::asin(val); }, py::arg("val"));
	module.def("acos", [](const librapid::mpfr & val) { return lrc::acos(val); }, py::arg("val"));
	module.def("atan", [](const librapid::mpfr & val) { return lrc::atan(val); }, py::arg("val"));
	module.def("csc", [](const librapid::mpfr & val) { return lrc::csc(val); }, py::arg("val"));
	module.def("sec", [](const librapid::mpfr & val) { return lrc::sec(val); }, py::arg("val"));
	module.def("cot", [](const librapid::mpfr & val) { return lrc::cot(val); }, py::arg("val"));
	module.def("acsc", [](const librapid::mpfr & val) { return lrc::acsc(val); }, py::arg("val"));
	module.def("asec", [](const librapid::mpfr & val) { return lrc::asec(val); }, py::arg("val"));
	module.def("acot", [](const librapid::mpfr & val) { return lrc::acot(val); }, py::arg("val"));
	module.def("sinh", [](const librapid::mpfr & val) { return lrc::sinh(val); }, py::arg("val"));
	module.def("cosh", [](const librapid::mpfr & val) { return lrc::cosh(val); }, py::arg("val"));
	module.def("tanh", [](const librapid::mpfr & val) { return lrc::tanh(val); }, py::arg("val"));
	module.def("asinh", [](const librapid::mpfr & val) { return lrc::asinh(val); }, py::arg("val"));
	module.def("acosh", [](const librapid::mpfr & val) { return lrc::acosh(val); }, py::arg("val"));
	module.def("atanh", [](const librapid::mpfr & val) { return lrc::atanh(val); }, py::arg("val"));
	module.def("mod", [](const librapid::mpfr & val, const librapid::mpfr & divisor) { return lrc::mod(val, divisor); }, py::arg("val"), py::arg("divisor"));
	module.def("round", [](const librapid::mpfr & val, int64_t dp) { return lrc::round(val, dp); }, py::arg("val"), py::arg("dp") = int64_t(0));
	module.def("roundSigFig", [](const librapid::mpfr & val, int64_t dp) { return lrc::roundSigFig(val, dp); }, py::arg("val"), py::arg("dp") = int64_t(3));
	module.def("roundTo", [](const librapid::mpfr & val, const librapid::mpfr & num) { return lrc::roundTo(val, num); }, py::arg("val"), py::arg("num") = librapid::mpfr(0));
	module.def("roundUpTo", [](const librapid::mpfr & val, const librapid::mpfr & num) { return lrc::roundUpTo(val, num); }, py::arg("val"), py::arg("num") = librapid::mpfr(0));
	module.def("map", [](double val, double start1, double stop1, double start2, double stop2) { return lrc::map(val, start1, stop1, start2, stop2); }, py::arg("val"), py::arg("start1"), py::arg("stop1"), py::arg("start2"), py::arg("stop2"));
	module.def("random", [](double lower, double upper, int64_t seed) { return librapid::random(lower, upper, seed); }, py::arg("lower") = double(0), py::arg("upper") = double(1), py::arg("seed") = int64_t(-1));
	module.def("randint", [](int64_t lower, int64_t upper, int64_t seed) { return librapid::randint(lower, upper, seed); }, py::arg("lower") = int64_t(0), py::arg("upper") = int64_t(0), py::arg("seed") = int64_t(-1));
	module.def("trueRandomEntropy", []() { return librapid::trueRandomEntropy(); });
	module.def("trueRandom", [](double lower, double upper) { return librapid::trueRandom(lower, upper); }, py::arg("lower") = double(0), py::arg("upper") = double(1));
	module.def("trueRandint", [](int64_t lower, int64_t upper) { return librapid::trueRandint(lower, upper); }, py::arg("lower") = int64_t(0), py::arg("upper") = int64_t(1));
	module.def("randomGaussian", []() { return librapid::randomGaussian(); });
	module.def("pow10", [](int64_t exponent) { return librapid::pow10(exponent); }, py::arg("exponent"));
	module.def("lerp", [](double a, double b, double t) { return librapid::lerp(a, b, t); }, py::arg("a"), py::arg("b"), py::arg("t"));

}