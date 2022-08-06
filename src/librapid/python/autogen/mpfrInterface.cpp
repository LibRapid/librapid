
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

void init_mpfr(py::module &module) {
py::class_<librapid::mpz>(module, "mpz")
	.def(py::init<>())
	.def(py::init<int64_t>())
	.def(py::init<double>())
	.def(py::init<const std::string &>())
	.def(py::init<const librapid::mpz &>())
	.def("__add__", [](const librapid::mpz & this_, const librapid::mpz & other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::mpz & this_, const librapid::mpz & other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::mpz & this_, const librapid::mpz & other) { return this_ * other; }, py::arg("other"))
	.def("__truediv__", [](const librapid::mpz & this_, const librapid::mpz & other) { return this_ / other; }, py::arg("other"))
	.def("__radd__", [](const librapid::mpz & this_, const librapid::mpz & other) { return other + this_; }, py::arg("other"))
	.def("__rsub__", [](const librapid::mpz & this_, const librapid::mpz & other) { return other - this_; }, py::arg("other"))
	.def("__rmul__", [](const librapid::mpz & this_, const librapid::mpz & other) { return other * this_; }, py::arg("other"))
	.def("__rtruediv__", [](const librapid::mpz & this_, const librapid::mpz & other) { return other / this_; }, py::arg("other"))
	.def("__iadd__", [](librapid::mpz & this_, const librapid::mpz & other) { this_ += other; return this_; }, py::arg("other"))
	.def("__isub__", [](librapid::mpz & this_, const librapid::mpz & other) { this_ -= other; return this_; }, py::arg("other"))
	.def("__imul__", [](librapid::mpz & this_, const librapid::mpz & other) { this_ *= other; return this_; }, py::arg("other"))
	.def("__itruediv__", [](librapid::mpz & this_, const librapid::mpz & other) { this_ /= other; return this_; }, py::arg("other"))
	.def("__lt__", [](const librapid::mpz & this_, const librapid::mpz & other) { return this_ < other; }, py::arg("other"))
	.def("__gt__", [](const librapid::mpz & this_, const librapid::mpz & other) { return this_ > other; }, py::arg("other"))
	.def("__lte__", [](const librapid::mpz & this_, const librapid::mpz & other) { return this_ <= other; }, py::arg("other"))
	.def("__gte__", [](const librapid::mpz & this_, const librapid::mpz & other) { return this_ >= other; }, py::arg("other"))
	.def("str", [](const librapid::mpz & this_, int64_t base) { return lrc::str(this_, {-1, base, false}); }, py::arg("base") = int64_t(10))
	.def("__str__", [](const librapid::mpz & this_) { return lrc::str(this_, {-1, 10, false}); })
	.def("__repr__", [](const librapid::mpz & this_) { return "librapid::mpz(\"" + lrc::str(this_, {-1, 10, false}) + "\")"; })
	.def("__lshift__", [](const librapid::mpz & this_, int64_t other) { return this_ << other; }, py::arg("other"))
	.def("__rshift__", [](const librapid::mpz & this_, int64_t other) { return this_ >> other; }, py::arg("other"))
	.def("__ilshift__", [](librapid::mpz & this_, int64_t other) { this_ <<= other; return this_; }, py::arg("other"))
	.def("__irshift__", [](librapid::mpz & this_, int64_t other) { this_ >>= other; return this_; }, py::arg("other"));

py::class_<librapid::mpf>(module, "mpf")
	.def(py::init<>())
	.def(py::init<int64_t>())
	.def(py::init<double>())
	.def(py::init<const std::string &>())
	.def(py::init<const librapid::mpf &>())
	.def("__add__", [](const librapid::mpf & this_, const librapid::mpf & other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::mpf & this_, const librapid::mpf & other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::mpf & this_, const librapid::mpf & other) { return this_ * other; }, py::arg("other"))
	.def("__truediv__", [](const librapid::mpf & this_, const librapid::mpf & other) { return this_ / other; }, py::arg("other"))
	.def("__radd__", [](const librapid::mpf & this_, const librapid::mpf & other) { return other + this_; }, py::arg("other"))
	.def("__rsub__", [](const librapid::mpf & this_, const librapid::mpf & other) { return other - this_; }, py::arg("other"))
	.def("__rmul__", [](const librapid::mpf & this_, const librapid::mpf & other) { return other * this_; }, py::arg("other"))
	.def("__rtruediv__", [](const librapid::mpf & this_, const librapid::mpf & other) { return other / this_; }, py::arg("other"))
	.def("__iadd__", [](librapid::mpf & this_, const librapid::mpf & other) { this_ += other; return this_; }, py::arg("other"))
	.def("__isub__", [](librapid::mpf & this_, const librapid::mpf & other) { this_ -= other; return this_; }, py::arg("other"))
	.def("__imul__", [](librapid::mpf & this_, const librapid::mpf & other) { this_ *= other; return this_; }, py::arg("other"))
	.def("__itruediv__", [](librapid::mpf & this_, const librapid::mpf & other) { this_ /= other; return this_; }, py::arg("other"))
	.def("__lt__", [](const librapid::mpf & this_, const librapid::mpf & other) { return this_ < other; }, py::arg("other"))
	.def("__gt__", [](const librapid::mpf & this_, const librapid::mpf & other) { return this_ > other; }, py::arg("other"))
	.def("__lte__", [](const librapid::mpf & this_, const librapid::mpf & other) { return this_ <= other; }, py::arg("other"))
	.def("__gte__", [](const librapid::mpf & this_, const librapid::mpf & other) { return this_ >= other; }, py::arg("other"))
	.def("str", [](const librapid::mpf & this_, int64_t base) { return lrc::str(this_, {-1, base, false}); }, py::arg("base") = int64_t(10))
	.def("__str__", [](const librapid::mpf & this_) { return lrc::str(this_, {-1, 10, false}); })
	.def("__repr__", [](const librapid::mpf & this_) { return "librapid::mpf(\"" + lrc::str(this_, {-1, 10, false}) + "\")"; })
	.def("__lshift__", [](const librapid::mpf & this_, int64_t other) { return this_ << other; }, py::arg("other"))
	.def("__rshift__", [](const librapid::mpf & this_, int64_t other) { return this_ >> other; }, py::arg("other"))
	.def("__ilshift__", [](librapid::mpf & this_, int64_t other) { this_ <<= other; return this_; }, py::arg("other"))
	.def("__irshift__", [](librapid::mpf & this_, int64_t other) { this_ >>= other; return this_; }, py::arg("other"));

py::class_<librapid::mpq>(module, "mpq")
	.def(py::init<>())
	.def(py::init<int64_t>())
	.def(py::init<double>())
	.def(py::init<const std::string &>())
	.def(py::init<const librapid::mpq &>())
	.def("__add__", [](const librapid::mpq & this_, const librapid::mpq & other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::mpq & this_, const librapid::mpq & other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::mpq & this_, const librapid::mpq & other) { return this_ * other; }, py::arg("other"))
	.def("__truediv__", [](const librapid::mpq & this_, const librapid::mpq & other) { return this_ / other; }, py::arg("other"))
	.def("__radd__", [](const librapid::mpq & this_, const librapid::mpq & other) { return other + this_; }, py::arg("other"))
	.def("__rsub__", [](const librapid::mpq & this_, const librapid::mpq & other) { return other - this_; }, py::arg("other"))
	.def("__rmul__", [](const librapid::mpq & this_, const librapid::mpq & other) { return other * this_; }, py::arg("other"))
	.def("__rtruediv__", [](const librapid::mpq & this_, const librapid::mpq & other) { return other / this_; }, py::arg("other"))
	.def("__iadd__", [](librapid::mpq & this_, const librapid::mpq & other) { this_ += other; return this_; }, py::arg("other"))
	.def("__isub__", [](librapid::mpq & this_, const librapid::mpq & other) { this_ -= other; return this_; }, py::arg("other"))
	.def("__imul__", [](librapid::mpq & this_, const librapid::mpq & other) { this_ *= other; return this_; }, py::arg("other"))
	.def("__itruediv__", [](librapid::mpq & this_, const librapid::mpq & other) { this_ /= other; return this_; }, py::arg("other"))
	.def("__lt__", [](const librapid::mpq & this_, const librapid::mpq & other) { return this_ < other; }, py::arg("other"))
	.def("__gt__", [](const librapid::mpq & this_, const librapid::mpq & other) { return this_ > other; }, py::arg("other"))
	.def("__lte__", [](const librapid::mpq & this_, const librapid::mpq & other) { return this_ <= other; }, py::arg("other"))
	.def("__gte__", [](const librapid::mpq & this_, const librapid::mpq & other) { return this_ >= other; }, py::arg("other"))
	.def("str", [](const librapid::mpq & this_, int64_t base) { return lrc::str(this_, {-1, base, false}); }, py::arg("base") = int64_t(10))
	.def("__str__", [](const librapid::mpq & this_) { return lrc::str(this_, {-1, 10, false}); })
	.def("__repr__", [](const librapid::mpq & this_) { return "librapid::mpq(\"" + lrc::str(this_, {-1, 10, false}) + "\")"; })
	.def("__lshift__", [](const librapid::mpq & this_, int64_t other) { return this_ << other; }, py::arg("other"))
	.def("__rshift__", [](const librapid::mpq & this_, int64_t other) { return this_ >> other; }, py::arg("other"))
	.def("__ilshift__", [](librapid::mpq & this_, int64_t other) { this_ <<= other; return this_; }, py::arg("other"))
	.def("__irshift__", [](librapid::mpq & this_, int64_t other) { this_ >>= other; return this_; }, py::arg("other"));

py::class_<librapid::mpfr>(module, "mpfr")
	.def(py::init<>())
	.def(py::init<int64_t>())
	.def(py::init<double>())
	.def(py::init<const std::string &>())
	.def(py::init<const librapid::mpfr &>())
	.def("__add__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return this_ * other; }, py::arg("other"))
	.def("__truediv__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return this_ / other; }, py::arg("other"))
	.def("__radd__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return other + this_; }, py::arg("other"))
	.def("__rsub__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return other - this_; }, py::arg("other"))
	.def("__rmul__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return other * this_; }, py::arg("other"))
	.def("__rtruediv__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return other / this_; }, py::arg("other"))
	.def("__iadd__", [](librapid::mpfr & this_, const librapid::mpfr & other) { this_ += other; return this_; }, py::arg("other"))
	.def("__isub__", [](librapid::mpfr & this_, const librapid::mpfr & other) { this_ -= other; return this_; }, py::arg("other"))
	.def("__imul__", [](librapid::mpfr & this_, const librapid::mpfr & other) { this_ *= other; return this_; }, py::arg("other"))
	.def("__itruediv__", [](librapid::mpfr & this_, const librapid::mpfr & other) { this_ /= other; return this_; }, py::arg("other"))
	.def("__lt__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return this_ < other; }, py::arg("other"))
	.def("__gt__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return this_ > other; }, py::arg("other"))
	.def("__lte__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return this_ <= other; }, py::arg("other"))
	.def("__gte__", [](const librapid::mpfr & this_, const librapid::mpfr & other) { return this_ >= other; }, py::arg("other"))
	.def("str", [](const librapid::mpfr & this_, int64_t base) { return lrc::str(this_, {-1, base, false}); }, py::arg("base") = int64_t(10))
	.def("__str__", [](const librapid::mpfr & this_) { return lrc::str(this_, {-1, 10, false}); })
	.def("__repr__", [](const librapid::mpfr & this_) { return "librapid::mpfr(\"" + lrc::str(this_, {-1, 10, false}) + "\")"; });

module.def("toMpz", [](const librapid::mpfr & this_) { return librapid::toMpz(this_); });
module.def("toMpq", [](const librapid::mpfr & this_) { return librapid::toMpq(this_); });
module.def("toMpfr", [](const librapid::mpfr & this_) { return librapid::toMpfr(this_); });
module.def("toMpz", [](const librapid::mpfr & this_) { return librapid::toMpz(this_); });
module.def("toMpq", [](const librapid::mpfr & this_) { return librapid::toMpq(this_); });
module.def("toMpfr", [](const librapid::mpfr & this_) { return librapid::toMpfr(this_); });
module.def("toMpz", [](const librapid::mpfr & this_) { return librapid::toMpz(this_); });
module.def("toMpq", [](const librapid::mpfr & this_) { return librapid::toMpq(this_); });
module.def("toMpfr", [](const librapid::mpfr & this_) { return librapid::toMpfr(this_); });
module.def("toMpz", [](const librapid::mpfr & this_) { return librapid::toMpz(this_); });
module.def("toMpq", [](const librapid::mpfr & this_) { return librapid::toMpq(this_); });
module.def("toMpfr", [](const librapid::mpfr & this_) { return librapid::toMpfr(this_); });

}