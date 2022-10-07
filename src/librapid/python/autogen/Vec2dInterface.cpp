
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

void init_Vec2d(py::module &module) {
py::class_<librapid::Vec2d>(module, "Vec2d")
	.def(py::init<>())
	.def(py::init<const librapid::Vec2d &>())
	.def(py::init<librapid::f64>())
	.def(py::init<librapid::f64, librapid::f64>())
	.def("__getitem__", [](const librapid::Vec2d & this_, int64_t index) { return this_[index]; }, py::arg("index"))
	.def("__setitem__", [](librapid::Vec2d & this_, int64_t index, librapid::f64 val) { this_[index] = val; }, py::arg("index"), py::arg("val"))
	.def("__add__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_ * other; }, py::arg("other"))
	.def("__div__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_ / other; }, py::arg("other"))
	.def("__iadd__", [](librapid::Vec2d & this_, const librapid::Vec2d & other) { this_ += other; return this_; }, py::arg("other"))
	.def("__isub__", [](librapid::Vec2d & this_, const librapid::Vec2d & other) { this_ -= other; return this_; }, py::arg("other"))
	.def("__imul__", [](librapid::Vec2d & this_, const librapid::Vec2d & other) { this_ *= other; return this_; }, py::arg("other"))
	.def("__idiv__", [](librapid::Vec2d & this_, const librapid::Vec2d & other) { this_ /= other; return this_; }, py::arg("other"))
	.def("__radd__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return other + this_; }, py::arg("other"))
	.def("__rsub__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return other - this_; }, py::arg("other"))
	.def("__rmul__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return other * this_; }, py::arg("other"))
	.def("__rdiv__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return other / this_; }, py::arg("other"))
	.def("__add__", [](const librapid::Vec2d & this_, librapid::f64 other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::Vec2d & this_, librapid::f64 other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::Vec2d & this_, librapid::f64 other) { return this_ * other; }, py::arg("other"))
	.def("__div__", [](const librapid::Vec2d & this_, librapid::f64 other) { return this_ / other; }, py::arg("other"))
	.def("__iadd__", [](librapid::Vec2d & this_, librapid::f64 other) { this_ += other; return this_; }, py::arg("other"))
	.def("__isub__", [](librapid::Vec2d & this_, librapid::f64 other) { this_ -= other; return this_; }, py::arg("other"))
	.def("__imul__", [](librapid::Vec2d & this_, librapid::f64 other) { this_ *= other; return this_; }, py::arg("other"))
	.def("__idiv__", [](librapid::Vec2d & this_, librapid::f64 other) { this_ /= other; return this_; }, py::arg("other"))
	.def("__radd__", [](const librapid::Vec2d & this_, librapid::f64 other) { return other + this_; }, py::arg("other"))
	.def("__rsub__", [](const librapid::Vec2d & this_, librapid::f64 other) { return other - this_; }, py::arg("other"))
	.def("__rmul__", [](const librapid::Vec2d & this_, librapid::f64 other) { return other * this_; }, py::arg("other"))
	.def("__rdiv__", [](const librapid::Vec2d & this_, librapid::f64 other) { return other / this_; }, py::arg("other"))
	.def("__neg__", [](const librapid::Vec2d & this_) { return -this_; })
	.def("cmp", [](const librapid::Vec2d & this_, const librapid::Vec2d & other, const char * mode) { return this_.cmp(other, mode); }, py::arg("other"), py::arg("mode"))
	.def("cmp", [](const librapid::Vec2d & this_, librapid::f64 other, const char * mode) { return this_.cmp(other, mode); }, py::arg("other"), py::arg("mode"))
	.def("__lt__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_ < other; }, py::arg("other"))
	.def("__le__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_ <= other; }, py::arg("other"))
	.def("__gt__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_ > other; }, py::arg("other"))
	.def("__ge__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_ >= other; }, py::arg("other"))
	.def("__eq__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_ == other; }, py::arg("other"))
	.def("__ne__", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_ != other; }, py::arg("other"))
	.def("__lt__", [](const librapid::Vec2d & this_, librapid::f64 other) { return this_ < other; }, py::arg("other"))
	.def("__le__", [](const librapid::Vec2d & this_, librapid::f64 other) { return this_ <= other; }, py::arg("other"))
	.def("__gt__", [](const librapid::Vec2d & this_, librapid::f64 other) { return this_ > other; }, py::arg("other"))
	.def("__ge__", [](const librapid::Vec2d & this_, librapid::f64 other) { return this_ >= other; }, py::arg("other"))
	.def("__eq__", [](const librapid::Vec2d & this_, librapid::f64 other) { return this_ == other; }, py::arg("other"))
	.def("__ne__", [](const librapid::Vec2d & this_, librapid::f64 other) { return this_ != other; }, py::arg("other"))
	.def("mag2", [](const librapid::Vec2d & this_) { return this_.mag2(); })
	.def("mag", [](const librapid::Vec2d & this_) { return this_.mag(); })
	.def("invMag", [](const librapid::Vec2d & this_) { return this_.invMag(); })
	.def("norm", [](const librapid::Vec2d & this_) { return this_.norm(); })
	.def("dot", [](const librapid::Vec2d & this_, const librapid::Vec2d & other) { return this_.dot(other); }, py::arg("other"))
	.def("__bool__", [](const librapid::Vec2d & this_) { return (bool) this_; })
	.def("__str__", [](const librapid::Vec2d & this_) { return this_.str(); })
	.def("__repr__", [](const librapid::Vec2d & this_) { return std::string("librapid::Vec2d") + this_.str(); })
	.def("x", [](const librapid::Vec2d & this_) { return this_.x(); })
	.def("y", [](const librapid::Vec2d & this_) { return this_.y(); })
	.def("z", [](const librapid::Vec2d & this_) { return this_.z(); })
	.def("w", [](const librapid::Vec2d & this_) { return this_.w(); })
	.def("xy", [](const librapid::Vec2d & this_) { return this_.xy(); })
	.def("yx", [](const librapid::Vec2d & this_) { return this_.yx(); })
	.def("xz", [](const librapid::Vec2d & this_) { return this_.xz(); })
	.def("zx", [](const librapid::Vec2d & this_) { return this_.zx(); })
	.def("yz", [](const librapid::Vec2d & this_) { return this_.yz(); })
	.def("zy", [](const librapid::Vec2d & this_) { return this_.zy(); })
	.def("xyz", [](const librapid::Vec2d & this_) { return this_.xyz(); })
	.def("xzy", [](const librapid::Vec2d & this_) { return this_.xzy(); })
	.def("yxz", [](const librapid::Vec2d & this_) { return this_.yxz(); })
	.def("yzx", [](const librapid::Vec2d & this_) { return this_.yzx(); })
	.def("zxy", [](const librapid::Vec2d & this_) { return this_.zxy(); })
	.def("zyx", [](const librapid::Vec2d & this_) { return this_.zyx(); })
	.def("xyw", [](const librapid::Vec2d & this_) { return this_.xyw(); })
	.def("xwy", [](const librapid::Vec2d & this_) { return this_.xwy(); })
	.def("yxw", [](const librapid::Vec2d & this_) { return this_.yxw(); })
	.def("ywx", [](const librapid::Vec2d & this_) { return this_.ywx(); })
	.def("wxy", [](const librapid::Vec2d & this_) { return this_.wxy(); })
	.def("wyx", [](const librapid::Vec2d & this_) { return this_.wyx(); })
	.def("xzw", [](const librapid::Vec2d & this_) { return this_.xzw(); })
	.def("xwz", [](const librapid::Vec2d & this_) { return this_.xwz(); })
	.def("zxw", [](const librapid::Vec2d & this_) { return this_.zxw(); })
	.def("zwx", [](const librapid::Vec2d & this_) { return this_.zwx(); })
	.def("wxz", [](const librapid::Vec2d & this_) { return this_.wxz(); })
	.def("wzx", [](const librapid::Vec2d & this_) { return this_.wzx(); })
	.def("yzw", [](const librapid::Vec2d & this_) { return this_.yzw(); })
	.def("ywz", [](const librapid::Vec2d & this_) { return this_.ywz(); })
	.def("zyw", [](const librapid::Vec2d & this_) { return this_.zyw(); })
	.def("zwy", [](const librapid::Vec2d & this_) { return this_.zwy(); })
	.def("wyz", [](const librapid::Vec2d & this_) { return this_.wyz(); })
	.def("wzy", [](const librapid::Vec2d & this_) { return this_.wzy(); })
	.def("xyzw", [](const librapid::Vec2d & this_) { return this_.xyzw(); })
	.def("xywz", [](const librapid::Vec2d & this_) { return this_.xywz(); })
	.def("xzyw", [](const librapid::Vec2d & this_) { return this_.xzyw(); })
	.def("xzwy", [](const librapid::Vec2d & this_) { return this_.xzwy(); })
	.def("xwyz", [](const librapid::Vec2d & this_) { return this_.xwyz(); })
	.def("xwzy", [](const librapid::Vec2d & this_) { return this_.xwzy(); })
	.def("yxzw", [](const librapid::Vec2d & this_) { return this_.yxzw(); })
	.def("yxwz", [](const librapid::Vec2d & this_) { return this_.yxwz(); })
	.def("yzxw", [](const librapid::Vec2d & this_) { return this_.yzxw(); })
	.def("yzwx", [](const librapid::Vec2d & this_) { return this_.yzwx(); })
	.def("ywxz", [](const librapid::Vec2d & this_) { return this_.ywxz(); })
	.def("ywzx", [](const librapid::Vec2d & this_) { return this_.ywzx(); })
	.def("zxyw", [](const librapid::Vec2d & this_) { return this_.zxyw(); })
	.def("zxwy", [](const librapid::Vec2d & this_) { return this_.zxwy(); })
	.def("zyxw", [](const librapid::Vec2d & this_) { return this_.zyxw(); })
	.def("zywx", [](const librapid::Vec2d & this_) { return this_.zywx(); })
	.def("zwxy", [](const librapid::Vec2d & this_) { return this_.zwxy(); })
	.def("zwyx", [](const librapid::Vec2d & this_) { return this_.zwyx(); })
	.def("wxyz", [](const librapid::Vec2d & this_) { return this_.wxyz(); })
	.def("wxzy", [](const librapid::Vec2d & this_) { return this_.wxzy(); })
	.def("wyxz", [](const librapid::Vec2d & this_) { return this_.wyxz(); })
	.def("wyzx", [](const librapid::Vec2d & this_) { return this_.wyzx(); })
	.def("wzxy", [](const librapid::Vec2d & this_) { return this_.wzxy(); })
	.def("wzyx", [](const librapid::Vec2d & this_) { return this_.wzyx(); });


module.def("dist2", [](const librapid::Vec2d & lhs, const librapid::Vec2d & rhs) { return lrc::dist2(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"));
module.def("dist2", [](const librapid::Vec2d & lhs, const librapid::Vec2d & rhs) { return lrc::dist(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"));
module.def("abs", [](const librapid::Vec2d & val) { return lrc::abs(val); }, py::arg("val"));
module.def("sin", [](const librapid::Vec2d & val) { return lrc::sin(val); }, py::arg("val"));
module.def("cos", [](const librapid::Vec2d & val) { return lrc::cos(val); }, py::arg("val"));
module.def("tan", [](const librapid::Vec2d & val) { return lrc::tan(val); }, py::arg("val"));
module.def("asin", [](const librapid::Vec2d & val) { return lrc::asin(val); }, py::arg("val"));
module.def("acos", [](const librapid::Vec2d & val) { return lrc::acos(val); }, py::arg("val"));
module.def("atan", [](const librapid::Vec2d & val) { return lrc::atan(val); }, py::arg("val"));
module.def("sinh", [](const librapid::Vec2d & val) { return lrc::sinh(val); }, py::arg("val"));
module.def("cosh", [](const librapid::Vec2d & val) { return lrc::cosh(val); }, py::arg("val"));
module.def("tanh", [](const librapid::Vec2d & val) { return lrc::tanh(val); }, py::arg("val"));
module.def("asinh", [](const librapid::Vec2d & val) { return lrc::asinh(val); }, py::arg("val"));
module.def("acosh", [](const librapid::Vec2d & val) { return lrc::acosh(val); }, py::arg("val"));
module.def("atanh", [](const librapid::Vec2d & val) { return lrc::atanh(val); }, py::arg("val"));
module.def("exp", [](const librapid::Vec2d & val) { return lrc::exp(val); }, py::arg("val"));
module.def("log", [](const librapid::Vec2d & val) { return lrc::log(val); }, py::arg("val"));
module.def("sqrt", [](const librapid::Vec2d & val) { return lrc::sqrt(val); }, py::arg("val"));
module.def("floor", [](const librapid::Vec2d & val) { return lrc::floor(val); }, py::arg("val"));
module.def("ceil", [](const librapid::Vec2d & val) { return lrc::ceil(val); }, py::arg("val"));

}