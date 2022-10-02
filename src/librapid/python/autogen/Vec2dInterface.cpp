
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


// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

py::class_<librapid::Vec2d>(module, "Vec2d")
	.def(py::init<>())
	.def(py::init<double, double>(), py::arg("x"), py::arg("y") = 0)
	.def(py::init<const librapid::Vec2d>())

	.def("__getitem__", [](const librapid::Vec2d &vec, int64_t index) { return vec[index]; })
	.def("__setitem__", [](librapid::Vec2d &vec, int64_t index, double val) { vec[index] = val; })
	.def("__setitem__", [](librapid::Vec2d &vec, int64_t index, double val) { vec[index] = val; })

	.def("__neg__", [](const librapid::Vec2d &lhs) { return -lhs; })

	.def("__add__", [](const librapid::Vec2d &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2d &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2d &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2d &lhs, double rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec2d &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2d &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2d &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2d &lhs, double rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec2d &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2d &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2d &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2d &lhs, double rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2d &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2d &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2d &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2d &lhs, double rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec2d &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2d &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2d &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2d &lhs, double rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec2d &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2d &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2d &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2d &lhs, double rhs) { lhs /= rhs; })
	

	.def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

	.def("mag2", &librapid::Vec2d::mag2)
	.def("mag", &librapid::Vec2d::mag)
	.def("invMag", &librapid::Vec2d::invMag)

	.def("dot", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("str", &librapid::Vec2d::str)
	.def("__str__", &librapid::Vec2d::str)
	.def("__repr__", [](const librapid::Vec2d &vec) { return "Vec2d" + vec.str(); })
	.def("__len__", [](const librapid::Vec2d &vec) { return 2; })
	.def_property("x", &librapid::Vec2d::getX, &librapid::Vec2d::setX)
	.def_property("y", &librapid::Vec2d::getY, &librapid::Vec2d::setY)
	.def_property("z", &librapid::Vec2d::getZ, &librapid::Vec2d::setZ)
	.def_property("w", &librapid::Vec2d::getW, &librapid::Vec2d::setW)
	.def("xy", &librapid::Vec2d::xy)
	.def("yx", &librapid::Vec2d::yx)
	.def("xyz", &librapid::Vec2d::xyz)
	.def("xzy", &librapid::Vec2d::xzy)
	.def("yxz", &librapid::Vec2d::yxz)
	.def("yzx", &librapid::Vec2d::yzx)
	.def("zxy", &librapid::Vec2d::zxy)
	.def("zyx", &librapid::Vec2d::zyx)
	.def("xyzw", &librapid::Vec2d::xyzw)
	.def("xywz", &librapid::Vec2d::xywz)
	.def("xzyw", &librapid::Vec2d::xzyw)
	.def("xzwy", &librapid::Vec2d::xzwy)
	.def("xwyz", &librapid::Vec2d::xwyz)
	.def("xwzy", &librapid::Vec2d::xwzy)
	.def("yxzw", &librapid::Vec2d::yxzw)
	.def("yxwz", &librapid::Vec2d::yxwz)
	.def("yzxw", &librapid::Vec2d::yzxw)
	.def("yzwx", &librapid::Vec2d::yzwx)
	.def("ywxz", &librapid::Vec2d::ywxz)
	.def("ywzx", &librapid::Vec2d::ywzx)
	.def("zxyw", &librapid::Vec2d::zxyw)
	.def("zxwy", &librapid::Vec2d::zxwy)
	.def("zyxw", &librapid::Vec2d::zyxw)
	.def("zywx", &librapid::Vec2d::zywx)
	.def("zwxy", &librapid::Vec2d::zwxy)
	.def("zwyx", &librapid::Vec2d::zwyx)
	.def("wxyz", &librapid::Vec2d::wxyz)
	.def("wxzy", &librapid::Vec2d::wxzy)
	.def("wyxz", &librapid::Vec2d::wyxz)
	.def("wyzx", &librapid::Vec2d::wyzx)
	.def("wzxy", &librapid::Vec2d::wzxy)
	.def("wzyx", &librapid::Vec2d::wzyx);
}