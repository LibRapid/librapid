
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

	void init_Vec4d(py::module &module) {


// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

py::class_<librapid::Vec4d>(module, "Vec4d")
	.def(py::init<>())
	.def(py::init<double, double, double, double>(), py::arg("x"), py::arg("y") = 0, py::arg("z") = 0, py::arg("w") = 0)
	.def(py::init<const librapid::Vec4d>())

	.def("__getitem__", [](const librapid::Vec4d &vec, int64_t index) { return vec[index]; })
	.def("__setitem__", [](librapid::Vec4d &vec, int64_t index, double val) { vec[index] = val; })
	.def("__setitem__", [](librapid::Vec4d &vec, int64_t index, double val) { vec[index] = val; })

	.def("__neg__", [](const librapid::Vec4d &lhs) { return -lhs; })

	.def("__add__", [](const librapid::Vec4d &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4d &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4d &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4d &lhs, double rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec4d &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4d &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4d &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4d &lhs, double rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec4d &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4d &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4d &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4d &lhs, double rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4d &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4d &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4d &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4d &lhs, double rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec4d &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4d &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4d &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4d &lhs, double rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec4d &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4d &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4d &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4d &lhs, double rhs) { lhs /= rhs; })
	

	.def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

	.def("mag2", &librapid::Vec4d::mag2)
	.def("mag", &librapid::Vec4d::mag)
	.def("invMag", &librapid::Vec4d::invMag)

	.def("dot", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("str", &librapid::Vec4d::str)
	.def("__str__", &librapid::Vec4d::str)
	.def("__repr__", [](const librapid::Vec4d &vec) { return "Vec4d" + vec.str(); })
	.def("__len__", [](const librapid::Vec4d &vec) { return 4; })
	.def_property("x", &librapid::Vec4d::getX, &librapid::Vec4d::setX)
	.def_property("y", &librapid::Vec4d::getY, &librapid::Vec4d::setY)
	.def_property("z", &librapid::Vec4d::getZ, &librapid::Vec4d::setZ)
	.def_property("w", &librapid::Vec4d::getW, &librapid::Vec4d::setW)
	.def("xy", &librapid::Vec4d::xy)
	.def("yx", &librapid::Vec4d::yx)
	.def("xyz", &librapid::Vec4d::xyz)
	.def("xzy", &librapid::Vec4d::xzy)
	.def("yxz", &librapid::Vec4d::yxz)
	.def("yzx", &librapid::Vec4d::yzx)
	.def("zxy", &librapid::Vec4d::zxy)
	.def("zyx", &librapid::Vec4d::zyx)
	.def("xyzw", &librapid::Vec4d::xyzw)
	.def("xywz", &librapid::Vec4d::xywz)
	.def("xzyw", &librapid::Vec4d::xzyw)
	.def("xzwy", &librapid::Vec4d::xzwy)
	.def("xwyz", &librapid::Vec4d::xwyz)
	.def("xwzy", &librapid::Vec4d::xwzy)
	.def("yxzw", &librapid::Vec4d::yxzw)
	.def("yxwz", &librapid::Vec4d::yxwz)
	.def("yzxw", &librapid::Vec4d::yzxw)
	.def("yzwx", &librapid::Vec4d::yzwx)
	.def("ywxz", &librapid::Vec4d::ywxz)
	.def("ywzx", &librapid::Vec4d::ywzx)
	.def("zxyw", &librapid::Vec4d::zxyw)
	.def("zxwy", &librapid::Vec4d::zxwy)
	.def("zyxw", &librapid::Vec4d::zyxw)
	.def("zywx", &librapid::Vec4d::zywx)
	.def("zwxy", &librapid::Vec4d::zwxy)
	.def("zwyx", &librapid::Vec4d::zwyx)
	.def("wxyz", &librapid::Vec4d::wxyz)
	.def("wxzy", &librapid::Vec4d::wxzy)
	.def("wyxz", &librapid::Vec4d::wyxz)
	.def("wyzx", &librapid::Vec4d::wyzx)
	.def("wzxy", &librapid::Vec4d::wzxy)
	.def("wzyx", &librapid::Vec4d::wzyx);
}