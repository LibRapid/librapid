
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

	void init_Vec3d(py::module &module) {


// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

py::class_<librapid::Vec3d>(module, "Vec3d")
	.def(py::init<>())
	.def(py::init<double, double, double>(), py::arg("x"), py::arg("y") = 0, py::arg("z") = 0)
	.def(py::init<const librapid::Vec3d>())

	.def("__getitem__", [](const librapid::Vec3d &vec, int64_t index) { return vec[index]; })
	.def("__setitem__", [](librapid::Vec3d &vec, int64_t index, int64_t val) { vec[index] = val; })
	.def("__setitem__", [](librapid::Vec3d &vec, int64_t index, double val) { vec[index] = val; })

	.def("__add__", [](const librapid::Vec3d &lhs, int64_t rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, int64_t rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, int64_t rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, int64_t rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec3d &lhs, float rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, float rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, float rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, float rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec3d &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, double rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, int64_t rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, int64_t rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, int64_t rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, int64_t rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, float rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, float rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, float rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, float rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, double rhs) { lhs /= rhs; })
	

	.def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec2i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec2i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec2i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec2i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec2i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec2i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec2i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec2i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3d &lhs, const librapid::Vec2i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3d &lhs, const librapid::Vec2i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { return lhs.dist(rhs); })

	.def("mag2", &librapid::Vec3d::mag2)
	.def("mag", &librapid::Vec3d::mag)
	.def("invMag", &librapid::Vec3d::invMag)

	.def("dot", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("str", &librapid::Vec3d::str)
	.def("__str__", &librapid::Vec3d::str)
	.def("__repr__", [](const librapid::Vec3d &vec) { return "Vec3d" + vec.str(); })
	.def("__len__", [](const librapid::Vec3d &vec) { return 3; })
	.def_property("x", &librapid::Vec3d::getX, &librapid::Vec3d::setX)
	.def_property("y", &librapid::Vec3d::getY, &librapid::Vec3d::setY)
	.def_property("z", &librapid::Vec3d::getZ, &librapid::Vec3d::setZ)
	.def_property("w", &librapid::Vec3d::getW, &librapid::Vec3d::setW)
	.def("xy", &librapid::Vec3d::xy)
	.def("yx", &librapid::Vec3d::yx)
	.def("xyz", &librapid::Vec3d::xyz)
	.def("xzy", &librapid::Vec3d::xzy)
	.def("yxz", &librapid::Vec3d::yxz)
	.def("yzx", &librapid::Vec3d::yzx)
	.def("zxy", &librapid::Vec3d::zxy)
	.def("zyx", &librapid::Vec3d::zyx)
	.def("xyzw", &librapid::Vec3d::xyzw)
	.def("xywz", &librapid::Vec3d::xywz)
	.def("xzyw", &librapid::Vec3d::xzyw)
	.def("xzwy", &librapid::Vec3d::xzwy)
	.def("xwyz", &librapid::Vec3d::xwyz)
	.def("xwzy", &librapid::Vec3d::xwzy)
	.def("yxzw", &librapid::Vec3d::yxzw)
	.def("yxwz", &librapid::Vec3d::yxwz)
	.def("yzxw", &librapid::Vec3d::yzxw)
	.def("yzwx", &librapid::Vec3d::yzwx)
	.def("ywxz", &librapid::Vec3d::ywxz)
	.def("ywzx", &librapid::Vec3d::ywzx)
	.def("zxyw", &librapid::Vec3d::zxyw)
	.def("zxwy", &librapid::Vec3d::zxwy)
	.def("zyxw", &librapid::Vec3d::zyxw)
	.def("zywx", &librapid::Vec3d::zywx)
	.def("zwxy", &librapid::Vec3d::zwxy)
	.def("zwyx", &librapid::Vec3d::zwyx)
	.def("wxyz", &librapid::Vec3d::wxyz)
	.def("wxzy", &librapid::Vec3d::wxzy)
	.def("wyxz", &librapid::Vec3d::wyxz)
	.def("wyzx", &librapid::Vec3d::wyzx)
	.def("wzxy", &librapid::Vec3d::wzxy)
	.def("wzyx", &librapid::Vec3d::wzyx);
}