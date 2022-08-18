
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

	void init_Vec3i(py::module &module) {


// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

py::class_<librapid::Vec3i>(module, "Vec3i")
	.def(py::init<>())
	.def(py::init<int64_t, int64_t, int64_t>(), py::arg("x"), py::arg("y") = 0, py::arg("z") = 0)
	.def(py::init<const librapid::Vec3i>())

	.def("__getitem__", [](const librapid::Vec3i &vec, int64_t index) { return vec[index]; })
	.def("__setitem__", [](librapid::Vec3i &vec, int64_t index, int64_t val) { vec[index] = val; })
	.def("__setitem__", [](librapid::Vec3i &vec, int64_t index, double val) { vec[index] = val; })

	.def("__add__", [](const librapid::Vec3i &lhs, int64_t rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, int64_t rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, int64_t rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, int64_t rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec3i &lhs, float rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, float rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, float rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, float rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec3i &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, double rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, int64_t rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, int64_t rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, int64_t rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, int64_t rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, float rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, float rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, float rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, float rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, double rhs) { lhs /= rhs; })
	

	.def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec2i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec2i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec2i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec2i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec2i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec2i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec2i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec2i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3i &lhs, const librapid::Vec2i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3i &lhs, const librapid::Vec2i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { return lhs.dist(rhs); })

	.def("mag2", &librapid::Vec3i::mag2)
	.def("mag", &librapid::Vec3i::mag)
	.def("invMag", &librapid::Vec3i::invMag)

	.def("dot", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("str", &librapid::Vec3i::str)
	.def("__str__", &librapid::Vec3i::str)
	.def("__repr__", [](const librapid::Vec3i &vec) { return "Vec3i" + vec.str(); })
	.def("__len__", [](const librapid::Vec3i &vec) { return 3; })
	.def_property("x", &librapid::Vec3i::getX, &librapid::Vec3i::setX)
	.def_property("y", &librapid::Vec3i::getY, &librapid::Vec3i::setY)
	.def_property("z", &librapid::Vec3i::getZ, &librapid::Vec3i::setZ)
	.def_property("w", &librapid::Vec3i::getW, &librapid::Vec3i::setW)
	.def("xy", &librapid::Vec3i::xy)
	.def("yx", &librapid::Vec3i::yx)
	.def("xyz", &librapid::Vec3i::xyz)
	.def("xzy", &librapid::Vec3i::xzy)
	.def("yxz", &librapid::Vec3i::yxz)
	.def("yzx", &librapid::Vec3i::yzx)
	.def("zxy", &librapid::Vec3i::zxy)
	.def("zyx", &librapid::Vec3i::zyx)
	.def("xyzw", &librapid::Vec3i::xyzw)
	.def("xywz", &librapid::Vec3i::xywz)
	.def("xzyw", &librapid::Vec3i::xzyw)
	.def("xzwy", &librapid::Vec3i::xzwy)
	.def("xwyz", &librapid::Vec3i::xwyz)
	.def("xwzy", &librapid::Vec3i::xwzy)
	.def("yxzw", &librapid::Vec3i::yxzw)
	.def("yxwz", &librapid::Vec3i::yxwz)
	.def("yzxw", &librapid::Vec3i::yzxw)
	.def("yzwx", &librapid::Vec3i::yzwx)
	.def("ywxz", &librapid::Vec3i::ywxz)
	.def("ywzx", &librapid::Vec3i::ywzx)
	.def("zxyw", &librapid::Vec3i::zxyw)
	.def("zxwy", &librapid::Vec3i::zxwy)
	.def("zyxw", &librapid::Vec3i::zyxw)
	.def("zywx", &librapid::Vec3i::zywx)
	.def("zwxy", &librapid::Vec3i::zwxy)
	.def("zwyx", &librapid::Vec3i::zwyx)
	.def("wxyz", &librapid::Vec3i::wxyz)
	.def("wxzy", &librapid::Vec3i::wxzy)
	.def("wyxz", &librapid::Vec3i::wyxz)
	.def("wyzx", &librapid::Vec3i::wyzx)
	.def("wzxy", &librapid::Vec3i::wzxy)
	.def("wzyx", &librapid::Vec3i::wzyx);
}