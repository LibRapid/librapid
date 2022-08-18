
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

	void init_Vec4i(py::module &module) {


// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

py::class_<librapid::Vec4i>(module, "Vec4i")
	.def(py::init<>())
	.def(py::init<int64_t, int64_t, int64_t, int64_t>(), py::arg("x"), py::arg("y") = 0, py::arg("z") = 0, py::arg("w") = 0)
	.def(py::init<const librapid::Vec4i>())

	.def("__getitem__", [](const librapid::Vec4i &vec, int64_t index) { return vec[index]; })
	.def("__setitem__", [](librapid::Vec4i &vec, int64_t index, int64_t val) { vec[index] = val; })
	.def("__setitem__", [](librapid::Vec4i &vec, int64_t index, double val) { vec[index] = val; })

	.def("__add__", [](const librapid::Vec4i &lhs, int64_t rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, int64_t rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, int64_t rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, int64_t rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec4i &lhs, float rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, float rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, float rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, float rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec4i &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, double rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, int64_t rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, int64_t rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, int64_t rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, int64_t rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, float rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, float rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, float rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, float rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, double rhs) { lhs /= rhs; })
	

	.def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec2i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec2i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec2i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec2i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec2i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec2i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec2i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec2i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4i &lhs, const librapid::Vec2i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4i &lhs, const librapid::Vec2i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs.dist(rhs); })

	.def("mag2", &librapid::Vec4i::mag2)
	.def("mag", &librapid::Vec4i::mag)
	.def("invMag", &librapid::Vec4i::invMag)

	.def("dot", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("str", &librapid::Vec4i::str)
	.def("__str__", &librapid::Vec4i::str)
	.def("__repr__", [](const librapid::Vec4i &vec) { return "Vec4i" + vec.str(); })
	.def("__len__", [](const librapid::Vec4i &vec) { return 4; })
	.def_property("x", &librapid::Vec4i::getX, &librapid::Vec4i::setX)
	.def_property("y", &librapid::Vec4i::getY, &librapid::Vec4i::setY)
	.def_property("z", &librapid::Vec4i::getZ, &librapid::Vec4i::setZ)
	.def_property("w", &librapid::Vec4i::getW, &librapid::Vec4i::setW)
	.def("xy", &librapid::Vec4i::xy)
	.def("yx", &librapid::Vec4i::yx)
	.def("xyz", &librapid::Vec4i::xyz)
	.def("xzy", &librapid::Vec4i::xzy)
	.def("yxz", &librapid::Vec4i::yxz)
	.def("yzx", &librapid::Vec4i::yzx)
	.def("zxy", &librapid::Vec4i::zxy)
	.def("zyx", &librapid::Vec4i::zyx)
	.def("xyzw", &librapid::Vec4i::xyzw)
	.def("xywz", &librapid::Vec4i::xywz)
	.def("xzyw", &librapid::Vec4i::xzyw)
	.def("xzwy", &librapid::Vec4i::xzwy)
	.def("xwyz", &librapid::Vec4i::xwyz)
	.def("xwzy", &librapid::Vec4i::xwzy)
	.def("yxzw", &librapid::Vec4i::yxzw)
	.def("yxwz", &librapid::Vec4i::yxwz)
	.def("yzxw", &librapid::Vec4i::yzxw)
	.def("yzwx", &librapid::Vec4i::yzwx)
	.def("ywxz", &librapid::Vec4i::ywxz)
	.def("ywzx", &librapid::Vec4i::ywzx)
	.def("zxyw", &librapid::Vec4i::zxyw)
	.def("zxwy", &librapid::Vec4i::zxwy)
	.def("zyxw", &librapid::Vec4i::zyxw)
	.def("zywx", &librapid::Vec4i::zywx)
	.def("zwxy", &librapid::Vec4i::zwxy)
	.def("zwyx", &librapid::Vec4i::zwyx)
	.def("wxyz", &librapid::Vec4i::wxyz)
	.def("wxzy", &librapid::Vec4i::wxzy)
	.def("wyxz", &librapid::Vec4i::wyxz)
	.def("wyzx", &librapid::Vec4i::wyzx)
	.def("wzxy", &librapid::Vec4i::wzxy)
	.def("wzyx", &librapid::Vec4i::wzyx);
}