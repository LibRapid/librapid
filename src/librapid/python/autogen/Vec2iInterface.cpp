
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

	void init_Vec2i(py::module &module) {


// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

py::class_<librapid::Vec2i>(module, "Vec2i")
	.def(py::init<>())
	.def(py::init<int64_t, int64_t>(), py::arg("x"), py::arg("y") = 0)
	.def(py::init<const librapid::Vec2i>())

	.def("__getitem__", [](const librapid::Vec2i &vec, int64_t index) { return vec[index]; })
	.def("__setitem__", [](librapid::Vec2i &vec, int64_t index, int64_t val) { vec[index] = val; })
	.def("__setitem__", [](librapid::Vec2i &vec, int64_t index, double val) { vec[index] = val; })

	.def("__add__", [](const librapid::Vec2i &lhs, int64_t rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, int64_t rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, int64_t rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, int64_t rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec2i &lhs, float rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, float rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, float rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, float rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec2i &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, double rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, int64_t rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, int64_t rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, int64_t rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, int64_t rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, float rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, float rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, float rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, float rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, double rhs) { lhs /= rhs; })
	

	.def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { return lhs.dist(rhs); })

	.def("mag2", &librapid::Vec2i::mag2)
	.def("mag", &librapid::Vec2i::mag)
	.def("invMag", &librapid::Vec2i::invMag)

	.def("dot", [](const librapid::Vec2i &lhs, const librapid::Vec2i &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("str", &librapid::Vec2i::str)
	.def("__str__", &librapid::Vec2i::str)
	.def("__repr__", [](const librapid::Vec2i &vec) { return "Vec2i" + vec.str(); })
	.def("__len__", [](const librapid::Vec2i &vec) { return 2; })
	.def_property("x", &librapid::Vec2i::getX, &librapid::Vec2i::setX)
	.def_property("y", &librapid::Vec2i::getY, &librapid::Vec2i::setY)
	.def_property("z", &librapid::Vec2i::getZ, &librapid::Vec2i::setZ)
	.def_property("w", &librapid::Vec2i::getW, &librapid::Vec2i::setW)
	.def("xy", &librapid::Vec2i::xy)
	.def("yx", &librapid::Vec2i::yx)
	.def("xyz", &librapid::Vec2i::xyz)
	.def("xzy", &librapid::Vec2i::xzy)
	.def("yxz", &librapid::Vec2i::yxz)
	.def("yzx", &librapid::Vec2i::yzx)
	.def("zxy", &librapid::Vec2i::zxy)
	.def("zyx", &librapid::Vec2i::zyx)
	.def("xyzw", &librapid::Vec2i::xyzw)
	.def("xywz", &librapid::Vec2i::xywz)
	.def("xzyw", &librapid::Vec2i::xzyw)
	.def("xzwy", &librapid::Vec2i::xzwy)
	.def("xwyz", &librapid::Vec2i::xwyz)
	.def("xwzy", &librapid::Vec2i::xwzy)
	.def("yxzw", &librapid::Vec2i::yxzw)
	.def("yxwz", &librapid::Vec2i::yxwz)
	.def("yzxw", &librapid::Vec2i::yzxw)
	.def("yzwx", &librapid::Vec2i::yzwx)
	.def("ywxz", &librapid::Vec2i::ywxz)
	.def("ywzx", &librapid::Vec2i::ywzx)
	.def("zxyw", &librapid::Vec2i::zxyw)
	.def("zxwy", &librapid::Vec2i::zxwy)
	.def("zyxw", &librapid::Vec2i::zyxw)
	.def("zywx", &librapid::Vec2i::zywx)
	.def("zwxy", &librapid::Vec2i::zwxy)
	.def("zwyx", &librapid::Vec2i::zwyx)
	.def("wxyz", &librapid::Vec2i::wxyz)
	.def("wxzy", &librapid::Vec2i::wxzy)
	.def("wyxz", &librapid::Vec2i::wyxz)
	.def("wyzx", &librapid::Vec2i::wyzx)
	.def("wzxy", &librapid::Vec2i::wzxy)
	.def("wzyx", &librapid::Vec2i::wzyx);
}