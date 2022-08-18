
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

	void init_Vec3f(py::module &module) {


// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

py::class_<librapid::Vec3f>(module, "Vec3f")
	.def(py::init<>())
	.def(py::init<float, float, float>(), py::arg("x"), py::arg("y") = 0, py::arg("z") = 0)
	.def(py::init<const librapid::Vec3f>())

	.def("__getitem__", [](const librapid::Vec3f &vec, int64_t index) { return vec[index]; })
	.def("__setitem__", [](librapid::Vec3f &vec, int64_t index, int64_t val) { vec[index] = val; })
	.def("__setitem__", [](librapid::Vec3f &vec, int64_t index, double val) { vec[index] = val; })

	.def("__add__", [](const librapid::Vec3f &lhs, int64_t rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, int64_t rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, int64_t rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, int64_t rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec3f &lhs, float rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, float rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, float rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, float rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec3f &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, double rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, int64_t rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, int64_t rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, int64_t rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, int64_t rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, float rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, float rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, float rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, float rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, double rhs) { lhs /= rhs; })
	

	.def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec2i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec2i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec2i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec2i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec2i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec2i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec2i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec2i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3f &lhs, const librapid::Vec2i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3f &lhs, const librapid::Vec2i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { return lhs.dist(rhs); })

	.def("mag2", &librapid::Vec3f::mag2)
	.def("mag", &librapid::Vec3f::mag)
	.def("invMag", &librapid::Vec3f::invMag)

	.def("dot", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("str", &librapid::Vec3f::str)
	.def("__str__", &librapid::Vec3f::str)
	.def("__repr__", [](const librapid::Vec3f &vec) { return "Vec3f" + vec.str(); })
	.def("__len__", [](const librapid::Vec3f &vec) { return 3; })
	.def_property("x", &librapid::Vec3f::getX, &librapid::Vec3f::setX)
	.def_property("y", &librapid::Vec3f::getY, &librapid::Vec3f::setY)
	.def_property("z", &librapid::Vec3f::getZ, &librapid::Vec3f::setZ)
	.def_property("w", &librapid::Vec3f::getW, &librapid::Vec3f::setW)
	.def("xy", &librapid::Vec3f::xy)
	.def("yx", &librapid::Vec3f::yx)
	.def("xyz", &librapid::Vec3f::xyz)
	.def("xzy", &librapid::Vec3f::xzy)
	.def("yxz", &librapid::Vec3f::yxz)
	.def("yzx", &librapid::Vec3f::yzx)
	.def("zxy", &librapid::Vec3f::zxy)
	.def("zyx", &librapid::Vec3f::zyx)
	.def("xyzw", &librapid::Vec3f::xyzw)
	.def("xywz", &librapid::Vec3f::xywz)
	.def("xzyw", &librapid::Vec3f::xzyw)
	.def("xzwy", &librapid::Vec3f::xzwy)
	.def("xwyz", &librapid::Vec3f::xwyz)
	.def("xwzy", &librapid::Vec3f::xwzy)
	.def("yxzw", &librapid::Vec3f::yxzw)
	.def("yxwz", &librapid::Vec3f::yxwz)
	.def("yzxw", &librapid::Vec3f::yzxw)
	.def("yzwx", &librapid::Vec3f::yzwx)
	.def("ywxz", &librapid::Vec3f::ywxz)
	.def("ywzx", &librapid::Vec3f::ywzx)
	.def("zxyw", &librapid::Vec3f::zxyw)
	.def("zxwy", &librapid::Vec3f::zxwy)
	.def("zyxw", &librapid::Vec3f::zyxw)
	.def("zywx", &librapid::Vec3f::zywx)
	.def("zwxy", &librapid::Vec3f::zwxy)
	.def("zwyx", &librapid::Vec3f::zwyx)
	.def("wxyz", &librapid::Vec3f::wxyz)
	.def("wxzy", &librapid::Vec3f::wxzy)
	.def("wyxz", &librapid::Vec3f::wyxz)
	.def("wyzx", &librapid::Vec3f::wyzx)
	.def("wzxy", &librapid::Vec3f::wzxy)
	.def("wzyx", &librapid::Vec3f::wzyx);
}