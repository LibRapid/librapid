
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

	void init_Vec4f(py::module &module) {


// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

py::class_<librapid::Vec4f>(module, "Vec4f")
	.def(py::init<>())
	.def(py::init<float, float, float, float>(), py::arg("x"), py::arg("y") = 0, py::arg("z") = 0, py::arg("w") = 0)
	.def(py::init<const librapid::Vec4f>())

	.def("__getitem__", [](const librapid::Vec4f &vec, int64_t index) { return vec[index]; })
	.def("__setitem__", [](librapid::Vec4f &vec, int64_t index, int64_t val) { vec[index] = val; })
	.def("__setitem__", [](librapid::Vec4f &vec, int64_t index, double val) { vec[index] = val; })

	.def("__add__", [](const librapid::Vec4f &lhs, int64_t rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, int64_t rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, int64_t rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, int64_t rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec4f &lhs, float rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, float rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, float rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, float rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec4f &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, double rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, int64_t rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, int64_t rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, int64_t rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, int64_t rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, float rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, float rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, float rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, float rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, double rhs) { lhs /= rhs; })
	

	.def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec2i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec2i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec2i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec2i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec2i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec2i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec2i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec2i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4f &lhs, const librapid::Vec2i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4f &lhs, const librapid::Vec2i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs.dist(rhs); })

	.def("mag2", &librapid::Vec4f::mag2)
	.def("mag", &librapid::Vec4f::mag)
	.def("invMag", &librapid::Vec4f::invMag)

	.def("dot", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("str", &librapid::Vec4f::str)
	.def("__str__", &librapid::Vec4f::str)
	.def("__repr__", [](const librapid::Vec4f &vec) { return "Vec4f" + vec.str(); })
	.def("__len__", [](const librapid::Vec4f &vec) { return 4; })
	.def_property("x", &librapid::Vec4f::getX, &librapid::Vec4f::setX)
	.def_property("y", &librapid::Vec4f::getY, &librapid::Vec4f::setY)
	.def_property("z", &librapid::Vec4f::getZ, &librapid::Vec4f::setZ)
	.def_property("w", &librapid::Vec4f::getW, &librapid::Vec4f::setW)
	.def("xy", &librapid::Vec4f::xy)
	.def("yx", &librapid::Vec4f::yx)
	.def("xyz", &librapid::Vec4f::xyz)
	.def("xzy", &librapid::Vec4f::xzy)
	.def("yxz", &librapid::Vec4f::yxz)
	.def("yzx", &librapid::Vec4f::yzx)
	.def("zxy", &librapid::Vec4f::zxy)
	.def("zyx", &librapid::Vec4f::zyx)
	.def("xyzw", &librapid::Vec4f::xyzw)
	.def("xywz", &librapid::Vec4f::xywz)
	.def("xzyw", &librapid::Vec4f::xzyw)
	.def("xzwy", &librapid::Vec4f::xzwy)
	.def("xwyz", &librapid::Vec4f::xwyz)
	.def("xwzy", &librapid::Vec4f::xwzy)
	.def("yxzw", &librapid::Vec4f::yxzw)
	.def("yxwz", &librapid::Vec4f::yxwz)
	.def("yzxw", &librapid::Vec4f::yzxw)
	.def("yzwx", &librapid::Vec4f::yzwx)
	.def("ywxz", &librapid::Vec4f::ywxz)
	.def("ywzx", &librapid::Vec4f::ywzx)
	.def("zxyw", &librapid::Vec4f::zxyw)
	.def("zxwy", &librapid::Vec4f::zxwy)
	.def("zyxw", &librapid::Vec4f::zyxw)
	.def("zywx", &librapid::Vec4f::zywx)
	.def("zwxy", &librapid::Vec4f::zwxy)
	.def("zwyx", &librapid::Vec4f::zwyx)
	.def("wxyz", &librapid::Vec4f::wxyz)
	.def("wxzy", &librapid::Vec4f::wxzy)
	.def("wyxz", &librapid::Vec4f::wyxz)
	.def("wyzx", &librapid::Vec4f::wyzx)
	.def("wzxy", &librapid::Vec4f::wzxy)
	.def("wzyx", &librapid::Vec4f::wzyx);
}