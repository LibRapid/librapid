
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

	void init_Vec2f(py::module &module) {


// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

py::class_<librapid::Vec2f>(module, "Vec2f")
	.def(py::init<>())
	.def(py::init<float, float>(), py::arg("x"), py::arg("y") = 0)
	.def(py::init<const librapid::Vec2f>())

	.def("__getitem__", [](const librapid::Vec2f &vec, int64_t index) { return vec[index]; })
	.def("__setitem__", [](librapid::Vec2f &vec, int64_t index, int64_t val) { vec[index] = val; })
	.def("__setitem__", [](librapid::Vec2f &vec, int64_t index, double val) { vec[index] = val; })

	.def("__add__", [](const librapid::Vec2f &lhs, int64_t rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, int64_t rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, int64_t rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, int64_t rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec2f &lhs, float rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, float rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, float rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, float rhs) { return lhs / rhs; })

	.def("__add__", [](const librapid::Vec2f &lhs, double rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, double rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, double rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, double rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, int64_t rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, int64_t rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, int64_t rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, int64_t rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, float rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, float rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, float rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, float rhs) { lhs /= rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, double rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, double rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, double rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, double rhs) { lhs /= rhs; })
	

	.def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { return lhs.dist(rhs); })

	.def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
	.def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
	.def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
	.def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

	.def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
	.def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
	.def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
	.def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

	.def("dist2", [](const librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { return lhs.dist2(rhs); })
	.def("dist", [](const librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { return lhs.dist(rhs); })

	.def("mag2", &librapid::Vec2f::mag2)
	.def("mag", &librapid::Vec2f::mag)
	.def("invMag", &librapid::Vec2f::invMag)

	.def("dot", [](const librapid::Vec2f &lhs, const librapid::Vec2i &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("dot", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs.dot(rhs); }, py::arg("other"))

	.def("str", &librapid::Vec2f::str)
	.def("__str__", &librapid::Vec2f::str)
	.def("__repr__", [](const librapid::Vec2f &vec) { return "Vec2f" + vec.str(); })
	.def("__len__", [](const librapid::Vec2f &vec) { return 2; })
	.def_property("x", &librapid::Vec2f::getX, &librapid::Vec2f::setX)
	.def_property("y", &librapid::Vec2f::getY, &librapid::Vec2f::setY)
	.def_property("z", &librapid::Vec2f::getZ, &librapid::Vec2f::setZ)
	.def_property("w", &librapid::Vec2f::getW, &librapid::Vec2f::setW)
	.def("xy", &librapid::Vec2f::xy)
	.def("yx", &librapid::Vec2f::yx)
	.def("xyz", &librapid::Vec2f::xyz)
	.def("xzy", &librapid::Vec2f::xzy)
	.def("yxz", &librapid::Vec2f::yxz)
	.def("yzx", &librapid::Vec2f::yzx)
	.def("zxy", &librapid::Vec2f::zxy)
	.def("zyx", &librapid::Vec2f::zyx)
	.def("xyzw", &librapid::Vec2f::xyzw)
	.def("xywz", &librapid::Vec2f::xywz)
	.def("xzyw", &librapid::Vec2f::xzyw)
	.def("xzwy", &librapid::Vec2f::xzwy)
	.def("xwyz", &librapid::Vec2f::xwyz)
	.def("xwzy", &librapid::Vec2f::xwzy)
	.def("yxzw", &librapid::Vec2f::yxzw)
	.def("yxwz", &librapid::Vec2f::yxwz)
	.def("yzxw", &librapid::Vec2f::yzxw)
	.def("yzwx", &librapid::Vec2f::yzwx)
	.def("ywxz", &librapid::Vec2f::ywxz)
	.def("ywzx", &librapid::Vec2f::ywzx)
	.def("zxyw", &librapid::Vec2f::zxyw)
	.def("zxwy", &librapid::Vec2f::zxwy)
	.def("zyxw", &librapid::Vec2f::zyxw)
	.def("zywx", &librapid::Vec2f::zywx)
	.def("zwxy", &librapid::Vec2f::zwxy)
	.def("zwyx", &librapid::Vec2f::zwyx)
	.def("wxyz", &librapid::Vec2f::wxyz)
	.def("wxzy", &librapid::Vec2f::wxzy)
	.def("wyxz", &librapid::Vec2f::wyxz)
	.def("wyzx", &librapid::Vec2f::wyzx)
	.def("wzxy", &librapid::Vec2f::wzxy)
	.def("wzyx", &librapid::Vec2f::wzyx);
}