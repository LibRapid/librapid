#ifndef LIBRAPID_VEC_INTERFACE
#define LIBRAPID_VEC_INTERFACE



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
py::class_<librapid::Vec2d>(module, "Vec2d")
    .def(py::init<>())
    .def(py::init<double, double>(), py::arg("x"), py::arg("y") = 0)
    .def(py::init<const librapid::Vec2d>())

    .def("__getitem__", [](const librapid::Vec2d &vec, int64_t index) { return vec[index]; })
    .def("__setitem__", [](librapid::Vec2d &vec, int64_t index, int64_t val) { vec[index] = val; })
    .def("__setitem__", [](librapid::Vec2d &vec, int64_t index, double val) { vec[index] = val; })

    .def("__add__", [](const librapid::Vec2d &lhs, int64_t rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, int64_t rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, int64_t rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, int64_t rhs) { return lhs / rhs; })

    .def("__add__", [](const librapid::Vec2d &lhs, float rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, float rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, float rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, float rhs) { return lhs / rhs; })

    .def("__add__", [](const librapid::Vec2d &lhs, double rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, double rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, double rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, double rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, int64_t rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, int64_t rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, int64_t rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, int64_t rhs) { lhs /= rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, float rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, float rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, float rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, float rhs) { lhs /= rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, double rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, double rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, double rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, double rhs) { lhs /= rhs; })
    

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { return lhs.dist(rhs); })

    .def("mag2", &librapid::Vec2d::mag2)
    .def("mag", &librapid::Vec2d::mag)
    .def("invMag", &librapid::Vec2d::invMag)

    .def("dot", [](const librapid::Vec2d &lhs, const librapid::Vec2i &rhs) { return lhs.dot(rhs); }, py::arg("other"))

    .def("dot", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs.dot(rhs); }, py::arg("other"))

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
py::class_<librapid::Vec4d>(module, "Vec4d")
    .def(py::init<>())
    .def(py::init<double, double, double, double>(), py::arg("x"), py::arg("y") = 0, py::arg("z") = 0, py::arg("w") = 0)
    .def(py::init<const librapid::Vec4d>())

    .def("__getitem__", [](const librapid::Vec4d &vec, int64_t index) { return vec[index]; })
    .def("__setitem__", [](librapid::Vec4d &vec, int64_t index, int64_t val) { vec[index] = val; })
    .def("__setitem__", [](librapid::Vec4d &vec, int64_t index, double val) { vec[index] = val; })

    .def("__add__", [](const librapid::Vec4d &lhs, int64_t rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, int64_t rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, int64_t rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, int64_t rhs) { return lhs / rhs; })

    .def("__add__", [](const librapid::Vec4d &lhs, float rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, float rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, float rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, float rhs) { return lhs / rhs; })

    .def("__add__", [](const librapid::Vec4d &lhs, double rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, double rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, double rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, double rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, int64_t rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, int64_t rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, int64_t rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, int64_t rhs) { lhs /= rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, float rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, float rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, float rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, float rhs) { lhs /= rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, double rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, double rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, double rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, double rhs) { lhs /= rhs; })
    

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec2i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec2i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec2i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec2i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec2i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec2i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec2i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec2i &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec4d &lhs, const librapid::Vec2i &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec4d &lhs, const librapid::Vec2i &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs.dist(rhs); })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

    .def("dist2", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs.dist2(rhs); })
    .def("dist", [](const librapid::Vec4d &lhs, const librapid::Vec4d &rhs) { return lhs.dist(rhs); })

    .def("mag2", &librapid::Vec4d::mag2)
    .def("mag", &librapid::Vec4d::mag)
    .def("invMag", &librapid::Vec4d::invMag)

    .def("dot", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs.dot(rhs); }, py::arg("other"))

    .def("dot", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs.dot(rhs); }, py::arg("other"))

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

#endif // LIBRAPID_VEC_INTERFACE
