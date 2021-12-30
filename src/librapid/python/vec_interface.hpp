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

    .def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2i &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

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
    .def_property("w", &librapid::Vec2i::getW, &librapid::Vec2i::setW);

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

    .def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2f &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

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
    .def_property("w", &librapid::Vec2f::getW, &librapid::Vec2f::setW);

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

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec2d &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

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
    .def_property("w", &librapid::Vec2d::getW, &librapid::Vec2d::setW);

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

    .def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3i &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

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
    .def_property("w", &librapid::Vec3i::getW, &librapid::Vec3i::setW);

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

    .def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3f &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

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
    .def_property("w", &librapid::Vec3f::getW, &librapid::Vec3f::setW);

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

    .def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec3d &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

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
    .def_property("w", &librapid::Vec3d::getW, &librapid::Vec3d::setW);

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

    .def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4i &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

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
    .def_property("w", &librapid::Vec4i::getW, &librapid::Vec4i::setW);

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

    .def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4f &lhs, const librapid::Vec4d &rhs) { lhs /= rhs; })

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
    .def_property("w", &librapid::Vec4f::getW, &librapid::Vec4f::setW);

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

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec2f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec2d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec3i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec3f &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec3d &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec4i &rhs) { lhs /= rhs; })

    .def("__add__", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs + rhs; })
    .def("__sub__", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs - rhs; })
    .def("__mul__", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs * rhs; })
    .def("__truediv__", [](const librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { return lhs / rhs; })

    .def("__iadd__", [](librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { lhs += rhs; })
    .def("__isub__", [](librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { lhs -= rhs; })
    .def("__imul__", [](librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { lhs *= rhs; })
    .def("__itruediv__", [](librapid::Vec4d &lhs, const librapid::Vec4f &rhs) { lhs /= rhs; })

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
    .def_property("w", &librapid::Vec4d::getW, &librapid::Vec4d::setW);


#endif // LIBRAPID_VEC_INTERFACE
