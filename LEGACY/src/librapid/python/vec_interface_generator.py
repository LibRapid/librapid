import itertools


def generate(types):
    res = """
// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

"""

    for dtype in types:
        res += f"""
py::class_<librapid::{dtype[0]}>(module, \"{dtype[0]}\")
    .def(py::init<>())
    .def(py::init<{", ".join([dtype[-2] for _ in range(dtype[-1])])}>(), {dtype[2]})
    .def(py::init<const librapid::{dtype[0]}>())

    .def("__getitem__", [](const librapid::{dtype[0]} &vec, int64_t index) {{ return vec[index]; }})
    .def("__setitem__", [](librapid::{dtype[0]} &vec, int64_t index, int64_t val) {{ vec[index] = val; }})
    .def("__setitem__", [](librapid::{dtype[0]} &vec, int64_t index, double val) {{ vec[index] = val; }})

    .def("__add__", [](const librapid::{dtype[0]} &lhs, int64_t rhs) {{ return lhs + rhs; }})
    .def("__sub__", [](const librapid::{dtype[0]} &lhs, int64_t rhs) {{ return lhs - rhs; }})
    .def("__mul__", [](const librapid::{dtype[0]} &lhs, int64_t rhs) {{ return lhs * rhs; }})
    .def("__truediv__", [](const librapid::{dtype[0]} &lhs, int64_t rhs) {{ return lhs / rhs; }})

    .def("__add__", [](const librapid::{dtype[0]} &lhs, float rhs) {{ return lhs + rhs; }})
    .def("__sub__", [](const librapid::{dtype[0]} &lhs, float rhs) {{ return lhs - rhs; }})
    .def("__mul__", [](const librapid::{dtype[0]} &lhs, float rhs) {{ return lhs * rhs; }})
    .def("__truediv__", [](const librapid::{dtype[0]} &lhs, float rhs) {{ return lhs / rhs; }})

    .def("__add__", [](const librapid::{dtype[0]} &lhs, double rhs) {{ return lhs + rhs; }})
    .def("__sub__", [](const librapid::{dtype[0]} &lhs, double rhs) {{ return lhs - rhs; }})
    .def("__mul__", [](const librapid::{dtype[0]} &lhs, double rhs) {{ return lhs * rhs; }})
    .def("__truediv__", [](const librapid::{dtype[0]} &lhs, double rhs) {{ return lhs / rhs; }})

    .def("__iadd__", [](librapid::{dtype[0]} &lhs, int64_t rhs) {{ lhs += rhs; }})
    .def("__isub__", [](librapid::{dtype[0]} &lhs, int64_t rhs) {{ lhs -= rhs; }})
    .def("__imul__", [](librapid::{dtype[0]} &lhs, int64_t rhs) {{ lhs *= rhs; }})
    .def("__itruediv__", [](librapid::{dtype[0]} &lhs, int64_t rhs) {{ lhs /= rhs; }})

    .def("__iadd__", [](librapid::{dtype[0]} &lhs, float rhs) {{ lhs += rhs; }})
    .def("__isub__", [](librapid::{dtype[0]} &lhs, float rhs) {{ lhs -= rhs; }})
    .def("__imul__", [](librapid::{dtype[0]} &lhs, float rhs) {{ lhs *= rhs; }})
    .def("__itruediv__", [](librapid::{dtype[0]} &lhs, float rhs) {{ lhs /= rhs; }})

    .def("__iadd__", [](librapid::{dtype[0]} &lhs, double rhs) {{ lhs += rhs; }})
    .def("__isub__", [](librapid::{dtype[0]} &lhs, double rhs) {{ lhs -= rhs; }})
    .def("__imul__", [](librapid::{dtype[0]} &lhs, double rhs) {{ lhs *= rhs; }})
    .def("__itruediv__", [](librapid::{dtype[0]} &lhs, double rhs) {{ lhs /= rhs; }})
    
"""

        for dtype2 in types:
            res += f"""
    .def("__add__", [](const librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ return lhs + rhs; }})
    .def("__sub__", [](const librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ return lhs - rhs; }})
    .def("__mul__", [](const librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ return lhs * rhs; }})
    .def("__truediv__", [](const librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ return lhs / rhs; }})

    .def("__iadd__", [](librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ lhs += rhs; }})
    .def("__isub__", [](librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ lhs -= rhs; }})
    .def("__imul__", [](librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ lhs *= rhs; }})
    .def("__itruediv__", [](librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ lhs /= rhs; }})

    .def("dist2", [](const librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ return lhs.dist2(rhs); }})
    .def("dist", [](const librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ return lhs.dist(rhs); }})
"""

        res += f"""
    .def("mag2", &librapid::{dtype[0]}::mag2)
    .def("mag", &librapid::{dtype[0]}::mag)
    .def("invMag", &librapid::{dtype[0]}::invMag)
"""

        for dtype2 in types:
            if dtype[-1] != dtype2[-1]:
                continue

            res += f"""
    .def("dot", [](const librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ return lhs.dot(rhs); }}, py::arg("other"))
"""

            if dtype[-2] == 3 and dtype2[-2] == 3:
                res += f"""
    .def("cross", [](const librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ return lhs.cross(rhs); }}, py::arg("other"))
"""

        res += f"""
    .def("str", &librapid::{dtype[0]}::str)
    .def("__str__", &librapid::{dtype[0]}::str)
    .def("__repr__", [](const librapid::{dtype[0]} &vec) {{ return "{dtype[0]}" + vec.str(); }})
    .def("__len__", [](const librapid::{dtype[0]} &vec) {{ return {dtype[-1]}; }})
    .def_property("x", &librapid::{dtype[0]}::getX, &librapid::{dtype[0]}::setX)
    .def_property("y", &librapid::{dtype[0]}::getY, &librapid::{dtype[0]}::setY)
    .def_property("z", &librapid::{dtype[0]}::getZ, &librapid::{dtype[0]}::setZ)
    .def_property("w", &librapid::{dtype[0]}::getW, &librapid::{dtype[0]}::setW)
"""

        toSwizzle = ["xy", "xyz", "xyzw"]
        for swiz in toSwizzle:
            for perm in itertools.permutations(list(swiz)):
                joined = "".join(perm)
                res += f"\t.def(\"{joined}\", &librapid::{dtype[0]}::{joined})\n"
        res = res[:-1]
        res += ";"

    return res


types = [
    ["Vec2i", "int64_t x, int64_t y", "py::arg(\"x\"), py::arg(\"y\") = 0", "int64_t", 2],
    ["Vec2f", "float x, float y", "py::arg(\"x\"), py::arg(\"y\") = 0", "float", 2],
    ["Vec2d", "double x, double y", "py::arg(\"x\"), py::arg(\"y\") = 0", "double", 2],

    ["Vec3i", "int64_t x, int64_t y, int64_t z", "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0", "int64_t",
     3],
    ["Vec3f", "float x, float y, float z", "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0", "float", 3],
    ["Vec3d", "double x, double y, double z", "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0", "double", 3],

    ["Vec4i", "int64_t x, int64_t y, int64_t z, int64_t w",
     "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0, py::arg(\"w\") = 0", "int64_t", 4],
    ["Vec4f", "float x, float y, float z, float w",
     "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0, py::arg(\"w\") = 0", "float", 4],
    ["Vec4d", "double x, double y, double z, double w",
     "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0, py::arg(\"w\") = 0", "double", 4]
]

with open("./vec_interface.hpp", "w") as file:
    print("Writing contents")

    file.write("#ifndef LIBRAPID_VEC_INTERFACE\n")
    file.write("#define LIBRAPID_VEC_INTERFACE\n")
    file.write("\n\n")

    file.write(generate(types))

    file.write("\n\n")
    file.write("#endif // LIBRAPID_VEC_INTERFACE\n")
