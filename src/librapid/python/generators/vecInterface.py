import itertools


def generate(types):
	res = []
	for dtype in types:
		tmp = f"""

// ====================================================== //
// The code in this file is GENERATED. DO NOT CHANGE IT.  //
// To change this file's contents, please edit and run    //
// "vec_interface_generator.py" in the same directory     //
// ====================================================== //

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
			tmp += f"""
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

		tmp += f"""
	.def("mag2", &librapid::{dtype[0]}::mag2)
	.def("mag", &librapid::{dtype[0]}::mag)
	.def("invMag", &librapid::{dtype[0]}::invMag)
"""

		for dtype2 in types:
			if dtype[-1] != dtype2[-1]:
				continue

			tmp += f"""
	.def("dot", [](const librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ return lhs.dot(rhs); }}, py::arg("other"))
"""

			if dtype[-2] == 3 and dtype2[-2] == 3:
				tmp += f"""
	.def("cross", [](const librapid::{dtype[0]} &lhs, const librapid::{dtype2[0]} &rhs) {{ return lhs.cross(rhs); }}, py::arg("other"))
"""

		tmp += f"""
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
				tmp += f"\t.def(\"{joined}\", &librapid::{dtype[0]}::{joined})\n"
		tmp = tmp[:-1]
		tmp += ";"

		res.append((dtype[0], tmp))
	return res


types = [
	["Vec2i", "int64_t x, int64_t y", "py::arg(\"x\"), py::arg(\"y\") = 0", "int64_t", 2],
	["Vec2f", "float x, float y", "py::arg(\"x\"), py::arg(\"y\") = 0", "float", 2],
	["Vec2d", "double x, double y", "py::arg(\"x\"), py::arg(\"y\") = 0", "double", 2],

	["Vec3i", "int64_t x, int64_t y, int64_t z", "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0", "int64_t", 3],
	["Vec3f", "float x, float y, float z", "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0", "float", 3],
	["Vec3d", "double x, double y, double z", "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0", "double", 3],

	["Vec4i", "int64_t x, int64_t y, int64_t z, int64_t w", "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0, py::arg(\"w\") = 0", "int64_t", 4],
	["Vec4f", "float x, float y, float z, float w", "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0, py::arg(\"w\") = 0", "float", 4],
	["Vec4d", "double x, double y, double z, double w", "py::arg(\"x\"), py::arg(\"y\") = 0, py::arg(\"z\") = 0, py::arg(\"w\") = 0", "double", 4]
]

def write(path:str):
	res = generate(types)
	for tmp in res:
		with open(f"{path}/{tmp[0]}Interface.cpp", "w") as file:
			file.write("""
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

	""")

			file.write("void init_{}(py::module &module) {{\n".format(tmp[0]))
			
			file.write(tmp[1])

			file.write("\n}")

if __name__ == "__main__":
	write("../autogen")

	res = generate(types)

	for tmp in res:
		print("\"${{CMAKE_CURRENT_SOURCE_DIR}}/src/librapid/python/autogen/{}\"".format(f"../autogen/{tmp[0]}Interface.cpp"))
	print()
	for tmp in res:
		print(f"void init_{tmp[0]}(py::module &);")
	print()
	for tmp in res:
		print(f"init_{tmp[0]}(module);")
	print()
