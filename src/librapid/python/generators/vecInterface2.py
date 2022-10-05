from argumentHelper import *

# Detect LibRapid features
features = []
try:
	with open("../../configuration.txt", "r") as file:
		for line in file:
			if len(line) == 0 or line.startswith("#"):
				continue
			
			print("FEATURE:", line.strip())
			try:
				args = line.split()
				features.append((args[0], args[2])) # Cut out the "="
			except:
				pass
except:
	pass

vecTypes = [
	["Vec2i", "i32", 2],
	["Vec3i", "i32", 2],
	["Vec4i", "i32", 2],
	["Vec2f", "f32", 3],
	["Vec3f", "f32", 3],
	["Vec4f", "f32", 3],
	["Vec2d", "f64", 4],
	["Vec3d", "f64", 4],
	["Vec4d", "f64", 4],
]

classStr = ""
moduleStr = ""

interfaceList = {}

for t in vecTypes:
	typename = "librapid::{}".format(t[0])
	constRef = "const librapid::{} &".format(t[0])
	ref = "librapid::{} &".format(t[0])
	int64_t = "int64_t"
	scalar = "librapid::{}".format(t[1])
	extent = "librapid::Extent"
	extentConstRef = "const librapid::Extent &"
	stdString = "std::string"
	stdStringConstRef = "const std::string &"
	dims = t[2]

	# Class Definition
	classStr += "py::class_<{0}>(module, \"{1}\")\n".format(typename, t[0])

	# Constructors
	classStr += "\t.def(py::init<>())\n"
	classStr += "\t.def(py::init<{}>())\n".format(constRef)

	# Multi-value constructors
	for i in range(dims):
		args = "{}".format(scalar) * (i + 1)
		classStr += "\t.def(py::init<{}>())\n".format(args)

	fCopy = [
		Function("copy", [Argument(constRef, "this_")], "return this_.copy();"),
	]

	fIndex = [
		Function("__getitem__", [Argument(constRef, "this_"), Argument(int64_t, "index")], "return this_[index];"),
		Function("__setitem__", [Argument(ref, "this_"), Argument(int64_t, "index"), Argument(scalar, "val")], "this_[index] = val;"),
		Function("__setitem__", [Argument(ref, "this_"), Argument(int64_t, "index"), Argument(scalar, "val")], "this_[index] = val;")
	]

	functions = [
		Function("__getitem__", [Argument(constRef, "this_"), Argument(int64_t, "index")], "return this_[index];"),
		Function("__setitem__", [Argument(ref, "this_"), Argument(int64_t, "index"), Argument(scalar, "val")], "this_[index] = val;"),

		Function("__add__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ + other;"),
		Function("__add__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ + other;"),
		Function("__add__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ + other;"),
		Function("__add__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ + other;"),
	]

	for i in range(len(functions)):
		function = functions[i]
		if isinstance(function,Function):
			classStr += "\t" + function.gen()
		else:
			classStr += "\t" + function
		
		if i + 1 < len(functions):
			classStr += "\n"
		else:
			classStr += ";\n"

	# Module-based functions

	forceTmpFunc = []
	if t not in ("ArrayB", "ArrayBG"):
		forceTmpFunc += [
			Function("add", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::add(lhs, rhs, dst);"),
			Function("sub", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::sub(lhs, rhs, dst);"),
			Function("mul", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::mul(lhs, rhs, dst);"),
			Function("div", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::div(lhs, rhs, dst);"),

			Function("add", [Argument(constRef, "lhs"), Argument(scalar, "rhs"), Argument(ref, "dst")], "librapid::add(lhs, rhs, dst);"),
			Function("sub", [Argument(constRef, "lhs"), Argument(scalar, "rhs"), Argument(ref, "dst")], "librapid::sub(lhs, rhs, dst);"),
			Function("mul", [Argument(constRef, "lhs"), Argument(scalar, "rhs"), Argument(ref, "dst")], "librapid::mul(lhs, rhs, dst);"),
			Function("div", [Argument(constRef, "lhs"), Argument(scalar, "rhs"), Argument(ref, "dst")], "librapid::div(lhs, rhs, dst);"),

			Function("add", [Argument(scalar, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::add(lhs, rhs, dst);"),
			Function("sub", [Argument(scalar, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::sub(lhs, rhs, dst);"),
			Function("mul", [Argument(scalar, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::mul(lhs, rhs, dst);"),
			Function("div", [Argument(scalar, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::div(lhs, rhs, dst);"),

			Function("negate", [Argument(constRef, "lhs"), Argument(ref, "dst")], "librapid::negate(lhs, dst);"),
		]

	if not any([t.startswith(prefix) for prefix in ["ArrayF", "ArrayCF", "ArrayMP", "ArrayCMP"]]):
		forceTmpFunc += [
			Function("bitwiseOr", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::bitwiseOr(lhs, rhs, dst);"),
			Function("bitwiseAnd", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::bitwiseAnd(lhs, rhs, dst);"),
			Function("bitwiseXor", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::bitwiseXor(lhs, rhs, dst);"),

			Function("bitwiseOr", [Argument(constRef, "lhs"), Argument(scalar, "rhs"), Argument(ref, "dst")], "librapid::bitwiseOr(lhs, rhs, dst);"),
			Function("bitwiseAnd", [Argument(constRef, "lhs"), Argument(scalar, "rhs"), Argument(ref, "dst")], "librapid::bitwiseAnd(lhs, rhs, dst);"),
			Function("bitwiseXor", [Argument(constRef, "lhs"), Argument(scalar, "rhs"), Argument(ref, "dst")], "librapid::bitwiseXor(lhs, rhs, dst);"),

			Function("bitwiseOr", [Argument(scalar, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::bitwiseOr(lhs, rhs, dst);"),
			Function("bitwiseAnd", [Argument(scalar, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::bitwiseAnd(lhs, rhs, dst);"),
			Function("bitwiseXor", [Argument(scalar, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::bitwiseXor(lhs, rhs, dst);"),
			
			Function("bitwiseNot", [Argument(constRef, "lhs"), Argument(ref, "dst")], "librapid::bitwiseNot(lhs, dst);"),
		]

	unaryFunctions = []

	if not any([t.startswith(prefix) for prefix in ["ArrayB", "ArrayC", "ArrayI"]]) and t != "ArrayMPZ" and t != "ArrayMPQ" and t != "ArrayMPF":
		unaryFunctions += [
			Function("sin", [Argument(constRef, "val")], "return lrc::sin(val);"),
			Function("cos", [Argument(constRef, "val")], "return lrc::cos(val);"),
			Function("tan", [Argument(constRef, "val")], "return lrc::tan(val);"),
			Function("asin", [Argument(constRef, "val")], "return lrc::asin(val);"),
			Function("acos", [Argument(constRef, "val")], "return lrc::acos(val);"),
			Function("atan", [Argument(constRef, "val")], "return lrc::atan(val);"),
			Function("sinh", [Argument(constRef, "val")], "return lrc::sinh(val);"),
			Function("cosh", [Argument(constRef, "val")], "return lrc::cosh(val);"),
			Function("tanh", [Argument(constRef, "val")], "return lrc::tanh(val);"),
			Function("asinh", [Argument(constRef, "val")], "return lrc::asinh(val);"),
			Function("acosh", [Argument(constRef, "val")], "return lrc::acosh(val);"),
			Function("atanh", [Argument(constRef, "val")], "return lrc::atanh(val);"),
			Function("exp", [Argument(constRef, "val")], "return lrc::exp(val);"),
			Function("log", [Argument(constRef, "val")], "return lrc::log(val);"),
			Function("sqrt", [Argument(constRef, "val")], "return lrc::sqrt(val);"),
			Function("abs", [Argument(constRef, "val")], "return lrc::abs(val);"),
			Function("floor", [Argument(constRef, "val")], "return lrc::floor(val);"),
			Function("ceil", [Argument(constRef, "val")], "return lrc::ceil(val);"),
		]

	moduleFunctions = forceTmpFunc + unaryFunctions

	for function in moduleFunctions:
		moduleStr += "module" + function.gen() + ";\n"

	interfaceList[t] = classStr + "\n\n" + moduleStr
	classStr = ""
	moduleStr = ""

# def write(path:str):
# 	with open(path, "w") as file:
# 		file.write(classStr)
# 		file.write("\n")
# 		file.write(moduleStr)

def write(path:str):
	for type, interface in interfaceList.items():
		with open(path + "/{}Interface.cpp".format(type), "w") as file:
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

			file.write("void init_{}(py::module &module) {{\n".format(type))
			if type[-1] == "G":
				file.write("#if defined(LIBRAPID_HAS_CUDA)\n")
			file.write(interface)
			if type[-1] == "G":
				file.write("#endif\n")
			file.write("\n}")

if __name__ == "__main__":
	write("../autogen")

	for type, interface in interfaceList.items():
		print("\"${{CMAKE_CURRENT_SOURCE_DIR}}/src/librapid/python/autogen/{}Interface.cpp\"".format(type))

	print("\n")

	for type, interface in interfaceList.items():
		print("void init_{}(py::module &);".format(type))

	print("\n")

	for type, interface in interfaceList.items():
		print("init_{}(module);".format(type))
