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

arrayTypes = [
	"ArrayBG",
	"ArrayCG",
	"ArrayF16G",
	"ArrayF32G",
	"ArrayF64G",
	"ArrayI16G",
	"ArrayI32G",
	"ArrayI64G",

	"ArrayB",
	"ArrayC",
	"ArrayF16",
	"ArrayF32",
	"ArrayF64",
	"ArrayI16",
	"ArrayI32",
	"ArrayI64",
	
	"ArrayMPZ",
	"ArrayMPQ",
	"ArrayMPFR",

	"ArrayCF32",
	"ArrayCF64",
	"ArrayCMPFR",
]

classStr = ""
moduleStr = ""

interfaceList = {}

for t in arrayTypes:
	typename = "librapid::{}".format(t)
	constRef = "const librapid::{} &".format(t)
	ref = "librapid::{} &".format(t)
	int64_t = "int64_t"
	scalar = "typename librapid::internal::traits<{}>::Scalar".format(typename)
	extent = "librapid::Extent"
	extentConstRef = "const librapid::Extent &"
	stdString = "std::string"
	stdStringConstRef = "const std::string &"

	# Class Definition
	classStr += "py::class_<{0}>(module, \"{1}\")\n".format(typename, t)

	# Constructors
	classStr += "\t.def(py::init<>())\n"
	classStr += "\t.def(py::init<librapid::Extent>())\n"
	classStr += "\t.def(py::init<{}>())\n".format(constRef)
	classStr += "\t.def(py::init<librapid::internal::traits<{}>::Scalar>())\n".format(typename)

	fCopy = [
		Function("copy", [Argument(constRef, "this_")], "return this_.copy();"),
	]

	fIndex = [
		Function("__getitem__", [Argument(constRef, "this_"), Argument(int64_t, "index")], "return this_[index];"),
		Function("__setitem__", [Argument(ref, "this_"), Argument(int64_t, "index"), Argument(scalar, "val")], "this_[index] = val;"),
		Function("__setitem__", [Argument(ref, "this_"), Argument(int64_t, "index"), Argument(scalar, "val")], "this_[index] = val;")
	]

	for i in range(1, 32):
		args = [Argument("int64_t", "index{}".format(ind)) for ind in range(i)]
		params = ", ".join(["index{}".format(ind) for ind in range(i)])
		fIndex.append(Function("__call__", [Argument(constRef, "this_")] + args, "return this_({}).get();".format(params)))
		fIndex.append(Function("get", [Argument(constRef, "this_")] + args, "return this_({}).get();".format(params)))
		fIndex.append(Function("set", [Argument(constRef, "this_"), Argument(scalar, "val")] + args, "this_({}) = val;".format(params)))

	fMove = [
		Function("move_CPU", [Argument(constRef, "this_")], "return this_.move<librapid::device::CPU>();"),
	]

	fMove.append("#if defined(LIBRAPID_HAS_CUDA)")
	fMove.append(Function("move_GPU", [Argument(constRef, "this_")], "return this_.move<librapid::device::GPU>();"))
	fMove.append("#endif // LIBRAPID_HAS_CUDA")

	fCast = []
	
	for t2 in arrayTypes:
		if t2[0] == "#":
			continue

		typename2 = "librapid::{}".format(t2)
		scalar2 = "typename librapid::internal::traits<{}>::Scalar".format(typename2)
		fCast += [
			Function("cast_{}".format(t2), [Argument(constRef, "this_")], "return this_.cast<{}>();".format(scalar2)),
			# Function("castMove_{}_CPU".format(t2), [Argument(constRef, "this_")], "return this_.castMove<{}, librapid::device::CPU>();".format(scalar2)),
		]

		# fCast.append("#if defined(LIBRAPID_HAS_CUDA)")
		# fCast.append(Function("castMove_{}_GPU".format(t2), [Argument(constRef, "this_")], "return this_.castMove<{}, librapid::device::GPU>();".format(scalar2)))
		# fCast.append("#endif // LIBRAPID_HAS_CUDA")

	if t not in ["ArrayB", "ArrayBG"]:
		fArithmetic = [
			Function("__add__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ + other;"),
			Function("__add__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ + other;"),

			Function("__sub__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ - other;"),
			Function("__sub__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ - other;"),

			Function("__mul__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ * other;"),
			Function("__mul__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ * other;"),

			Function("__div__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ / other;"),
			Function("__div__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ / other;"),
		]
	else:
		fArithmetic = []

	if t not in ["ArrayF16", "ArrayF16G", "ArrayF32", "ArrayF32G", "ArrayF64", "ArrayF64G", "ArrayMPFR", "ArrayMPQ", "ArrayCF32", "ArrayCF64", "ArrayCMPFR"]:
		fBitwise = [
			Function("__or__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ | other;"),
			Function("__and__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ & other;"),
			Function("__xor__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ ^ other;"),
		]
	else:
		fBitwise = []

	fUnary = []
	if t not in ["ArrayF16", "ArrayF16G", "ArrayF32", "ArrayF32G", "ArrayF64", "ArrayF64G", "ArrayMPFR", "ArrayMPQ", "ArrayCF32", "ArrayCF64", "ArrayCMPFR"]:
		fUnary += [Function("__invert__", [Argument(constRef, "this_")], "return ~this_;")]

	if not t.startswith("ArrayB"):
		fUnary += [Function("__neg__", [Argument(constRef, "this_")], "return -this_;")]

	fMatrix = [
		Function("transpose", [Argument(ref, "this_"), Argument(extentConstRef, "order", "{}")], "this_.transpose(order);"),
		Function("transposed", [Argument(constRef, "this_"), Argument(extentConstRef, "order", "{}")], "return this_.transposed(order);"),
		Function("dot", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_.dot(other);"),
	]

	fString = [
		Function("str", [Argument(constRef, "this_"),
						 Argument(stdStringConstRef, "format", "\"{}\""),
						 Argument(stdStringConstRef, "delim", "\" \""),
						 Argument(int64_t, "stripWidth", "-1"),
						 Argument(int64_t, "beforePoint", "-1"),
						 Argument(int64_t, "afterPoint", "-1"),
						 Argument(int64_t, "depth", "0")], "return this_.str(format, delim, stripWidth, beforePoint, afterPoint, depth);"),
		Function("__str__", [Argument(constRef, "this_")], "return this_.str();"),
		Function("__repr__", [Argument(constRef, "this_")], "return \"<{0}\\n\" + this_.str(\"{{}}\", \",\") + \"\\n>\";".format(typename))
	]
	
	fProperties = [
		Function("isScalar", [Argument(constRef, "this_")], "return this_.isScalar();"),
		Function("extent", [Argument(constRef, "this_")], "return this_.extent();"),
	]

	functions = fCopy + fIndex + fMove + fCast + fArithmetic + fBitwise + fUnary + fMatrix + fString + fProperties

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

	if t not in ("ArrayB", "ArrayBG"):
		forceTmpFunc = [
			Function("add", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::add(lhs, rhs, dst);"),
			Function("sub", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::sub(lhs, rhs, dst);"),
			Function("mul", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::mul(lhs, rhs, dst);"),
			Function("div", [Argument(constRef, "lhs"), Argument(constRef, "rhs"), Argument(ref, "dst")], "librapid::div(lhs, rhs, dst);"),
		]
	else:
		forceTmpFunc = []

	moduleFunctions = forceTmpFunc

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
