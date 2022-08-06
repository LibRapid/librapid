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

classDefs = ""
moduleDefs = ""

types = [
	["Complex<float>", "ComplexF32", "float"],
	["Complex<double>", "ComplexF64", "double"],
	["Complex<librapid::mpfr>", "ComplexMPFR", "const librapid::mpfr &"]
]

for type in types:
	t = type[0]
	pretty = type[1]
	scalar = type[2]
	typename = "librapid::{}".format(t)
	constRef = "const librapid::{} &".format(t)
	ref = "librapid::{} &".format(t)
	int64_t = "int64_t"
	extent = "librapid::Extent"
	stdString = "std::string"
	stdStringConstRef = "const std::string &"

	# Class Definition
	classDefs += "py::class_<{0}>(module, \"{1}\")\n".format(typename, pretty)

	# Constructors
	classDefs += "\t.def(py::init<>())\n"
	classDefs += "\t.def(py::init<int64_t>())\n".format(scalar)
	classDefs += "\t.def(py::init<int64_t, int64_t>())\n".format(scalar)

	classDefs += "\t.def(py::init<double>())\n".format(scalar)
	classDefs += "\t.def(py::init<double, double>())\n".format(scalar)

	classDefs += "\t.def(py::init<const lrc::mpfr &>())\n".format(scalar)
	classDefs += "\t.def(py::init<const lrc::mpfr &, const lrc::mpfr &>())\n".format(scalar)

	classDefs += "\t.def(py::init<{}>())\n".format(constRef)

	functions = [
		Function("__add__", [Argument(constRef, "this_"), Argument(constRef, "other")],     "return this_ + other;"),
		Function("__sub__", [Argument(constRef, "this_"), Argument(constRef, "other")],     "return this_ - other;"),
		Function("__mul__", [Argument(constRef, "this_"), Argument(constRef, "other")],     "return this_ * other;"),
		Function("__truediv__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ / other;"),

		Function("__radd__", [Argument(constRef, "this_"), Argument(constRef, "other")],     "return other + this_;"),
		Function("__rsub__", [Argument(constRef, "this_"), Argument(constRef, "other")],     "return other - this_;"),
		Function("__rmul__", [Argument(constRef, "this_"), Argument(constRef, "other")],     "return other * this_;"),
		Function("__rtruediv__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return other / this_;"),

		Function("__iadd__", [Argument(ref, "this_"), Argument(constRef, "other")],     "this_ += other; return this_;"),
		Function("__isub__", [Argument(ref, "this_"), Argument(constRef, "other")],     "this_ -= other; return this_;"),
		Function("__imul__", [Argument(ref, "this_"), Argument(constRef, "other")],     "this_ *= other; return this_;"),
		Function("__itruediv__", [Argument(ref, "this_"), Argument(constRef, "other")], "this_ /= other; return this_;"),
		
		Function("str", [Argument(constRef, "this_"), Argument("int64_t", "base", "10")], "return librapid::str(this_, {-1, base, false});"),
		Function("__str__", [Argument(constRef, "this_")], "return librapid::str(this_, {-1, 10, false});"),
		Function("__repr__", [Argument(constRef, "this_")], "return \"librapid::{}(\\\"\" + librapid::str(this_, {{-1, 10, false}}) + \"\\\")\";".format(pretty)),
	]

	for i in range(len(functions)):
		function = functions[i]
		if isinstance(function,Function):
			classDefs += "\t" + function.gen()
		else:
			classDefs += "\t" + function
		
		if i + 1 < len(functions):
			classDefs += "\n"
		else:
			classDefs += ";\n\n"

functions = []

for type in types:
	t = type[0]
	pretty = type[1]
	scalar = type[2]
	typename = "librapid::{}".format(t)
	constRef = "const librapid::{} &".format(t)
	ref = "librapid::{} &".format(t)
	int64_t = "int64_t"
	extent = "librapid::Extent"
	stdString = "std::string"
	stdStringConstRef = "const std::string &"

	if type not in ["librapid::mpz", "librapid::mpq", "librapid::mpf"]:
		functions += [
			Function("abs", [Argument(constRef, "val")], "return librapid::abs(val);"),
			Function("pow", [Argument(scalar, "base"), Argument(constRef, "power")], "return librapid::pow(base, power);"),
			Function("pow", [Argument(constRef, "base"), Argument(scalar, "power")], "return librapid::pow(base, power);"),
			Function("pow", [Argument(constRef, "base"), Argument(constRef, "power")], "return librapid::pow(base, power);"),
			Function("sqrt", [Argument(constRef, "val")], "return librapid::sqrt(val);"),
			Function("exp", [Argument(constRef, "val")], "return librapid::exp(val);"),
			Function("exp2", [Argument(constRef, "val")], "return librapid::exp2(val);"),
			Function("exp10", [Argument(constRef, "val")], "return librapid::exp10(val);"),
			Function("log", [Argument(constRef, "val")], "return librapid::log(val);"),
			Function("log2", [Argument(constRef, "val")], "return librapid::log2(val);"),
			Function("log10", [Argument(constRef, "val")], "return librapid::log10(val);"),
			Function("log", [Argument(constRef, "val"), Argument(constRef, "base")], "return librapid::log(val, base);"),
			Function("log", [Argument(constRef, "val"), Argument(scalar, "base")], "return librapid::log(val, base);"),

			Function("sin", [Argument(constRef, "val")], "return librapid::sin(val);"),
			Function("cos", [Argument(constRef, "val")], "return librapid::cos(val);"),
			Function("tan", [Argument(constRef, "val")], "return librapid::tan(val);"),

			Function("asin", [Argument(constRef, "val")], "return librapid::asin(val);"),
			Function("acos", [Argument(constRef, "val")], "return librapid::acos(val);"),
			Function("atan", [Argument(constRef, "val")], "return librapid::atan(val);"),
			Function("atan2", [Argument(constRef, "a"), Argument(constRef, "b")], "return librapid::atan2(a, b);"),

			Function("csc", [Argument(constRef, "val")], "return librapid::csc(val);"),
			Function("sec", [Argument(constRef, "val")], "return librapid::sec(val);"),
			Function("cot", [Argument(constRef, "val")], "return librapid::cot(val);"),

			Function("acsc", [Argument(constRef, "val")], "return librapid::acsc(val);"),
			Function("asec", [Argument(constRef, "val")], "return librapid::asec(val);"),
			Function("acot", [Argument(constRef, "val")], "return librapid::acot(val);"),

			Function("sinh", [Argument(constRef, "val")], "return librapid::sinh(val);"),
			Function("cosh", [Argument(constRef, "val")], "return librapid::cosh(val);"),
			Function("tanh", [Argument(constRef, "val")], "return librapid::tanh(val);"),

			Function("asinh", [Argument(constRef, "val")], "return librapid::asinh(val);"),
			Function("acosh", [Argument(constRef, "val")], "return librapid::acosh(val);"),
			Function("atanh", [Argument(constRef, "val")], "return librapid::atanh(val);"),

			Function("arg", [Argument(constRef, "val")], "return librapid::arg(val);"),
			Function("proj", [Argument(constRef, "val")], "return librapid::proj(val);"),
			Function("norm", [Argument(constRef, "val")], "return librapid::norm(val);"),
			Function("polar", [Argument(scalar, "rho"), Argument(scalar, "theta")], "return librapid::polar(rho, theta);"),
		]

for i in range(len(functions)):
	function = functions[i]
	if isinstance(function,Function):
		moduleDefs += "\tmodule" + function.gen()
	else:
		moduleDefs += "\t" + function
	moduleDefs += ";\n"

def write(path:str):
	with open(path, "w") as file:
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

		file.write("void init_complex(py::module &module) {\n")
		
		file.write(classDefs)
		file.write("\n\n")
		file.write(moduleDefs)

		file.write("\n}")

if __name__ == "__main__":
	write("../autogen/complexInterface.cpp")
	
	print("\"${{CMAKE_CURRENT_SOURCE_DIR}}/src/librapid/python/autogen/complexInterface.cpp\"".format(type))
	print("void init_complex(py::module &);")
	print("init_complex(module);")
	print()
