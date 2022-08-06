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

resStr = ""

types = [
	"mpz",
	"mpf",
	"mpq",
	"mpfr"
]

for type in types:
	t = type
	typename = "librapid::{}".format(t)
	constRef = "const librapid::{} &".format(t)
	ref = "librapid::{} &".format(t)
	int64_t = "int64_t"
	extent = "librapid::Extent"
	stdString = "std::string"
	stdStringConstRef = "const std::string &"

	# Class Definition
	resStr += "py::class_<{0}>(module, \"{1}\")\n".format(typename, t)

	# Constructors
	resStr += "\t.def(py::init<>())\n"
	resStr += "\t.def(py::init<int64_t>())\n"
	resStr += "\t.def(py::init<double>())\n"
	resStr += "\t.def(py::init<const std::string &>())\n"
	resStr += "\t.def(py::init<{}>())\n".format(constRef)

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
		
		Function("__lt__", [Argument(constRef, "this_"), Argument(constRef, "other")],  "return this_ < other;"),
		Function("__gt__", [Argument(constRef, "this_"), Argument(constRef, "other")],  "return this_ > other;"),
		Function("__lte__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ <= other;"),
		Function("__gte__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ >= other;"),

		Function("str", [Argument(constRef, "this_"), Argument("int64_t", "base", "10")], "return lrc::str(this_, {-1, base, false});"),
		Function("__str__", [Argument(constRef, "this_")], "return lrc::str(this_, {-1, 10, false});"),
		Function("__repr__", [Argument(constRef, "this_")], "return \"librapid::{}(\\\"\" + lrc::str(this_, {{-1, 10, false}}) + \"\\\")\";".format(t)),
	]

	if type != "mpfr":
		functions += [
			Function("__lshift__", [Argument(constRef, "this_"), Argument("int64_t", "other")], "return this_ << other;"),
			Function("__rshift__", [Argument(constRef, "this_"), Argument("int64_t", "other")], "return this_ >> other;"),
			Function("__ilshift__", [Argument(ref, "this_"), Argument("int64_t", "other")], "this_ <<= other; return this_;"),
			Function("__irshift__", [Argument(ref, "this_"), Argument("int64_t", "other")], "this_ >>= other; return this_;"),
		]

	for i in range(len(functions)):
		function = functions[i]
		if isinstance(function,Function):
			resStr += "\t" + function.gen()
		else:
			resStr += "\t" + function
		
		if i + 1 < len(functions):
			resStr += "\n"
		else:
			resStr += ";\n\n"

for type in types:
	functions = [
		Function("toMpz", [Argument(constRef, "this_")], "return librapid::toMpz(this_);"),
		Function("toMpq", [Argument(constRef, "this_")], "return librapid::toMpq(this_);"),
		Function("toMpfr", [Argument(constRef, "this_")], "return librapid::toMpfr(this_);"),
	]

	for function in functions:
		resStr += "module" + function.gen() + ";\n"

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

		file.write("void init_{}(py::module &module) {{\n".format(type))
		
		file.write(resStr)

		file.write("\n}")

if __name__ == "__main__":
	write("../autogen/mpfrInterface.cpp")

	print("\"${{CMAKE_CURRENT_SOURCE_DIR}}/src/librapid/python/autogen/mpfrInterface.cpp\"".format(type))
	print("void init_mpfr(py::module &);")
	print("init_mpfr(module);")
	print()
