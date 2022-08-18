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
	# Scalar types
	"int64_t",
	"double",
	"librapid::mpz",
	"librapid::mpf",
	"librapid::mpq",
	"librapid::mpfr",

	# Complex number types
	"librapid::Complex<float>",
	"librapid::Complex<double>",
	"librapid::Complex<librapid::mpfr>"
]

functions = []

for type in types:
	t = type
	typename = "{}".format(t)
	constRef = "const {} &".format(t)
	ref = "{} &".format(t)
	int64_t = "int64_t"
	extent = "librapid::Extent"
	stdString = "std::string"
	stdStringConstRef = "const std::string &"

	if type not in ["librapid::mpz", "librapid::mpq", "librapid::mpf"]:
		functions += [
			Function("abs", [Argument(constRef, "val")], "return lrc::abs(val);"),
			Function("floor", [Argument(constRef, "val")], "return lrc::floor(val);"),
			Function("ceil", [Argument(constRef, "val")], "return lrc::ceil(val);"),
			Function("pow", [Argument(constRef, "base"), Argument(constRef, "power")], "return lrc::pow(base, power);"),
			Function("sqrt", [Argument(constRef, "val")], "return lrc::sqrt(val);"),
			Function("exp", [Argument(constRef, "val")], "return lrc::exp(val);"),
			Function("exp2", [Argument(constRef, "val")], "return lrc::exp2(val);"),
			Function("exp10", [Argument(constRef, "val")], "return lrc::exp10(val);"),
			Function("log", [Argument(constRef, "val")], "return lrc::log(val);"),
			Function("log2", [Argument(constRef, "val")], "return lrc::log2(val);"),
			Function("log10", [Argument(constRef, "val")], "return lrc::log10(val);"),

			Function("sin", [Argument(constRef, "val")], "return lrc::sin(val);"),
			Function("cos", [Argument(constRef, "val")], "return lrc::cos(val);"),
			Function("tan", [Argument(constRef, "val")], "return lrc::tan(val);"),

			Function("asin", [Argument(constRef, "val")], "return lrc::asin(val);"),
			Function("acos", [Argument(constRef, "val")], "return lrc::acos(val);"),
			Function("atan", [Argument(constRef, "val")], "return lrc::atan(val);"),
			Function("atan2", [Argument(constRef, "a"), Argument(constRef, "b")], "return lrc::atan2(a, b);"),

			Function("csc", [Argument(constRef, "val")], "return lrc::csc(val);"),
			Function("sec", [Argument(constRef, "val")], "return lrc::sec(val);"),
			Function("cot", [Argument(constRef, "val")], "return lrc::cot(val);"),

			Function("acsc", [Argument(constRef, "val")], "return lrc::acsc(val);"),
			Function("asec", [Argument(constRef, "val")], "return lrc::asec(val);"),
			Function("acot", [Argument(constRef, "val")], "return lrc::acot(val);"),

			Function("sinh", [Argument(constRef, "val")], "return lrc::sinh(val);"),
			Function("cosh", [Argument(constRef, "val")], "return lrc::cosh(val);"),
			Function("tanh", [Argument(constRef, "val")], "return lrc::tanh(val);"),

			Function("asinh", [Argument(constRef, "val")], "return lrc::asinh(val);"),
			Function("acosh", [Argument(constRef, "val")], "return lrc::acosh(val);"),
			Function("atanh", [Argument(constRef, "val")], "return lrc::atanh(val);"),

			Function("mod", [Argument(constRef, "val"), Argument(constRef, "divisor")], "return lrc::mod(val, divisor);"),
			Function("round", [Argument(constRef, "val"), Argument("int64_t", "dp", "0")], "return lrc::round(val, dp);"),
			Function("roundSigFig", [Argument(constRef, "val"), Argument("int64_t", "dp", "3")], "return lrc::roundSigFig(val, dp);"),
			Function("roundTo", [Argument(constRef, "val"), Argument(constRef, "num")], "return lrc::roundTo(val, num);"),
			Function("roundUpTo", [Argument(constRef, "val"), Argument(constRef, "num")], "return lrc::roundUpTo(val, num);"),
		]

functions += [
	Function("map", [Argument("double", "val"), 
					 Argument("double", "start1"),
					 Argument("double", "stop1"),
					 Argument("double", "start2"),
					 Argument("double", "stop2")], "return lrc::map(val, start1, stop1, start2, stop2);"),
	Function("random", [Argument("double", "lower", "0"), Argument("double", "upper", "1"), Argument("int64_t", "seed", "-1")], "return librapid::random(lower, upper, seed);"),
	Function("randint", [Argument("int64_t", "lower", "0"), Argument("int64_t", "upper", "0"), Argument("int64_t", "seed", "-1")], "return librapid::randint(lower, upper, seed);"),
	Function("trueRandomEntropy", [], "return librapid::trueRandomEntropy();"),
	Function("trueRandom", [Argument("double", "lower", "0"), Argument("double", "upper", "1")], "return librapid::trueRandom(lower, upper);"),
	Function("trueRandint", [Argument("int64_t", "lower", "0"), Argument("int64_t", "upper", "1")], "return librapid::trueRandint(lower, upper);"),
	Function("randomGaussian", [], "return librapid::randomGaussian();"),
	Function("pow10", [Argument("int64_t", "exponent")], "return librapid::pow10(exponent);"),
	Function("lerp", [Argument("double", "a"), Argument("double", "b"), Argument("double", "t")], "return librapid::lerp(a, b, t);"),
]

for i in range(len(functions)):
	function = functions[i]
	if isinstance(function,Function):
		resStr += "\tmodule" + function.gen()
	else:
		resStr += "\t" + function
	resStr += ";\n"

def write(path:str):
	with open(path.format(type), "w") as file:
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

		file.write("void init_math(py::module &module) {\n")
		file.write(resStr)
		file.write("\n}")

if __name__ == "__main__":
	write("../autogen/mathInterface.cpp")

	print("\"${CMAKE_CURRENT_SOURCE_DIR}/src/librapid/python/autogen/mathInterface.cpp\"")
	print("\n")
	print("void init_math(py::module &);")
	print("\n")
	print("init_math(module);")