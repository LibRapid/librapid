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
	["Vec3i", "i32", 3],
	["Vec4i", "i32", 4],
	["Vec2f", "f32", 2],
	["Vec3f", "f32", 3],
	["Vec4f", "f32", 4],
	["Vec2d", "f64", 2],
	["Vec3d", "f64", 3],
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
		args = ", ".join([scalar] * (i + 1))
		classStr += "\t.def(py::init<{}>())\n".format(args)

	functions = [
		Function("__getitem__", [Argument(constRef, "this_"), Argument(int64_t, "index")], "return this_[index];"),
		Function("__setitem__", [Argument(ref, "this_"), Argument(int64_t, "index"), Argument(scalar, "val")], "this_[index] = val;"),

		Function("__add__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ + other;"),
		Function("__sub__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ - other;"),
		Function("__mul__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ * other;"),
		Function("__truediv__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ / other;"),

		Function("__iadd__", [Argument(ref, "this_"), Argument(constRef, "other")], "this_ += other; return this_;"),
		Function("__isub__", [Argument(ref, "this_"), Argument(constRef, "other")], "this_ -= other; return this_;"),
		Function("__imul__", [Argument(ref, "this_"), Argument(constRef, "other")], "this_ *= other; return this_;"),
		Function("__idiv__", [Argument(ref, "this_"), Argument(constRef, "other")], "this_ /= other; return this_;"),

		Function("__radd__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return other + this_;"),
		Function("__rsub__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return other - this_;"),
		Function("__rmul__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return other * this_;"),
		Function("__rtruediv__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return other / this_;"),

		Function("__add__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ + other;"),
		Function("__sub__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ - other;"),
		Function("__mul__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ * other;"),
		Function("__truediv__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ / other;"),

		Function("__iadd__", [Argument(ref, "this_"), Argument(scalar, "other")], "this_ += other; return this_;"),
		Function("__isub__", [Argument(ref, "this_"), Argument(scalar, "other")], "this_ -= other; return this_;"),
		Function("__imul__", [Argument(ref, "this_"), Argument(scalar, "other")], "this_ *= other; return this_;"),
		Function("__itruediv__", [Argument(ref, "this_"), Argument(scalar, "other")], "this_ /= other; return this_;"),

		Function("__radd__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return other + this_;"),
		Function("__rsub__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return other - this_;"),
		Function("__rmul__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return other * this_;"),
		Function("__rtruediv__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return other / this_;"),

		Function("__neg__", [Argument(constRef, "this_")], "return -this_;"),

		Function("cmp", [Argument(constRef, "this_"), Argument(constRef, "other"), Argument("const char *", "mode")], "return this_.cmp(other, mode);"),
		Function("cmp", [Argument(constRef, "this_"), Argument(scalar, "other"), Argument("const char *", "mode")], "return this_.cmp(other, mode);"),

		Function("__lt__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ < other;"),
		Function("__le__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ <= other;"),
		Function("__gt__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ > other;"),
		Function("__ge__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ >= other;"),
		Function("__eq__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ == other;"),
		Function("__ne__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ != other;"),

		Function("__lt__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ < other;"),
		Function("__le__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ <= other;"),
		Function("__gt__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ > other;"),
		Function("__ge__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ >= other;"),
		Function("__eq__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ == other;"),
		Function("__ne__", [Argument(constRef, "this_"), Argument(scalar, "other")], "return this_ != other;"),

		Function("mag2", [Argument(constRef, "this_")], "return this_.mag2();"),
		Function("mag", [Argument(constRef, "this_")], "return this_.mag();"),
		Function("invMag", [Argument(constRef, "this_")], "return this_.invMag();"),
		
		Function("norm", [Argument(constRef, "this_")], "return this_.norm();"),
		Function("dot", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_.dot(other);"),
		
		Function("__bool__", [Argument(constRef, "this_")], "return (bool) this_;"),
		Function("__str__", [Argument(constRef, "this_")], "return this_.str();"),
		Function("__repr__", [Argument(constRef, "this_")], "return std::string(\"librapid::{}\") + this_.str();".format(t[0])),

		Function("x", [Argument(constRef, "this_")], "return this_.x();"),
		Function("y", [Argument(constRef, "this_")], "return this_.y();"),
		Function("z", [Argument(constRef, "this_")], "return this_.z();"),
		Function("w", [Argument(constRef, "this_")], "return this_.w();"),
		Function("xy", [Argument(constRef, "this_")], "return this_.xy();"),
		Function("yx", [Argument(constRef, "this_")], "return this_.yx();"),
		Function("xz", [Argument(constRef, "this_")], "return this_.xz();"),
		Function("zx", [Argument(constRef, "this_")], "return this_.zx();"),
		Function("yz", [Argument(constRef, "this_")], "return this_.yz();"),
		Function("zy", [Argument(constRef, "this_")], "return this_.zy();"),
		Function("xyz", [Argument(constRef, "this_")], "return this_.xyz();"),
		Function("xzy", [Argument(constRef, "this_")], "return this_.xzy();"),
		Function("yxz", [Argument(constRef, "this_")], "return this_.yxz();"),
		Function("yzx", [Argument(constRef, "this_")], "return this_.yzx();"),
		Function("zxy", [Argument(constRef, "this_")], "return this_.zxy();"),
		Function("zyx", [Argument(constRef, "this_")], "return this_.zyx();"),
		Function("xyw", [Argument(constRef, "this_")], "return this_.xyw();"),
		Function("xwy", [Argument(constRef, "this_")], "return this_.xwy();"),
		Function("yxw", [Argument(constRef, "this_")], "return this_.yxw();"),
		Function("ywx", [Argument(constRef, "this_")], "return this_.ywx();"),
		Function("wxy", [Argument(constRef, "this_")], "return this_.wxy();"),
		Function("wyx", [Argument(constRef, "this_")], "return this_.wyx();"),
		Function("xzw", [Argument(constRef, "this_")], "return this_.xzw();"),
		Function("xwz", [Argument(constRef, "this_")], "return this_.xwz();"),
		Function("zxw", [Argument(constRef, "this_")], "return this_.zxw();"),
		Function("zwx", [Argument(constRef, "this_")], "return this_.zwx();"),
		Function("wxz", [Argument(constRef, "this_")], "return this_.wxz();"),
		Function("wzx", [Argument(constRef, "this_")], "return this_.wzx();"),
		Function("yzw", [Argument(constRef, "this_")], "return this_.yzw();"),
		Function("ywz", [Argument(constRef, "this_")], "return this_.ywz();"),
		Function("zyw", [Argument(constRef, "this_")], "return this_.zyw();"),
		Function("zwy", [Argument(constRef, "this_")], "return this_.zwy();"),
		Function("wyz", [Argument(constRef, "this_")], "return this_.wyz();"),
		Function("wzy", [Argument(constRef, "this_")], "return this_.wzy();"),
		Function("xyzw", [Argument(constRef, "this_")], "return this_.xyzw();"),
		Function("xywz", [Argument(constRef, "this_")], "return this_.xywz();"),
		Function("xzyw", [Argument(constRef, "this_")], "return this_.xzyw();"),
		Function("xzwy", [Argument(constRef, "this_")], "return this_.xzwy();"),
		Function("xwyz", [Argument(constRef, "this_")], "return this_.xwyz();"),
		Function("xwzy", [Argument(constRef, "this_")], "return this_.xwzy();"),
		Function("yxzw", [Argument(constRef, "this_")], "return this_.yxzw();"),
		Function("yxwz", [Argument(constRef, "this_")], "return this_.yxwz();"),
		Function("yzxw", [Argument(constRef, "this_")], "return this_.yzxw();"),
		Function("yzwx", [Argument(constRef, "this_")], "return this_.yzwx();"),
		Function("ywxz", [Argument(constRef, "this_")], "return this_.ywxz();"),
		Function("ywzx", [Argument(constRef, "this_")], "return this_.ywzx();"),
		Function("zxyw", [Argument(constRef, "this_")], "return this_.zxyw();"),
		Function("zxwy", [Argument(constRef, "this_")], "return this_.zxwy();"),
		Function("zyxw", [Argument(constRef, "this_")], "return this_.zyxw();"),
		Function("zywx", [Argument(constRef, "this_")], "return this_.zywx();"),
		Function("zwxy", [Argument(constRef, "this_")], "return this_.zwxy();"),
		Function("zwyx", [Argument(constRef, "this_")], "return this_.zwyx();"),
		Function("wxyz", [Argument(constRef, "this_")], "return this_.wxyz();"),
		Function("wxzy", [Argument(constRef, "this_")], "return this_.wxzy();"),
		Function("wyxz", [Argument(constRef, "this_")], "return this_.wyxz();"),
		Function("wyzx", [Argument(constRef, "this_")], "return this_.wyzx();"),
		Function("wzxy", [Argument(constRef, "this_")], "return this_.wzxy();"),
		Function("wzyx", [Argument(constRef, "this_")], "return this_.wzyx();"),
	]

	if dims == 3:
		functions.append(Function("cross", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_.cross(other);"))

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
	moduleFunctions = [
		Function("dist2", [Argument(constRef, "lhs"), Argument(constRef, "rhs")], "return lrc::dist2(lhs, rhs);"),
		Function("dist2", [Argument(constRef, "lhs"), Argument(constRef, "rhs")], "return lrc::dist(lhs, rhs);"),
		Function("abs", [Argument(constRef, "val")], "return lrc::abs(val);"),
	]

	if "i" not in t[0]:
		moduleFunctions += [
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
			Function("floor", [Argument(constRef, "val")], "return lrc::floor(val);"),
			Function("ceil", [Argument(constRef, "val")], "return lrc::ceil(val);"),
		]

	for function in moduleFunctions:
		moduleStr += "module" + function.gen() + ";\n"

	interfaceList[t[0]] = classStr + "\n\n" + moduleStr
	classStr = ""
	moduleStr = ""

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
