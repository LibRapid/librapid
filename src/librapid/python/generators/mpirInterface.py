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
	"mpq"
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
	resStr += "\t.def(py::init<const lrc::mpz>())\n"
	resStr += "\t.def(py::init<const lrc::mpf>())\n"
	resStr += "\t.def(py::init<const lrc::mpq>())\n"

	functions = [
		Function("__add__", [Argument(constRef, "this_"), Argument("int64_t", "other")],               "return {}(this_ + other);".format(typename)),
		Function("__add__", [Argument(constRef, "this_"), Argument("double", "other")],                "return {}(this_ + other);".format(typename)),
		Function("__add__", [Argument(constRef, "this_"), Argument("const librapid::mpz &", "other")], "return {}(this_ + other);".format(typename)),
		Function("__add__", [Argument(constRef, "this_"), Argument("const librapid::mpf &", "other")], "return {}(this_ + other);".format(typename)),
		Function("__add__", [Argument(constRef, "this_"), Argument("const librapid::mpq &", "other")], "return {}(this_ + other);".format(typename)),

		Function("__sub__", [Argument(constRef, "this_"), Argument("int64_t", "other")],               "return {}(this_ - other);".format(typename)),
		Function("__sub__", [Argument(constRef, "this_"), Argument("double", "other")],                "return {}(this_ - other);".format(typename)),
		Function("__sub__", [Argument(constRef, "this_"), Argument("const librapid::mpz &", "other")], "return {}(this_ - other);".format(typename)),
		Function("__sub__", [Argument(constRef, "this_"), Argument("const librapid::mpf &", "other")], "return {}(this_ - other);".format(typename)),
		Function("__sub__", [Argument(constRef, "this_"), Argument("const librapid::mpq &", "other")], "return {}(this_ - other);".format(typename)),

		Function("__mul__", [Argument(constRef, "this_"), Argument("int64_t", "other")],               "return {}(this_ * other);".format(typename)),
		Function("__mul__", [Argument(constRef, "this_"), Argument("double", "other")],                "return {}(this_ * other);".format(typename)),
		Function("__mul__", [Argument(constRef, "this_"), Argument("const librapid::mpz &", "other")], "return {}(this_ * other);".format(typename)),
		Function("__mul__", [Argument(constRef, "this_"), Argument("const librapid::mpf &", "other")], "return {}(this_ * other);".format(typename)),
		Function("__mul__", [Argument(constRef, "this_"), Argument("const librapid::mpq &", "other")], "return {}(this_ * other);".format(typename)),

		Function("__truediv__", [Argument(constRef, "this_"), Argument("int64_t", "other")],               "return {}(this_ / other);".format(typename)),
		Function("__truediv__", [Argument(constRef, "this_"), Argument("double", "other")],                "return {}(this_ / other);".format(typename)),
		Function("__truediv__", [Argument(constRef, "this_"), Argument("const librapid::mpz &", "other")], "return {}(this_ / other);".format(typename)),
		Function("__truediv__", [Argument(constRef, "this_"), Argument("const librapid::mpf &", "other")], "return {}(this_ / other);".format(typename)),
		Function("__truediv__", [Argument(constRef, "this_"), Argument("const librapid::mpq &", "other")], "return {}(this_ / other);".format(typename)),


		Function("__radd__", [Argument(constRef, "this_"), Argument("int64_t", "other")],               "return {}(other + this_);".format(typename)),
		Function("__radd__", [Argument(constRef, "this_"), Argument("double", "other")],                "return {}(other + this_);".format(typename)),
		Function("__radd__", [Argument(constRef, "this_"), Argument("const librapid::mpz &", "other")], "return {}(other + this_);".format(typename)),
		Function("__radd__", [Argument(constRef, "this_"), Argument("const librapid::mpf &", "other")], "return {}(other + this_);".format(typename)),
		Function("__radd__", [Argument(constRef, "this_"), Argument("const librapid::mpq &", "other")], "return {}(other + this_);".format(typename)),

		Function("__rsub__", [Argument(constRef, "this_"), Argument("int64_t", "other")],               "return {}(other - this_);".format(typename)),
		Function("__rsub__", [Argument(constRef, "this_"), Argument("double", "other")],                "return {}(other - this_);".format(typename)),
		Function("__rsub__", [Argument(constRef, "this_"), Argument("const librapid::mpz &", "other")], "return {}(other - this_);".format(typename)),
		Function("__rsub__", [Argument(constRef, "this_"), Argument("const librapid::mpf &", "other")], "return {}(other - this_);".format(typename)),
		Function("__rsub__", [Argument(constRef, "this_"), Argument("const librapid::mpq &", "other")], "return {}(other - this_);".format(typename)),

		Function("__rmul__", [Argument(constRef, "this_"), Argument("int64_t", "other")],               "return {}(other * this_);".format(typename)),
		Function("__rmul__", [Argument(constRef, "this_"), Argument("double", "other")],                "return {}(other * this_);".format(typename)),
		Function("__rmul__", [Argument(constRef, "this_"), Argument("const librapid::mpz &", "other")], "return {}(other * this_);".format(typename)),
		Function("__rmul__", [Argument(constRef, "this_"), Argument("const librapid::mpf &", "other")], "return {}(other * this_);".format(typename)),
		Function("__rmul__", [Argument(constRef, "this_"), Argument("const librapid::mpq &", "other")], "return {}(other * this_);".format(typename)),

		Function("__rtruediv__", [Argument(constRef, "this_"), Argument("int64_t", "other")],               "return {}(other / this_);".format(typename)),
		Function("__rtruediv__", [Argument(constRef, "this_"), Argument("double", "other")],                "return {}(other / this_);".format(typename)),
		Function("__rtruediv__", [Argument(constRef, "this_"), Argument("const librapid::mpz &", "other")], "return {}(other / this_);".format(typename)),
		Function("__rtruediv__", [Argument(constRef, "this_"), Argument("const librapid::mpf &", "other")], "return {}(other / this_);".format(typename)),
		Function("__rtruediv__", [Argument(constRef, "this_"), Argument("const librapid::mpq &", "other")], "return {}(other / this_);".format(typename)),

		Function("__iadd__", [Argument(ref, "this_"), Argument("int64_t", "other")],               "this_ += other; return this_;"),
		Function("__iadd__", [Argument(ref, "this_"), Argument("double", "other")],                "this_ += other; return this_;"),
		Function("__iadd__", [Argument(ref, "this_"), Argument("const librapid::mpz &", "other")], "this_ += other; return this_;"),
		Function("__iadd__", [Argument(ref, "this_"), Argument("const librapid::mpf &", "other")], "this_ += other; return this_;"),
		Function("__iadd__", [Argument(ref, "this_"), Argument("const librapid::mpq &", "other")], "this_ += other; return this_;"),

		Function("__isub__", [Argument(ref, "this_"), Argument("int64_t", "other")],               "this_ -= other; return this_;"),
		Function("__isub__", [Argument(ref, "this_"), Argument("double", "other")],                "this_ -= other; return this_;"),
		Function("__isub__", [Argument(ref, "this_"), Argument("const librapid::mpz &", "other")], "this_ -= other; return this_;"),
		Function("__isub__", [Argument(ref, "this_"), Argument("const librapid::mpf &", "other")], "this_ -= other; return this_;"),
		Function("__isub__", [Argument(ref, "this_"), Argument("const librapid::mpq &", "other")], "this_ -= other; return this_;"),

		Function("__imul__", [Argument(ref, "this_"), Argument("int64_t", "other")],               "this_ *= other; return this_;"),
		Function("__imul__", [Argument(ref, "this_"), Argument("double", "other")],                "this_ *= other; return this_;"),
		Function("__imul__", [Argument(ref, "this_"), Argument("const librapid::mpz &", "other")], "this_ *= other; return this_;"),
		Function("__imul__", [Argument(ref, "this_"), Argument("const librapid::mpf &", "other")], "this_ *= other; return this_;"),
		Function("__imul__", [Argument(ref, "this_"), Argument("const librapid::mpq &", "other")], "this_ *= other; return this_;"),

		Function("__itruediv__", [Argument(ref, "this_"), Argument("int64_t", "other")],               "this_ /= other; return this_;"),
		Function("__itruediv__", [Argument(ref, "this_"), Argument("double", "other")],                "this_ /= other; return this_;"),
		Function("__itruediv__", [Argument(ref, "this_"), Argument("const librapid::mpz &", "other")], "this_ /= other; return this_;"),
		Function("__itruediv__", [Argument(ref, "this_"), Argument("const librapid::mpf &", "other")], "this_ /= other; return this_;"),
		Function("__itruediv__", [Argument(ref, "this_"), Argument("const librapid::mpq &", "other")], "this_ /= other; return this_;"),

		Function("__lshift__", [Argument(constRef, "this_"), Argument("int64_t", "other")], "return {}(this_ << other);".format(typename)),
		Function("__rshift__", [Argument(constRef, "this_"), Argument("int64_t", "other")], "return {}(this_ >> other);".format(typename)),
		Function("__ilshift__", [Argument(ref, "this_"), Argument("int64_t", "other")], "this_ <<= other; return this_;"),
		Function("__irshift__", [Argument(ref, "this_"), Argument("int64_t", "other")], "this_ >>= other; return this_;"),

		Function("str", [Argument(constRef, "this_"), Argument("int64_t", "base", "10")], "return lrc::str(this_, {-1, base, false});"),
		Function("__str__", [Argument(constRef, "this_")], "return lrc::str(this_, {-1, 10, false});"),
		Function("__repr__", [Argument(constRef, "this_")], "return \"librapid::{}(\\\"\" + lrc::str(this_, {{-1, 10, false}}) + \"\\\")\";".format(t)),
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

def write(path:str):
	with open(path, "w") as file:
		file.write(resStr)

if __name__ == "__main__":
	print(resStr)
	write("../autogen/mpirInterface.hpp")
