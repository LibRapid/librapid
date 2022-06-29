# Detect LibRapid features

arrayTypes = [
	"ArrayB",
	"ArrayC",
	"ArrayF16",
	"ArrayF32",
	"ArrayF64",
	"ArrayI16",
	"ArrayI32",
	"ArrayI64",

	"#if defined(LIBRAPID_HAS_CUDA)",

	"ArrayBG",
	"ArrayCG",
	"ArrayF16G",
	"ArrayF32G",
	"ArrayF64G",
	"ArrayI16G",
	"ArrayI32G",
	"ArrayI64G",
	
	"#endif // LIBRAPID_HAS_CUDA"
]

class Argument:
	def __init__(self, type:str, name:str, default:str = None):
		self.type = type.strip().lstrip()
		self.name = name.strip().lstrip()
		self.default = default

	def setType(self, type:str):
		self.type = type

	def hasDefault(self):
		return self.default is not None

	def __str__(self) -> str:
		return self.type + " " + self.name

class Function:
	def __init__(self, name:str, args:list, op:str):
		self.name = name
		self.args = args
		self.op = op

	def gen(self, type:str):
		inputArgs = ""
		if len(self.args) != 0:
			for i in range(len(self.args)):
				inputArgs += "{0} {1}".format(self.args[i].type, self.args[i].name)

				if i + 1 < len(self.args):
					inputArgs += ", "
		
		arguments = ""
		if len(self.args) > 0 and self.args[0].name == "this_":
			tmpArgs = self.args[1:]
		else:
			tmpArgs = self.args[:]

		if len(tmpArgs) > 0:
			arguments = ", "
			for i in range(len(tmpArgs)):
				arguments += "py::arg(\"{0}\")".format(tmpArgs[i].name, tmpArgs[i].type)

				if tmpArgs[i].hasDefault():
					arguments += " = {0}(".format(tmpArgs[i].type.strip("&").lstrip("const").strip()) + tmpArgs[i].default + ")"

				if i + 1 < len(tmpArgs):
					arguments += ", "

		return ".def(\"{0}\", []({1}) {{ {2} }}{3})".format(self.name, inputArgs, self.op, arguments)

classStr = ""
moduleStr = ""

for t in arrayTypes:
	if t[0] == "#":
		classStr += "\n" + t + "\n"
		moduleStr += "\n" + t + "\n"
		continue

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
		Function("copy", [Argument(constRef, "this_")], "return this_.copy();")
	]

	fIndex = [
		Function("__getitem__", [Argument(constRef, "this_"), Argument(int64_t, "index")], "return this_[index];"),
		Function("__setitem__", [Argument(ref, "this_"), Argument(int64_t, "index"), Argument(scalar, "val")], "this_[index] = val;"),
		Function("__setitem__", [Argument(ref, "this_"), Argument(int64_t, "index"), Argument(scalar, "val")], "this_[index] = val;")
	]

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
			Function("cast_{}".format(t), [Argument(constRef, "this_")], "return this_.cast<{}>();".format(scalar2)),
			Function("castMove_{}_CPU".format(t2), [Argument(constRef, "this_")], "return this_.castMove<{}, librapid::device::CPU>();".format(scalar2)),
		]

		fCast.append("#if defined(LIBRAPID_HAS_CUDA)")
		fCast.append(Function("castMove_{}_GPU".format(t2), [Argument(constRef, "this_")], "return this_.castMove<{}, librapid::device::GPU>();".format(scalar2)))
		fCast.append("#endif // LIBRAPID_HAS_CUDA")

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

	if t not in ["ArrayF16", "ArrayF16G", "ArrayF32", "ArrayF32G", "ArrayF64", "ArrayF64G"]:
		fBitwise = [
			Function("__or__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ | other;"),
			Function("__and__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ & other;"),
			Function("__xor__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ ^ other;"),
		]
	else:
		fBitwise = []

	if t not in ["ArrayF16", "ArrayF16G", "ArrayF32", "ArrayF32G", "ArrayF64", "ArrayF64G"]:
		fUnary = [Function("__invert__", [Argument(constRef, "this_")], "return ~this_;")]
		if not t.startswith("ArrayB"):
			fUnary = [Function("__neg__", [Argument(constRef, "this_")], "return -this_;")]
	else:
		fUnary = []

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

	functions = fCopy + fIndex + fMove + fCast + fArithmetic + fBitwise + fUnary + fMatrix + fString

	for i in range(len(functions)):
		function = functions[i]
		if isinstance(function,Function):
			classStr += "\t" + function.gen(t)
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
		moduleStr += "module" + function.gen(t) + ";\n"

def write(path:str):
	with open(path, "w") as file:
		file.write(classStr)
		file.write("\n")
		file.write(moduleStr)

if __name__ == "__main__":
	print(classStr)
	print("\n")
	print(moduleStr)
	write("../autogen/arrayInterface.hpp")
