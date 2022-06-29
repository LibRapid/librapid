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
	def __init__(self, name:str, args:list, op:str, **kwargs):
		self.name = name
		self.args = args
		self.op = op
		self.static = False

		if "static" in kwargs:
			self.static = kwargs["static"]

	def gen(self, type:str):
		inputArgs = ""
		if len(self.args) != 0:
			for i in range(len(self.args)):
				inputArgs += "{0} {1}".format(self.args[i].type, self.args[i].name)

				if i + 1 < len(self.args):
					inputArgs += ", "
		
		arguments = ""
		if len(self.args) > 1:
			arguments = ", "
			for i in range(1, len(self.args)):
				arguments += "py::arg(\"{0}\")".format(self.args[i].name, self.args[i].type)

				if self.args[i].hasDefault():
					arguments += " = {0}(".format(self.args[i].type.strip("&").lstrip("const").strip()) + self.args[i].default + ")"

				if i + 1 < len(self.args):
					arguments += ", "

		funcType = None
		if self.static:
			funcType = ".def_static"
		else:
			funcType = ".def"

		return "{4}(\"{0}\", []({1}) {{ {2} }}{3})".format(self.name, inputArgs, self.op, arguments, funcType)

resStr = ""

t = "Extent"
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
resStr += "\t.def(py::init<const std::vector<int64_t> &>())\n"
resStr += "\t.def(py::init<{}>())\n".format(constRef)

functions = [
	Function("zero", [Argument(int64_t, "dims")], "return librapid::Extent::zero(dims);", static=True),
	Function("stride", [Argument(constRef, "this_")], "return this_.stride();"),
	Function("strideAdjusted", [Argument(constRef, "this_")], "return this_.strideAdjusted();"),
	Function("index", [Argument(constRef, "this_"), Argument(constRef, "ind")], "return this_.index(ind);"),
	Function("indexAdjusted", [Argument(constRef, "this_"), Argument(constRef, "ind")], "return this_.indexAdjusted(ind);"),
	Function("indexAdjusted", [Argument(constRef, "this_"), Argument(constRef, "ind")], "return this_.indexAdjusted(ind);"),
	Function("reverseIndex", [Argument(constRef, "this_"), Argument(int64_t, "ind")], "return this_.reverseIndex(ind);"),
	Function("reverseIndex", [Argument(constRef, "this_"), Argument(int64_t, "ind")], "return this_.reverseIndex(ind);"),
	Function("partial", [Argument(constRef, "this_"), Argument(int64_t, "start", "0"), Argument(int64_t, "end", "-1")], "return this_.partial(start, end);"),
	Function("swivel", [Argument(constRef, "this_"), Argument(constRef, "order")], "return this_.swivel(order);"),
	Function("swivelInplace", [Argument(ref, "this_"), Argument(constRef, "order")], "this_.swivelInplace(order);"),
	Function("size", [Argument(constRef, "this_")], "return this_.size();"),
	Function("sizeAdjusted", [Argument(constRef, "this_")], "return this_.sizeAdjusted();"),
	Function("__getitem__", [Argument(constRef, "this_"), Argument(int64_t, "index")], "return this_[index];"),
	Function("__setitem__", [Argument(ref, "this_"), Argument(int64_t, "index"), Argument(int64_t, "val")], "this_[index] = val;"),
	Function("adjusted", [Argument(constRef, "this_"), Argument(int64_t, "ind")], "return this_.adjusted(ind);"),
	Function("__eq__", [Argument(constRef, "this_"), Argument(constRef, "other")], "return this_ == other;"),
	Function("str", [Argument(constRef, "this_")], "return this_.str();"),
	Function("__str__", [Argument(constRef, "this_")], "return this_.str();"),
	Function("__repr__", [Argument(constRef, "this_")], "return \"<librapid::\" + this_.str() + \">\";"),
]

for i in range(len(functions)):
	function = functions[i]
	if isinstance(function,Function):
		resStr += "\t" + function.gen(t)
	else:
		resStr += "\t" + function
	
	if i + 1 < len(functions):
		resStr += "\n"
	else:
		resStr += ";\n"

def write(path:str):
	with open(path, "w") as file:
		file.write(resStr)

if __name__ == "__main__":
	print(resStr)
	write("../autogen/extentInterface.hpp")
