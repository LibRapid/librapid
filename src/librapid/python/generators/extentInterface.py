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
	Function("swivelled", [Argument(constRef, "this_"), Argument(constRef, "order")], "return this_.swivelled(order);"),
	Function("swivel", [Argument(ref, "this_"), Argument(constRef, "order")], "this_.swivel(order);"),
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
		resStr += "\t" + function.gen()
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
	write("../autogen/extentInterface.hpp")
