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

	def gen(self):
		inputArgs = ""
		if len(self.args) != 0:
			for i in range(len(self.args)):
				inputArgs += "{0} {1}".format(self.args[i].type, self.args[i].name)

				if i + 1 < len(self.args):
					inputArgs += ", "
		
		start = -1
		if len(self.args) > 0:
			start = 0
			if self.args[0].name == "this_":
				start = 1

		arguments = ""
		if len(self.args) > 1 or start == 0:
			arguments = ", "
			for i in range(start, len(self.args)):
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