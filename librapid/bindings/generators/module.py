from class_ import Class


class Module:
    def __init__(self, name, parentModule=None, docstring=None):
        self.name = name
        self.parent = parentModule
        self.docstring = docstring if docstring is not None else f"Bindings for {name}"
        self.classes = []
        self.functions = []

    def addClass(self, class_):
        self.classes.append(class_)
        return self

    def addFunction(self, func):
        self.functions.append(func)
        return self

    def genInterfaceDefinition(self):
        tmpName = self.name.replace(".", "_")
        return f"void genInterface_{tmpName}(py::module& module)"

    def genInterfaceCall(self, moduleName):
        tmpName = self.name.replace(".", "_")
        return f"genInterface_{tmpName}({moduleName})"

    def genInterface(self):
        ret = f"{self.genInterfaceDefinition()} {{\n"

        if self.parent is None:
            moduleName = "module"
        else:
            ret += f"module.def_submodule(\"{self.name}\", \"{self.parent.name}.{self.name}\") {{\n"
            moduleName = self.name

        ret += f"{moduleName}.doc() = \"{self.docstring}\";\n\n"

        for class_ in self.classes:
            ret += class_.genInterface(moduleName)
            ret += "\n"

        for func in self.functions:
            ret += func.gen(moduleName)
            ret += "\n"

        ret += "}\n"
        return ret


if __name__ == "__main__":
    testModule = Module("test")
