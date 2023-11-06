import textwrap


class File:
    def __init__(self, path=None, docstring=None):
        self.path = path if path is not None else "./module.cpp"
        self.modules = []

    def addModule(self, module):
        self.modules.append(module)
        return self

    def genInterface(self):
        interfaceFunctions = []
        ret = ""
        for module in self.modules:
            ret += module.genInterface()
            ret += "\n"

            interfaceFunctions.append((module.genInterfaceDefinition, module.genInterfaceCall))

        return ret, interfaceFunctions

    def write(self, path=None):
        interfaceFunctions = []
        with open(path if path is not None else self.path, "w") as f:
            f.write("#include \"librapidPython.hpp\"\n\n")
            interface, interfaceFunctionsTmp = self.genInterface()
            f.write(interface.strip())
            interfaceFunctions.extend(interfaceFunctionsTmp)

        return interfaceFunctions
