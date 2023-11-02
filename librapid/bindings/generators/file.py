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
        includes = []
        ret = ""

        root = self.path[:self.path.rfind("/")]

        for module in self.modules:
            moduleInterface, moduleIncludes = module.genInterface(root=root)
            ret += moduleInterface
            ret += "\n"

            interfaceFunctions.append((module.genInterfaceDefinition, module.genInterfaceCall))
            includes += moduleIncludes

        return ret, interfaceFunctions, includes

    def write(self, path=None):
        interfaceFunctions = []
        with open(path if path is not None else self.path, "w") as f:
            f.write("#include \"librapidPython.hpp\"\n\n")
            interface, interfaceFunctionsTmp, includes = self.genInterface()
            for include in includes:
                f.write(f"#include \"{include.strip('../python/generated/')}.hpp\"\n")
            f.write(interface.strip())
            interfaceFunctions.extend(interfaceFunctionsTmp)

        return interfaceFunctions
