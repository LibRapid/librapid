import re


def isPrimitive(type):
    if type in ["bool", "int", "char", "float", "double"]:
        return True

    # Match int, int32_t, int64_t, uint32_t etc.
    if re.match(r"u?int\d+_t", type):
        return True

    return False


class Argument:
    def __init__(self, *args, **kwargs):
        """
        Arguments may include (in order):
        - name
        - type
        - default
        - const
        - ref
        - pointer
        - noConvert
        - returnPolicy

        :param args:
        :param kwargs (optional):
        """

        self.name = kwargs.get("name", None)
        self.type = kwargs.get("type", None)
        self.default = kwargs.get("default", None)
        self.const = kwargs.get("const", True)
        self.ref = kwargs.get("ref", True)
        self.pointer = kwargs.get("pointer", False)
        self.noConvert = kwargs.get("noConvert", False)

        for i in range(len(args)):
            if i == 0 and self.name is None:
                self.name = args[i]
            elif i == 1 and self.type is None:
                self.type = args[i]
            elif i == 2 and self.default is None:
                self.default = args[i]
            elif i == 3 and self.const is None:
                self.const = args[i]
            elif i == 4 and self.ref is None:
                self.ref = args[i]
            elif i == 5 and self.pointer is None:
                self.pointer = args[i]
            elif i == 6 and self.noConvert is None:
                self.noConvert = args[i]
            else:
                raise ValueError("Too many arguments")

        if self.name is None:
            raise ValueError("Argument must have a name")

        if self.type is None:
            raise ValueError("Argument must have a type")

        self.isArgs = self.name == "*args"
        self.isKwargs = self.name == "**kwargs"

    def param(self):
        if self.isArgs:
            return f"py::args args"
        elif self.isKwargs:
            return f"py::kwargs kwargs"
        else:
            isPrimitiveType = isPrimitive(self.type)
            return f"{'const ' if self.const and not isPrimitiveType else ''}{self.type} {'&' if self.ref and not isPrimitiveType else ''}{'*' if self.pointer else ''}{self.name}"

    def declaration(self):
        if self.default is None:
            return f"{self.type} {self.name}"
        else:
            return f"{self.type} {self.name} = {self.default}"

    def pyarg(self):
        if self.default is not None:
            return f"py::arg_v(\"{self.name}\", {self.default}, \"{self.name} = {self.default}\"){'.noconvert()' if self.noConvert else ''}"
        else:
            return f"py::arg(\"{self.name}\"){'.noconvert()' if self.noConvert else ''}"

    def __str__(self):
        return self.name


if __name__ == "__main__":
    # Run some tests

    normalArgNoDefault1 = Argument("arg1", "int")
    normalArgNoDefault2 = Argument(name="arg1", type="int")

    normalArgDefault1 = Argument("arg1", "int", 0)
    normalArgDefault2 = Argument(name="arg1", type="int", default=0)

    constArgNoDefault1 = Argument("arg1", "int", const=True)
    constArgNoDefault2 = Argument(name="arg1", type="int", const=True)

    refArgNoDefault1 = Argument("arg1", "int", ref=True, noConvert=True)
    refArgNoDefault2 = Argument(name="arg1", type="int", ref=True, noConvert=True)

    noConvertArgDefault = Argument(name="arg1", type="int", default=0, noConvert=True)

    print(f"Normal Argument ~ No Default ~ 1 ~ Param: {normalArgNoDefault1.param()}")
    print(f"Normal Argument ~ No Default ~ 2 ~ Param: {normalArgNoDefault2.param()}")
    print(f"Normal Argument ~ No Default ~ 1 ~ Declaration: {normalArgNoDefault1.declaration()}")
    print(f"Normal Argument ~ No Default ~ 2 ~ Declaration: {normalArgNoDefault2.declaration()}")
    print(f"Normal Argument ~ No Default ~ 1 ~ PyArg: {normalArgNoDefault1.pyarg()}")
    print(f"Normal Argument ~ No Default ~ 2 ~ PyArg: {normalArgNoDefault2.pyarg()}")

    print(f"Normal Argument ~ Default ~ 1 ~ Param: {normalArgDefault1.param()}")
    print(f"Normal Argument ~ Default ~ 2 ~ Param: {normalArgDefault2.param()}")
    print(f"Normal Argument ~ Default ~ 1 ~ Declaration: {normalArgDefault1.declaration()}")
    print(f"Normal Argument ~ Default ~ 2 ~ Declaration: {normalArgDefault2.declaration()}")
    print(f"Normal Argument ~ Default ~ 1 ~ PyArg: {normalArgDefault1.pyarg()}")
    print(f"Normal Argument ~ Default ~ 2 ~ PyArg: {normalArgDefault2.pyarg()}")

    print(f"Const Argument ~ No Default ~ 1 ~ Param: {constArgNoDefault1.param()}")
    print(f"Const Argument ~ No Default ~ 2 ~ Param: {constArgNoDefault2.param()}")
    print(f"Const Argument ~ No Default ~ 1 ~ Declaration: {constArgNoDefault1.declaration()}")
    print(f"Const Argument ~ No Default ~ 2 ~ Declaration: {constArgNoDefault2.declaration()}")
    print(f"Const Argument ~ No Default ~ 1 ~ PyArg: {constArgNoDefault1.pyarg()}")
    print(f"Const Argument ~ No Default ~ 2 ~ PyArg: {constArgNoDefault2.pyarg()}")

    print(f"Ref Argument ~ No Default ~ 1 ~ Param: {refArgNoDefault1.param()}")
    print(f"Ref Argument ~ No Default ~ 2 ~ Param: {refArgNoDefault2.param()}")
    print(f"Ref Argument ~ No Default ~ 1 ~ Declaration: {refArgNoDefault1.declaration()}")
    print(f"Ref Argument ~ No Default ~ 2 ~ Declaration: {refArgNoDefault2.declaration()}")
    print(f"Ref Argument ~ No Default ~ 1 ~ PyArg: {refArgNoDefault1.pyarg()}")
    print(f"Ref Argument ~ No Default ~ 2 ~ PyArg: {refArgNoDefault2.pyarg()}")

    print(f"No Convert Argument ~ Default ~ 1 ~ Param: {noConvertArgDefault.param()}")
    print(f"No Convert Argument ~ Default ~ 1 ~ Declaration: {noConvertArgDefault.declaration()}")
    print(f"No Convert Argument ~ Default ~ 1 ~ PyArg: {noConvertArgDefault.pyarg()}")
