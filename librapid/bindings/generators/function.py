from argument import Argument
import textwrap

RETURN_TAKE_OWNERSHIP = "take_ownership"
RETURN_COPY = "copy"
RETURN_MOVE = "move"
RETURN_REFERENCE = "reference"
RETURN_REFERENCE_INTERNAL = "reference_internal"
RETURN_AUTO = "automatic"
RETURN_AUTO_REFERENCE = "automatic_reference"


class Function:
    def __init__(self, *args, **kwargs):
        """
        Arguments in order:
        - name
        - args
        - op
        - static
        - property
        - returnPolicy
        - isOperator

        :param args:
        :param kwargs:
        """

        self.name = kwargs.get("name", None)
        self.args = kwargs.get("args", [])
        self.op = kwargs.get("op", None)
        self.static = kwargs.get("static", False)
        self.property = kwargs.get("property", False)
        self.returnPolicy = kwargs.get("returnPolicy", None)
        self.isOperator = kwargs.get("isOperator", False)

        for i in range(len(args)):
            if i == 0 and self.name is None:
                self.name = args[i]
            elif i == 1 and self.args is None:
                self.args = args[i]
            elif i == 2 and self.op is None:
                self.op = args[i]
            elif i == 3 and self.static is None:
                self.static = args[i]
            elif i == 4 and self.property is None:
                self.property = args[i]
            elif i == 5 and self.returnPolicy is None:
                self.returnPolicy = args[i]
            elif i == 6 and self.isOperator is None:
                self.isOperator = args[i]
            else:
                raise ValueError("Too many arguments")

        if self.name is None:
            raise ValueError("Function must have a name")

        self.isConstructor = self.name == "__init__"

        if self.op is None:
            if not self.isConstructor:
                raise ValueError("A non-constructor function must have an operation")
        else:
            self.op = textwrap.dedent(self.op).strip()

        if not isinstance(self.args, (list, tuple)):
            self.args = [self.args]

    def arguments(self, ):
        for arg in self.args:
            if isinstance(arg, Argument):
                yield arg
            else:
                yield Argument(arg)

    def genArgumentStr(self):
        return ", ".join([arg.param() for arg in self.arguments()])

    def pyargs(self):
        args = [arg.pyarg() for arg in self.arguments() if arg.name not in ["self", "*args", "**kwargs"]]
        if self.returnPolicy is not None:
            args.append(f"py::return_value_policy::{self.returnPolicy}")

        if self.isOperator:
            args.append("py::is_operator()")

        return ", ".join(args)

    def gen(self, parent, haveParent=True):
        defType = "def"
        if self.static:
            defType = "def_static"
        elif self.property:
            defType = "def_property"

        pyArgStr = self.pyargs()
        if pyArgStr != "":
            pyArgStr = ", " + pyArgStr

        # Special case for constructors
        if self.isConstructor and self.op is None:
            return textwrap.dedent(f"""
                {parent if haveParent else ""}.def(py::init([]({", ".join([arg.param() for arg in self.arguments()])}) {{
                    return {parent.type}({", ".join([arg.name for arg in self.arguments()])});
                }}){pyArgStr})
            """).strip()

        return textwrap.dedent(f"""
            {parent if haveParent else ""}.{defType}(\"{self.name}\", []({self.genArgumentStr()}) {{
                {self.op}
            }}{pyArgStr})
        """).strip()

    def __str__(self):
        return self.gen("module")


if __name__ == "__main__":
    testFunction = Function(
        name="test",
        args=[
            Argument("a", "int"),
            Argument("b", "int", default=0),
        ],
        op="""
            return a + b;
        """
    )

    print(testFunction.genArgumentStr())
    print(testFunction.pyargs())

    print(testFunction.gen("module"))
