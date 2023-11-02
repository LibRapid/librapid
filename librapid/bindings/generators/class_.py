from argument import Argument
import function


class Class:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.functions = []
        self.implicitConversions = []

    def addFunction(self, func):
        self.functions.append(func)
        return self

    def addImplicitConversion(self, other):
        self.implicitConversions.append(other)
        return self

    def genImplicitConversions(self):
        ret = ""
        for other in self.implicitConversions:
            ret += f"py::implicitly_convertible<{other.type}, {self.type}>();\n"

        return ret

    def genInterface(self, parent="module"):
        ret = f"py::class_<{self.type}>({parent}, \"{self.name}\")\n"
        for func in self.functions:
            ret += func.gen(self, False)

            if func is not self.functions[-1]:
                ret += "\n"

        ret += ";\n"

        if len(self.implicitConversions) > 0:
            ret += "\n"
            ret += self.genImplicitConversions()

        return ret

    def __str__(self):
        return self.name


if __name__ == "__main__":
    vector = Class("Vec2d", "lrc::Vec2d")
    vector.addFunction(function.Function(
        name="__init__",
        args=[]
    ))
    vector.addFunction(function.Function(
        name="__init__",
        args=[
            Argument("x", "double"),
            Argument("y", "double", default=0)
        ]
    ))
    vector.addFunction(function.Function(
        name="__init__",
        args=[
            Argument(
                name="other",
                type="lrc::Vec2d",
                const=True,
                ref=True
            )
        ]
    ))

    vector.addFunction(function.Function(
        name="__add__",
        args=[
            Argument(
                name="self",
                type="lrc::Vec2d",
                const=True,
                ref=True
            ),
            Argument(
                name="other",
                type="lrc::Vec2d",
                const=True,
                ref=True
            )
        ],
        op="""
            return (self + other).eval();
        """,
        isOperator=True
    ))

    vector.addFunction(function.Function(
        name="__str__",
        args=[
            Argument(
                name="self",
                type="lrc::Vec2d",
                const=True,
                ref=True
            )
        ],
        op="""
            return fmt::format("{}", self);
        """,
        isOperator=True
    ))

    vector.addImplicitConversion(vector)

    print(vector.genInterface())
