from argument import Argument
import function
import os
import textwrap
import boilerplate


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

    def genInterface(self, parent="module", root="./", includeGuard=None):
        mainInterface = f"py::class_<{self.type}> {self.name}Class({parent}, \"{self.name}\");\n"
        includes = []

        # Ensure directory exists
        if not os.path.exists(f"{root}/{self.name}"):
            os.makedirs(f"{root}/{self.name}")

        # Ensure function names are unique
        functionCount = 0

        for func in self.functions:
            functionName = f"librapidPython_{self.name}_{func.name}_{functionCount}"
            fileName = f"{func.name}_{functionCount}"
            filePath = f"{root}/{self.name}/{fileName}"

            # Write definition
            with open(f"{filePath}.hpp", "w") as f:
                f.write(textwrap.dedent(f"""
                    {boilerplate.boilerplate()}
                    
                    void {functionName}(py::class_<{self.type}>& module);
                """))
                includes.append(f"{filePath}.hpp")

            # Write implementation
            with open(f"{filePath}.cpp", "w") as f:
                f.write(f"#include \"{fileName}.hpp\"\n")

                if includeGuard is not None:
                    f.write(f"#if {includeGuard}\n")

                f.write(textwrap.dedent(f"""
                    void {functionName}(py::class_<{self.type}>& {self.name}) {{
                        {func.gen(self, True)};
                    }}
                """))

                if includeGuard is not None:
                    f.write(f"#else\n")
                    f.write(textwrap.dedent(f"""
                        void {functionName}(py::class_<{self.type}>& module) {{
                            return;
                        }}
                    """))
                    f.write(f"#endif\n")

            # Add function call to interface
            mainInterface += f"{functionName}({self.name}Class);\n"

            functionCount += 1

            # ret += func.gen(self, False)
            #
            # if func is not self.functions[-1]:
            #     ret += "\n"

        if len(self.implicitConversions) > 0:
            mainInterface += "\n"
            mainInterface += self.genImplicitConversions()

        return mainInterface, includes

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

    mainInterface, includes = vector.genInterface(root="../python/generated", includeGuard="defined(LIBRAPID_HAS_CUDA)")
    print(mainInterface)
    print(includes)
