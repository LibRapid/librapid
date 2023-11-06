import argument
import function
import class_
import module
import file

import itertools

# The set of Array types we support in Python
arrayTypes = []

for scalar in [("int32_t", "Int32"),
               ("int64_t", "Int64"),
               ("uint32_t", "UInt32"),
               ("uint64_t", "UInt64"),
               ("float", "Float"),
               ("double", "Double"),
               # ("lrc::Complex<float>", "ComplexFloat"),
               # ("lrc::Complex<double>", "ComplexDouble")
               ]:
    for backend in ["CPU"]:  # ["CPU", "OpenCL", "CUDA"]:
        arrayTypes.append({
            "scalar": scalar[0],
            "backend": backend,
            "name": f"GeneralArrayView{scalar[1]}{backend}"
        })


def generateCppArrayType(config):
    return f"lrc::Array<{config['scalar']}, lrc::backend::{config['backend']}>"


def generateCppArrayViewType(config):
    return f"lrc::array::GeneralArrayView<{generateCppArrayType(config)} &, lrc::Shape>"


def generateFunctionsForGeneralArrayView(config):
    methods = [
        # Create a new GeneralArrayView
        function.Function(
            name="createFromArray",
            args=[
                argument.Argument(
                    name="array",
                    type=generateCppArrayType(config),
                    const=False,
                    ref=True
                )
            ],
            op=f"""
                return lrc::createGeneralArrayView(array);
            """,
            static=True
        ),

        # Get item
        function.Function(
            name="__getitem__",
            args=[
                argument.Argument(
                    name="self",
                    type=generateCppArrayViewType(config),
                    const=True,
                    ref=True
                ),
                argument.Argument(
                    name="index",
                    type="int64_t"
                )
            ],
            op="""
                return self[index];
            """
        ),

        # Set item (GeneralArrayView)
        function.Function(
            name="__setitem__",
            args=[
                argument.Argument(
                    name="self",
                    type=generateCppArrayViewType(config),
                    const=False,
                    ref=True
                ),
                argument.Argument(
                    name="index",
                    type="int64_t"
                ),
                argument.Argument(
                    name="other",
                    type=generateCppArrayViewType(config),
                    const=True,
                    ref=True
                )
            ],
            op="""
                self[index] = other;
                return self;
            """
        ),

        # Set item (Array)
        function.Function(
            name="__setitem__",
            args=[
                argument.Argument(
                    name="self",
                    type=generateCppArrayViewType(config),
                    const=False,
                    ref=True
                ),
                argument.Argument(
                    name="index",
                    type="int64_t"
                ),
                argument.Argument(
                    name="other",
                    type=generateCppArrayType(config),
                    const=True,
                    ref=True
                )
            ],
            op="""
                self[index] = other;
                return self;
            """
        ),

        # Set item (Scalar)
        function.Function(
            name="__setitem__",
            args=[
                argument.Argument(
                    name="self",
                    type=generateCppArrayViewType(config),
                    const=False,
                    ref=True
                ),
                argument.Argument(
                    name="index",
                    type="int64_t"
                ),
                argument.Argument(
                    name="other",
                    type=config["scalar"],
                    const=True,
                    ref=True
                )
            ],
            op="""
                self[index] = other;
                return self;
            """
        )
    ]

    for operation in [("add", "+"), ("sub", "-"), ("mul", "*"), ("div", "/")]:
        for dtype in [generateCppArrayViewType(config), generateCppArrayType(config), config["scalar"]]:
            methods.append(
                function.Function(
                    name=f"__i{operation[0]}__",
                    args=[
                        argument.Argument(
                            name="self",
                            type=generateCppArrayViewType(config),
                            const=False,
                            ref=True
                        ),
                        argument.Argument(
                            name="other",
                            type=dtype,
                            const=True,
                            ref=True
                        )
                    ],
                    op=f"""
                        self += other;
                        return self;
                    """
                )
            )

            methods.append(
                function.Function(
                    name=f"__{operation[0]}__",
                    args=[
                        argument.Argument(
                            name="self",
                            type=generateCppArrayViewType(config),
                            const=True,
                            ref=True
                        ),
                        argument.Argument(
                            name="other",
                            type=dtype,
                            const=True,
                            ref=True
                        )
                    ],
                    op=f"""
                        // Release the GIL to improve performance
                        py::gil_scoped_release release;
                        return (self {operation[1]} other).eval();
                    """
                )
            )

            methods.append(
                function.Function(
                    name=f"__r{operation[0]}__",
                    args=[
                        argument.Argument(
                            name="self",
                            type=generateCppArrayViewType(config),
                            const=True,
                            ref=True
                        ),
                        argument.Argument(
                            name="other",
                            type=dtype,
                            const=True,
                            ref=True
                        )
                    ],
                    op=f"""
                        return (other {operation[1]} self).eval();
                    """
                )
            )

    methods += [
        # String representation
        function.Function(
            name="__str__",
            args=[
                argument.Argument(
                    name="self",
                    type=generateCppArrayViewType(config),
                    const=True,
                    ref=True
                )
            ],
            op="""
                return fmt::format("{}", self);
            """
        ),

        # String representation
        function.Function(
            name="__repr__",
            args=[
                argument.Argument(
                    name="self",
                    type=generateCppArrayViewType(config),
                    const=True,
                    ref=True
                )
            ],
            op=f"""
                std::string thisStr = fmt::format("{{}}", self);
                std::string padded;
                for (const auto &c : thisStr) {{
                    padded += c;
                    if (c == '\\n') {{
                        padded += std::string(27, ' ');
                    }}
                }}
                return fmt::format("<librapid.GeneralArrayView {{}} dtype={config['scalar']} backend={config['backend']}>", padded);
            """
        ),

        # Format (__format__)
        function.Function(
            name="__format__",
            args=[
                argument.Argument(
                    name="self",
                    type=generateCppArrayViewType(config),
                    const=True,
                    ref=True
                ),
                argument.Argument(
                    name="formatSpec",
                    type="std::string",
                    const=True,
                    ref=True
                )
            ],
            op="""
                std::string format = fmt::format("{{:{}}}", formatSpec);
                return fmt::format(fmt::runtime(format), self);
            """
        )
    ]

    return methods, []


def generateGeneralArrayViewModule(config):
    generalArrayViewClass = class_.Class(
        name=config["name"],
        type=generateCppArrayViewType(config)
    )

    methods, functions = generateFunctionsForGeneralArrayView(config)
    generalArrayViewClass.functions.extend(methods)

    includeGuard = None
    if config["backend"] == "CUDA":
        includeGuard = "defined(LIBRAPID_HAS_CUDA)"
    elif config["backend"] == "OpenCL":
        includeGuard = "defined(LIBRAPID_HAS_OPENCL)"

    generalArrayViewModule = module.Module(
        name=f"librapid.GeneralArrayView.{config['name']}",
        includeGuard=includeGuard
    )
    generalArrayViewModule.addClass(generalArrayViewClass)
    generalArrayViewModule.functions.extend(functions)

    return generalArrayViewModule


def writeGeneralArrayView(root, config):
    fileType = file.File(
        path=f"{root}/GeneralArrayView_{config['name']}.cpp"
    )
    fileType.modules.append(generateGeneralArrayViewModule(config))
    interfaceFunctions = fileType.write()
    return interfaceFunctions


def write(root):
    interfaces = []
    for config in arrayTypes:
        interfaces.extend(writeGeneralArrayView(root, config))
    return interfaces
