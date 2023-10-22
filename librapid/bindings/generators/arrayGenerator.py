import argument
import function
import class_
import module
import file

import itertools

# The set of Array types we support in Python
arrayTypes = []

for scalar in [("int16_t", "Int16"),
               ("int32_t", "Int32"),
               ("int64_t", "Int64"),
               ("float", "Float"),
               ("double", "Double"),
               ("lrc::Complex<float>", "ComplexFloat"),
               ("lrc::Complex<double>", "ComplexDouble")]:
    for backend in ["CPU", "OpenCL", "CUDA"]:
        arrayTypes.append({
            "scalar": scalar[0],
            "backend": backend,
            "name": f"Array{scalar[1]}{backend}"
        })


def generateCppArrayType(config):
    return f"lrc::Array<{config['scalar']}, lrc::backend::{config['backend']}>"


def generateCppArrayViewType(config):
    return f"lrc::array::GeneralArrayView<{generateCppArrayType(config)}>"


def generateFunctionsForArray(config):
    methods = [
        # Default constructor
        function.Function(
            name="__init__",
            args=[]
        )
    ]

    # Static fromData (n dimensions)
    for n in range(1, 9):
        cppType = ""
        for j in range(n):
            cppType += "std::vector<"
        cppType += config['scalar'] + ">" * n

        methods.append(
            function.Function(
                name="fromData",
                args=[
                    argument.Argument(
                        name=f"array{n}D",
                        type=cppType,
                        const=True,
                        ref=True,
                    )
                ],
                static=True,
                op=f"""
                    return {generateCppArrayType(config)}::fromData(array{n}D);
                """
            )
        )

    methods += [
        # Shape
        function.Function(
            name="__init__",
            args=[
                argument.Argument(
                    name="shape",
                    type="lrc::Shape",
                    const=True,
                    ref=True
                )
            ]
        ),

        # Shape and fill
        function.Function(
            name="__init__",
            args=[
                argument.Argument(
                    name="shape",
                    type="lrc::Shape",
                    const=True,
                    ref=True
                ),
                argument.Argument(
                    name="val",
                    type=config['scalar'],
                    const=True,
                    ref=True
                )
            ]
        ),

        # Copy constructor
        function.Function(
            name="__init__",
            args=[
                argument.Argument(
                    name="other",
                    type=generateCppArrayType(config),
                    const=True,
                    ref=True
                )
            ]
        ),

        # String representation
        function.Function(
            name="__str__",
            args=[
                argument.Argument(
                    name="self",
                    type=generateCppArrayType(config),
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
                    type=generateCppArrayType(config),
                    const=True,
                    ref=True
                )
            ],
            op=f"""
                return fmt::format("<librapid.{config['name']} ~ {{}}>", self.shape());
            """
        ),

        # Format (__format__)
        function.Function(
            name="__format__",
            args=[
                argument.Argument(
                    name="self",
                    type=generateCppArrayType(config),
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


def generateArrayModule(config):
    arrayClass = class_.Class(
        name=config["name"],
        type=generateCppArrayType(config)
    )

    methods, functions = generateFunctionsForArray(config)
    arrayClass.functions.extend(methods)

    includeGuard = None
    if config["backend"] == "CUDA":
        includeGuard = "defined(LIBRAPID_HAS_CUDA)"
    elif config["backend"] == "OpenCL":
        includeGuard = "defined(LIBRAPID_HAS_OPENCL)"

    arrayModule = module.Module(
        name=f"librapid.{config['name']}",
        includeGuard=includeGuard
    )
    arrayModule.addClass(arrayClass)
    arrayModule.functions.extend(functions)

    return arrayModule


def writeArray(root, config):
    fileType = file.File(
        path=f"{root}/{config['name']}.cpp"
    )

    fileType.modules.append(generateArrayModule(config))

    interfaceFunctions = fileType.write()
    # Run clang-format if possible
    try:
        import subprocess

        subprocess.run(["clang-format", "-i", fileType.path])
    except Exception as e:
        print("Unable to run clang-format:", e)

    return interfaceFunctions


def write(root):
    interfaces = []
    for config in arrayTypes:
        interfaces.extend(writeArray(root, config))
    return interfaces
