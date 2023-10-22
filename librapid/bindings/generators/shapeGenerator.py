import argument
import function
import class_
import module
import file

methods = [
    # Default constructor
    function.Function(
        name="__init__",
        args=[]
    ),

    # List of values
    function.Function(
        name="__init__",
        args=[
            argument.Argument(
                name="vals",
                type="std::vector<uint32_t>",
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
                type="lrc::Shape",
                const=True,
                ref=True
            )
        ]
    ),

    # zeros
    function.Function(
        name="zeros",
        args=[
            argument.Argument(
                name="dims",
                type="int",
                const=True,
                ref=False
            )
        ],
        static=True,
        op="""
            return lrc::Shape::zeros(dims);
        """
    ),

    # ones
    function.Function(
        name="ones",
        args=[
            argument.Argument(
                name="dims",
                type="int",
                const=True,
                ref=False
            )
        ],
        static=True,
        op="""
            return lrc::Shape::ones(dims);
        """
    ),

    # Indexing (get)
    function.Function(
        name="__getitem__",
        args=[
            argument.Argument(
                name="self",
                type="lrc::Shape",
                const=True,
                ref=True
            ),
            argument.Argument(
                name="index",
                type="int",
                const=True,
                ref=False
            )
        ],
        op="""
            return self[index];
        """
    ),

    # Indexing (set)
    function.Function(
        name="__setitem__",
        args=[
            argument.Argument(
                name="self",
                type="lrc::Shape",
                const=False,
                ref=True
            ),
            argument.Argument(
                name="index",
                type="int",
                const=True,
                ref=False
            ),
            argument.Argument(
                name="value",
                type="uint32_t",
                const=True,
                ref=False
            )
        ],
        op="""
            self[index] = value;
        """,
        isOperator=True
    ),

    # Equality
    function.Function(
        name="__eq__",
        args=[
            argument.Argument(
                name="self",
                type="lrc::Shape",
                const=True,
                ref=True
            ),
            argument.Argument(
                name="other",
                type="lrc::Shape",
                const=True,
                ref=True
            )
        ],
        op="""
            return self == other;
        """,
        isOperator=True
    ),

    # Inequality
    function.Function(
        name="__ne__",
        args=[
            argument.Argument(
                name="self",
                type="lrc::Shape",
                const=True,
                ref=True
            ),
            argument.Argument(
                name="other",
                type="lrc::Shape",
                const=True,
                ref=True
            )
        ],
        op="""
            return self != other;
        """,
        isOperator=True
    ),

    # Number of dimensions
    function.Function(
        name="ndim",
        args=[
            argument.Argument(
                name="self",
                type="lrc::Shape",
                const=True,
                ref=True
            )
        ],
        op="""
            return self.ndim();
        """
    ),

    # Subshape
    function.Function(
        name="subshape",
        args=[
            argument.Argument(
                name="self",
                type="lrc::Shape",
                const=True,
                ref=True
            ),
            argument.Argument(
                name="start",
                type="int",
                const=True,
                ref=False
            ),
            argument.Argument(
                name="end",
                type="int",
                const=True,
                ref=False
            )
        ],
        op="""
            return self.subshape(start, end);
        """
    ),

    # Size
    function.Function(
        name="size",
        args=[
            argument.Argument(
                name="self",
                type="lrc::Shape",
                const=True,
                ref=True
            )
        ],
        op="""
            return self.size();
        """
    ),

    # To string (__str__)
    function.Function(
        name="__str__",
        args=[
            argument.Argument(
                name="self",
                type="lrc::Shape",
                const=True,
                ref=True
            )
        ],
        op="""
            return fmt::format("{}", self);
        """
    ),

    # To string (__repr__)
    function.Function(
        name="__repr__",
        args=[
            argument.Argument(
                name="self",
                type="lrc::Shape",
                const=True,
                ref=True
            )
        ],
        op="""
            return fmt::format("_librapid.{}", self);
        """
    ),

    # Format (__format__)
    function.Function(
        name="__format__",
        args=[
            argument.Argument(
                name="self",
                type="lrc::Shape",
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

classType = class_.Class(
    name="Shape",
    type="lrc::Shape",
)

classType.functions = methods

moduleType = module.Module(
    name="librapid.shape"
)

moduleType.classes.append(classType)


def write(root):
    fileType = file.File(
        path=f"{root}/shape.cpp"
    )

    fileType.modules.append(moduleType)

    interfaceFunctions = fileType.write()
    # Run clang-format if possible
    try:
        import subprocess

        subprocess.run(["clang-format", "-i", fileType.path])
    except Exception as e:
        print("Unable to run clang-format:", e)

    return interfaceFunctions
