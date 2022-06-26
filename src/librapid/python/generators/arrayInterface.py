# Detect LibRapid features
features = []
try:
    with open("../configuration.txt", "r") as file:
        for line in file:
            if line.startswith("#"):
                continue

            args = line.split()
            features.append((args[0], args[2])) # Cut out the "="
except:
    pass

arrayTypes = [
    "ArrayB",
    # "ArrayC",
    # "ArrayF16",
    "ArrayF32",
    # "ArrayF64",
    # "ArrayI16",
    # "ArrayI32",
    # "ArrayI64"
]

# GPU Arrays
if ("LIBRAPID_HAS_CUDA", "TRUE") in features:
    arrayTypes += [
        "ArrayBG",
        "ArrayCG",
        "ArrayF16G",
        "ArrayF32G",
        "ArrayF64G",
        "ArrayI16G",
        "ArrayI32G",
        "ArrayI64G"
    ]

class Argument:
    def __init__(self, type:str, name:str):
        self.type = type.strip().lstrip()
        self.name = name.strip().lstrip()

    def setType(self, type:str):
        self.type = type

    def __str__(self) -> str:
        return self.type + " " + self.name

class Function:
    def __init__(self, prettyName:str, name:str, templateArgs:list):
        self.prettyName = prettyName
        self.name = name
        self.templateArgs = templateArgs

    def gen(self, type:str):
        templates = ""
        if len(self.templateArgs) != 0:
            templates = "<"
            for i in range(len(self.templateArgs)):
                templates += "librapid::" + self.templateArgs[i]
                if i + 1 < len(self.templateArgs):
                    templates += ", "
            templates += ">"

        return ".def(\"{0}\", &librapid::{1}::{2}{3})".format(self.prettyName, type, self.name, templates)

    def __str__(self):
        return "help me"

for t in arrayTypes:
    # Class Definition
    print("\tpy::class_<librapid::{0}>(module, \"{0}\")".format(t))

    # Constructors
    print("\t\t.def(py::init<>())")
    print("\t\t.def(py::init<librapid::Extent>())")
    print("\t\t.def(py::init<librapid::{}>())".format(t))
    print("\t\t.def(py::init<librapid::internal::traits<librapid::{}>::Scalar>())".format(t))

    fCopy = [
        Function("copy", "copy", []), # Copy
    ]

    fIndex = [
        ".def(\"__getitem__\", [](const librapid::{} &arr, int64_t index) {{ return arr[index]; }})".format(t), # Get Item Indexing
        ".def(\"__setitem__\", [](const librapid::{0} &arr, int64_t index, librapid::internal::traits<librapid::{0}>::Scalar val) {{ arr[index] = val; }})".format(t), # Set Item (Scalar)
        ".def(\"__setitem__\", [](const librapid::{0} &arr, int64_t index, const librapid::{0} &val) {{ arr[index] = val; }})".format(t), # Set Item (Array)
    ]

    fMove = [
        Function("move_CPU", "move", ["device::CPU"]), # Move CPU
        Function("move_GPU", "move", ["device::GPU"]), # Move GPU
    ]

    if t not in ["ArrayB", "ArrayBG"]:
        fArithmetic = [
            Function("__add__", "operator+", [t]), 
            Function("__add__", "operator+", ["internal::traits<librapid::{}>::Scalar".format(t)]), 

            Function("__sub__", "operator-", [t]), 
            Function("__sub__", "operator-", ["internal::traits<librapid::{}>::Scalar".format(t)]), 

            Function("__mul__", "operator*", [t]), 
            Function("__mul__", "operator*", ["internal::traits<librapid::{}>::Scalar".format(t)]), 

            Function("__div__", "operator/", [t]), 
            Function("__div__", "operator/", ["internal::traits<librapid::{}>::Scalar".format(t)]), 
        ]
    else:
        fArithmetic = []

    if t not in ["ArrayF16", "ArrayF16G", "ArrayF32", "ArrayF32G", "ArrayF64", "ArrayF64G"]:
        fBitwise = [
            Function("__or__", "operator|", [t]), 
            # Function("__or__", "operator|", ["internal::traits<librapid::{}>::Scalar".format(t)]), 

            Function("__and__", "operator&", [t]), 
            # Function("__and__", "operator&", ["internal::traits<librapid::{}>::Scalar".format(t)]), 

            Function("__xor__", "operator^", [t]), 
            # Function("__xor__", "operator^", ["internal::traits<librapid::{}>::Scalar".format(t)]), 
        ]
    else:
        fBitwise = []

    if t not in ["ArrayF16", "ArrayF16G", "ArrayF32", "ArrayF32G", "ArrayF64", "ArrayF64G"]:
        fUnary = [Function("__invert__", "operator~", [])]
        if t != "ArrayB":
            fUnary.append(Function("__neg__", "operator-", []))
    else:
        fUnary = []

    fMatrix = [
        Function("transpose", "transpose", []), # Transpose
        Function("transposed", "transposed", []), # Transposed
        Function("dot", "dot", [t]), # Transposed
    ]

    fString = [
        Function("str", "str", []), # String
        Function("__str__", "str", []), # String
        ".def(\"__repr__\", [](const librapid::{0} &arr) {{ return \"<librapid.{0}\\n\" + arr.str() + \"\\n>\"; }})".format(t)
    ]

    functions = fCopy + fIndex + fMove + fArithmetic + fBitwise + fUnary + fMatrix + fString

    # Casting
    for t2 in arrayTypes:
        functions.append(Function("cast_{}".format(t2), "cast", [t2]))

        functions.append(Function("castMove_{}_CPU".format(t2), "castMove", ["internal::traits<{}>::Scalar".format(t2), "device::CPU"])) # Cast and Move to CPU
        functions.append(Function("castMove_{}_GPU".format(t2), "castMove", ["internal::traits<{}>::Scalar".format(t2), "device::GPU"])) # Cast and Move to GPU

    for i in range(len(functions)):
        function = functions[i]
        if isinstance(function, Function):
            print("\t\t" + function.gen(t), end="")
        else:
            print("\t\t" + function, end="")
        
        if i + 1 < len(functions):
            print("")
        else:
            print(";\n")
