DTYPES = [
    # {
    #     "name": "char",
    #     "cname": "char",
    #     "floating": False
    # },
    # {
    #     "name": "uchar",
    #     "cname": "unsigned char",
    #     "floating": False
    # },
    # {
    #     "name": "short",
    #     "cname": "short",
    #     "floating": False
    # },
    # {
    #     "name": "ushort",
    #     "cname": "unsigned short",
    #     "floating": False
    # },
    # {
    #     "name": "int",
    #     "cname": "int",
    #     "floating": False
    # },
    # {
    #     "name": "uint",
    #     "cname": "unsigned int",
    #     "floating": False
    # },
    # {
    #     "name": "long",
    #     "cname": "long",
    #     "floating": False
    # },
    # {
    #     "name": "ulong",
    #     "cname": "unsigned long",
    #     "floating": False
    # },
    # {
    #     "name": "longlong",
    #     "cname": "long long",
    #     "floating": False
    # },
    # {
    #     "name": "ulonglong",
    #     "cname": "unsigned long long",
    #     "floating": False
    # },
    {
        "name": "F",
        "cname": "FL",
        "floating": True
    },
    {
        "name": "D",
        "cname": "DL",
        "floating": True
    }
]
FLOATING = ["float", "double"]
MIN_DIMS = 2
MAX_DIMS = 4

FUNCS = [
    {
        "name": "+",
        "op": "+",
        "opType": "binary",
        "inplace": False,
        "requireFloating": False
    },
    {
        "name": "-",
        "op": "-",
        "opType": "binary",
        "inplace": False,
        "requireFloating": False
    },
    {
        "name": "*",
        "op": "*",
        "opType": "binary",
        "inplace": False,
        "requireFloating": False
    },
    {
        "name": "/",
        "op": "/",
        "opType": "binary",
        "inplace": False,
        "requireFloating": False
    },
    {
        "name": "+=",
        "op": "+",
        "opType": "binary",
        "inplace": True,
        "requireFloating": False
    },
    {
        "name": "-=",
        "op": "-",
        "opType": "binary",
        "inplace": True,
        "requireFloating": False
    },
    {
        "name": "*=",
        "op": "*",
        "opType": "binary",
        "inplace": True,
        "requireFloating": False
    },
    {
        "name": "/=",
        "op": "/",
        "opType": "binary",
        "inplace": True,
        "requireFloating": False
    },
    {
        "name": ">",
        "op": ">",
        "opType": "binary",
        "inplace": False,
        "requireFloating": False
    },
    {
        "name": "<",
        "op": "<",
        "opType": "binary",
        "inplace": False,
        "requireFloating": False
    },
    {
        "name": ">=",
        "op": ">=",
        "opType": "binary",
        "inplace": False,
        "requireFloating": False
    },
    {
        "name": "<=",
        "op": "<=",
        "opType": "binary",
        "inplace": False,
        "requireFloating": False
    },
    {
        "name": "==",
        "op": "==",
        "opType": "binary",
        "inplace": False,
        "requireFloating": False
    },
    {
        "name": "!=",
        "op": "!=",
        "opType": "binary",
        "inplace": False,
        "requireFloating": False
    },
    {
        "name": "sin",
        "op": "sin",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "cos",
        "op": "cos",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "tan",
        "op": "tan",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "asin",
        "op": "asin",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "acos",
        "op": "acos",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "atan",
        "op": "atan",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "sinh",
        "op": "sinh",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "cosh",
        "op": "cosh",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "tanh",
        "op": "tanh",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "asinh",
        "op": "asinh",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "acosh",
        "op": "acosh",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "atanh",
        "op": "atanh",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "exp",
        "op": "exp",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "log",
        "op": "log",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "log2",
        "op": "log2",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "log10",
        "op": "log10",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "sqrt",
        "op": "sqrt",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "cbrt",
        "op": "cbrt",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "abs",
        "op": "abs",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "ceil",
        "op": "ceil",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    },
    {
        "name": "floor",
        "op": "floor",
        "opType": "unary",
        "inplace": False,
        "requireFloating": True
    }
]

def gen():
    ret = ""
    for type in DTYPES:
        for func in FUNCS:
            if func["requireFloating"] and not type["floating"]:
                continue

            for dims in range(MIN_DIMS, MAX_DIMS + 1):
                if func["opType"] == "binary":
                    if func["inplace"]:
                        op = func["op"]
                        block = f"""IOF({type["name"]}{dims},{func["name"]})({type["name"]}{dims} &a,CO {type["name"]}{dims} &b){{M{type["name"]}{dims}N({",".join([f"a.{var}{op}b.{var}" for var in "xyzw"[:dims]])});}}
IOF({type["name"]}{dims},{func["name"]})({type["name"]}{dims} &a,CO {type["cname"]} &b){{M{type["name"]}{dims}N({",".join([f"a.{var} {op} b" for var in "xyzw"[:dims]])});}}
"""

                        ret += block
                    else:
                        op = func["op"]
                        block = f"""IOF({type["name"]}{dims},{func["name"]})(CO {type["name"]}{dims} &a,CO {type["name"]}{dims} &b){{M{type["name"]}{dims}({",".join([f"a.{var}{op}b.{var}" for var in "xyzw"[:dims]])});}}
IOF({type["name"]}{dims},{func["name"]})(CO {type["name"]}{dims} &a,CO {type["cname"]} &b){{M{type["name"]}{dims}({",".join([f"a.{var}{op}b" for var in "xyzw"[:dims]])});}}
IOF({type["name"]}{dims},{func["name"]})(CO {type["cname"]} &a,CO {type["name"]}{dims} &b){{M{type["name"]}{dims}({",".join([f"a{op}b.{var}" for var in "xyzw"[:dims]])});}}
"""

                        ret += block
                elif func["opType"] == "unary":
                    op = func["op"]
                    block = f"""inline {type["name"]}{dims} {func["name"]}(CO {type["name"]}{dims} &a){{M{type["name"]}{dims}({",".join([f"{op}(a.{var})" for var in "xyzw"[:dims]])});}}"""
                    ret += block + "\n"

    return ret

print(gen())
