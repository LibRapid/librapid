import itertools

toSwizzle = ["xy", "xyz", "xyzw"]

for swiz in toSwizzle:
    for perm in itertools.permutations(list(swiz)):
        term = ", ".join([x + "()" for x in perm])
        print(f"""inline VecImpl<Scalar, {len(perm)}, StorageType> {"".join(perm)}() const {{ return {{ {term} }}; }}""")
