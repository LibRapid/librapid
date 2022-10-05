import itertools

toSwizzle = ["xy", "xz", "yz", "xyz", "xyw", "xzw", "yzw", "xyzw"]

for swiz in toSwizzle:
    for perm in itertools.permutations(list(swiz)):
        term = ", ".join([x + "()" for x in perm])
        print(f"""LR_FORCE_INLINE Vec<Scalar, {len(perm)}> {"".join(perm)}() const {{ return {{ {term} }}; }}""")
