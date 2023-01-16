import itertools

toSwizzle = ["xy", "xz", "yz", "xyz", "xyw", "xzw", "yzw", "xyzw"]

for swiz in toSwizzle:
    for perm in itertools.permutations(list(swiz)):
        term = ", ".join([x + "()" for x in perm])
        print(f"""LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Vec<Scalar, {len(perm)}> {"".join(perm)}() const {{ return {{ {term} }}; }}""")

tmpArgs = {"x": "1", "y": "2", "z": "3", "w": "4"}
for swiz in toSwizzle:
    for perm in itertools.permutations(list(swiz)):
        term = ", ".join([x + "()" for x in perm])
        print(f"""REQUIRE(testC.{"".join(perm)}() == lrc::Vec{len(perm)}d({", ".join([tmpArgs[v] for v in perm])}));""")
