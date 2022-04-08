import itertools

toSwizzle = ["xy", "xyz", "xyzw"]

for swiz in toSwizzle:
    for perm in itertools.permutations(list(swiz)):
        term = ", ".join(perm)
        print(f"""
inline Vec<DTYPE, {len(perm)}> {"".join(perm)}() const
{{
    return {{ {term} }};
}}
""")
