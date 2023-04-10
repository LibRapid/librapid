import itertools


class Swizzler:
    def __init__(self, swiz, vecName="GenericVector"):
        self.swiz = "".join(swiz)
        self.vec_name = vecName

    def getter_definition(self):
        # "LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE GenericVector<Scalar, 2> xy() const;"
        return f"LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE {self.vec_name}<Scalar, {len(self.swiz)}> {self.swiz}() const;"

    def setter_definition(self):
        # "LIBRAPID_ALWAYS_INLINE void xy(const GenericVector<Scalar, 2> &v);"
        return f"LIBRAPID_ALWAYS_INLINE void {self.swiz}(const {self.vec_name}<Scalar, {len(self.swiz)}> &v);"

    def getter_implementation(self):
        """
        template<typename Scalar, int64_t Dims>
        auto GenericVector<Scalar, Dims>::xy() const -> GenericVector<Scalar, 2> {
            return {x(), y()};
        }
        """

        params = ", ".join([x + "()" for x in self.swiz])
        return f"template<typename Scalar, int64_t Dims>\nauto {self.vec_name}<Scalar, Dims>::{self.swiz}() const -> {self.vec_name}<Scalar, {len(self.swiz)}> {{\n    return {{{params}}};\n}}"

    def setter_implementation(self):
        """
        template<typename Scalar, int64_t Dims>
        void GenericVector<Scalar, Dims>::xy(const GenericVector<Scalar, 2> &v) {
            x(v.x());
            y(v.y());
        }
        """

        assignments = ""
        for a, b in zip(self.swiz, list("xyzw")):
            assignments += f"{a}(v.{b}());\n"
        return f"template<typename Scalar, int64_t Dims>\nvoid {self.vec_name}<Scalar, Dims>::{self.swiz}(const {self.vec_name}<Scalar, {len(self.swiz)}> &v) {{\n{assignments}}}"

    def test_getter(self):
        """
        REQUIRE(testC.xy() == lrc::Vec2d(1, 2));
        """
        tmpArgs = {"x": "1", "y": "2", "z": "3", "w": "4"}
        return f"REQUIRE(testC.{self.swiz}() == lrc::Vec{len(self.swiz)}d({', '.join([tmpArgs[v] for v in self.swiz])}));"

    def test_setter(self):
        """
        testC.xy(lrc::Vec2d(1, 2));
        REQUIRE(testC.x() == 1);
        REQUIRE(testC.y() == 2);
        """
        tmpArgs = {"x": "1", "y": "2", "z": "3", "w": "4"}
        return f"testC.{self.swiz}(lrc::Vec{len(self.swiz)}d({', '.join([tmpArgs[v] for v in self.swiz])}));\n" + "\n".join(
            [f"REQUIRE(testC.{v}() == {tmpArgs[v]});" for v in self.swiz])


to_swizzle = ["xy", "xz", "yz", "xyz", "xyw", "xzw", "yzw", "xyzw"]

vecName = "GenericVector"

getter_implementations = []
setter_implementations = []
getter_definitions = []
setter_definitions = []
setters = []
for swiz in to_swizzle:
    for perm in itertools.permutations(list(swiz)):
        swizzler = Swizzler(perm, vecName)
        getter_definitions.append(swizzler.getter_definition())
        setter_definitions.append(swizzler.setter_definition())
        getter_implementations.append(swizzler.getter_implementation())
        setter_implementations.append(swizzler.setter_implementation())

print("============== GETTER DEFINITIONS ==============")
print("\n".join(getter_definitions))
print("============== SETTER DEFINITIONS ==============")
print("\n".join(setter_definitions))
print("============== GETTER IMPLEMENTATIONS ==============")
print("\n\n".join(getter_implementations))
print("============== SETTER IMPLEMENTATIONS ==============")
print("\n\n".join(setter_implementations))

# for swiz in to_swizzle:
#     for perm in itertools.permutations(list(swiz)):
#         term = ", ".join([x + "()" for x in perm])
#         print(
#             f"""LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE {vecName}<Scalar, {len(perm)}> {"".join(perm)}() const {{ return {{ {term} }}; }}""")
#
#         assignments = ""
#         for a, b in zip(perm, list("xyzw")):
#             assignments += f"{a}(vec.{b}());\n"
#         print(
#             f"""template<typename S, size_t D> LIBRAPID_ALWAYS_INLINE void {"".join(perm)}(const {vecName}<S, D> &vec) const {{ {assignments} }}""")

# tmpArgs = {"x": "1", "y": "2", "z": "3", "w": "4"}
# for swiz in to_swizzle:
#     for perm in itertools.permutations(list(swiz)):
#         term = ", ".join([x + "()" for x in perm])
#         print(f"""REQUIRE(testC.{"".join(perm)}() == lrc::Vec{len(perm)}d({", ".join([tmpArgs[v] for v in perm])}));""")
