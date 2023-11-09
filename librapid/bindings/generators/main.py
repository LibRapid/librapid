import os
import textwrap

import boilerplate
import shapeGenerator
import arrayGenerator
import generalArrayViewGenerator

outputDir = "../python/generated"


def main():
    # Ensure the output directory exists
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    interfaceFunctions = []

    interfaceFunctions += shapeGenerator.write(outputDir)
    interfaceFunctions += arrayGenerator.write(outputDir)
    interfaceFunctions += generalArrayViewGenerator.write(outputDir)

    with open(f"{outputDir}/librapidPython.hpp", "w") as f:
        f.write(boilerplate.boilerplate)
        f.write("\n\n")
        for interfaceDef, _ in interfaceFunctions:
            f.write(f"{interfaceDef()};\n")

    with open(f"{outputDir}/librapidPython.cpp", "w") as f:
        f.write("#include \"librapidPython.hpp\"\n\n")
        f.write("PYBIND11_MODULE(_librapid, module) {\n")
        f.write("    module.doc() = \"Python bindings for librapid\";\n")
        for _, interfaceCall in interfaceFunctions:
            f.write(f"    {interfaceCall('module')};\n")

        f.write("\n")
        f.write(boilerplate.postBoilerplate)
        f.write("\n")

        f.write("}\n")

    # Apply clang-format to the generated files
    import subprocess
    for file in os.listdir("../python/generated"):
        if file.endswith(".hpp") or file.endswith(".cpp"):
            try:
                subprocess.run(["clang-format", "-i", "-style=llvm", f"librapid/bindings/python/generated/{file}"], cwd="../../../")
                print(f"Ran clang-format on {file}", end="\r")
            except Exception as e:
                print("Unable to run clang-format:", e)


if __name__ == "__main__":
    main()
