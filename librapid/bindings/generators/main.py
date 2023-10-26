import os
import textwrap

import shapeGenerator
import arrayGenerator
import generalArrayViewGenerator

outputDir = "../python/generated"

boilerplate = textwrap.dedent(f"""
            #pragma once

            #ifndef LIBRAPID_DEBUG
                #define LIBRAPID_DEBUG
            #endif

            #include <librapid/librapid.hpp>
            #include <pybind11/pybind11.h>
            #include <pybind11/stl.h>
            #include <pybind11/functional.h>
            
            namespace py  = pybind11;
            namespace lrc = librapid;
        """).strip()


def main():
    # Ensure the output directory exists
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)

    interfaceFunctions = []

    interfaceFunctions += shapeGenerator.write(outputDir)
    interfaceFunctions += arrayGenerator.write(outputDir)
    interfaceFunctions += generalArrayViewGenerator.write(outputDir)

    with open(f"{outputDir}/librapidPython.hpp", "w") as f:
        f.write(boilerplate)
        f.write("\n\n")
        for interfaceDef, _ in interfaceFunctions:
            f.write(f"{interfaceDef()};\n")

    with open(f"{outputDir}/librapidPython.cpp", "w") as f:
        f.write("#include \"librapidPython.hpp\"\n\n")
        f.write("PYBIND11_MODULE(_librapid, module) {\n")
        f.write("    module.doc() = \"Python bindings for librapid\";\n")
        for _, interfaceCall in interfaceFunctions:
            f.write(f"    {interfaceCall('module')};\n")
        f.write("}\n")


if __name__ == "__main__":
    main()
