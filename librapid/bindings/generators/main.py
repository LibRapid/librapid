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
            
            #include <pybind11/pybind11.h>
            #include <pybind11/stl.h>
            #include <pybind11/functional.h>

            #include <librapid/librapid.hpp>
            
            namespace py  = pybind11;
            namespace lrc = librapid;
        """).strip()

postBoilerplate = textwrap.dedent(f"""
#if defined(LIBRAPID_HAS_OPENCL)
            module.def("configureOpenCL", [](bool verbose, bool ask) {{
                lrc::configureOpenCL(verbose, ask);
            }}, py::arg("verbose") = false, py::arg("ask") = false);
#else
            module.def("configureOpenCL", [](bool verbose, bool ask) {{
                throw std::runtime_error("OpenCL is not supported in this build "
                                         "of LibRapid. Please ensure OpenCL is "
                                         "installed on your system and reinstall "
                                         "LibRapid from source.");
            }}, py::arg("verbose") = false, py::arg("ask") = false);
#endif

            module.def("hasOpenCL", []() {{ 
            #if defined(LIBRAPID_HAS_OPENCL)
                return true;
            #else
                return false;
            #endif
            }});
            
            module.def("hasCUDA", []() {{
            #if defined(LIBRAPID_HAS_CUDA)
                return true;
            #else
                return false;
            #endif
            }});
            
            module.def("setNumThreads", [](size_t numThreads) {{
                lrc::setNumThreads(numThreads);
            }}, py::arg("numThreads"));
            
            module.def("getNumThreads", []() {{
                return lrc::getNumThreads();
            }});
            
            module.def("setSeed", [](size_t seed) {{
                lrc::setSeed(seed);
            }}, py::arg("seed"));
            
            module.def("getSeed", []() {{
                return lrc::getSeed();
            }});
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

        f.write("\n")
        f.write(postBoilerplate)
        f.write("\n")

        f.write("}\n")

    # Apply clang-format to the generated files
    import subprocess
    for file in os.listdir("../python/generated"):
        if file.endswith(".hpp") or file.endswith(".cpp"):
            try:
                subprocess.run(["clang-format", "-i", "-style=llvm", f"librapid/bindings/python/generated/{file}"], cwd="../../../")
            except Exception as e:
                print("Unable to run clang-format:", e)


if __name__ == "__main__":
    main()
