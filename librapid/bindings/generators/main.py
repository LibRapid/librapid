import os
import textwrap

import shapeGenerator
import arrayGenerator
import generalArrayViewGenerator
import boilerplate

outputDir = "../python/generated"

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
        f.write(boilerplate.boilerplate())
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
    # import subprocess
    # for file in os.listdir("../python/generated"):
    #     if file.endswith(".hpp") or file.endswith(".cpp"):
    #         try:
    #             subprocess.run(["clang-format", "-i", "-style=llvm", f"librapid/bindings/python/generated/{file}"], cwd="../../../")
    #         except Exception as e:
    #             print("Unable to run clang-format:", e)

    # Apply clang-format to the generated files (recursive)
    import subprocess
    prevChars = 0
    for root, dirs, files in os.walk("../python/generated"):
        for file in files:

            print(" " * prevChars, end='\r')
            text = f"Formatting {root}/{file}"
            print(text, end='\r')
            prevChars = len(text)


            if file.endswith(".hpp") or file.endswith(".cpp"):
                try:
                    subprocess.run(["clang-format", "-i", "-style=llvm", f"{root}/{file}"], cwd="./")
                except Exception as e:
                    print(f"Unable to run clang-format on {root}/{file}:", e)


if __name__ == "__main__":
    main()
