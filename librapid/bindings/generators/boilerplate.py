import textwrap

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

boilerplateLibrapidOnly = textwrap.dedent(f"""
            #pragma once

            #ifndef LIBRAPID_DEBUG
                #define LIBRAPID_DEBUG
            #endif
            
            #include <librapid/librapid.hpp>            
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
