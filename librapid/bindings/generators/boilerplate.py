import textwrap


def boilerplate():
    return textwrap.dedent(f"""
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
