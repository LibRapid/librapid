#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <librapid/ndarray/ndarray.hpp>

#include <string>

namespace py = pybind11;

// Docstring for the module
std::string module_docstring = "A fast math and neural network library for Python and C++";

PYBIND11_MODULE(librapid, module)
{
    module.doc() = module_docstring;

    py::class_<ndarray::extent>(module, "extent")
        .def(py::init<>())
        .def(py::init<const std::vector<nd_int> &>())
        .def(py::init<nd_int>())
        .def(py::init<const ndarray::extent &>())
        .def("__getitem__", &ndarray::extent::__py_getitem)
        .def("__setitem__", &ndarray::extent::__py_setitem)
        .def("compressed", &ndarray::extent::compressed)
        .def_property_readonly("ndim", &ndarray::extent::ndim)
        .def_property_readonly("is_valid", &ndarray::extent::is_valid)
        .def("reshape", &ndarray::extent::__py_reshape)
        .def("__str__", &ndarray::extent::str);
}
