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
        .def("__eq__", &ndarray::extent::operator==)
        .def("__str__", &ndarray::extent::str)
        .def("__repr__", &ndarray::extent::str);

    py::class_<ndarray::stride>(module, "stride")
        .def(py::init<>())
        .def(py::init<std::vector<nd_int>>())
        .def(py::init<nd_int>())
        .def(py::init<const ndarray::stride &>())
        .def("from_extent", &ndarray::stride::__py_from_extent)
        .def("__getitem__", &ndarray::stride::__py_getitem)
        .def("__setitem__", &ndarray::stride::__py_setitem)
        .def_property_readonly("ndim", &ndarray::stride::ndim)
        .def_property_readonly("is_valid", &ndarray::stride::is_valid)
        .def_property_readonly("is_trivial", &ndarray::stride::is_trivial)
        .def("set_dimensions", &ndarray::stride::set_dimensions)
        .def("reshape", &ndarray::stride::__py_reshape)
        .def("__eq__", &ndarray::stride::operator==)
        .def("__str__", &ndarray::stride::str)
        .def("__repr__", &ndarray::stride::str);
}
