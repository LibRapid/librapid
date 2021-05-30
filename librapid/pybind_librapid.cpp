#include <librapid/ndarray/ndarray.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


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
        .def(py::init<py::args>())

        .def("__getitem__", [](const ndarray::extent &e, nd_int index) { return e[index]; })
        .def("__setitem__", [](ndarray::extent &e, nd_int index, nd_int val) { e[index] = val; })

        .def("compressed", &ndarray::extent::compressed)
        .def_property_readonly("ndim", &ndarray::extent::ndim)
        .def_property_readonly("is_valid", &ndarray::extent::is_valid)
        .def("reshape", [](ndarray::extent &e, const std::vector<nd_int> &order) { e.reshape(order); })

        .def("__eq__", &ndarray::extent::operator==)
        .def("__str__", &ndarray::extent::str)
        .def("__repr__", [](const ndarray::extent &e) { return "<librapid." + e.str() + ">"; });


    py::class_<ndarray::stride>(module, "stride")
        .def(py::init<>())
        .def(py::init<std::vector<nd_int>>())
        .def(py::init<nd_int>())
        .def(py::init<const ndarray::stride &>())
        .def(py::init<py::args>())
        .def("from_extent", [](const std::vector<nd_int> &extent) { return ndarray::stride::from_extent(extent); })

        .def("__getitem__", [](const ndarray::stride &s, nd_int index) { return s[index]; })
        .def("__setitem__", [](ndarray::stride &s, nd_int index, nd_int val) { s[index] = val; })

        .def_property_readonly("ndim", &ndarray::stride::ndim)
        .def_property_readonly("is_valid", &ndarray::stride::is_valid)
        .def_property_readonly("is_trivial", &ndarray::stride::is_trivial)
        .def("set_dimensions", &ndarray::stride::set_dimensions)
        .def("reshape", [](ndarray::stride &s, const std::vector<nd_int> &order) { s.reshape(order); })

        .def("__eq__", &ndarray::stride::operator==)
        .def("__str__", &ndarray::stride::str)
        .def("__repr__", [](const ndarray::stride &s) { return "<librapid." + s.str() + ">"; });


    py::class_<ndarray::ndarray>(module, "ndarray")
        .def(py::init<>())
        .def(py::init<const ndarray::extent &>())
        .def(py::init<const ndarray::extent &, double>())
        .def(py::init<const ndarray::ndarray &>())

        .def_property_readonly("ndim", &ndarray::ndarray::ndim)
        .def_property_readonly("is_initialized", &ndarray::ndarray::is_initialized)
        .def_property_readonly("is_scalar", &ndarray::ndarray::is_scalar)

        .def("get_extent", &ndarray::ndarray::get_extent)
        .def("get_stride", &ndarray::ndarray::get_stride)
        .def_property_readonly("extent", &ndarray::ndarray::get_extent)
        .def_property_readonly("stride", &ndarray::ndarray::get_stride)

        .def("__getitem__", [](const ndarray::ndarray &arr, nd_int index) { return arr[index]; })
        .def("__setitem__", [](ndarray::ndarray &arr, nd_int index, const ndarray::ndarray &value) { arr[index] = value; })
        .def("__setitem__", [](ndarray::ndarray &arr, nd_int index, double value) { arr[index] = value; })

        .def("fill", [](ndarray::ndarray &arr, double filler) { arr.fill(filler); })
        .def("filled", [](const ndarray::ndarray &arr, double filler) { return arr.filled(filler); })
        .def("clone", &ndarray::ndarray::clone)
        .def("set_value", &ndarray::ndarray::set_value)

        .def("__add__", [](const ndarray::ndarray &lhs, const ndarray::ndarray &rhs) { return lhs + rhs; })
        .def("__sub__", [](const ndarray::ndarray &lhs, const ndarray::ndarray &rhs) { return lhs - rhs; })
        .def("__mul__", [](const ndarray::ndarray &lhs, const ndarray::ndarray &rhs) { return lhs * rhs; })
        .def("__truediv__", [](const ndarray::ndarray &lhs, const ndarray::ndarray &rhs) { return lhs / rhs; })

        .def("__add__", [](const ndarray::ndarray &lhs, double rhs) { return lhs + rhs; })
        .def("__sub__", [](const ndarray::ndarray &lhs, double rhs) { return lhs - rhs; })
        .def("__mul__", [](const ndarray::ndarray &lhs, double rhs) { return lhs * rhs; })
        .def("__truediv__", [](const ndarray::ndarray &lhs, double rhs) { return lhs / rhs; })

        .def("__add__", [](double lhs, const ndarray::ndarray &rhs) { return lhs + rhs; })
        .def("__sub__", [](double lhs, const ndarray::ndarray &rhs) { return lhs - rhs; })
        .def("__mul__", [](double lhs, const ndarray::ndarray &rhs) { return lhs * rhs; })
        .def("__truediv__", [](double lhs, const ndarray::ndarray &rhs) { return lhs / rhs; })

        .def("__radd__", [](const ndarray::ndarray &lhs, double rhs) { return rhs + lhs; })
        .def("__rsub__", [](const ndarray::ndarray &lhs, double rhs) { return rhs - lhs; })
        .def("__rmul__", [](const ndarray::ndarray &lhs, double rhs) { return rhs * lhs; })
        .def("__rtruediv__", [](const ndarray::ndarray &lhs, double rhs) { return rhs / lhs; })

        .def("__neg__", [](const ndarray::ndarray &arr) { return -arr; })

        .def("reshape", [](ndarray::ndarray &arr, const std::vector<nd_int> &order) { arr.reshape(order); })
        .def("reshaped", [](const ndarray::ndarray &arr, const std::vector<nd_int> &order) { return arr.reshaped(order); })
        .def("reshape", [](ndarray::ndarray &arr, py::args args) { arr.reshape(py::cast<std::vector<nd_int>>(args)); })
        .def("reshaped", [](const ndarray::ndarray &arr, py::args args) { return arr.reshaped(py::cast<std::vector<nd_int>>(args)); })

        .def("strip_front", &ndarray::ndarray::strip_front)
        .def("strip_back", &ndarray::ndarray::strip_back)
        .def("strip", &ndarray::ndarray::strip)
        .def("stripped_front", &ndarray::ndarray::stripped_front)
        .def("stripped_back", &ndarray::ndarray::stripped_back)
        .def("stripped", &ndarray::ndarray::stripped)

        .def("transpose", [](ndarray::ndarray &arr, const std::vector<nd_int> &order) { arr.transpose(order); })
        .def("transpose", [](ndarray::ndarray &arr) { arr.transpose(); })
        .def("transposed", [](const ndarray::ndarray &arr, const std::vector<nd_int> &order) { return arr.transposed(order); })
        .def("transposed", [](ndarray::ndarray &arr) { return arr.transposed(); })

        .def("__str__", [](const ndarray::ndarray &arr) { return arr.str(0); })
        .def("__int__", [](const ndarray::ndarray &arr) { if (!arr.is_scalar()) throw py::value_error("Cannot convert non-scalar array to int"); return (nd_int) *arr.get_data_start(); })
        .def("__float__", [](const ndarray::ndarray &arr) { if (!arr.is_scalar()) throw py::value_error("Cannot convert non-scalar array to float"); return (double) *arr.get_data_start(); })
        .def("__repr__", [](const ndarray::ndarray &arr) { return "<librapid.ndarray " + arr.str(18) + ">"; });
}
