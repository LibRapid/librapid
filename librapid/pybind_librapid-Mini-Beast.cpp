#include <librapid/librapid.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


#include <string>

namespace py = pybind11;

// Docstring for the module
std::string module_docstring = "A fast math and neural network library for Python and C++";

const char *class_extent_docstring = R"(
.. hint::
	This is a test thing. Does it work?"

.. tip::
	Can I use variable names? This should be a variable `abc123`

Description of parameter `x` (the default is -1, which implies summation
over all axes).

Parameters
----------
x : type
	Description of parameter `x`.
y
	Description of parameter `y` (with type not specified).
)";

PYBIND11_MODULE(librapid_, module)
{
	module.doc() = module_docstring;

	py::class_<librapid::ndarray::extent>(module, "extent", class_extent_docstring)
		.def(py::init<>())
		.def(py::init<const std::vector<nd_int> &>())
		.def(py::init<const librapid::ndarray::extent &>())
		.def(py::init<py::args>())

		.def("__getitem__", [](const librapid::ndarray::extent &e, nd_int index) { return e[index]; })
		.def("__setitem__", [](librapid::ndarray::extent &e, nd_int index, nd_int val) { e[index] = val; })

		.def("compressed", &librapid::ndarray::extent::compressed)
		.def_property_readonly("ndim", &librapid::ndarray::extent::ndim)
		.def_property_readonly("is_valid", &librapid::ndarray::extent::is_valid)
		.def("reshape", [](librapid::ndarray::extent &e, const std::vector<nd_int> &order) { e.reshape(order); })

		.def("__len__", &librapid::ndarray::extent::ndim)

		.def("__iter__", [](const librapid::ndarray::extent &e) { return py::make_iterator(e.begin(), e.end()); }, py::keep_alive<0, 1>())

		.def("__eq__", &librapid::ndarray::extent::operator==)
		.def("__str__", &librapid::ndarray::extent::str)
		.def("__repr__", [](const librapid::ndarray::extent &e) { return "<librapid." + e.str() + ">"; });


	py::class_<librapid::ndarray::stride>(module, "stride")
		.def(py::init<>())
		.def(py::init<std::vector<nd_int>>())
		.def(py::init<nd_int>())
		.def(py::init<const librapid::ndarray::stride &>())
		.def(py::init<py::args>())
		.def("from_extent", [](const std::vector<nd_int> &extent) { return librapid::ndarray::stride::from_extent(extent); })

		.def("__getitem__", [](const librapid::ndarray::stride &s, nd_int index) { return s[index]; })
		.def("__setitem__", [](librapid::ndarray::stride &s, nd_int index, nd_int val) { s[index] = val; })

		.def_property_readonly("ndim", &librapid::ndarray::stride::ndim)
		.def_property_readonly("is_valid", &librapid::ndarray::stride::is_valid)
		.def_property_readonly("is_trivial", &librapid::ndarray::stride::is_trivial)
		.def("set_dimensions", &librapid::ndarray::stride::set_dimensions)
		.def("reshape", [](librapid::ndarray::stride &s, const std::vector<nd_int> &order) { s.reshape(order); })

		.def("__len__", &librapid::ndarray::stride::ndim)

		.def("__iter__", [](const librapid::ndarray::stride &s) { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())

		.def("__eq__", &librapid::ndarray::stride::operator==)
		.def("__str__", &librapid::ndarray::stride::str)
		.def("__repr__", [](const librapid::ndarray::stride &s) { return "<librapid." + s.str() + ">"; });


	py::class_<librapid::ndarray::ndarray>(module, "ndarray")
		.def(py::init<>())
		.def(py::init<const librapid::ndarray::extent &>())
		.def(py::init<const librapid::ndarray::extent &, double>())
		.def(py::init<const librapid::ndarray::ndarray &>())

		.def_property_readonly("ndim", &librapid::ndarray::ndarray::ndim)
		.def_property_readonly("size", &librapid::ndarray::ndarray::size)
		.def_property_readonly("is_initialized", &librapid::ndarray::ndarray::is_initialized)
		.def_property_readonly("is_scalar", &librapid::ndarray::ndarray::is_scalar)

		.def("get_extent", &librapid::ndarray::ndarray::get_extent)
		.def("get_stride", &librapid::ndarray::ndarray::get_stride)
		.def_property_readonly("extent", &librapid::ndarray::ndarray::get_extent)
		.def_property_readonly("stride", &librapid::ndarray::ndarray::get_stride)

		.def("__getitem__", [](const librapid::ndarray::ndarray &arr, nd_int index) { return arr[index]; })
		.def("__setitem__", [](librapid::ndarray::ndarray &arr, nd_int index, const librapid::ndarray::ndarray &value) { arr[index] = value; })
		.def("__setitem__", [](librapid::ndarray::ndarray &arr, nd_int index, double value) { arr[index] = value; })

		.def("fill", [](librapid::ndarray::ndarray &arr, double filler) { arr.fill(filler); })
		.def("filled", [](const librapid::ndarray::ndarray &arr, double filler) { return arr.filled(filler); })
		.def("clone", &librapid::ndarray::ndarray::clone)
		.def("set_value", &librapid::ndarray::ndarray::set_value)

		.def("__add__", [](const librapid::ndarray::ndarray &lhs, const librapid::ndarray::ndarray &rhs) { return lhs + rhs; })
		.def("__sub__", [](const librapid::ndarray::ndarray &lhs, const librapid::ndarray::ndarray &rhs) { return lhs - rhs; })
		.def("__mul__", [](const librapid::ndarray::ndarray &lhs, const librapid::ndarray::ndarray &rhs) { return lhs * rhs; })
		.def("__truediv__", [](const librapid::ndarray::ndarray &lhs, const librapid::ndarray::ndarray &rhs) { return lhs / rhs; })

		.def("__add__", [](const librapid::ndarray::ndarray &lhs, double rhs) { return lhs + rhs; })
		.def("__sub__", [](const librapid::ndarray::ndarray &lhs, double rhs) { return lhs - rhs; })
		.def("__mul__", [](const librapid::ndarray::ndarray &lhs, double rhs) { return lhs * rhs; })
		.def("__truediv__", [](const librapid::ndarray::ndarray &lhs, double rhs) { return lhs / rhs; })

		.def("__add__", [](double lhs, const librapid::ndarray::ndarray &rhs) { return lhs + rhs; })
		.def("__sub__", [](double lhs, const librapid::ndarray::ndarray &rhs) { return lhs - rhs; })
		.def("__mul__", [](double lhs, const librapid::ndarray::ndarray &rhs) { return lhs * rhs; })
		.def("__truediv__", [](double lhs, const librapid::ndarray::ndarray &rhs) { return lhs / rhs; })

		.def("__radd__", [](const librapid::ndarray::ndarray &lhs, double rhs) { return rhs + lhs; })
		.def("__rsub__", [](const librapid::ndarray::ndarray &lhs, double rhs) { return rhs - lhs; })
		.def("__rmul__", [](const librapid::ndarray::ndarray &lhs, double rhs) { return rhs * lhs; })
		.def("__rtruediv__", [](const librapid::ndarray::ndarray &lhs, double rhs) { return rhs / lhs; })

		.def("__neg__", [](const librapid::ndarray::ndarray &arr) { return -arr; })

		.def("dot", [](const librapid::ndarray::ndarray &lhs, const librapid::ndarray::ndarray &rhs) { return lhs.dot(rhs); })
		.def("__matmul__", [](const librapid::ndarray::ndarray &lhs, const librapid::ndarray::ndarray &rhs) { return lhs.dot(rhs); })

		.def("reshape", [](librapid::ndarray::ndarray &arr, const std::vector<nd_int> &order) { arr.reshape(order); })
		.def("reshape", [](librapid::ndarray::ndarray &e, const librapid::ndarray::extent &order) { e.reshape(order); })
		.def("reshaped", [](const librapid::ndarray::ndarray &arr, const std::vector<nd_int> &order) { return arr.reshaped(order); })
		.def("reshape", [](librapid::ndarray::ndarray &arr, py::args args) { arr.reshape(py::cast<std::vector<nd_int>>(args)); })
		.def("reshaped", [](const librapid::ndarray::ndarray &arr, py::args args) { return arr.reshaped(py::cast<std::vector<nd_int>>(args)); })
		.def("subarray", [](const librapid::ndarray::ndarray &arr, const std::vector<nd_int> &axes) { return arr.subarray(axes); })

		.def("strip_front", &librapid::ndarray::ndarray::strip_front)
		.def("strip_back", &librapid::ndarray::ndarray::strip_back)
		.def("strip", &librapid::ndarray::ndarray::strip)
		.def("stripped_front", &librapid::ndarray::ndarray::stripped_front)
		.def("stripped_back", &librapid::ndarray::ndarray::stripped_back)
		.def("stripped", &librapid::ndarray::ndarray::stripped)

		.def("transpose", [](librapid::ndarray::ndarray &arr, const std::vector<nd_int> &order) { arr.transpose(order); })
		.def("transpose", [](librapid::ndarray::ndarray &arr) { arr.transpose(); })
		.def("transposed", [](const librapid::ndarray::ndarray &arr, const std::vector<nd_int> &order) { return arr.transposed(order); })
		.def("transposed", [](librapid::ndarray::ndarray &arr) { return arr.transposed(); })

		.def("__str__", [](const librapid::ndarray::ndarray &arr) { return arr.str(0); })
		.def("__int__", [](const librapid::ndarray::ndarray &arr) { if (!arr.is_scalar()) throw py::value_error("Cannot convert non-scalar array to int"); return (nd_int) *arr.get_data_start(); })
		.def("__float__", [](const librapid::ndarray::ndarray &arr) { if (!arr.is_scalar()) throw py::value_error("Cannot convert non-scalar array to float"); return (double) *arr.get_data_start(); })
		.def("__repr__", [](const librapid::ndarray::ndarray &arr) { return "<librapid.ndarray " + arr.str(18) + ">"; });

	module.def("add", [](double lhs, double rhs) { return lhs + rhs; });
	module.def("add", [](librapid::ndarray::ndarray lhs, double rhs) { return lhs + rhs; });
	module.def("add", [](double lhs, librapid::ndarray::ndarray rhs) { return lhs + rhs; });

	module.def("sub", [](double lhs, double rhs) { return lhs - rhs; });
	module.def("sub", [](librapid::ndarray::ndarray lhs, double rhs) { return lhs - rhs; });
	module.def("sub", [](double lhs, librapid::ndarray::ndarray rhs) { return lhs - rhs; });
	
	module.def("mul", [](double lhs, double rhs) { return lhs * rhs; });
	module.def("mul", [](librapid::ndarray::ndarray lhs, double rhs) { return lhs * rhs; });
	module.def("mul", [](double lhs, librapid::ndarray::ndarray rhs) { return lhs * rhs; });

	module.def("div", [](double lhs, double rhs) { return lhs / rhs; });
	module.def("div", [](librapid::ndarray::ndarray lhs, double rhs) { return lhs / rhs; });
	module.def("div", [](double lhs, librapid::ndarray::ndarray rhs) { return lhs / rhs; });

	module.def("reshape", [](const librapid::ndarray::ndarray &arr, const librapid::ndarray::extent &shape) { return librapid::ndarray::reshape(arr, shape); });
	module.def("reshape", [](const librapid::ndarray::ndarray &arr, const std::vector<nd_int> &shape) { return librapid::ndarray::reshape(arr, librapid::ndarray::extent(shape)); });
}
