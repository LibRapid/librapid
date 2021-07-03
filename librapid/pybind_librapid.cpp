#include <librapid/librapid.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <functional>

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

template<typename T>
using V = std::vector<T>;

template<class T>
struct python_activation {
	T *activation;

	python_activation() {
		activation = new T();
	}

	~python_activation() {
		delete activation;
	}
};

PYBIND11_MODULE(librapid_, module)
{
	module.doc() = module_docstring;

	module.def("has_blas", &librapid::has_blas);
	module.def("set_blas_threads", &librapid::set_blas_threads);
	module.def("get_blas_threads", &librapid::get_blas_threads);
	
	module.def("set_num_threads", &librapid::set_num_threads);
	module.def("get_num_threads", &librapid::get_num_threads);

	module.def("time", [](){ return TIME; });
	module.def("sleep", &librapid::sleep);

	module.attr("pi") = librapid::math::pi;
	module.attr("twopi") = librapid::math::twopi;
	module.attr("halfpi") = librapid::math::halfpi;
	module.attr("e") = librapid::math::e;
	module.attr("sqrt2") = librapid::math::sqrt2;
	module.attr("sqrt3") = librapid::math::sqrt3;
	module.attr("sqrt5") = librapid::math::sqrt5;

	module.def("product", [](const std::vector<double> &vals) { return librapid::math::product(vals); }, py::arg("vals"));

	module.def("min", [](const std::vector<double> &vals) { return librapid::math::min(vals); }, py::arg("vals"));
	module.def("max", [](const std::vector<double> &vals) { return librapid::math::max(vals); }, py::arg("vals"));

	module.def("abs", [](double val) { return librapid::math::abs(val); }, py::arg("val"));
	module.def("map", [](double val, double start1, double stop1, double start2, double stop2) { return librapid::math::map(val, start1, stop1, start2, stop2); }, py::arg("val"), py::arg("start1") = double(0), py::arg("stop1") = double(1), py::arg("start2") = double(0), py::arg("stop2") = double(1));
	module.def("random", [](double min, double max) { return librapid::math::random(min, max); }, py::arg("min") = 0, py::arg("max") = 1);
	module.def("pow10", &librapid::math::pow10);
	module.def("round", [](double val, lr_int places) { return librapid::math::round(val, places); }, py::arg("val"), py::arg("places") = 0);

	// The librapid extent object
	py::class_<librapid::extent>(module, "extent", class_extent_docstring)
		.def(py::init<>())
		.def(py::init<const std::vector<lr_int> &>())
		.def(py::init<const librapid::extent &>())
		.def(py::init<py::args>())

		.def("__getitem__", [](const librapid::extent &e, lr_int index) { return e[index]; })
		.def("__setitem__", [](librapid::extent &e, lr_int index, lr_int val) { e[index] = val; })

		.def("compressed", &librapid::extent::compressed)
		.def_property_readonly("ndim", &librapid::extent::ndim)
		.def_property_readonly("is_valid", &librapid::extent::is_valid)
		.def("reshape", [](librapid::extent &e, const std::vector<lr_int> &order) { e.reshape(order); })

		.def("fix_automatic", &librapid::extent::fix_automatic)

		.def("__len__", &librapid::extent::ndim)

		.def("__iter__", [](const librapid::extent &e) { return py::make_iterator(e.begin(), e.end()); }, py::keep_alive<0, 1>())

		.def("__eq__", &librapid::extent::operator==)
		.def("__ne__", &librapid::extent::operator!=)
		.def("__str__", &librapid::extent::str)
		.def("__repr__", [](const librapid::extent &e) { return "<librapid." + e.str() + ">"; });

	// The librapid stride object
	py::class_<librapid::stride>(module, "stride")
		.def(py::init<>())
		.def(py::init<std::vector<lr_int>>())
		.def(py::init<lr_int>())
		.def(py::init<const librapid::stride &>())
		.def(py::init<py::args>())
		.def("from_extent", [](const std::vector<lr_int> &extent) { return librapid::stride::from_extent(extent); })

		.def("__getitem__", [](const librapid::stride &s, lr_int index) { return s[index]; })
		.def("__setitem__", [](librapid::stride &s, lr_int index, lr_int val) { s[index] = val; })

		.def_property_readonly("ndim", &librapid::stride::ndim)
		.def_property_readonly("is_valid", &librapid::stride::is_valid)
		.def_property_readonly("is_trivial", &librapid::stride::is_trivial)
		.def("set_dimensions", &librapid::stride::set_dimensions)
		.def("reshape", [](librapid::stride &s, const std::vector<lr_int> &order) { s.reshape(order); })

		.def("__len__", &librapid::stride::ndim)

		.def("__iter__", [](const librapid::stride &s) { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())

		.def("__eq__", &librapid::stride::operator==)
		.def("__str__", &librapid::stride::str)
		.def("__repr__", [](const librapid::stride &s) { return "<librapid." + s.str() + ">"; });

	// The librapid ndarray object
	py::class_<librapid::ndarray>(module, "ndarray")
		.def(py::init<>())
		.def(py::init<const librapid::extent &>())
		.def(py::init<const librapid::extent &, double>())
		.def(py::init<const librapid::ndarray &>())
		
		.def_static("from_scalar", [](double x) { return librapid::ndarray::from_scalar(x); }, py::arg("x") = double(0))

		.def_static("from_data", [](V<double> &vals) { return librapid::ndarray::from_data(vals); }, py::arg("vals") = V<double>())
		.def_static("from_data", [](V<V<double>> &vals) { return librapid::ndarray::from_data(vals); }, py::arg("vals") = V<V<double>>())
		.def_static("from_data", [](V<V<V<double>>> &vals) { return librapid::ndarray::from_data(vals); }, py::arg("vals") = V<V<V<double>>>())
		.def_static("from_data", [](V<V<V<V<double>>>> &vals) { return librapid::ndarray::from_data(vals); }, py::arg("vals") = V<V<V<V<double>>>>())
		.def_static("from_data", [](V<V<V<V<V<double>>>>> &vals) { return librapid::ndarray::from_data(vals); }, py::arg("vals") = V<V<V<V<V<double>>>>>())
		.def_static("from_data", [](V<V<V<V<V<V<double>>>>>> &vals) { return librapid::ndarray::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<double>>>>>>())
		.def_static("from_data", [](V<V<V<V<V<V<V<double>>>>>>> &vals) { return librapid::ndarray::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<double>>>>>>>())
		.def_static("from_data", [](V<V<V<V<V<V<V<V<double>>>>>>>> &vals) { return librapid::ndarray::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<V<double>>>>>>>>())
		.def_static("from_data", [](V<V<V<V<V<V<V<V<V<double>>>>>>>>> &vals) { return librapid::ndarray::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<V<V<double>>>>>>>>>())
		.def_static("from_data", [](V<V<V<V<V<V<V<V<V<V<double>>>>>>>>>> &vals) { return librapid::ndarray::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<V<V<V<double>>>>>>>>>>())

		.def("set_to", &librapid::ndarray::set_to)

		.def_property_readonly("ndim", &librapid::ndarray::ndim)
		.def_property_readonly("size", &librapid::ndarray::size)
		.def_property_readonly("is_initialized", &librapid::ndarray::is_initialized)
		.def_property_readonly("is_scalar", &librapid::ndarray::is_scalar)

		.def("get_extent", &librapid::ndarray::get_extent)
		.def("get_stride", &librapid::ndarray::get_stride)
		.def_property_readonly("extent", &librapid::ndarray::get_extent)
		.def_property_readonly("stride", &librapid::ndarray::get_stride)

		.def("__eq__", [](const librapid::ndarray &arr, double val) { return arr == val; }, py::arg("val"))
		.def("__ne__", [](const librapid::ndarray &arr, double val) { return arr != val; }, py::arg("val"))

		.def("__getitem__", [](const librapid::ndarray &arr, lr_int index) { return arr[index]; }, py::arg("index"))
		.def("__setitem__", [](librapid::ndarray &arr, lr_int index, const librapid::ndarray &value) { arr[index] = value; }, py::arg("index"), py::arg("value"))
		.def("__setitem__", [](librapid::ndarray &arr, lr_int index, double value) { arr[index] = value; }, py::arg("index"), py::arg("value"))

		.def("fill", [](librapid::ndarray &arr, double filler) { arr.fill(filler); }, py::arg("filler") = double(0))
		.def("filled", [](const librapid::ndarray &arr, double filler) { return arr.filled(filler); }, py::arg("filler") = double(0))
		.def("fill_random", [](librapid::ndarray &arr, double min, double max) { arr.fill_random(min, max); }, py::arg("min") = double(0), py::arg("max") = double(1))
		.def("filled_random", [](const librapid::ndarray &arr, double min, double max) { return arr.filled_random(min, max); }, py::arg("min") = double(0), py::arg("max") = double(1))
		.def("map", [](librapid::ndarray &arr, const std::function<double(double)> &func) { arr.map(func); })
		.def("mapped", [](const librapid::ndarray &arr, const std::function<double(double)> &func) { return arr.mapped(func); })
		.def("clone", &librapid::ndarray::clone)
		.def("set_value", &librapid::ndarray::set_value)

		.def("__add__", [](const librapid::ndarray &lhs, const librapid::ndarray &rhs) { return lhs + rhs; }, py::arg("rhs"))
		.def("__sub__", [](const librapid::ndarray &lhs, const librapid::ndarray &rhs) { return lhs - rhs; }, py::arg("rhs"))
		.def("__mul__", [](const librapid::ndarray &lhs, const librapid::ndarray &rhs) { return lhs * rhs; }, py::arg("rhs"))
		.def("__truediv__", [](const librapid::ndarray &lhs, const librapid::ndarray &rhs) { return lhs / rhs; }, py::arg("rhs"))

		.def("__add__", [](const librapid::ndarray &lhs, double rhs) { return lhs + rhs; }, py::arg("rhs"))
		.def("__sub__", [](const librapid::ndarray &lhs, double rhs) { return lhs - rhs; }, py::arg("rhs"))
		.def("__mul__", [](const librapid::ndarray &lhs, double rhs) { return lhs * rhs; }, py::arg("rhs"))
		.def("__truediv__", [](const librapid::ndarray &lhs, double rhs) { return lhs / rhs; }, py::arg("rhs"))

		.def("__radd__", [](const librapid::ndarray &rhs, double lhs) { return lhs + rhs; }, py::arg("lhs"))
		.def("__rsub__", [](const librapid::ndarray &rhs, double lhs) { return lhs - rhs; }, py::arg("lhs"))
		.def("__rmul__", [](const librapid::ndarray &rhs, double lhs) { return lhs * rhs; }, py::arg("lhs"))
		.def("__rtruediv__", [](const librapid::ndarray &rhs, double lhs) { return lhs / rhs; }, py::arg("lhs"))

		.def("__iadd__", [](librapid::ndarray &lhs, const librapid::ndarray &rhs) { lhs += rhs; return lhs; }, py::arg("rhs"))
		.def("__isub__", [](librapid::ndarray &lhs, const librapid::ndarray &rhs) { lhs -= rhs; return lhs; }, py::arg("rhs"))
		.def("__imul__", [](librapid::ndarray &lhs, const librapid::ndarray &rhs) { lhs *= rhs; return lhs; }, py::arg("rhs"))
		.def("__itruediv__", [](librapid::ndarray &lhs, const librapid::ndarray &rhs) { lhs /= rhs; return lhs; }, py::arg("rhs"))

		.def("__iadd__", [](librapid::ndarray &lhs, double rhs) { lhs += rhs; return lhs; }, py::arg("rhs"))
		.def("__isub__", [](librapid::ndarray &lhs, double rhs) { lhs -= rhs; return lhs; }, py::arg("rhs"))
		.def("__imul__", [](librapid::ndarray &lhs, double rhs) { lhs *= rhs; return lhs; }, py::arg("rhs"))
		.def("__itruediv__", [](librapid::ndarray &lhs, double rhs) { lhs /= rhs; return lhs; }, py::arg("rhs"))

		.def("__neg__", [](const librapid::ndarray &arr) { return -arr; })

		.def("minimum", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::minimum(x1, x2); }, py::arg("x2"))
		.def("minimum", [](const librapid::ndarray &x1, double x2) { return librapid::minimum(x1, x2); }, py::arg("x2"))
		.def("maximum", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::maximum(x1, x2); }, py::arg("x2"))
		.def("maximum", [](const librapid::ndarray &x1, double x2) { return librapid::maximum(x1, x2); }, py::arg("x2"))

		.def("less_than", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::less_than(x1, x2); }, py::arg("x2"))
		.def("less_than", [](const librapid::ndarray &x1, double x2) { return librapid::less_than(x1, x2); }, py::arg("x2"))
		.def("greater_than", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::greater_than(x1, x2); }, py::arg("x2"))
		.def("greater_than", [](const librapid::ndarray &x1, double x2) { return librapid::greater_than(x1, x2); }, py::arg("x2"))
		.def("less_than_or_equal", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::less_than_or_equal(x1, x2); }, py::arg("x2"))
		.def("less_than_or_equal", [](const librapid::ndarray &x1, double x2) { return librapid::less_than_or_equal(x1, x2); }, py::arg("x2"))
		.def("greater_than_or_equal", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::greater_than_or_equal(x1, x2); }, py::arg("x2"))
		.def("greater_than_or_equal", [](const librapid::ndarray &x1, double x2) { return librapid::greater_than_or_equal(x1, x2); }, py::arg("x2"))

		.def("dot", [](const librapid::ndarray &lhs, const librapid::ndarray &rhs) { return lhs.dot(rhs); }, py::arg("rhs"))
		.def("__matmul__", [](const librapid::ndarray &lhs, const librapid::ndarray &rhs) { return lhs.dot(rhs); }, py::arg("rhs"))

		.def("reshape", [](librapid::ndarray &arr, const std::vector<lr_int> &order) { arr.reshape(order); }, py::arg("order"))
		.def("reshape", [](librapid::ndarray &arr, const librapid::extent &order) { arr.reshape(order); }, py::arg("order"))
		.def("reshaped", [](const librapid::ndarray &arr, const std::vector<lr_int> &order) { return arr.reshaped(order); }, py::arg("order"))
		.def("reshape", [](librapid::ndarray &arr, py::args args) { arr.reshape(py::cast<std::vector<lr_int>>(args)); })
		.def("reshaped", [](const librapid::ndarray &arr, py::args args) { return arr.reshaped(py::cast<std::vector<lr_int>>(args)); })
		.def("subarray", [](const librapid::ndarray &arr, const std::vector<lr_int> &axes) { return arr.subarray(axes); }, py::arg("axes"))

		.def("strip_front", &librapid::ndarray::strip_front)
		.def("strip_back", &librapid::ndarray::strip_back)
		.def("strip", &librapid::ndarray::strip)
		.def("stripped_front", &librapid::ndarray::stripped_front)
		.def("stripped_back", &librapid::ndarray::stripped_back)
		.def("stripped", &librapid::ndarray::stripped)

		.def("transpose", [](librapid::ndarray &arr, const std::vector<lr_int> &order) { arr.transpose(order); }, py::arg("order"))
		.def("transpose", [](librapid::ndarray &arr) { arr.transpose(); })
		.def("transposed", [](const librapid::ndarray &arr, const std::vector<lr_int> &order) { return arr.transposed(order); }, py::arg("order"))
		.def("transposed", [](const librapid::ndarray &arr) { return arr.transposed(); })

		.def("sum", [](const librapid::ndarray &arr, lr_int axis) { return arr.sum(axis); }, py::arg("axis") = librapid::AUTO)
		.def("product", [](const librapid::ndarray &arr, lr_int axis) { return arr.product(axis); }, py::arg("axis") = librapid::AUTO)
		.def("mean", [](const librapid::ndarray &arr, lr_int axis) { return arr.mean(axis); }, py::arg("axis") = librapid::AUTO)
		.def("abs", [](const librapid::ndarray &arr) { return arr.abs(); })
		.def("square", [](const librapid::ndarray &arr) { return arr.square(); })
		.def("variance", [](const librapid::ndarray &arr, lr_int axis) { return arr.variance(axis); }, py::arg("axis") = librapid::AUTO)

		.def("__str__", [](const librapid::ndarray &arr) { return arr.str(0); })
		.def("__int__", [](const librapid::ndarray &arr) { if (!arr.is_scalar()) throw py::value_error("Cannot convert non-scalar array to int"); return (lr_int) *arr.get_data_start(); })
		.def("__float__", [](const librapid::ndarray &arr) { if (!arr.is_scalar()) throw py::value_error("Cannot convert non-scalar array to float"); return (double) *arr.get_data_start(); })
		.def("__repr__", [](const librapid::ndarray &arr) { return "<librapid.ndarray " + arr.str(18) + ">"; });

	module.def("zeros_like", [](const librapid::ndarray &arr) { return librapid::zeros_like(arr); }, py::arg("arr"));
	module.def("ones_like", [](const librapid::ndarray &arr) { return librapid::ones_like(arr); }, py::arg("arr"));
	module.def("random_like", [](const librapid::ndarray &arr, double min, double max) { return librapid::random_like(arr, min, max); }, py::arg("arr"), py::arg("min") = 0, py::arg("max") = 1);

	module.def("add", [](double lhs, double rhs) { return lhs + rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("add", [](librapid::ndarray lhs, double rhs) { return lhs + rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("add", [](double lhs, librapid::ndarray rhs) { return lhs + rhs; }, py::arg("lhs"), py::arg("rhs"));

	module.def("sub", [](double lhs, double rhs) { return lhs - rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("sub", [](librapid::ndarray lhs, double rhs) { return lhs - rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("sub", [](double lhs, librapid::ndarray rhs) { return lhs - rhs; }, py::arg("lhs"), py::arg("rhs"));
	
	module.def("mul", [](double lhs, double rhs) { return lhs * rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("mul", [](librapid::ndarray lhs, double rhs) { return lhs * rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("mul", [](double lhs, librapid::ndarray rhs) { return lhs * rhs; }, py::arg("lhs"), py::arg("rhs"));

	module.def("div", [](double lhs, double rhs) { return lhs / rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("div", [](librapid::ndarray lhs, double rhs) { return lhs / rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("div", [](double lhs, librapid::ndarray rhs) { return lhs / rhs; }, py::arg("lhs"), py::arg("rhs"));

	module.def("exp", [](const librapid::ndarray &arr) { return librapid::exp(arr); }, py::arg("arr"));

	module.def("sin", [](const librapid::ndarray &arr) { return librapid::sin(arr); }, py::arg("arr"));
	module.def("cos", [](const librapid::ndarray &arr) { return librapid::cos(arr); }, py::arg("arr"));
	module.def("tan", [](const librapid::ndarray &arr) { return librapid::tan(arr); }, py::arg("arr"));

	module.def("asin", [](const librapid::ndarray &arr) { return librapid::asin(arr); }, py::arg("arr"));
	module.def("acos", [](const librapid::ndarray &arr) { return librapid::acos(arr); }, py::arg("arr"));
	module.def("atan", [](const librapid::ndarray &arr) { return librapid::atan(arr); }, py::arg("arr"));

	module.def("sinh", [](const librapid::ndarray &arr) { return librapid::sinh(arr); }, py::arg("arr"));
	module.def("cosh", [](const librapid::ndarray &arr) { return librapid::cosh(arr); }, py::arg("arr"));
	module.def("tanh", [](const librapid::ndarray &arr) { return librapid::tanh(arr); }, py::arg("arr"));

	module.def("reshape", [](const librapid::ndarray &arr, const librapid::extent &shape) { return librapid::reshape(arr, shape); }, py::arg("arr"), py::arg("shape"));
	module.def("reshape", [](const librapid::ndarray &arr, const std::vector<lr_int> &shape) { return librapid::reshape(arr, librapid::extent(shape)); }, py::arg("arr"), py::arg("shape"));

	module.def("linear", [](double start, double end, lr_int len) { return librapid::linear(start, end, len); }, py::arg("start") = double(0), py::arg("end"), py::arg("len"));
	module.def("range", [](double start, double end, double inc) { return librapid::range(start, end, inc); }, py::arg("start") = double(0), py::arg("end"), py::arg("inc") = double(1));

	module.def("minimum", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::minimum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("minimum", [](const librapid::ndarray &x1, double x2) { return librapid::minimum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("minimum", [](double x1, const librapid::ndarray &x2) { return librapid::minimum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("minimum", [](double x1, double x2) { return librapid::minimum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	
	module.def("maximum", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::maximum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("maximum", [](const librapid::ndarray &x1, double x2) { return librapid::maximum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("maximum", [](double x1, const librapid::ndarray &x2) { return librapid::maximum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("maximum", [](double x1, double x2) { return librapid::maximum(x1, x2); }, py::arg("x1"), py::arg("x2"));

	module.def("less_than", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::less_than(x1, x2); });
	module.def("less_than", [](const librapid::ndarray &x1, double x2) { return librapid::less_than(x1, x2); });
	module.def("less_than", [](double x1, const librapid::ndarray &x2) { return librapid::greater_than(x2, x1); });
	module.def("less_than", [](double x1, double x2) { return librapid::less_than(x1, x2); });

	module.def("greater_than", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::greater_than(x1, x2); });
	module.def("greater_than", [](const librapid::ndarray &x1, double x2) { return librapid::greater_than(x1, x2); });
	module.def("greater_than", [](double x1, const librapid::ndarray &x2) { return librapid::less_than(x2, x1); });
	module.def("greater_than", [](double x1, double x2) { return librapid::greater_than(x1, x2); });

	module.def("less_than_or_equal", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::less_than_or_equal(x1, x2); });
	module.def("less_than_or_equal", [](const librapid::ndarray &x1, double x2) { return librapid::less_than_or_equal(x1, x2); });
	module.def("less_than_or_equal", [](double x1, const librapid::ndarray &x2) { return librapid::greater_than_or_equal(x2, x1); });
	module.def("less_than_or_equal", [](double x1, double x2) { return librapid::less_than_or_equal(x1, x2); });

	module.def("greater_than_or_equal", [](const librapid::ndarray &x1, const librapid::ndarray &x2) { return librapid::greater_than_or_equal(x1, x2); });
	module.def("greater_than_or_equal", [](const librapid::ndarray &x1, double x2) { return librapid::greater_than_or_equal(x1, x2); });
	module.def("greater_than_or_equal", [](double x1, const librapid::ndarray &x2) { return librapid::less_than_or_equal(x2, x1); });
	module.def("greater_than_or_equal", [](double x1, double x2) { return librapid::greater_than_or_equal(x1, x2); });

	module.def("sum", [](const librapid::ndarray &arr, lr_int axis) { return librapid::sum(arr, axis); }, py::arg("arr"), py::arg("axis") = librapid::AUTO);
	module.def("product", [](const librapid::ndarray &arr, lr_int axis) { return librapid::product(arr, axis); }, py::arg("arr"), py::arg("axis") = librapid::AUTO);
	module.def("mean", [](const librapid::ndarray &arr, lr_int axis) { return librapid::mean(arr, axis); }, py::arg("arr"), py::arg("axis") = librapid::AUTO);
	module.def("abs", [](const librapid::ndarray &arr) { return librapid::abs(arr); }, py::arg("arr"));
	module.def("square", [](const librapid::ndarray &arr) { return librapid::square(arr); }, py::arg("arr"));
	module.def("variance", [](const librapid::ndarray &arr, lr_int axis) { return librapid::sum(arr, axis); }, py::arg("arr"), py::arg("axis") = librapid::AUTO);

	py::module_ activations = module.def_submodule("activations", "LibRapid neural network activations");

	py::class_<python_activation<librapid::activations::sigmoid<double>>>(activations, "sigmoid")
		.def(py::init<>())
		.def("construct", [](python_activation<librapid::activations::sigmoid<double>> &activation, lr_int prev_nodes) { activation.activation->construct(prev_nodes); }, py::arg("prev_nodes"))
		.def("f", [](const python_activation<librapid::activations::sigmoid<double>> &activation, const librapid::ndarray &arr) { return activation.activation->f(arr); }, py::arg("arr"))
		.def("df", [](const python_activation<librapid::activations::sigmoid<double>> &activation, const librapid::ndarray &arr) { return activation.activation->df(arr); }, py::arg("arr"))
		.def("weight", [](const python_activation<librapid::activations::sigmoid<double>> &activation, const librapid::extent &shape) { return activation.activation->weight(shape); }, py::arg("shape"));

	py::class_<python_activation<librapid::activations::tanh<double>>>(activations, "tanh")
		.def(py::init<>())
		.def("construct", [](python_activation<librapid::activations::tanh<double>> &activation, lr_int prev_nodes) { activation.activation->construct(prev_nodes); }, py::arg("prev_nodes"))
		.def("f", [](const python_activation<librapid::activations::tanh<double>> &activation, const librapid::ndarray &arr) { return activation.activation->f(arr); }, py::arg("arr"))
		.def("df", [](const python_activation<librapid::activations::tanh<double>> &activation, const librapid::ndarray &arr) { return activation.activation->df(arr); }, py::arg("arr"))
		.def("weight", [](const python_activation<librapid::activations::tanh<double>> &activation, const librapid::extent &shape) { return activation.activation->weight(shape); }, py::arg("shape"));

	py::class_<python_activation<librapid::activations::relu<double>>>(activations, "relu")
		.def(py::init<>())
		.def("construct", [](python_activation<librapid::activations::relu<double>> &activation, lr_int prev_nodes) { activation.activation->construct(prev_nodes); }, py::arg("prev_nodes"))
		.def("f", [](const python_activation<librapid::activations::relu<double>> &activation, const librapid::ndarray &arr) { return activation.activation->f(arr); }, py::arg("arr"))
		.def("df", [](const python_activation<librapid::activations::relu<double>> &activation, const librapid::ndarray &arr) { return activation.activation->df(arr); }, py::arg("arr"))
		.def("weight", [](const python_activation<librapid::activations::relu<double>> &activation, const librapid::extent &shape) { return activation.activation->weight(shape); }, py::arg("shape"));

	py::class_<python_activation<librapid::activations::leaky_relu<double>>>(activations, "leaky_relu")
		.def(py::init<>())
		.def("construct", [](python_activation<librapid::activations::leaky_relu<double>> &activation, lr_int prev_nodes) { activation.activation->construct(prev_nodes); }, py::arg("prev_nodes"))
		.def("f", [](const python_activation<librapid::activations::leaky_relu<double>> &activation, const librapid::ndarray &arr) { return activation.activation->f(arr); }, py::arg("arr"))
		.def("df", [](const python_activation<librapid::activations::leaky_relu<double>> &activation, const librapid::ndarray &arr) { return activation.activation->df(arr); }, py::arg("arr"))
		.def("weight", [](const python_activation<librapid::activations::leaky_relu<double>> &activation, const librapid::extent &shape) { return activation.activation->weight(shape); }, py::arg("shape"));
}
