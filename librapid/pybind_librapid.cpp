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

struct python_sgd {
	librapid::optimizers::basic_optimizer<python_dtype> *optimizer;

	python_sgd(python_dtype learning_rate = 1e-2) { optimizer = new librapid::optimizers::sgd<python_dtype>(learning_rate); }
	~python_sgd() { delete optimizer; }
	librapid::basic_ndarray<python_dtype> apply(const librapid::basic_ndarray<python_dtype> &w, const librapid::basic_ndarray<python_dtype> &dw) { return optimizer->apply(w, dw); }
	void set_param(const std::string &name, const python_dtype val) { optimizer->set_param(name, val); }
	void set_param(const std::string &name, const librapid::basic_ndarray<python_dtype> &val) { optimizer->set_param(name, val); }
	const librapid::basic_ndarray<python_dtype> get_param(const std::string &name) { return optimizer->get_param(name); }
};

struct python_sgd_momentum {
	librapid::optimizers::basic_optimizer<python_dtype> *optimizer;

	python_sgd_momentum(python_dtype learning_rate = 1e-2, python_dtype momentum = 0.9, const librapid::basic_ndarray<python_dtype> &velocity = librapid::basic_ndarray<python_dtype>()) { optimizer = new librapid::optimizers::sgd_momentum<python_dtype>(learning_rate, momentum, velocity); }
	~python_sgd_momentum() { delete optimizer; }
	librapid::basic_ndarray<python_dtype> apply(const librapid::basic_ndarray<python_dtype> &w, const librapid::basic_ndarray<python_dtype> &dw) { return optimizer->apply(w, dw); }
	void set_param(const std::string &name, const python_dtype val) { optimizer->set_param(name, val); }
	void set_param(const std::string &name, const librapid::basic_ndarray<python_dtype> &val) { optimizer->set_param(name, val); }
	const librapid::basic_ndarray<python_dtype> get_param(const std::string &name) { return optimizer->get_param(name); }
};

struct python_rmsprop {
	librapid::optimizers::basic_optimizer<python_dtype> *optimizer;

	python_rmsprop(python_dtype learning_rate = 1e-2, python_dtype decay_rate = 0.99, python_dtype epsilon = 1e-8, const librapid::basic_ndarray<python_dtype> &cache = librapid::basic_ndarray<python_dtype>()) { optimizer = new librapid::optimizers::rmsprop<python_dtype>(learning_rate, decay_rate, epsilon, cache); }
	~python_rmsprop() { delete optimizer; }
	librapid::basic_ndarray<python_dtype> apply(const librapid::basic_ndarray<python_dtype> &w, const librapid::basic_ndarray<python_dtype> &dw) { return optimizer->apply(w, dw); }
	void set_param(const std::string &name, const python_dtype val) { optimizer->set_param(name, val); }
	void set_param(const std::string &name, const librapid::basic_ndarray<python_dtype> &val) { optimizer->set_param(name, val); }
	const librapid::basic_ndarray<python_dtype> get_param(const std::string &name) { return optimizer->get_param(name); }
};

struct python_adam {
	librapid::optimizers::basic_optimizer<python_dtype> *optimizer;

	python_adam(python_dtype learning_rate = 1e-3, python_dtype beta1 = 0.9, python_dtype beta2 = 0.999, python_dtype epsilon = 1e-8, const librapid::basic_ndarray<python_dtype> &m = librapid::basic_ndarray<python_dtype>(), const librapid::basic_ndarray<python_dtype> &v = librapid::basic_ndarray<python_dtype>(), lr_int time = 0) { optimizer = new librapid::optimizers::adam<python_dtype>(learning_rate, beta1, beta2, epsilon, m, v, time); }
	~python_adam() { delete optimizer; }
	librapid::basic_ndarray<python_dtype> apply(const librapid::basic_ndarray<python_dtype> &w, const librapid::basic_ndarray<python_dtype> &dw) { return optimizer->apply(w, dw); }
	void set_param(const std::string &name, const python_dtype val) { optimizer->set_param(name, val); }
	void set_param(const std::string &name, const librapid::basic_ndarray<python_dtype> &val) { optimizer->set_param(name, val); }
	const librapid::basic_ndarray<python_dtype> get_param(const std::string &name) { return optimizer->get_param(name); }
};

PYBIND11_MODULE(librapid_, module)
{
	module.doc() = module_docstring;

	module.def("bitness", &librapid::python_bitness);
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

	module.def("product", [](const std::vector<python_dtype> &vals) { return librapid::math::product(vals); }, py::arg("vals"));

	module.def("min", [](const std::vector<python_dtype> &vals) { return librapid::math::min(vals); }, py::arg("vals"));
	module.def("max", [](const std::vector<python_dtype> &vals) { return librapid::math::max(vals); }, py::arg("vals"));

	module.def("abs", [](python_dtype val) { return librapid::math::abs(val); }, py::arg("val"));
	module.def("map", [](python_dtype val, python_dtype start1, python_dtype stop1, python_dtype start2, python_dtype stop2) { return librapid::math::map(val, start1, stop1, start2, stop2); }, py::arg("val"), py::arg("start1") = python_dtype(0), py::arg("stop1") = python_dtype(1), py::arg("start2") = python_dtype(0), py::arg("stop2") = python_dtype(1));
	module.def("random", [](python_dtype min, python_dtype max) { return librapid::math::random(min, max); }, py::arg("min") = 0, py::arg("max") = 1);
	module.def("pow10", &librapid::math::pow10);
	module.def("round", [](python_dtype val, lr_int places) { return librapid::math::round(val, places); }, py::arg("val"), py::arg("places") = 0);
	module.def("round_sigfig", [](python_dtype val, lr_int figs) { return librapid::math::round(val, figs); }, py::arg("val"), py::arg("figs") = 3);

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
	py::class_<librapid::basic_ndarray<python_dtype>>(module, "ndarray")
		.def(py::init<>())
		.def(py::init<const librapid::extent &>())
		.def(py::init<const librapid::extent &, python_dtype>())
		.def(py::init<const librapid::basic_ndarray<python_dtype> &>())
		
		.def_static("from_data", [](const python_dtype val) { return librapid::basic_ndarray<python_dtype>::from_data(val); }, py::arg("val") = python_dtype(0))
		.def_static("from_data", [](const V<python_dtype> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<python_dtype>())
		.def_static("from_data", [](const V<V<python_dtype>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<python_dtype>>())
		.def_static("from_data", [](const V<V<V<python_dtype>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<python_dtype>>>())
		.def_static("from_data", [](const V<V<V<V<python_dtype>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<python_dtype>>>>())
		.def_static("from_data", [](const V<V<V<V<V<python_dtype>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<python_dtype>>>>>())
		.def_static("from_data", [](const V<V<V<V<V<V<python_dtype>>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<python_dtype>>>>>>())
		.def_static("from_data", [](const V<V<V<V<V<V<V<python_dtype>>>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<python_dtype>>>>>>>())
		.def_static("from_data", [](const V<V<V<V<V<V<V<V<python_dtype>>>>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<V<python_dtype>>>>>>>>())
		.def_static("from_data", [](const V<V<V<V<V<V<V<V<V<python_dtype>>>>>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<V<V<python_dtype>>>>>>>>>())
		.def_static("from_data", [](const V<V<V<V<V<V<V<V<V<V<python_dtype>>>>>>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<V<V<V<python_dtype>>>>>>>>>>())

		.def("set_to", &librapid::basic_ndarray<python_dtype>::set_to)

		.def_property_readonly("ndim", &librapid::basic_ndarray<python_dtype>::ndim)
		.def_property_readonly("size", &librapid::basic_ndarray<python_dtype>::size)
		.def_property_readonly("is_initialized", &librapid::basic_ndarray<python_dtype>::is_initialized)
		.def_property_readonly("is_scalar", &librapid::basic_ndarray<python_dtype>::is_scalar)

		.def("get_extent", &librapid::basic_ndarray<python_dtype>::get_extent)
		.def("get_stride", &librapid::basic_ndarray<python_dtype>::get_stride)
		.def_property_readonly("extent", &librapid::basic_ndarray<python_dtype>::get_extent)
		.def_property_readonly("stride", &librapid::basic_ndarray<python_dtype>::get_stride)

		.def("__eq__", [](const librapid::basic_ndarray<python_dtype> &arr, python_dtype val) { return arr == val; }, py::arg("val"))
		.def("__ne__", [](const librapid::basic_ndarray<python_dtype> &arr, python_dtype val) { return arr != val; }, py::arg("val"))

		.def("__getitem__", [](const librapid::basic_ndarray<python_dtype> &arr, lr_int index) { return arr[index]; }, py::arg("index"))
		.def("__setitem__", [](librapid::basic_ndarray<python_dtype> &arr, lr_int index, const librapid::basic_ndarray<python_dtype> &value) { arr[index] = value; }, py::arg("index"), py::arg("value"))
		.def("__setitem__", [](librapid::basic_ndarray<python_dtype> &arr, lr_int index, python_dtype value) { arr[index] = value; }, py::arg("index"), py::arg("value"))

		.def("fill", [](librapid::basic_ndarray<python_dtype> &arr, python_dtype filler) { arr.fill(filler); }, py::arg("filler") = python_dtype(0))
		.def("filled", [](const librapid::basic_ndarray<python_dtype> &arr, python_dtype filler) { return arr.filled(filler); }, py::arg("filler") = python_dtype(0))
		.def("fill_random", [](librapid::basic_ndarray<python_dtype> &arr, python_dtype min, python_dtype max) { arr.fill_random(min, max); }, py::arg("min") = python_dtype(0), py::arg("max") = python_dtype(1))
		.def("filled_random", [](const librapid::basic_ndarray<python_dtype> &arr, python_dtype min, python_dtype max) { return arr.filled_random(min, max); }, py::arg("min") = python_dtype(0), py::arg("max") = python_dtype(1))
		.def("map", [](librapid::basic_ndarray<python_dtype> &arr, const std::function<python_dtype(python_dtype)> &func) { arr.map(func); })
		.def("mapped", [](const librapid::basic_ndarray<python_dtype> &arr, const std::function<python_dtype(python_dtype)> &func) { return arr.mapped(func); })
		.def("clone", &librapid::basic_ndarray<python_dtype>::clone)
		.def("set_value", &librapid::basic_ndarray<python_dtype>::set_value)

		.def("__add__", [](const librapid::basic_ndarray<python_dtype> &lhs, const librapid::basic_ndarray<python_dtype> &rhs) { return lhs + rhs; }, py::arg("rhs"))
		.def("__sub__", [](const librapid::basic_ndarray<python_dtype> &lhs, const librapid::basic_ndarray<python_dtype> &rhs) { return lhs - rhs; }, py::arg("rhs"))
		.def("__mul__", [](const librapid::basic_ndarray<python_dtype> &lhs, const librapid::basic_ndarray<python_dtype> &rhs) { return lhs * rhs; }, py::arg("rhs"))
		.def("__truediv__", [](const librapid::basic_ndarray<python_dtype> &lhs, const librapid::basic_ndarray<python_dtype> &rhs) { return lhs / rhs; }, py::arg("rhs"))

		.def("__add__", [](const librapid::basic_ndarray<python_dtype> &lhs, python_dtype rhs) { return lhs + rhs; }, py::arg("rhs"))
		.def("__sub__", [](const librapid::basic_ndarray<python_dtype> &lhs, python_dtype rhs) { return lhs - rhs; }, py::arg("rhs"))
		.def("__mul__", [](const librapid::basic_ndarray<python_dtype> &lhs, python_dtype rhs) { return lhs * rhs; }, py::arg("rhs"))
		.def("__truediv__", [](const librapid::basic_ndarray<python_dtype> &lhs, python_dtype rhs) { return lhs / rhs; }, py::arg("rhs"))

		.def("__radd__", [](const librapid::basic_ndarray<python_dtype> &rhs, python_dtype lhs) { return lhs + rhs; }, py::arg("lhs"))
		.def("__rsub__", [](const librapid::basic_ndarray<python_dtype> &rhs, python_dtype lhs) { return lhs - rhs; }, py::arg("lhs"))
		.def("__rmul__", [](const librapid::basic_ndarray<python_dtype> &rhs, python_dtype lhs) { return lhs * rhs; }, py::arg("lhs"))
		.def("__rtruediv__", [](const librapid::basic_ndarray<python_dtype> &rhs, python_dtype lhs) { return lhs / rhs; }, py::arg("lhs"))

		.def("__iadd__", [](librapid::basic_ndarray<python_dtype> &lhs, const librapid::basic_ndarray<python_dtype> &rhs) { lhs += rhs; return lhs; }, py::arg("rhs"))
		.def("__isub__", [](librapid::basic_ndarray<python_dtype> &lhs, const librapid::basic_ndarray<python_dtype> &rhs) { lhs -= rhs; return lhs; }, py::arg("rhs"))
		.def("__imul__", [](librapid::basic_ndarray<python_dtype> &lhs, const librapid::basic_ndarray<python_dtype> &rhs) { lhs *= rhs; return lhs; }, py::arg("rhs"))
		.def("__itruediv__", [](librapid::basic_ndarray<python_dtype> &lhs, const librapid::basic_ndarray<python_dtype> &rhs) { lhs /= rhs; return lhs; }, py::arg("rhs"))

		.def("__iadd__", [](librapid::basic_ndarray<python_dtype> &lhs, python_dtype rhs) { lhs += rhs; return lhs; }, py::arg("rhs"))
		.def("__isub__", [](librapid::basic_ndarray<python_dtype> &lhs, python_dtype rhs) { lhs -= rhs; return lhs; }, py::arg("rhs"))
		.def("__imul__", [](librapid::basic_ndarray<python_dtype> &lhs, python_dtype rhs) { lhs *= rhs; return lhs; }, py::arg("rhs"))
		.def("__itruediv__", [](librapid::basic_ndarray<python_dtype> &lhs, python_dtype rhs) { lhs /= rhs; return lhs; }, py::arg("rhs"))

		.def("__neg__", [](const librapid::basic_ndarray<python_dtype> &arr) { return -arr; })

		.def("minimum", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::minimum(x1, x2); }, py::arg("x2"))
		.def("minimum", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::minimum(x1, x2); }, py::arg("x2"))
		.def("maximum", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::maximum(x1, x2); }, py::arg("x2"))
		.def("maximum", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::maximum(x1, x2); }, py::arg("x2"))

		.def("less_than", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::less_than(x1, x2); }, py::arg("x2"))
		.def("less_than", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::less_than(x1, x2); }, py::arg("x2"))
		.def("greater_than", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::greater_than(x1, x2); }, py::arg("x2"))
		.def("greater_than", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::greater_than(x1, x2); }, py::arg("x2"))
		.def("less_than_or_equal", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::less_than_or_equal(x1, x2); }, py::arg("x2"))
		.def("less_than_or_equal", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::less_than_or_equal(x1, x2); }, py::arg("x2"))
		.def("greater_than_or_equal", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::greater_than_or_equal(x1, x2); }, py::arg("x2"))
		.def("greater_than_or_equal", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::greater_than_or_equal(x1, x2); }, py::arg("x2"))

		.def("dot", [](const librapid::basic_ndarray<python_dtype> &lhs, const librapid::basic_ndarray<python_dtype> &rhs) { return lhs.dot(rhs); }, py::arg("rhs"))
		.def("__matmul__", [](const librapid::basic_ndarray<python_dtype> &lhs, const librapid::basic_ndarray<python_dtype> &rhs) { return lhs.dot(rhs); }, py::arg("rhs"))

		.def("reshape", [](librapid::basic_ndarray<python_dtype> &arr, const std::vector<lr_int> &order) { arr.reshape(order); }, py::arg("order"))
		.def("reshape", [](librapid::basic_ndarray<python_dtype> &arr, const librapid::extent &order) { arr.reshape(order); }, py::arg("order"))
		.def("reshaped", [](const librapid::basic_ndarray<python_dtype> &arr, const std::vector<lr_int> &order) { return arr.reshaped(order); }, py::arg("order"))
		.def("reshape", [](librapid::basic_ndarray<python_dtype> &arr, py::args args) { arr.reshape(py::cast<std::vector<lr_int>>(args)); })
		.def("reshaped", [](const librapid::basic_ndarray<python_dtype> &arr, py::args args) { return arr.reshaped(py::cast<std::vector<lr_int>>(args)); })
		.def("subarray", [](const librapid::basic_ndarray<python_dtype> &arr, const std::vector<lr_int> &axes) { return arr.subarray(axes); }, py::arg("axes"))

		.def("strip_front", &librapid::basic_ndarray<python_dtype>::strip_front)
		.def("strip_back", &librapid::basic_ndarray<python_dtype>::strip_back)
		.def("strip", &librapid::basic_ndarray<python_dtype>::strip)
		.def("stripped_front", &librapid::basic_ndarray<python_dtype>::stripped_front)
		.def("stripped_back", &librapid::basic_ndarray<python_dtype>::stripped_back)
		.def("stripped", &librapid::basic_ndarray<python_dtype>::stripped)

		.def("transpose", [](librapid::basic_ndarray<python_dtype> &arr, const std::vector<lr_int> &order) { arr.transpose(order); }, py::arg("order"))
		.def("transpose", [](librapid::basic_ndarray<python_dtype> &arr) { arr.transpose(); })
		.def("transposed", [](const librapid::basic_ndarray<python_dtype> &arr, const std::vector<lr_int> &order) { return arr.transposed(order); }, py::arg("order"))
		.def("transposed", [](const librapid::basic_ndarray<python_dtype> &arr) { return arr.transposed(); })

		.def("sum", [](const librapid::basic_ndarray<python_dtype> &arr, lr_int axis) { return arr.sum(axis); }, py::arg("axis") = librapid::AUTO)
		.def("product", [](const librapid::basic_ndarray<python_dtype> &arr, lr_int axis) { return arr.product(axis); }, py::arg("axis") = librapid::AUTO)
		.def("mean", [](const librapid::basic_ndarray<python_dtype> &arr, lr_int axis) { return arr.mean(axis); }, py::arg("axis") = librapid::AUTO)
		.def("abs", [](const librapid::basic_ndarray<python_dtype> &arr) { return arr.abs(); })
		.def("square", [](const librapid::basic_ndarray<python_dtype> &arr) { return arr.square(); })
		.def("variance", [](const librapid::basic_ndarray<python_dtype> &arr, lr_int axis) { return arr.variance(axis); }, py::arg("axis") = librapid::AUTO)

		.def("__str__", [](const librapid::basic_ndarray<python_dtype> &arr) { return arr.str(0); })
		.def("__int__", [](const librapid::basic_ndarray<python_dtype> &arr) { if (!arr.is_scalar()) throw py::value_error("Cannot convert non-scalar array to int"); return (lr_int) *arr.get_data_start(); })
		.def("__float__", [](const librapid::basic_ndarray<python_dtype> &arr) { if (!arr.is_scalar()) throw py::value_error("Cannot convert non-scalar array to float"); return (python_dtype) *arr.get_data_start(); })
		.def("__repr__", [](const librapid::basic_ndarray<python_dtype> &arr) { return "<librapid.ndarray " + arr.str(18) + ">"; });

	module.def("zeros_like", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::zeros_like(arr); }, py::arg("arr"));
	module.def("ones_like", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::ones_like(arr); }, py::arg("arr"));
	module.def("random_like", [](const librapid::basic_ndarray<python_dtype> &arr, python_dtype min, python_dtype max) { return librapid::random_like(arr, min, max); }, py::arg("arr"), py::arg("min") = 0, py::arg("max") = 1);

	module.def("add", [](python_dtype lhs, python_dtype rhs) { return lhs + rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("add", [](librapid::basic_ndarray<python_dtype> lhs, python_dtype rhs) { return lhs + rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("add", [](python_dtype lhs, librapid::basic_ndarray<python_dtype> rhs) { return lhs + rhs; }, py::arg("lhs"), py::arg("rhs"));

	module.def("sub", [](python_dtype lhs, python_dtype rhs) { return lhs - rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("sub", [](librapid::basic_ndarray<python_dtype> lhs, python_dtype rhs) { return lhs - rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("sub", [](python_dtype lhs, librapid::basic_ndarray<python_dtype> rhs) { return lhs - rhs; }, py::arg("lhs"), py::arg("rhs"));
	
	module.def("mul", [](python_dtype lhs, python_dtype rhs) { return lhs * rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("mul", [](librapid::basic_ndarray<python_dtype> lhs, python_dtype rhs) { return lhs * rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("mul", [](python_dtype lhs, librapid::basic_ndarray<python_dtype> rhs) { return lhs * rhs; }, py::arg("lhs"), py::arg("rhs"));

	module.def("div", [](python_dtype lhs, python_dtype rhs) { return lhs / rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("div", [](librapid::basic_ndarray<python_dtype> lhs, python_dtype rhs) { return lhs / rhs; }, py::arg("lhs"), py::arg("rhs"));
	module.def("div", [](python_dtype lhs, librapid::basic_ndarray<python_dtype> rhs) { return lhs / rhs; }, py::arg("lhs"), py::arg("rhs"));

	module.def("exp", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::exp(arr); }, py::arg("arr"));

	module.def("sin", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::sin(arr); }, py::arg("arr"));
	module.def("cos", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::cos(arr); }, py::arg("arr"));
	module.def("tan", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::tan(arr); }, py::arg("arr"));

	module.def("asin", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::asin(arr); }, py::arg("arr"));
	module.def("acos", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::acos(arr); }, py::arg("arr"));
	module.def("atan", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::atan(arr); }, py::arg("arr"));

	module.def("sinh", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::sinh(arr); }, py::arg("arr"));
	module.def("cosh", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::cosh(arr); }, py::arg("arr"));
	module.def("tanh", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::tanh(arr); }, py::arg("arr"));

	module.def("reshape", [](const librapid::basic_ndarray<python_dtype> &arr, const librapid::extent &shape) { return librapid::reshape(arr, shape); }, py::arg("arr"), py::arg("shape"));
	module.def("reshape", [](const librapid::basic_ndarray<python_dtype> &arr, const std::vector<lr_int> &shape) { return librapid::reshape(arr, librapid::extent(shape)); }, py::arg("arr"), py::arg("shape"));

	module.def("linear", [](python_dtype start, python_dtype end, lr_int len) { return librapid::linear(start, end, len); }, py::arg("start") = python_dtype(0), py::arg("end"), py::arg("len"));
	module.def("range", [](python_dtype start, python_dtype end, python_dtype inc) { return librapid::range(start, end, inc); }, py::arg("start") = python_dtype(0), py::arg("end"), py::arg("inc") = python_dtype(1));

	module.def("minimum", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::minimum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("minimum", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::minimum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("minimum", [](python_dtype x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::minimum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("minimum", [](python_dtype x1, python_dtype x2) { return librapid::minimum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	
	module.def("maximum", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::maximum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("maximum", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::maximum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("maximum", [](python_dtype x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::maximum(x1, x2); }, py::arg("x1"), py::arg("x2"));
	module.def("maximum", [](python_dtype x1, python_dtype x2) { return librapid::maximum(x1, x2); }, py::arg("x1"), py::arg("x2"));

	module.def("less_than", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::less_than(x1, x2); });
	module.def("less_than", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::less_than(x1, x2); });
	module.def("less_than", [](python_dtype x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::greater_than(x2, x1); });
	module.def("less_than", [](python_dtype x1, python_dtype x2) { return librapid::less_than(x1, x2); });

	module.def("greater_than", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::greater_than(x1, x2); });
	module.def("greater_than", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::greater_than(x1, x2); });
	module.def("greater_than", [](python_dtype x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::less_than(x2, x1); });
	module.def("greater_than", [](python_dtype x1, python_dtype x2) { return librapid::greater_than(x1, x2); });

	module.def("less_than_or_equal", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::less_than_or_equal(x1, x2); });
	module.def("less_than_or_equal", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::less_than_or_equal(x1, x2); });
	module.def("less_than_or_equal", [](python_dtype x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::greater_than_or_equal(x2, x1); });
	module.def("less_than_or_equal", [](python_dtype x1, python_dtype x2) { return librapid::less_than_or_equal(x1, x2); });

	module.def("greater_than_or_equal", [](const librapid::basic_ndarray<python_dtype> &x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::greater_than_or_equal(x1, x2); });
	module.def("greater_than_or_equal", [](const librapid::basic_ndarray<python_dtype> &x1, python_dtype x2) { return librapid::greater_than_or_equal(x1, x2); });
	module.def("greater_than_or_equal", [](python_dtype x1, const librapid::basic_ndarray<python_dtype> &x2) { return librapid::less_than_or_equal(x2, x1); });
	module.def("greater_than_or_equal", [](python_dtype x1, python_dtype x2) { return librapid::greater_than_or_equal(x1, x2); });

	module.def("sum", [](const librapid::basic_ndarray<python_dtype> &arr, lr_int axis) { return librapid::sum(arr, axis); }, py::arg("arr"), py::arg("axis") = librapid::AUTO);
	module.def("product", [](const librapid::basic_ndarray<python_dtype> &arr, lr_int axis) { return librapid::product(arr, axis); }, py::arg("arr"), py::arg("axis") = librapid::AUTO);
	module.def("mean", [](const librapid::basic_ndarray<python_dtype> &arr, lr_int axis) { return librapid::mean(arr, axis); }, py::arg("arr"), py::arg("axis") = librapid::AUTO);
	module.def("abs", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::abs(arr); }, py::arg("arr"));
	module.def("square", [](const librapid::basic_ndarray<python_dtype> &arr) { return librapid::square(arr); }, py::arg("arr"));
	module.def("variance", [](const librapid::basic_ndarray<python_dtype> &arr, lr_int axis) { return librapid::sum(arr, axis); }, py::arg("arr"), py::arg("axis") = librapid::AUTO);

	module.def("from_data", [](python_dtype val) { return librapid::basic_ndarray<python_dtype>::from_data(val); }, py::arg("val") = python_dtype(0));
	module.def("from_data", [](const V<python_dtype> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<python_dtype>());
	module.def("from_data", [](const V<V<python_dtype>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<python_dtype>>());
	module.def("from_data", [](const V<V<V<python_dtype>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<python_dtype>>>());
	module.def("from_data", [](const V<V<V<V<python_dtype>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<python_dtype>>>>());
	module.def("from_data", [](const V<V<V<V<V<python_dtype>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<python_dtype>>>>>());
	module.def("from_data", [](const V<V<V<V<V<V<python_dtype>>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<python_dtype>>>>>>());
	module.def("from_data", [](const V<V<V<V<V<V<V<python_dtype>>>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<python_dtype>>>>>>>());
	module.def("from_data", [](const V<V<V<V<V<V<V<V<python_dtype>>>>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<V<python_dtype>>>>>>>>());
	module.def("from_data", [](const V<V<V<V<V<V<V<V<V<python_dtype>>>>>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<V<V<python_dtype>>>>>>>>>());
	module.def("from_data", [](const V<V<V<V<V<V<V<V<V<V<python_dtype>>>>>>>>>> &vals) { return librapid::basic_ndarray<python_dtype>::from_data(vals); }, py::arg("vals") = V<V<V<V<V<V<V<V<V<V<python_dtype>>>>>>>>>>());

	py::module_ activations = module.def_submodule("activations", "LibRapid neural network activations");

	py::class_<python_activation<librapid::activations::sigmoid<python_dtype>>>(activations, "sigmoid")
		.def(py::init<>())
		.def("construct", [](python_activation<librapid::activations::sigmoid<python_dtype>> &activation, lr_int prev_nodes) { activation.activation->construct(prev_nodes); }, py::arg("prev_nodes"))
		.def("f", [](const python_activation<librapid::activations::sigmoid<python_dtype>> &activation, const librapid::basic_ndarray<python_dtype> &arr) { return activation.activation->f(arr); }, py::arg("arr"))
		.def("df", [](const python_activation<librapid::activations::sigmoid<python_dtype>> &activation, const librapid::basic_ndarray<python_dtype> &arr) { return activation.activation->df(arr); }, py::arg("arr"))
		.def("weight", [](const python_activation<librapid::activations::sigmoid<python_dtype>> &activation, const librapid::extent &shape) { return activation.activation->weight(shape); }, py::arg("shape"));

	py::class_<python_activation<librapid::activations::tanh<python_dtype>>>(activations, "tanh")
		.def(py::init<>())
		.def("construct", [](python_activation<librapid::activations::tanh<python_dtype>> &activation, lr_int prev_nodes) { activation.activation->construct(prev_nodes); }, py::arg("prev_nodes"))
		.def("f", [](const python_activation<librapid::activations::tanh<python_dtype>> &activation, const librapid::basic_ndarray<python_dtype> &arr) { return activation.activation->f(arr); }, py::arg("arr"))
		.def("df", [](const python_activation<librapid::activations::tanh<python_dtype>> &activation, const librapid::basic_ndarray<python_dtype> &arr) { return activation.activation->df(arr); }, py::arg("arr"))
		.def("weight", [](const python_activation<librapid::activations::tanh<python_dtype>> &activation, const librapid::extent &shape) { return activation.activation->weight(shape); }, py::arg("shape"));

	py::class_<python_activation<librapid::activations::relu<python_dtype>>>(activations, "relu")
		.def(py::init<>())
		.def("construct", [](python_activation<librapid::activations::relu<python_dtype>> &activation, lr_int prev_nodes) { activation.activation->construct(prev_nodes); }, py::arg("prev_nodes"))
		.def("f", [](const python_activation<librapid::activations::relu<python_dtype>> &activation, const librapid::basic_ndarray<python_dtype> &arr) { return activation.activation->f(arr); }, py::arg("arr"))
		.def("df", [](const python_activation<librapid::activations::relu<python_dtype>> &activation, const librapid::basic_ndarray<python_dtype> &arr) { return activation.activation->df(arr); }, py::arg("arr"))
		.def("weight", [](const python_activation<librapid::activations::relu<python_dtype>> &activation, const librapid::extent &shape) { return activation.activation->weight(shape); }, py::arg("shape"));

	py::class_<python_activation<librapid::activations::leaky_relu<python_dtype>>>(activations, "leaky_relu")
		.def(py::init<>())
		.def("construct", [](python_activation<librapid::activations::leaky_relu<python_dtype>> &activation, lr_int prev_nodes) { activation.activation->construct(prev_nodes); }, py::arg("prev_nodes"))
		.def("f", [](const python_activation<librapid::activations::leaky_relu<python_dtype>> &activation, const librapid::basic_ndarray<python_dtype> &arr) { return activation.activation->f(arr); }, py::arg("arr"))
		.def("df", [](const python_activation<librapid::activations::leaky_relu<python_dtype>> &activation, const librapid::basic_ndarray<python_dtype> &arr) { return activation.activation->df(arr); }, py::arg("arr"))
		.def("weight", [](const python_activation<librapid::activations::leaky_relu<python_dtype>> &activation, const librapid::extent &shape) { return activation.activation->weight(shape); }, py::arg("shape"));

	py::module_ optimizers = module.def_submodule("optimizers", "LibRapid neural network optimizers");

	py::class_<python_sgd>(optimizers, "sgd")
		.def(py::init<python_dtype>(), py::arg("learning_rate") = 1e-2)
		.def("apply", [](python_sgd &optim, const librapid::basic_ndarray<python_dtype> &w, const librapid::basic_ndarray<python_dtype> &dw) { return optim.apply(w, dw); }, py::arg("w"), py::arg("dw"))
		.def_property("learning_rate", [](python_sgd &optim) { optim.get_param("learning rate").to_scalar(); }, [](python_sgd &optim, const python_dtype val) { optim.set_param("learning rate", val); });

	py::class_<python_sgd_momentum>(optimizers, "sgd_momentum")
		.def(py::init<python_dtype, python_dtype, const librapid::basic_ndarray<python_dtype> &>(), py::arg("learning_rate") = 1e-2, py::arg("momentum") = 0.9, py::arg("velocity") = librapid::basic_ndarray<python_dtype>())
		.def("apply", [](python_sgd_momentum &optim, const librapid::basic_ndarray<python_dtype> &w, const librapid::basic_ndarray<python_dtype> &dw) { return optim.apply(w, dw); }, py::arg("w"), py::arg("dw"))
		.def_property("learning_rate", [](python_sgd_momentum &optim) { return optim.get_param("learning rate").to_scalar(); }, [](python_sgd_momentum &optim, const python_dtype val) { optim.set_param("learning rate", val); })
		.def_property("momentum", [](python_sgd_momentum &optim) { return optim.get_param("momentum").to_scalar(); }, [](python_sgd_momentum &optim, const python_dtype val) { optim.set_param("momentum", val); })
		.def_property("velocity", [](python_sgd_momentum &optim) { return optim.get_param("velocity"); }, [](python_sgd_momentum &optim, const librapid::basic_ndarray<python_dtype> &val) { optim.set_param("velocity", val); });

	py::class_<python_rmsprop>(optimizers, "rmsprop")
		.def(py::init<python_dtype, python_dtype, python_dtype, const librapid::basic_ndarray<python_dtype> &>(), py::arg("learning_rate") = 1e-2, py::arg("decay_rate") = 0.99, py::arg("epsilon") = 1e-8, py::arg("cache") = librapid::basic_ndarray<python_dtype>())
		.def("apply", [](python_rmsprop &optim, const librapid::basic_ndarray<python_dtype> &w, const librapid::basic_ndarray<python_dtype> &dw) { return optim.apply(w, dw); }, py::arg("w"), py::arg("dw"))
		.def_property("learning_rate", [](python_rmsprop &optim) { return optim.get_param("learning rate").to_scalar(); }, [](python_rmsprop &optim, const python_dtype val) { optim.set_param("learning rate", val); })
		.def_property("decay_rate", [](python_rmsprop &optim) { return optim.get_param("decay rate").to_scalar(); }, [](python_rmsprop &optim, const python_dtype val) { optim.set_param("decay rate", val); })
		.def_property("epsilon", [](python_rmsprop &optim) { return optim.get_param("epsilon").to_scalar(); }, [](python_rmsprop &optim, const python_dtype val) { optim.set_param("epsilon", val); })
		.def_property("cache", [](python_rmsprop &optim) { return optim.get_param("cache"); }, [](python_rmsprop &optim, const librapid::basic_ndarray<python_dtype> &val) { optim.set_param("cache", val); });

	py::class_<python_adam>(optimizers, "adam")
		.def(py::init<python_dtype, python_dtype, python_dtype, python_dtype, const librapid::basic_ndarray<python_dtype> &, const librapid::basic_ndarray<python_dtype> &, lr_int>(), py::arg("learning_rate") = 1e-3, py::arg("beta1") = 0.9, py::arg("beta2") = 0.999, py::arg("epsilon") = 1e-8, py::arg("m") = librapid::basic_ndarray<python_dtype>(), py::arg("v") = librapid::basic_ndarray<python_dtype>(), py::arg("time") = 0)
		.def("apply", [](python_adam &optim, const librapid::basic_ndarray<python_dtype> &w, const librapid::basic_ndarray<python_dtype> &dw) { return optim.apply(w, dw); }, py::arg("w"), py::arg("dw"))
		.def_property("learning_rate", [](python_adam &optim) { return optim.get_param("learning rate").to_scalar(); }, [](python_adam &optim, const python_dtype val) { optim.set_param("learning rate", val); })
		.def_property("beta1", [](python_adam &optim) { return optim.get_param("beta1").to_scalar(); }, [](python_adam &optim, const python_dtype val) { optim.set_param("beta1", val); })
		.def_property("beta2", [](python_adam &optim) { return optim.get_param("beta2").to_scalar(); }, [](python_adam &optim, const python_dtype val) { optim.set_param("beta1", val); })
		.def_property("epsilon", [](python_adam &optim) { return optim.get_param("epsilon").to_scalar(); }, [](python_adam &optim, const python_dtype val) { optim.set_param("epsilon", val); })
		.def_property("m", [](python_adam &optim) { return optim.get_param("m"); }, [](python_adam &optim, const librapid::basic_ndarray<python_dtype> &val) { optim.set_param("m", val); })
		.def_property("v", [](python_adam &optim) { return optim.get_param("v"); }, [](python_adam &optim, const librapid::basic_ndarray<python_dtype> &val) { optim.set_param("v", val); })
		.def_property("time", [](python_adam &optim) { return optim.get_param("time").to_scalar(); }, [](python_adam &optim, const python_dtype val) { optim.set_param("time", val); });
}
