#include <librapid/librapid.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <functional>

#include <string>

namespace py = pybind11;

// Docstring for the module
std::string module_docstring = "A fast math and neural network library for Python and C++";

template<typename T>
using V = std::vector<T>;

PYBIND11_MODULE(librapid_, module)
{
	module.doc() = module_docstring;

	py::module_ test = module.def_submodule("test", "Tests within the librapid library");
	test.def("test_librapid", &librapid::test::test_librapid);

	module.def("bitness", &librapid::python_bitness);
	module.def("has_blas", &librapid::has_blas);
	module.def("set_blas_threads", &librapid::set_blas_threads);
	module.def("get_blas_threads", &librapid::get_blas_threads);
	
	module.def("set_num_threads", &librapid::set_num_threads);
	module.def("get_num_threads", &librapid::get_num_threads);

	module.def("now", [](){ return NOW; });
	module.def("sleep", &librapid::sleep);

	module.def("get_console_size", []() { auto size = librapid::get_console_size(); return py::make_tuple(size.rows, size.cols); });

	module.attr("AUTO") = (lr_int) -1;

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
	module.def("round", [](double val, lr_int places) { return librapid::math::round<double>(val, places); }, py::arg("val"), py::arg("places") = 0);
	module.def("round_sigfig", [](double val, lr_int figs) { return librapid::math::round(val, figs); }, py::arg("val"), py::arg("figs") = 3);

	// // The librapid extent object
	// py::class_<librapid::extent>(module, "extent")
	// 	.def(py::init<>())
	// 	.def(py::init<const std::vector<lr_int> &>())
	// 	.def(py::init<const librapid::extent &>())
	// 	.def(py::init<py::args>())

	// 	.def("__getitem__", [](const librapid::extent &e, lr_int index) { return e[index]; })
	// 	.def("__setitem__", [](librapid::extent &e, lr_int index, lr_int val) { e[index] = val; })

	// 	.def("compressed", &librapid::extent::compressed)
	// 	.def_property_readonly("ndim", &librapid::extent::ndim)
	// 	.def_property_readonly("is_valid", &librapid::extent::is_valid)
	// 	.def("reshape", [](librapid::extent &e, const std::vector<lr_int> &order) { e.reshape(order); })

	// 	.def("fix_automatic", &librapid::extent::fix_automatic)

	// 	.def("__len__", &librapid::extent::ndim)

	// 	.def("__iter__", [](const librapid::extent &e) { return py::make_iterator(e.begin(), e.end()); }, py::keep_alive<0, 1>())

	// 	.def("__eq__", &librapid::extent::operator==)
	// 	.def("__ne__", &librapid::extent::operator!=)
	// 	.def("__str__", &librapid::extent::str)
	// 	.def("__repr__", [](const librapid::extent &e) { return "<librapid." + e.str() + ">"; });

	// // The librapid stride object
	// py::class_<librapid::stride>(module, "stride")
	// 	.def(py::init<>())
	// 	.def(py::init<std::vector<lr_int>>())
	// 	.def(py::init<lr_int>())
	// 	.def(py::init<const librapid::stride &>())
	// 	.def(py::init<py::args>())
	// 	.def("from_extent", [](const std::vector<lr_int> &extent) { return librapid::stride::from_extent(extent); })

	// 	.def("__getitem__", [](const librapid::stride &s, lr_int index) { return s[index]; })
	// 	.def("__setitem__", [](librapid::stride &s, lr_int index, lr_int val) { s[index] = val; })

	// 	.def_property_readonly("ndim", &librapid::stride::ndim)
	// 	.def_property_readonly("is_valid", &librapid::stride::is_valid)
	// 	.def_property_readonly("is_trivial", &librapid::stride::is_trivial)
	// 	.def_property_readonly("is_contiguous", &librapid::stride::is_contiguous)
	// 	.def("set_contiguous", &librapid::stride::set_contiguous)
	// 	.def("set_dimensions", &librapid::stride::set_dimensions)
	// 	.def("reshape", [](librapid::stride &s, const std::vector<lr_int> &order) { s.reshape(order); })

	// 	.def("__len__", &librapid::stride::ndim)

	// 	.def("__iter__", [](const librapid::stride &s) { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())

	// 	.def("__eq__", &librapid::stride::operator==)
	// 	.def("__str__", &librapid::stride::str)
	// 	.def("__repr__", [](const librapid::stride &s) { return "<librapid." + s.str() + ">"; });

	// The librapid ndarray object
	
	// Colours
	py::module_ color = module.def_submodule("color", "A simple text color library");
	
	py::class_<librapid::color::rgb>(color, "rgb")
		.def(py::init<int, int, int>(), py::arg("red") = 0, py::arg("green") = 0, py::arg("blue") = 0)
		.def("__str__", [](const librapid::color::rgb &col) { return librapid::color::fore(col); })
		.def("__repr__", [](const librapid::color::rgb &col) { return std::string("librapid.color.rgb(red: " + std::to_string(col.red) + ", green: " + std::to_string(col.green) + ", blue: " + std::to_string(col.blue)) + ")"; });

	py::class_<librapid::color::hsl>(color, "hsl")
		.def(py::init<int, int, int>(), py::arg("red") = 0, py::arg("green") = 0, py::arg("blue") = 0)
		.def("__str__", [](const librapid::color::hsl &col) { return librapid::color::fore(col); })
		.def("__repr__", [](const librapid::color::hsl &col) { return std::string("librapid.color.hsl(hue: " + std::to_string(col.hue) + ", saturation: " + std::to_string(col.saturation) + ", lightness: " + std::to_string(col.lightness)) + ")"; });

	color.def("rgb_to_hsl", &librapid::color::rgb_to_hsl);
	color.def("hsl_to_rgb", &librapid::color::hsl_to_rgb);
	
	color.def("merge_colors", [](const librapid::color::rgb &colorA, const librapid::color::rgb &colorB) { return librapid::color::merge_colors(colorA, colorB); });
	color.def("merge_colors", [](const librapid::color::rgb &colorA, const librapid::color::hsl &colorB) { return librapid::color::merge_colors(colorA, colorB); });
	color.def("merge_colors", [](const librapid::color::hsl &colorA, const librapid::color::rgb &colorB) { return librapid::color::merge_colors(colorA, colorB); });
	color.def("merge_colors", [](const librapid::color::hsl &colorA, const librapid::color::hsl &colorB) { return librapid::color::merge_colors(colorA, colorB); });

	color.attr("clear") = librapid::color::clear;
	color.attr("bold") = librapid::color::bold;
	color.attr("blink") = librapid::color::blink;

	color.attr("black") = librapid::color::black;
	color.attr("red") = librapid::color::red;
	color.attr("green") = librapid::color::green;
	color.attr("yellow") = librapid::color::yellow;
	color.attr("blue") = librapid::color::blue;
	color.attr("magenta") = librapid::color::magenta;
	color.attr("cyan") = librapid::color::cyan;
	color.attr("white") = librapid::color::white;
	color.attr("bright_black") = librapid::color::bright_black;
	color.attr("bright_red") = librapid::color::bright_red;
	color.attr("bright_green") = librapid::color::bright_green;
	color.attr("bright_yellow") = librapid::color::bright_yellow;
	color.attr("bright_blue") = librapid::color::bright_blue;
	color.attr("bright_magenta") = librapid::color::bright_magenta;
	color.attr("bright_cyan") = librapid::color::bright_cyan;
	color.attr("bright_white") = librapid::color::bright_white;

	color.attr("black_background") = librapid::color::black_background;
	color.attr("red_background") = librapid::color::red_background;
	color.attr("green_background") = librapid::color::green_background;
	color.attr("yellow_background") = librapid::color::yellow_background;
	color.attr("blue_background") = librapid::color::blue_background;
	color.attr("magenta_background") = librapid::color::magenta_background;
	color.attr("cyan_background") = librapid::color::cyan_background;
	color.attr("white_background") = librapid::color::white_background;
	color.attr("bright_black_background") = librapid::color::bright_black_background;
	color.attr("bright_red_background") = librapid::color::bright_red_background;
	color.attr("bright_green_background") = librapid::color::bright_green_background;
	color.attr("bright_yellow_background") = librapid::color::bright_yellow_background;
	color.attr("bright_blue_background") = librapid::color::bright_blue_background;
	color.attr("bright_magenta_background") = librapid::color::bright_magenta_background;
	color.attr("bright_cyan_background") = librapid::color::bright_cyan_background;
	color.attr("bright_white_background") = librapid::color::bright_white_background;

	color.def("fore", [](const librapid::color::rgb &col) { return librapid::color::fore(col); });
	color.def("fore", [](const librapid::color::hsl &col) { return librapid::color::fore(col); });
	color.def("fore", [](int r, int g, int b) { return librapid::color::fore(r, g, b); });

	color.def("back", [](const librapid::color::rgb &col) { return librapid::color::back(col); });
	color.def("back", [](const librapid::color::hsl &col) { return librapid::color::back(col); });
	color.def("back", [](int r, int g, int b) { return librapid::color::back(r, g, b); });

	// py::implicitly_convertible<long long, librapid::basic_ndarray<python_dtype>>();
	// py::implicitly_convertible<python_dtype, librapid::basic_ndarray<python_dtype>>();
	// py::implicitly_convertible<py::tuple, librapid::basic_ndarray<python_dtype>>();
	// py::implicitly_convertible<py::list, librapid::basic_ndarray<python_dtype>>();
}
