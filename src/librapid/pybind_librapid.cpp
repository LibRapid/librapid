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

PYBIND11_MODULE(_librapid, module)
{
	module.doc() = module_docstring;

	py::module_ test = module.def_submodule("test", "Tests within the librapid library");
	test.def("testLibrapid", &librapid::test::testLibrapid);
	test.def("streamTest", &librapid::test::streamTest);

	module.def("bitness", &librapid::pythonBitness);
	module.def("hasBlas", &librapid::hasBlas);
	module.def("setBlasThreads", &librapid::setBlasThreads);
	module.def("getBlasThreads", &librapid::getBlasThreads);
	
	module.def("setNumThreads", &librapid::setNumThreads);
	module.def("getNumThreads", &librapid::getNumThreads);

	module.def("seconds", [](){ return librapid::seconds(); });
	module.def("sleep", &librapid::sleep);

	module.def("getConsoleSize", []() { auto size = librapid::getConsoleSize(); return py::make_tuple(size.rows, size.cols); });

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

	 // The librapid Extent object
	 py::class_<librapid::Extent>(module, "Extent")
	 	.def(py::init<>())
	 	.def(py::init<const std::vector<lr_int> &>())
	 	.def(py::init<const librapid::Extent &>())
	 	.def(py::init<py::args>())

	 	.def("__getitem__", [](const librapid::Extent &e, lr_int index) { return e[index]; })
	 	.def("__setitem__", [](librapid::Extent &e, lr_int index, lr_int val) { e[index] = val; })

	 	.def_property_readonly("ndim", &librapid::Extent::ndim)
		.def_property_readonly("containsAutomatic", &librapid::Extent::containsAutomatic)
		.def_property_readonly("size", &librapid::Extent::size)
	 	.def("reorder", [](librapid::Extent &e, const std::vector<size_t> &order) { e.reorder(order); })
	 	.def("fixed", &librapid::Extent::fixed)

	 	.def("__len__", &librapid::Extent::ndim)
	 	.def("__iter__", [](const librapid::Extent &e) { return py::make_iterator(e.begin(), e.end()); }, py::keep_alive<0, 1>())
	 	.def("__eq__", &librapid::Extent::operator==)
	 	.def("__ne__", &librapid::Extent::operator!=)
	 	.def("__str__", &librapid::Extent::str)
	 	.def("__repr__", [](const librapid::Extent &e) { return "<librapid." + e.str() + ">"; });

	 // The librapid Stride object
	 py::class_<librapid::Stride>(module, "Stride")
	 	.def(py::init<>())
	 	.def(py::init<std::vector<lr_int>>())
	 	.def(py::init<lr_int>())
	 	.def(py::init<const librapid::Stride &>())
	 	.def(py::init<py::args>())
	 	.def_static("fromExtent", &librapid::Stride::fromExtent)

	 	.def("__getitem__", [](const librapid::Stride &s, lr_int index) { return s[index]; })
	 	.def("__setitem__", [](librapid::Stride &s, lr_int index, lr_int val) { s[index] = val; })

	 	.def_property_readonly("ndim", &librapid::Stride::ndim)
	 	.def_property_readonly("isTrivial", &librapid::Stride::isTrivial)
	 	.def_property_readonly("isContiguous", &librapid::Stride::isContiguous)
	 	.def("reorder", [](librapid::Stride &s, const std::vector<size_t> &order) { s.reorder(order); })

	 	.def("__len__", &librapid::Stride::ndim)

	 	.def("__iter__", [](const librapid::Stride &s) { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())

	 	.def("__eq__", &librapid::Stride::operator==)
	 	.def("__str__", &librapid::Stride::str)
	 	.def("__repr__", [](const librapid::Stride &s) { return "<librapid." + s.str() + ">"; });

	// The librapid ndarray object
	
	// Colours
	py::module_ color = module.def_submodule("color", "A simple text color library");
	
	py::class_<librapid::color::RGB>(color, "RGB")
		.def(py::init<int, int, int>(), py::arg("red") = 0, py::arg("green") = 0, py::arg("blue") = 0)
		.def("__str__", [](const librapid::color::RGB &col) { return librapid::color::fore(col); })
		.def("__repr__", [](const librapid::color::RGB &col) { return std::string("librapid.color.RGB(red: " + std::to_string(col.red) + ", green: " + std::to_string(col.green) + ", blue: " + std::to_string(col.blue)) + ")"; });

	py::class_<librapid::color::HSL>(color, "HSL")
		.def(py::init<int, int, int>(), py::arg("red") = 0, py::arg("green") = 0, py::arg("blue") = 0)
		.def("__str__", [](const librapid::color::HSL &col) { return librapid::color::fore(col); })
		.def("__repr__", [](const librapid::color::HSL &col) { return std::string("librapid.color.HSL(hue: " + std::to_string(col.hue) + ", saturation: " + std::to_string(col.saturation) + ", lightness: " + std::to_string(col.lightness)) + ")"; });

	color.def("rgbToHsl", &librapid::color::rgbToHsl);
	color.def("hslToRgb", &librapid::color::hslToRgb);
	
	color.def("mergeColors", [](const librapid::color::RGB &colorA, const librapid::color::RGB &colorB) { return librapid::color::mergeColors(colorA, colorB); });
	color.def("mergeColors", [](const librapid::color::RGB &colorA, const librapid::color::HSL &colorB) { return librapid::color::mergeColors(colorA, colorB); });
	color.def("mergeColors", [](const librapid::color::HSL &colorA, const librapid::color::RGB &colorB) { return librapid::color::mergeColors(colorA, colorB); });
	color.def("mergeColors", [](const librapid::color::HSL &colorA, const librapid::color::HSL &colorB) { return librapid::color::mergeColors(colorA, colorB); });

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
	color.attr("brightBlack") = librapid::color::brightBlack;
	color.attr("brightRed") = librapid::color::brightRed;
	color.attr("brightGreen") = librapid::color::brightGreen;
	color.attr("brightYellow") = librapid::color::brightYellow;
	color.attr("brightBlue") = librapid::color::brightBlue;
	color.attr("brightMagenta") = librapid::color::brightMagenta;
	color.attr("brightCyan") = librapid::color::brightCyan;
	color.attr("brightWhite") = librapid::color::brightWhite;

	color.attr("blackBackground") = librapid::color::blackBackground;
	color.attr("redBackground") = librapid::color::redBackground;
	color.attr("greenBackground") = librapid::color::greenBackground;
	color.attr("yellowBackground") = librapid::color::yellowBackground;
	color.attr("blueBackground") = librapid::color::blueBackground;
	color.attr("magentaBackground") = librapid::color::magentaBackground;
	color.attr("cyanBackground") = librapid::color::cyanBackground;
	color.attr("whiteBackground") = librapid::color::whiteBackground;
	color.attr("brightBlackBackground") = librapid::color::brightBlackBackground;
	color.attr("brightRedBackground") = librapid::color::brightRedBackground;
	color.attr("brightGreenBackground") = librapid::color::brightGreenBackground;
	color.attr("brightYellowBackground") = librapid::color::brightYellowBackground;
	color.attr("brightBlueBackground") = librapid::color::brightBlueBackground;
	color.attr("brightMagentaBackground") = librapid::color::brightMagentaBackground;
	color.attr("brightCyanBackground") = librapid::color::brightCyanBackground;
	color.attr("brightWhiteBackground") = librapid::color::brightWhiteBackground;

	color.def("fore", [](const librapid::color::RGB &col) { return librapid::color::fore(col); });
	color.def("fore", [](const librapid::color::HSL &col) { return librapid::color::fore(col); });
	color.def("fore", [](int r, int g, int b) { return librapid::color::fore(r, g, b); });

	color.def("back", [](const librapid::color::RGB &col) { return librapid::color::back(col); });
	color.def("back", [](const librapid::color::HSL &col) { return librapid::color::back(col); });
	color.def("back", [](int r, int g, int b) { return librapid::color::back(r, g, b); });

	// py::implicitly_convertible<long long, librapid::basic_ndarray<python_dtype>>();
	// py::implicitly_convertible<python_dtype, librapid::basic_ndarray<python_dtype>>();
	// py::implicitly_convertible<py::tuple, librapid::basic_ndarray<python_dtype>>();
	// py::implicitly_convertible<py::list, librapid::basic_ndarray<python_dtype>>();
}
