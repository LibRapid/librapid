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

template<typename T>
T testSum(const std::vector<T> &data)
{
	T total = 0;
	for (const auto &val : data)
		total += val;
	return total;
}

// Just remove these. They're pointless
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

PYBIND11_MODULE(_librapid, module)
{
	module.doc() = module_docstring;

	py::module_ test = module.def_submodule("test", "Tests within the librapid library");
	test.def("testLibrapid", &librapid::test::testLibrapid);
	test.def("streamTest", &librapid::test::streamTest);

	test.def("sum", [](const std::vector<double> &data) { return testSum(data); });
	test.def("sum", [](const std::vector<lr_int> &data) { return testSum(data); });
	test.def("sum", [](const std::vector<char> &data) { return testSum(data); });
	test.def("sum", [](const std::vector<bool> &data) { return testSum(data); });

	module.def("hasBlas", &librapid::hasBlas);
	module.def("setBlasThreads", &librapid::setBlasThreads);
	module.def("getBlasThreads", &librapid::getBlasThreads);	
	module.def("setNumThreads", &librapid::setNumThreads);
	module.def("getNumThreads", &librapid::getNumThreads);

	module.def("seconds", [](){ return librapid::seconds(); });
	module.def("sleep", &librapid::sleep);

	module.def("getConsoleSize", []() { auto size = librapid::getConsoleSize(); return py::make_tuple(size.rows, size.cols); });

	module.attr("AUTO") = (lr_int) -1;

	module.attr("pi") = librapid::pi;
	module.attr("twopi") = librapid::twopi;
	module.attr("halfpi") = librapid::halfpi;
	module.attr("e") = librapid::e;
	module.attr("sqrt2") = librapid::sqrt2;
	module.attr("sqrt3") = librapid::sqrt3;
	module.attr("sqrt5") = librapid::sqrt5;

	module.def("product", [](const std::vector<lr_int> &vals) { return librapid::product(vals); }, py::arg("vals"));
	module.def("product", [](const std::vector<double> &vals) { return librapid::product(vals); }, py::arg("vals"));

	module.def("min", [](const std::vector<double> &vals) { return librapid::min(vals); }, py::arg("vals"));
	module.def("max", [](const std::vector<double> &vals) { return librapid::max(vals); }, py::arg("vals"));

	module.def("map", [](double val, double start1, double stop1, double start2, double stop2) { return librapid::map(val, start1, stop1, start2, stop2); }, py::arg("val"), py::arg("start1") = double(0), py::arg("stop1") = double(1), py::arg("start2") = double(0), py::arg("stop2") = double(1));
	module.def("random", [](double min, double max) { return librapid::random(min, max); }, py::arg("min") = 0, py::arg("max") = 1);
	module.def("randint", [](lr_int min, lr_int max) { return librapid::randint(min, max); }, py::arg("min") = 0, py::arg("max") = 1);
	module.def("pow10", &librapid::pow10);
	module.def("round", [](double val, lr_int places) { return librapid::round(val, places); }, py::arg("val"), py::arg("places") = 0);
	module.def("roundSigFig", [](double val, lr_int figs) { return librapid::roundSigFig(val, figs); }, py::arg("val"), py::arg("figs") = 3);

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
	
	py::class_<librapid::RGB>(color, "RGB")
		.def(py::init<int, int, int>(), py::arg("red") = 0, py::arg("green") = 0, py::arg("blue") = 0)
		.def("__str__", [](const librapid::RGB &col) { return librapid::fore(col); })
		.def("__repr__", [](const librapid::RGB &col) { return std::string("librapid.color.RGB(red: " + std::to_string(col.red) + ", green: " + std::to_string(col.green) + ", blue: " + std::to_string(col.blue)) + ")"; });

	py::class_<librapid::HSL>(color, "HSL")
		.def(py::init<int, int, int>(), py::arg("red") = 0, py::arg("green") = 0, py::arg("blue") = 0)
		.def("__str__", [](const librapid::HSL &col) { return librapid::fore(col); })
		.def("__repr__", [](const librapid::HSL &col) { return std::string("librapid.color.HSL(hue: " + std::to_string(col.hue) + ", saturation: " + std::to_string(col.saturation) + ", lightness: " + std::to_string(col.lightness)) + ")"; });

	color.def("rgbToHsl", &librapid::rgbToHsl);
	color.def("hslToRgb", &librapid::hslToRgb);
	
	color.def("mergeColors", [](const librapid::RGB &colorA, const librapid::RGB &colorB) { return librapid::mergeColors(colorA, colorB); });
	color.def("mergeColors", [](const librapid::RGB &colorA, const librapid::HSL &colorB) { return librapid::mergeColors(colorA, colorB); });
	color.def("mergeColors", [](const librapid::HSL &colorA, const librapid::RGB &colorB) { return librapid::mergeColors(colorA, colorB); });
	color.def("mergeColors", [](const librapid::HSL &colorA, const librapid::HSL &colorB) { return librapid::mergeColors(colorA, colorB); });

	color.attr("clear") = librapid::clear;
	color.attr("bold") = librapid::bold;
	color.attr("blink") = librapid::blink;

	color.attr("black") = librapid::black;
	color.attr("red") = librapid::red;
	color.attr("green") = librapid::green;
	color.attr("yellow") = librapid::yellow;
	color.attr("blue") = librapid::blue;
	color.attr("magenta") = librapid::magenta;
	color.attr("cyan") = librapid::cyan;
	color.attr("white") = librapid::white;
	color.attr("brightBlack") = librapid::brightBlack;
	color.attr("brightRed") = librapid::brightRed;
	color.attr("brightGreen") = librapid::brightGreen;
	color.attr("brightYellow") = librapid::brightYellow;
	color.attr("brightBlue") = librapid::brightBlue;
	color.attr("brightMagenta") = librapid::brightMagenta;
	color.attr("brightCyan") = librapid::brightCyan;
	color.attr("brightWhite") = librapid::brightWhite;

	color.attr("blackBackground") = librapid::blackBackground;
	color.attr("redBackground") = librapid::redBackground;
	color.attr("greenBackground") = librapid::greenBackground;
	color.attr("yellowBackground") = librapid::yellowBackground;
	color.attr("blueBackground") = librapid::blueBackground;
	color.attr("magentaBackground") = librapid::magentaBackground;
	color.attr("cyanBackground") = librapid::cyanBackground;
	color.attr("whiteBackground") = librapid::whiteBackground;
	color.attr("brightBlackBackground") = librapid::brightBlackBackground;
	color.attr("brightRedBackground") = librapid::brightRedBackground;
	color.attr("brightGreenBackground") = librapid::brightGreenBackground;
	color.attr("brightYellowBackground") = librapid::brightYellowBackground;
	color.attr("brightBlueBackground") = librapid::brightBlueBackground;
	color.attr("brightMagentaBackground") = librapid::brightMagentaBackground;
	color.attr("brightCyanBackground") = librapid::brightCyanBackground;
	color.attr("brightWhiteBackground") = librapid::brightWhiteBackground;

	color.def("fore", [](const librapid::RGB &col) { return librapid::fore(col); });
	color.def("fore", [](const librapid::HSL &col) { return librapid::fore(col); });
	color.def("fore", [](int r, int g, int b) { return librapid::fore(r, g, b); });

	color.def("back", [](const librapid::RGB &col) { return librapid::back(col); });
	color.def("back", [](const librapid::HSL &col) { return librapid::back(col); });
	color.def("back", [](int r, int g, int b) { return librapid::back(r, g, b); });

	// py::implicitly_convertible<long long, librapid::basic_ndarray<double>>();
	// py::implicitly_convertible<double, librapid::basic_ndarray<double>>();
	// py::implicitly_convertible<py::tuple, librapid::basic_ndarray<double>>();
	// py::implicitly_convertible<py::list, librapid::basic_ndarray<double>>();
}
