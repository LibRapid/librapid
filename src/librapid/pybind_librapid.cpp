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
	test.def("sum", [](const std::vector<int64_t> &data) { return testSum(data); });
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

	module.attr("AUTO") = (int64_t) -1;

	module.attr("pi") = librapid::pi;
	module.attr("twopi") = librapid::twopi;
	module.attr("halfpi") = librapid::halfpi;
	module.attr("e") = librapid::e;
	module.attr("sqrt2") = librapid::sqrt2;
	module.attr("sqrt3") = librapid::sqrt3;
	module.attr("sqrt5") = librapid::sqrt5;

	module.def("product", [](const std::vector<int64_t> &vals) { return librapid::product(vals); }, py::arg("vals"));
	module.def("product", [](const std::vector<double> &vals) { return librapid::product(vals); }, py::arg("vals"));

	module.def("min", [](const std::vector<double> &vals) { return librapid::min(vals); }, py::arg("vals"));
	module.def("max", [](const std::vector<double> &vals) { return librapid::max(vals); }, py::arg("vals"));

	module.def("map", [](double val, double start1, double stop1, double start2, double stop2) { return librapid::map(val, start1, stop1, start2, stop2); }, py::arg("val"), py::arg("start1") = double(0), py::arg("stop1") = double(1), py::arg("start2") = double(0), py::arg("stop2") = double(1));
	module.def("random", [](double min, double max) { return librapid::random(min, max); }, py::arg("min") = 0, py::arg("max") = 1);
	module.def("randint", [](int64_t min, int64_t max) { return librapid::randint(min, max); }, py::arg("min") = 0, py::arg("max") = 1);
	module.def("pow10", &librapid::pow10);
	module.def("round", [](double val, int64_t places) { return librapid::round(val, places); }, py::arg("val"), py::arg("places") = 0);
	module.def("roundSigFig", [](double val, int64_t figs) { return librapid::roundSigFig(val, figs); }, py::arg("val"), py::arg("figs") = 3);

	// Create the Datatype enum
	py::enum_<librapid::Datatype>(module, "Datatype")
		.value("NONE", librapid::Datatype::NONE)
		.value("VALIDNONE", librapid::Datatype::VALIDNONE)
		.value("BOOL", librapid::Datatype::BOOL)
		.value("INT64", librapid::Datatype::INT64)
		.value("FLOAT32", librapid::Datatype::FLOAT32)
		.value("FLOAT64", librapid::Datatype::FLOAT64);

	// Create the Accelerator enum
	py::enum_<librapid::Accelerator>(module, "Accelerator")
		.value("CPU", librapid::Accelerator::CPU)
		.value("GPU", librapid::Accelerator::GPU);

	module.def("isIntegral", &librapid::isIntegral);
	module.def("isUnsigned", &librapid::isUnsigned);
	module.def("isFloating", &librapid::isFloating);
	module.def("datatypeBytes", &librapid::datatypeBytes);
	module.def("datatypeToString", &librapid::datatypeToString);

	// The librapid Extent object
	py::class_<librapid::Extent>(module, "Extent")
		.def(py::init<>())
		.def(py::init<const std::vector<int64_t> &>())
		.def(py::init<const librapid::Extent &>())
		.def(py::init<py::args>())

		.def("__getitem__", [](const librapid::Extent &e, int64_t index) { return e[index]; })
		.def("__setitem__", [](librapid::Extent &e, int64_t index, int64_t val) { e[index] = val; })

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
		.def(py::init<std::vector<int64_t>>())
		.def(py::init<int64_t>())
		.def(py::init<const librapid::Stride &>())
		.def(py::init<py::args>())
		.def_static("fromExtent", &librapid::Stride::fromExtent)

		.def("__getitem__", [](const librapid::Stride &s, int64_t index) { return s[index]; })
		.def("__setitem__", [](librapid::Stride &s, int64_t index, int64_t val) { s[index] = val; })

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
	py::class_<librapid::Array>(module, "Array")
		.def(py::init<>())
		.def(py::init<const librapid::Extent &, librapid::Datatype, librapid::Accelerator>(), py::arg("extent"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("location") = librapid::Accelerator::CPU)
		.def(py::init<const librapid::Extent &, const std::string &, librapid::Accelerator>(), py::arg("extent"), py::arg("dtype") = "float64", py::arg("location") = librapid::Accelerator::CPU)
		.def(py::init<const librapid::Extent &, librapid::Datatype, const std::string &>(), py::arg("extent"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("location") = "CPU")
		.def(py::init<const librapid::Extent &, const std::string &, const std::string &>(), py::arg("extent"), py::arg("dtype") = "float64", py::arg("location") = "CPU")
		.def(py::init<const librapid::Array &>())
		.def(py::init<int64_t>())
		.def(py::init<double>())

		.def_property_readonly("ndim", &librapid::Array::ndim)
		.def_property_readonly("extent", &librapid::Array::extent)
		.def_property_readonly("stride", &librapid::Array::stride)
		.def_property_readonly("dtype", &librapid::Array::dtype)
		.def_property_readonly("location", &librapid::Array::location)
		.def("__len__", &librapid::Array::ndim)

		.def("__getitem__", [](const librapid::Array &arr, int64_t index) { return arr[index]; })
		.def("__setitem__", [](librapid::Array &arr, int64_t index, int64_t val) { arr[index] = val; })
		.def("__setitem__", [](librapid::Array &arr, int64_t index, double val) { arr[index] = val; })
		.def("__setitem__", [](librapid::Array &arr, int64_t index, const librapid::Array &val) { arr[index] = val; })

		.def("fill", [](librapid::Array &arr, double val) { arr.fill(val); })

		.def("__add__", &librapid::Array::operator+)
		.def("__sub__", &librapid::Array::operator-)
		.def("__mul__", &librapid::Array::operator*)
		.def("__div__", &librapid::Array::operator/)

		.def("transpose", [](librapid::Array &arr, const librapid::Extent &order) { arr.transpose(order); }, py::arg("order") = librapid::Extent())

		.def("str", [](const librapid::Array &arr, uint64_t indent, bool showCommas) { return arr.str(indent, showCommas); }, py::arg("indent") = 0, py::arg("showCommas") = false)
		.def("__str__", [](const librapid::Array &arr) { return arr.str(); })
		.def("__repr__", [](const librapid::Array &arr) {
			int64_t rows, cols;

			std::string locnStr = (arr.location() == librapid::Accelerator::CPU ? "CPU" : "GPU");

			std::string res = "<librapid.Array ";
			res += arr.str(16, true, rows, cols) + "\n\n";
			res += std::string(16, ' ');
			res += "dtype=\"" + librapid::datatypeToString(arr.dtype()) + "\"\n";
			res += std::string(16, ' ');
			res += "location=\"" + locnStr + "\">";

			return res;
		});

	module.def("add", [](const librapid::Array &a, const librapid::Array &b, librapid::Array &res) { librapid::add(a, b, res); }, py::arg("a"), py::arg("b"), py::arg("res"));
	module.def("sub", [](const librapid::Array &a, const librapid::Array &b, librapid::Array &res) { librapid::sub(a, b, res); }, py::arg("a"), py::arg("b"), py::arg("res"));
	module.def("mul", [](const librapid::Array &a, const librapid::Array &b, librapid::Array &res) { librapid::mul(a, b, res); }, py::arg("a"), py::arg("b"), py::arg("res"));
	module.def("div", [](const librapid::Array &a, const librapid::Array &b, librapid::Array &res) { librapid::div(a, b, res); }, py::arg("a"), py::arg("b"), py::arg("res"));

	module.def("add", [](const librapid::Array &a, const librapid::Array &b) { return librapid::add(a, b); }, py::arg("a"), py::arg("b"));
	module.def("sub", [](const librapid::Array &a, const librapid::Array &b) { return librapid::sub(a, b); }, py::arg("a"), py::arg("b"));
	module.def("mul", [](const librapid::Array &a, const librapid::Array &b) { return librapid::mul(a, b); }, py::arg("a"), py::arg("b"));
	module.def("div", [](const librapid::Array &a, const librapid::Array &b) { return librapid::div(a, b); }, py::arg("a"), py::arg("b"));
	
	// Colours
	// py::module_ color = module.def_submodule("color", "A simple text color library");
	
	py::class_<librapid::RGB>(module, "RGB")
		.def(py::init<int, int, int>(), py::arg("red") = 0, py::arg("green") = 0, py::arg("blue") = 0)
		.def("__str__", [](const librapid::RGB &col) { return librapid::fore(col); })
		.def("__repr__", [](const librapid::RGB &col) { return std::string("librapid.RGB(red: " + std::to_string(col.red) + ", green: " + std::to_string(col.green) + ", blue: " + std::to_string(col.blue)) + ")"; });

	py::class_<librapid::HSL>(module, "HSL")
		.def(py::init<int, int, int>(), py::arg("red") = 0, py::arg("green") = 0, py::arg("blue") = 0)
		.def("__str__", [](const librapid::HSL &col) { return librapid::fore(col); })
		.def("__repr__", [](const librapid::HSL &col) { return std::string("librapid.HSL(hue: " + std::to_string(col.hue) + ", saturation: " + std::to_string(col.saturation) + ", lightness: " + std::to_string(col.lightness)) + ")"; });

	module.def("rgbToHsl", &librapid::rgbToHsl);
	module.def("hslToRgb", &librapid::hslToRgb);
	
	module.def("mergeColors", [](const librapid::RGB &colorA, const librapid::RGB &colorB) { return librapid::mergeColors(colorA, colorB); });
	module.def("mergeColors", [](const librapid::RGB &colorA, const librapid::HSL &colorB) { return librapid::mergeColors(colorA, colorB); });
	module.def("mergeColors", [](const librapid::HSL &colorA, const librapid::RGB &colorB) { return librapid::mergeColors(colorA, colorB); });
	module.def("mergeColors", [](const librapid::HSL &colorA, const librapid::HSL &colorB) { return librapid::mergeColors(colorA, colorB); });

	module.attr("clear") = librapid::clear;
	module.attr("bold") = librapid::bold;
	module.attr("blink") = librapid::blink;

	module.attr("black") = librapid::black;
	module.attr("red") = librapid::red;
	module.attr("green") = librapid::green;
	module.attr("yellow") = librapid::yellow;
	module.attr("blue") = librapid::blue;
	module.attr("magenta") = librapid::magenta;
	module.attr("cyan") = librapid::cyan;
	module.attr("white") = librapid::white;
	module.attr("brightBlack") = librapid::brightBlack;
	module.attr("brightRed") = librapid::brightRed;
	module.attr("brightGreen") = librapid::brightGreen;
	module.attr("brightYellow") = librapid::brightYellow;
	module.attr("brightBlue") = librapid::brightBlue;
	module.attr("brightMagenta") = librapid::brightMagenta;
	module.attr("brightCyan") = librapid::brightCyan;
	module.attr("brightWhite") = librapid::brightWhite;

	module.attr("blackBackground") = librapid::blackBackground;
	module.attr("redBackground") = librapid::redBackground;
	module.attr("greenBackground") = librapid::greenBackground;
	module.attr("yellowBackground") = librapid::yellowBackground;
	module.attr("blueBackground") = librapid::blueBackground;
	module.attr("magentaBackground") = librapid::magentaBackground;
	module.attr("cyanBackground") = librapid::cyanBackground;
	module.attr("whiteBackground") = librapid::whiteBackground;
	module.attr("brightBlackBackground") = librapid::brightBlackBackground;
	module.attr("brightRedBackground") = librapid::brightRedBackground;
	module.attr("brightGreenBackground") = librapid::brightGreenBackground;
	module.attr("brightYellowBackground") = librapid::brightYellowBackground;
	module.attr("brightBlueBackground") = librapid::brightBlueBackground;
	module.attr("brightMagentaBackground") = librapid::brightMagentaBackground;
	module.attr("brightCyanBackground") = librapid::brightCyanBackground;
	module.attr("brightWhiteBackground") = librapid::brightWhiteBackground;

	module.def("fore", [](const librapid::RGB &col) { return librapid::fore(col); });
	module.def("fore", [](const librapid::HSL &col) { return librapid::fore(col); });
	module.def("fore", [](int r, int g, int b) { return librapid::fore(r, g, b); });

	module.def("back", [](const librapid::RGB &col) { return librapid::back(col); });
	module.def("back", [](const librapid::HSL &col) { return librapid::back(col); });
	module.def("back", [](int r, int g, int b) { return librapid::back(r, g, b); });

	py::implicitly_convertible<int64_t, librapid::Array>();
	py::implicitly_convertible<double, librapid::Array>();
	py::implicitly_convertible<py::tuple, librapid::Array>();
	py::implicitly_convertible<py::list, librapid::Array>();

	py::implicitly_convertible<py::tuple, librapid::Extent>();
	py::implicitly_convertible<py::list, librapid::Extent>();
}
