#include <librapid/librapid.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <functional>

#include <string>

namespace py = pybind11;

// Docstring for the module
std::string module_docstring = "A fast math and neural network library for Python and C++";

// Make things a little shorter
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

PYBIND11_MODULE(_librapid, module) {
	module.doc() = module_docstring;

	py::module_ test = module.def_submodule("test", "Tests within the librapid library");
	test.def("testLibrapid", &librapid::test::testLibrapid);
	test.def("streamTest", &librapid::test::streamTest);
	test.def("empty", [](int x) { return x * 2; });

	test.def("sum", [](const std::vector<double> &data) { return testSum(data); });
	test.def("sum", [](const std::vector<int64_t> &data) { return testSum(data); });
	test.def("sum", [](const std::vector<char> &data) { return testSum(data); });

	module.def("hasBlas", &librapid::hasBlas);
	module.def("hasCuda", &librapid::hasCuda);
	module.def("setNumThreads", &librapid::setNumThreads);
	module.def("getNumThreads", &librapid::getNumThreads);
	module.def("optimiseThreads", &librapid::optimiseThreads, py::arg("timePerThread") = 2, py::arg("verbose") = false);

	module.def("seconds", [](){ return librapid::seconds(); });
	module.def("sleep", &librapid::sleep);

	module.def("getConsoleSize", []() { auto size = librapid::getConsoleSize(); return py::make_tuple(size.rows, size.cols); });

	module.attr("AUTO") = (int64_t) -1;

	module.attr("EPSILON32") = librapid::EPSILON32;
	module.attr("EPSILON64") = librapid::EPSILON64;
	module.attr("PISQRDIV6") = librapid::PISQRDIV6;
	module.attr("RADTODEG") = librapid::RADTODEG;
	module.attr("DEGTORAD") = librapid::DEGTORAD;
	module.attr("PI") = librapid::PI;
	module.attr("SQRTPI") = librapid::SQRTPI;
	module.attr("TAU") = librapid::TAU;
	module.attr("HALFPI") = librapid::HALFPI;
	module.attr("TWOPI") = librapid::TWOPI;
	module.attr("E") = librapid::E;
	module.attr("SQRTE") = librapid::SQRTE;
	module.attr("SQRT2") = librapid::SQRT2;
	module.attr("SQRT3") = librapid::SQRT3;
	module.attr("SQRT5") = librapid::SQRT5;
	module.attr("GOLDENRATIO") = librapid::GOLDENRATIO;
	module.attr("EULERMASCHERONI") = librapid::EULERMASCHERONI;
	module.attr("TWINPRIMES") = librapid::TWINPRIMES;
	module.attr("LN2") = librapid::LN2;
	module.attr("LN3") = librapid::LN3;
	module.attr("LN5") = librapid::LN5;
	module.attr("ZETA3") = librapid::ZETA3;
	module.attr("CUBEROOT2") = librapid::CUBEROOT2;
	module.attr("CUBEROOT3") = librapid::CUBEROOT3;
	module.attr("LIGHTSPEED") = librapid::LIGHTSPEED;
	module.attr("EARTHGRAVITY") = librapid::EARTHGRAVITY;
	module.attr("WALLISCONST") = librapid::WALLISCONST;
	module.attr("LAPLACELIMIT") = librapid::LAPLACELIMIT;
	module.attr("GAUSSCONST") = librapid::GAUSSCONST;
	module.attr("CAHENSCONST") = librapid::CAHENSCONST;
	module.attr("P2") = librapid::P2;
	module.attr("DOTTIENUMBER") = librapid::DOTTIENUMBER;
	module.attr("MEISSELMERTENS") = librapid::MEISSELMERTENS;
	module.attr("ETOPI") = librapid::ETOPI;
	module.attr("GOLDENANGLE") = librapid::GOLDENANGLE;
	module.attr("MANDELBROTAREA") = librapid::MANDELBROTAREA;
	module.attr("GIESEKINGCONST") = librapid::GIESEKINGCONST;
	module.attr("BLOCHLANDAU") = librapid::BLOCHLANDAU;
	module.attr("GOLOMBDICKMAN") = librapid::GOLOMBDICKMAN;
	module.attr("FELLERTORNIER") = librapid::FELLERTORNIER;
	module.attr("TWOTOSQRT2") = librapid::TWOTOSQRT2;
	module.attr("KHINCHINSCONST") = librapid::KHINCHINSCONST;
	module.attr("MILLSCONST") = librapid::MILLSCONST;
	module.attr("PIOVERLN2") = librapid::PIOVERLN2;
	module.attr("LOCHSCONST") = librapid::LOCHSCONST;
	module.attr("NIVENSCONST") = librapid::PIOVERLN2;
	module.attr("RECIPFIBCONST") = librapid::RECIPFIBCONST;
	module.attr("BACKHOUSECONST") = librapid::BACKHOUSECONST;
	module.attr("MRBCONST") = librapid::MRBCONST;
	module.attr("QUADRECURR") = librapid::QUADRECURR;
	module.attr("PLASTICNUMBER") = librapid::PLASTICNUMBER;

	module.def("product", [](const std::vector<int64_t> &vals) { return librapid::product(vals); }, py::arg("vals"));
	module.def("product", [](const std::vector<double> &vals) { return librapid::product(vals); }, py::arg("vals"));
	module.def("anyBelow", [](const std::vector<int64_t> &vals, int64_t bound) { return librapid::anyBelow(vals, bound); }, py::arg("vals"), py::arg("bound") = 0);
	module.def("min", [](const std::vector<double> &vals) { return librapid::min(vals); }, py::arg("vals"));
	module.def("max", [](const std::vector<double> &vals) { return librapid::max(vals); }, py::arg("vals"));
	module.def("abs", [](double val) { return librapid::abs(val); }, py::arg("val"));
	module.def("abs", [](int64_t val) { return librapid::abs(val); }, py::arg("val"));
	module.def("map", [](double val, double start1, double stop1, double start2, double stop2) { return librapid::map(val, start1, stop1, start2, stop2); }, py::arg("val"), py::arg("start1") = double(0), py::arg("stop1") = double(1), py::arg("start2") = double(0), py::arg("stop2") = double(1));
	module.def("random", [](double min, double max) { return librapid::random(min, max); }, py::arg("min") = 0, py::arg("max") = 1);
	module.def("randint", [](int64_t min, int64_t max) { return librapid::randint(min, max); }, py::arg("min") = 0, py::arg("max") = 1);
	module.def("trueRandomEntropy", []() { return librapid::trueRandomEntropy(); });
	module.def("trueRandom", [](double min, double max) { return librapid::trueRandom(min, max); }, py::arg("min") = 0, py::arg("max") = 1);
	module.def("trueRandint", [](int64_t min, int64_t max) { return librapid::trueRandint(min, max); }, py::arg("min") = 0, py::arg("max") = 1);
	module.def("randomGaussian", []() { return librapid::randomGaussian(); });
	module.def("pow10", &librapid::pow10);
	module.def("round", [](double val, int64_t places) { return librapid::round(val, places); }, py::arg("val"), py::arg("places") = 0);
	module.def("roundSigFig", [](double val, int64_t figs) { return librapid::roundSigFig(val, figs); }, py::arg("val"), py::arg("figs") = 3);

	module.def("nthFibonacci", &librapid::nthFibonacci, py::arg("n"));
	module.def("betterFcknBeEven", &librapid::betterFcknBeEven, py::arg("n"));

	// Create the Vector library
	#include "./python/vec_interface.hpp"

	// Create the Datatype enum
	py::enum_<librapid::Datatype>(module, "Datatype")
	.value("NONE", librapid::Datatype::NONE)
	.value("VALIDNONE", librapid::Datatype::VALIDNONE)
	.value("INT32", librapid::Datatype::INT32)
	.value("INT64", librapid::Datatype::INT64)
	.value("FLOAT32", librapid::Datatype::FLOAT32)
	.value("FLOAT64", librapid::Datatype::FLOAT64)
	.value("CFLOAT32", librapid::Datatype::CFLOAT32)
	.value("CFLOAT64", librapid::Datatype::CFLOAT64);

	// Create the Accelerator enum
	py::enum_<librapid::Accelerator>(module, "Accelerator")
	.value("CPU", librapid::Accelerator::CPU)
	.value("GPU", librapid::Accelerator::GPU);

	module.def("isIntegral", &librapid::isIntegral);
	module.def("isFloating", &librapid::isFloating);
	module.def("datatypeBytes", &librapid::datatypeBytes);
	module.def("datatypeToString", &librapid::datatypeToString);

	// Include the vector library
	#include "python/vec_interface.hpp"

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
		.def_property_readonly("size", [](const librapid::Extent &extent) { return static_cast<std::remove_const<librapid::Extent>::type>(extent).size(); })
		.def("reorder", [](librapid::Extent &e, const std::vector<int64_t> &order) { e.reorder(order); })
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
		.def("reorder", [](librapid::Stride &s, const std::vector<int64_t> &order) { s.reorder(order); })

		.def("__len__", &librapid::Stride::ndim)

		.def("__iter__", [](const librapid::Stride &s) { return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())

		.def("__eq__", &librapid::Stride::operator==)
		.def("__str__", &librapid::Stride::str)
		.def("__repr__", [](const librapid::Stride &s) { return "<librapid." + s.str() + ">"; });

	py::class_<librapid::GPUKernel>(module, "GPUKernel")
		.def(py::init<>())
		.def(py::init<const std::string &>(), py::arg("kernel"))
		.def(py::init<const std::string &, const std::string &>(), py::arg("name"), py::arg("kernel"))

		.def_property_readonly("name", &librapid::GPUKernel::getName)
		.def_property_readonly("kernel", &librapid::GPUKernel::getKernel)

		.def("__str__", [](const librapid::GPUKernel &e) { return e.str(); })
		.def("__repr__", [](const librapid::GPUKernel &e) { return "<librapid.GPUKernel \"\n" + e.str() + "\n\">"; });

	// The librapid ndarray object
	py::class_<librapid::Array>(module, "Array")
		.def(py::init<>())
		.def(py::init<const librapid::Extent &, librapid::Datatype, librapid::Accelerator>(), py::arg("extent"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<const librapid::Extent &, const std::string &, librapid::Accelerator>(), py::arg("extent"), py::arg("dtype") = "float64", py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<const librapid::Extent &, librapid::Datatype, const std::string &>(), py::arg("extent"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = "CPU")
		.def(py::init<const librapid::Extent &, const std::string &, const std::string &>(), py::arg("extent"), py::arg("dtype") = "float64", py::arg("locn") = "CPU")
		.def(py::init<const librapid::Array &>())
		
		.def(py::init<int64_t, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<int64_t>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<int64_t>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<int64_t>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<int64_t>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<int64_t>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<int64_t>>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<int64_t>>>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<int64_t>>>>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<V<int64_t>>>>>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<V<V<int64_t>>>>>>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::INT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<double, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<double>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<double>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<double>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<double>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<double>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<double>>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<double>>>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<double>>>>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<V<double>>>>>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<V<V<double>>>>>>>>>>, librapid::Datatype, librapid::Accelerator>(), py::arg("data"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU)

		.def(py::init<int64_t, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<int64_t>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<int64_t>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<int64_t>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<int64_t>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<int64_t>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<int64_t>>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<int64_t>>>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<int64_t>>>>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<V<int64_t>>>>>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<V<V<int64_t>>>>>>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<double, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<double>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<double>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<double>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<double>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<double>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<double>>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<double>>>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<double>>>>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<V<double>>>>>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)
		.def(py::init<V<V<V<V<V<V<V<V<V<V<double>>>>>>>>>>, const std::string &, librapid::Accelerator>(), py::arg("data"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU)

		.def(py::init<int64_t, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<int64_t>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<int64_t>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<int64_t>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<int64_t>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<int64_t>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<int64_t>>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<int64_t>>>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<int64_t>>>>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<V<int64_t>>>>>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<V<V<int64_t>>>>>>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<double, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<double>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<double>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<double>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<double>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<double>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<double>>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<double>>>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<double>>>>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<V<double>>>>>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<V<V<double>>>>>>>>>>, librapid::Datatype, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))

		.def(py::init<int64_t, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<int64_t>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<int64_t>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<int64_t>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<int64_t>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<int64_t>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<int64_t>>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<int64_t>>>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<int64_t>>>>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<V<int64_t>>>>>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<V<V<int64_t>>>>>>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<double, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<double>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<double>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<double>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<double>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<double>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<double>>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<double>>>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<double>>>>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<V<double>>>>>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))
		.def(py::init<V<V<V<V<V<V<V<V<V<V<double>>>>>>>>>>, const std::string &, const std::string &>(), py::arg("data"), py::arg("dtype"), py::arg("locn"))

		.def("set", [](librapid::Array &arr, const librapid::Array &other) { arr.set(other); }, py::arg("other"))
		.def("set", [](librapid::Array &arr, double other) { arr.set(other); }, py::arg("other"))
		.def("set", [](librapid::Array &arr, int64_t other) { arr.set(other); }, py::arg("other"))
		.def("set", [](librapid::Array &arr, librapid::Complex<double> other) { arr.set(other); }, py::arg("other"))

		.def("isSame", [](const librapid::Array &arr, const librapid::Array &other) { return arr.isSame(other); }, py::arg("other"))

		.def_property_readonly("ndim", &librapid::Array::ndim)
		.def_property_readonly("extent", &librapid::Array::extent)
		.def_property_readonly("stride", &librapid::Array::stride)
		.def_property_readonly("dtype", &librapid::Array::dtype)
		.def_property_readonly("isScalar", &librapid::Array::isScalar)
		.def_property_readonly("location", &librapid::Array::location)
		.def("__len__", [](const librapid::Array &arr) { return arr.len(); })

		.def("__iter__", [](librapid::Array &arr) { arr._increment(); return py::make_iterator(arr.begin(), arr.end()); }) // , py::keep_alive<0, 1>())
		// .def("__iter__", [](librapid::Array &arr) { /*arr._increment();*/ return py::make_iterator(arr.begin(true), arr.end()); }, py::keep_alive<0, 1>())
		// .def("__iter__", [](librapid::Array &arr) { return py::make_iterator(arr); })

		.def("subscript", [](const librapid::Array &arr, int64_t index) { return arr.subscript(index); }, py::arg("index"))
		.def("__getitem__", [](const librapid::Array &arr, int64_t index) { return arr[index]; }, py::arg("index"))
		.def("__setitem__", [](librapid::Array &arr, int64_t index, int64_t val) { arr[index] = val; }, py::arg("index"), py::arg("val"))
		.def("__setitem__", [](librapid::Array &arr, int64_t index, double val) { arr[index] = val; }, py::arg("index"), py::arg("val"))
		.def("__setitem__", [](librapid::Array &arr, int64_t index, const librapid::Array &val) { arr[index] = val; }, py::arg("index"), py::arg("val"))

		.def("fill", [](librapid::Array &arr, int64_t val) { arr.fill(val); })
		.def("filled", [](librapid::Array &arr, int64_t val) { return arr.filled(val); })
		.def("fill", [](librapid::Array &arr, double val) { arr.fill(val); })
		.def("filled", [](librapid::Array &arr, double val) { return arr.filled(val); })
		.def("fillRandom", [](librapid::Array &arr, int64_t min, int64_t max, int64_t seed) { arr.fillRandom(min, max, seed); }, py::arg("min") = 0, py::arg("max") = 1, py::arg("seed") = -1)
		.def("filledRandom", [](librapid::Array &arr, int64_t min, int64_t max, int64_t seed) { return arr.filledRandom(min, max, seed); }, py::arg("min") = 0, py::arg("max") = 1, py::arg("seed") = -1)
		.def("fillRandom", [](librapid::Array &arr, double min, double max, int64_t seed) { arr.fillRandom(min, max, seed); }, py::arg("min") = 0, py::arg("max") = 1, py::arg("seed") = -1)
		.def("filledRandom", [](librapid::Array &arr, double min, double max, int64_t seed) { return arr.filledRandom(min, max, seed); }, py::arg("min") = 0, py::arg("max") = 1, py::arg("seed") = -1)

		.def("clone", [](const librapid::Array &arr, librapid::Datatype dtype, librapid::Accelerator locn) { return arr.clone(dtype, locn); }, py::arg("dtype") = librapid::Datatype::NONE, py::arg("locn") = librapid::Accelerator::NONE)
		.def("clone", [](const librapid::Array &arr, const std::string &dtype, librapid::Accelerator locn) { return arr.clone(dtype, locn); }, py::arg("dtype") = "none", py::arg("locn") = librapid::Accelerator::NONE)
		.def("clone", [](const librapid::Array &arr, librapid::Datatype dtype, const std::string &locn) { return arr.clone(dtype, locn); }, py::arg("dtype") = librapid::Datatype::NONE, py::arg("locn") = "none")
		.def("clone", [](const librapid::Array &arr, const std::string &dtype, const std::string &locn) { return arr.clone(dtype, locn); }, py::arg("dtype") = "none", py::arg("locn") = "none")

		// Include the vector library
		#include "python/map_kernel_interface.hpp"

		.def("__neg__",     [](const librapid::Array &lhs) { return -lhs; })

		.def("__add__",     [](const librapid::Array &lhs, const librapid::Array &rhs) { return lhs + rhs; })
		.def("__sub__",     [](const librapid::Array &lhs, const librapid::Array &rhs) { return lhs - rhs; })
		.def("__mul__",     [](const librapid::Array &lhs, const librapid::Array &rhs) { return lhs * rhs; })
		.def("__truediv__", [](const librapid::Array &lhs, const librapid::Array &rhs) { return lhs / rhs; })

		.def("__add__",     [](const librapid::Array &lhs, int64_t rhs) { return lhs + rhs; })
		.def("__sub__",     [](const librapid::Array &lhs, int64_t rhs) { return lhs - rhs; })
		.def("__mul__",     [](const librapid::Array &lhs, int64_t rhs) { return lhs * rhs; })
		.def("__truediv__", [](const librapid::Array &lhs, int64_t rhs) { return lhs / rhs; })

		.def("__add__",     [](const librapid::Array &lhs, double rhs) { return lhs + rhs; })
		.def("__sub__",     [](const librapid::Array &lhs, double rhs) { return lhs - rhs; })
		.def("__mul__",     [](const librapid::Array &lhs, double rhs) { return lhs * rhs; })
		.def("__truediv__", [](const librapid::Array &lhs, double rhs) { return lhs / rhs; })

		.def("__radd__",     [](const librapid::Array &rhs, int64_t lhs) { return lhs + rhs; })
		.def("__rsub__",     [](const librapid::Array &rhs, int64_t lhs) { return lhs - rhs; })
		.def("__rmul__",     [](const librapid::Array &rhs, int64_t lhs) { return lhs * rhs; })
		.def("__rtruediv__", [](const librapid::Array &rhs, int64_t lhs) { return lhs / rhs; })

		.def("__radd__",     [](const librapid::Array &rhs, double lhs) { return lhs + rhs; })
		.def("__rsub__",     [](const librapid::Array &rhs, double lhs) { return lhs - rhs; })
		.def("__rmul__",     [](const librapid::Array &rhs, double lhs) { return lhs * rhs; })
		.def("__rtruediv__", [](const librapid::Array &rhs, double lhs) { return lhs / rhs; })

		.def("reshape", [](librapid::Array &arr, const librapid::Extent &shape) { arr.reshape(shape); })
		.def("reshape", [](librapid::Array &arr, const std::vector<int64_t> &shape) { arr.reshape(shape); })

		.def("reshaped", [](const librapid::Array &arr, const librapid::Extent &shape) { return arr.reshaped(shape); })
		.def("reshaped", [](const librapid::Array &arr, const std::vector<int64_t> &shape) { return arr.reshaped(shape); })

		.def("dot", [](const librapid::Array &arr, const librapid::Array &other) { return arr.dot(other); }, py::arg("other"))
		.def("transpose", [](librapid::Array &arr, const librapid::Extent &order) { arr.transpose(order); }, py::arg("order") = librapid::Extent())

		.def("__bool__", [](const librapid::Array &arr) { return (bool) arr; })
		.def("__int__", [](const librapid::Array &arr) { return (int64_t) arr; })
		.def("__float__", [](const librapid::Array &arr) { return (double) arr; })
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
			res += "locn=\"" + locnStr + "\">";

			return res;
		})

		.def("refCount", &librapid::Array::refCount);

	module.def("warmup", &librapid::warmup, py::arg("itersCPU") = 10, py::arg("itersGPU") = -1);

	module.def("zerosLike", [](const librapid::Array &arr) { return librapid::zerosLike(arr); }, py::arg("array"));
	module.def("onesLike", [](const librapid::Array &arr) { return librapid::onesLike(arr); }, py::arg("array"));
	module.def("randomLike", [](const librapid::Array &arr, int64_t min, int64_t max, int64_t seed) { return librapid::randomLike(arr, min, max, seed); }, py::arg("array"), py::arg("min") = 0, py::arg("max") = 1, py::arg("seed") = -1);
	module.def("randomLike", [](const librapid::Array &arr, double min, double max, int64_t seed) { return librapid::randomLike(arr, min, max, seed); }, py::arg("array"), py::arg("min") = 0, py::arg("max") = 1, py::arg("seed") = -1);

	module.def("linear", [](double start, double end, int64_t num, const librapid::Datatype &dtype, const librapid::Accelerator &locn) { return librapid::linear(start, end, num, dtype, locn); }, py::arg("start"), py::arg("end"), py::arg("num"), py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU);
	module.def("linear", [](double start, double end, int64_t num, const std::string &dtype, const librapid::Accelerator &locn) { return librapid::linear(start, end, num, dtype, locn); }, py::arg("start"), py::arg("end"), py::arg("num"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU);
	module.def("linear", [](double start, double end, int64_t num, const librapid::Datatype &dtype, const std::string &locn) { return librapid::linear(start, end, num, dtype, locn); }, py::arg("start"), py::arg("end"), py::arg("num"), py::arg("dtype"), py::arg("locn") = "CPU");
	module.def("linear", [](double start, double end, int64_t num, const std::string &dtype, const std::string &locn) { return librapid::linear(start, end, num, dtype, locn); }, py::arg("start"), py::arg("end"), py::arg("num"), py::arg("dtype"), py::arg("locn"));

	module.def("range", [](double start, double end, double inc, const librapid::Datatype &dtype, const librapid::Accelerator &locn) { return librapid::range(start, end, inc, dtype, locn); }, py::arg("start") = 0, py::arg("end") = std::numeric_limits<double>::infinity(), py::arg("inc") = 1, py::arg("dtype") = librapid::Datatype::FLOAT64, py::arg("locn") = librapid::Accelerator::CPU);
	module.def("range", [](double start, double end, double inc, const std::string &dtype, const librapid::Accelerator &locn) { return librapid::range(start, end, inc, dtype, locn); }, py::arg("start"), py::arg("end"), py::arg("inc"), py::arg("dtype"), py::arg("locn") = librapid::Accelerator::CPU);
	module.def("range", [](double start, double end, double inc, const librapid::Datatype &dtype, const std::string &locn) { return librapid::range(start, end, inc, dtype, locn); }, py::arg("start"), py::arg("end"), py::arg("inc"), py::arg("dtype"), py::arg("locn") = "CPU");
	module.def("range", [](double start, double end, double inc, const std::string &dtype, const std::string &locn) { return librapid::range(start, end, inc, dtype, locn); }, py::arg("start"), py::arg("end"), py::arg("inc"), py::arg("dtype"), py::arg("locn"));

	module.def("negate", [](const librapid::Array &a, librapid::Array &res) { librapid::negate(a, res); }, py::arg("a"), py::arg("res"));
	module.def("add", [](const librapid::Array &a, const librapid::Array &b, librapid::Array &res) { librapid::add(a, b, res); }, py::arg("a"), py::arg("b"), py::arg("res"));
	module.def("sub", [](const librapid::Array &a, const librapid::Array &b, librapid::Array &res) { librapid::sub(a, b, res); }, py::arg("a"), py::arg("b"), py::arg("res"));
	module.def("mul", [](const librapid::Array &a, const librapid::Array &b, librapid::Array &res) { librapid::mul(a, b, res); }, py::arg("a"), py::arg("b"), py::arg("res"));
	module.def("div", [](const librapid::Array &a, const librapid::Array &b, librapid::Array &res) { librapid::div(a, b, res); }, py::arg("a"), py::arg("b"), py::arg("res"));

	module.def("negate", [](const librapid::Array &a) { return librapid::negate(a); }, py::arg("a"));
	module.def("add", [](const librapid::Array &a, const librapid::Array &b) { return librapid::add(a, b); }, py::arg("a"), py::arg("b"));
	module.def("sub", [](const librapid::Array &a, const librapid::Array &b) { return librapid::sub(a, b); }, py::arg("a"), py::arg("b"));
	module.def("mul", [](const librapid::Array &a, const librapid::Array &b) { return librapid::mul(a, b); }, py::arg("a"), py::arg("b"));
	module.def("div", [](const librapid::Array &a, const librapid::Array &b) { return librapid::div(a, b); }, py::arg("a"), py::arg("b"));

	module.def("concatenate", [](const std::vector<librapid::Array> &arrays, int64_t axis) { return librapid::concatenate(arrays, axis); }, py::arg("arrays"), py::arg("axis") = 0);
	module.def("stack", [](const std::vector<librapid::Array> &arrays, int64_t axis) { return librapid::stack(arrays, axis); }, py::arg("arrays"), py::arg("axis") = 0);

	module.def("dot", [](const librapid::Array &lhs, const librapid::Array &rhs, librapid::Array &res) { librapid::dot(lhs, rhs, res); }, py::arg("lhs"), py::arg("rhs"), py::arg("res"));
	module.def("dot", [](const librapid::Array &lhs, const librapid::Array &rhs) { return librapid::dot(lhs, rhs); }, py::arg("lhs"), py::arg("rhs"));

	// Colours
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

	// py::implicitly_convertible<py::tuple, librapid::Extent>();
	// py::implicitly_convertible<py::list, librapid::Extent>();
}