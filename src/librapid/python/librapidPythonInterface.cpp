#define LIBRAPID_ASSERT

#include "../librapid.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <functional>

#include <string>

namespace py = pybind11;

// Docstring for the module
std::string module_docstring = "A highly-optimized Array library for Python";

// Just remove these. They're pointless
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

PYBIND11_MODULE(_librapid, module) {
	module.doc() = module_docstring;

	py::class_<librapid::Extent>(module, "Extent")
		.def(py::init<>())
		.def(py::init<const std::vector<int64_t> &>())
		.def(py::init<const librapid::Extent &>())

		.def_static("zero", &librapid::Extent::zero)
		.def("stride", &librapid::Extent::stride)
		.def("strideAdjusted", &librapid::Extent::strideAdjusted)
		.def("index", &librapid::Extent::index)
		.def("indexAdjusted", &librapid::Extent::indexAdjusted)
		.def("reverseIndex", &librapid::Extent::reverseIndex)
		.def("reverseIndexAdjusted", &librapid::Extent::reverseIndexAdjusted)
		.def("partial", &librapid::Extent::partial)
		.def("swivel", &librapid::Extent::swivel<librapid::Extent::Type, librapid::Extent::MaxDims, librapid::Extent::Align>)
		.def("swivelInplace", &librapid::Extent::swivelInplace<librapid::Extent::Type, librapid::Extent::MaxDims, librapid::Extent::Align>)
		.def("size", &librapid::Extent::size)
		.def("sizeAdjusted", &librapid::Extent::sizeAdjusted)
		.def("dims", &librapid::Extent::dims)

		.def("__getitem__", [](const librapid::Extent &e, int64_t index) { return e[index]; })
		.def("__setitem__", [](librapid::Extent &e, int64_t index, int64_t val) { e[index] = val; })

		.def("adjusted", &librapid::Extent::adjusted)

		.def("__eq__", &librapid::Extent::operator==<librapid::Extent::Type, librapid::Extent::MaxDims, librapid::Extent::Align>)

		.def("str", &librapid::Extent::str)
		.def("__str__", &librapid::Extent::str)
		.def("__repr__", [](const librapid::Extent &e) { return "<librapid." + e.str() + ">"; });

	// Include the Array library
    #include "cpp/arrayInterface.hpp"

	// Include the Vector library
	#include "cpp/vecInterface.hpp"

	// py::implicitly_convertible<int64_t, librapid::Array>();
	// py::implicitly_convertible<double, librapid::Array>();
	// py::implicitly_convertible<py::tuple, librapid::Array>();
	// py::implicitly_convertible<py::list, librapid::Array>();
}
