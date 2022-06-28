#define LIBRAPID_ASSERT

#include <librapid/librapid.hpp>

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

namespace lrc = librapid;

PYBIND11_MODULE(_librapid, module) {
	module.doc() = module_docstring;

	module.def("test", [](double n) {
		lrc::Array<float> myArray1(lrc::Extent(1000, 1000));
		lrc::Array<float> myArray2(lrc::Extent(1000, 1000));
		lrc::Array<float> myArray3(lrc::Extent(1000, 1000));

		lrc::timeFunction([&]() { auto res = myArray1 + myArray2; }, -1, -1, n);
		lrc::timeFunction([&]() { myArray3 = myArray1 + myArray2; }, -1, -1, n);
	});

	// Include the Extent type
	#include "autogen/extentInterface.hpp"

	// Include the Array library
    #include "autogen/arrayInterface.hpp"

	// Include the Vector library
	#include "autogen/vecInterface.hpp"

	// py::implicitly_convertible<int64_t, librapid::Array>();
	// py::implicitly_convertible<double, librapid::Array>();
	// py::implicitly_convertible<py::tuple, librapid::Array>();
	// py::implicitly_convertible<py::list, librapid::Array>();
}
