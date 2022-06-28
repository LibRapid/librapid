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

PYBIND11_MODULE(_librapid, module) {
	module.doc() = module_docstring;

	// Include the Extent type
	#include <librapid/python/cpp/extentInterface.hpp> //LIBRAPID_SOURCE_DIR "/cpp/extentInterface.hpp"

	// Include the Array library
    #include <librapid/python/cpp/arrayInterface.hpp> // LIBRAPID_SOURCE_DIR "/cpp/arrayInterface.hpp"

	// Include the Vector library
	#include <librapid/python/cpp/vecInterface.hpp> // LIBRAPID_SOURCE_DIR "/cpp/vecInterface.hpp"

	// py::implicitly_convertible<int64_t, librapid::Array>();
	// py::implicitly_convertible<double, librapid::Array>();
	// py::implicitly_convertible<py::tuple, librapid::Array>();
	// py::implicitly_convertible<py::list, librapid::Array>();
}
