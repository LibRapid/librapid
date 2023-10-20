#define LIBRAPID_ASSERT

#include <librapid/librapid.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;
namespace lrc = librapid;

// Docstring for the module
std::string moduleDocstring = "A highly-optimised Python library for numeric calculations";

PYBIND11_MODULE(_librapid, module) {
	module.doc() = moduleDocstring;

	module.def("test", [](uint64_t n) {
		if (n & 1) {
			return 3 * n + 1;
		} else {
			return n / 2;
		}
	});
}
