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

void init_extent(py::module &);
void init_vec(py::module &);

void init_ArrayBG(py::module &);
void init_ArrayCG(py::module &);
void init_ArrayF16G(py::module &);
void init_ArrayF32G(py::module &);
void init_ArrayF64G(py::module &);
void init_ArrayI16G(py::module &);
void init_ArrayI32G(py::module &);
void init_ArrayI64G(py::module &);
void init_ArrayB(py::module &);
void init_ArrayC(py::module &);
void init_ArrayF16(py::module &);
void init_ArrayF32(py::module &);
void init_ArrayF64(py::module &);
void init_ArrayI16(py::module &);
void init_ArrayI32(py::module &);
void init_ArrayI64(py::module &);
void init_ArrayMPZ(py::module &);
void init_ArrayMPF(py::module &);
void init_ArrayMPQ(py::module &);

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
    init_ArrayBG(module);
	init_ArrayCG(module);
	init_ArrayF16G(module);
	init_ArrayF32G(module);
	init_ArrayF64G(module);
	init_ArrayI16G(module);
	init_ArrayI32G(module);
	init_ArrayI64G(module);
	init_ArrayB(module);
	init_ArrayC(module);
	init_ArrayF16(module);
	init_ArrayF32(module);
	init_ArrayF64(module);
	init_ArrayI16(module);
	init_ArrayI32(module);
	init_ArrayI64(module);
	init_ArrayMPZ(module);
	init_ArrayMPF(module);
	init_ArrayMPQ(module);

	// Include the Vector library
	#include "autogen/vecInterface.hpp"

	// py::implicitly_convertible<int64_t, librapid::Array>();
	// py::implicitly_convertible<double, librapid::Array>();
	// py::implicitly_convertible<py::tuple, librapid::Array>();
	// py::implicitly_convertible<py::list, librapid::Array>();
}
