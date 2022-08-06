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
void init_ArrayMPQ(py::module &);
void init_ArrayMPFR(py::module &);
void init_ArrayCF32(py::module &);
void init_ArrayCF64(py::module &);
void init_ArrayCMPFR(py::module &);

void init_math(py::module &);

void init_mpfr(py::module &);

void init_complex(py::module &);

void init_Vec2i(py::module &);
void init_Vec2f(py::module &);
void init_Vec2d(py::module &);
void init_Vec3i(py::module &);
void init_Vec3f(py::module &);
void init_Vec3d(py::module &);
void init_Vec4i(py::module &);
void init_Vec4f(py::module &);
void init_Vec4d(py::module &);

PYBIND11_MODULE(_librapid, module) {
	module.doc() = module_docstring;

	module.def("test", [](double n) {
		lrc::Array<float> myArray1(lrc::Extent(1000, 1000));
		lrc::Array<float> myArray2(lrc::Extent(1000, 1000));
		lrc::Array<float> myArray3(lrc::Extent(1000, 1000));

		lrc::timeFunction([&]() { auto res = myArray1 + myArray2; }, -1, -1, n);
		lrc::timeFunction([&]() { myArray3 = myArray1 + myArray2; }, -1, -1, n);
	});

	module.def("prec", [](int64_t n) {
		lrc::prec(n);
	});

	module.def("constPi", &lrc::constPi);
	module.def("constEuler", &lrc::constEuler);
	module.def("constLog2", &lrc::constLog2);
	module.def("constCatalan", &lrc::constCatalan);

	// Include the Extent type
	#include "autogen/extentInterface.hpp"

	// MPFR support
	init_mpfr(module);

	// Math module
	init_math(module);

	// Complex numbers
	init_complex(module);

	// Vector interface
	init_Vec2i(module);
	init_Vec2f(module);
	init_Vec2d(module);
	init_Vec3i(module);
	init_Vec3f(module);
	init_Vec3d(module);
	init_Vec4i(module);
	init_Vec4f(module);
	init_Vec4d(module);

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
	init_ArrayMPQ(module);
	init_ArrayMPFR(module);
	init_ArrayCF32(module);
	init_ArrayCF64(module);
	init_ArrayCMPFR(module);

	py::implicitly_convertible<int64_t, librapid::mpz>();
	py::implicitly_convertible<const std::string &, librapid::mpz>();
	py::implicitly_convertible<int64_t, librapid::mpfr>();
	py::implicitly_convertible<double, librapid::mpfr>();
	py::implicitly_convertible<const std::string &, librapid::mpfr>();

	py::implicitly_convertible<int64_t, librapid::Complex<float>>();
	py::implicitly_convertible<double, librapid::Complex<float>>();
	py::implicitly_convertible<int64_t, librapid::Complex<double>>();
	py::implicitly_convertible<double, librapid::Complex<double>>();
	py::implicitly_convertible<int64_t, librapid::Complex<librapid::mpfr>>();
	py::implicitly_convertible<double, librapid::Complex<librapid::mpfr>>();
	py::implicitly_convertible<const std::string &, librapid::Complex<librapid::mpfr>>();

	// Allow implicit casting between Complex types
	py::implicitly_convertible<librapid::Complex<float>, librapid::Complex<double>>();
	py::implicitly_convertible<librapid::Complex<float>, librapid::Complex<librapid::mpfr>>();
	py::implicitly_convertible<librapid::Complex<double>, librapid::Complex<float>>();
	py::implicitly_convertible<librapid::Complex<double>, librapid::Complex<librapid::mpfr>>();
	py::implicitly_convertible<librapid::Complex<librapid::mpfr>, librapid::Complex<float>>();
	py::implicitly_convertible<librapid::Complex<librapid::mpfr>, librapid::Complex<double>>();

	// py::implicitly_convertible<int64_t, librapid::Array>();
	// py::implicitly_convertible<double, librapid::Array>();
	// py::implicitly_convertible<py::tuple, librapid::Array>();
	// py::implicitly_convertible<py::list, librapid::Array>();
}
