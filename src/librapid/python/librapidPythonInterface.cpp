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

        py::class_<librapid::ArrayB>(module, "ArrayB")
                .def(py::init<>())
                .def(py::init<librapid::Extent>())
                .def(py::init<librapid::ArrayB>())
                .def(py::init<librapid::internal::traits<librapid::ArrayB>::Scalar>())
                .def("copy", &librapid::ArrayB::copy)
                .def("__getitem__", [](const librapid::ArrayB &arr, int64_t index) { return arr[index]; })
                .def("__setitem__", [](const librapid::ArrayB &arr, int64_t index, librapid::internal::traits<librapid::ArrayB>::Scalar val) { arr[index] = val; })
                .def("__setitem__", [](const librapid::ArrayB &arr, int64_t index, const librapid::ArrayB &val) { arr[index] = val; })
                .def("move_CPU", &librapid::ArrayB::move<librapid::device::CPU>)
                .def("move_GPU", &librapid::ArrayB::move<librapid::device::GPU>)
                .def("__or__", &librapid::ArrayB::operator|<librapid::ArrayB>)
                .def("__and__", &librapid::ArrayB::operator&<librapid::ArrayB>)
                .def("__xor__", &librapid::ArrayB::operator^<librapid::ArrayB>)
                .def("__invert__", &librapid::ArrayB::operator~)
                .def("transpose", &librapid::ArrayB::transpose)
                .def("transposed", &librapid::ArrayB::transposed)
                .def("dot", &librapid::ArrayB::dot<librapid::ArrayB>)
                .def("str", &librapid::ArrayB::str)
                .def("__str__", &librapid::ArrayB::str)
                .def("__repr__", [](const librapid::ArrayB &arr) { return "<librapid.ArrayB\n" + arr.str() + "\n>"; })
                .def("cast_ArrayB", &librapid::ArrayB::cast<librapid::ArrayB>)
                .def("castMove_ArrayB_CPU", &librapid::ArrayB::castMove<librapid::internal::traits<ArrayB>::Scalar, librapid::device::CPU>)
                .def("castMove_ArrayB_GPU", &librapid::ArrayB::castMove<librapid::internal::traits<ArrayB>::Scalar, librapid::device::GPU>)
                .def("cast_ArrayF32", &librapid::ArrayB::cast<librapid::ArrayF32>)
                .def("castMove_ArrayF32_CPU", &librapid::ArrayB::castMove<librapid::internal::traits<ArrayF32>::Scalar, librapid::device::CPU>)
                .def("castMove_ArrayF32_GPU", &librapid::ArrayB::castMove<librapid::internal::traits<ArrayF32>::Scalar, librapid::device::GPU>);

        py::class_<librapid::ArrayF32>(module, "ArrayF32")
                .def(py::init<>())
                .def(py::init<librapid::Extent>())
                .def(py::init<librapid::ArrayF32>())
                .def(py::init<librapid::internal::traits<librapid::ArrayF32>::Scalar>())
                .def("copy", &librapid::ArrayF32::copy)
                .def("__getitem__", [](const librapid::ArrayF32 &arr, int64_t index) { return arr[index]; })
                .def("__setitem__", [](const librapid::ArrayF32 &arr, int64_t index, librapid::internal::traits<librapid::ArrayF32>::Scalar val) { arr[index] = val; })
                .def("__setitem__", [](const librapid::ArrayF32 &arr, int64_t index, const librapid::ArrayF32 &val) { arr[index] = val; })
                .def("move_CPU", &librapid::ArrayF32::move<librapid::device::CPU>)
                .def("move_GPU", &librapid::ArrayF32::move<librapid::device::GPU>)
                .def("__add__", &librapid::ArrayF32::operator+<librapid::ArrayF32>)
                .def("__add__", &librapid::ArrayF32::operator+<librapid::internal::traits<librapid::ArrayF32>::Scalar>)
                .def("__sub__", &librapid::ArrayF32::operator-<librapid::ArrayF32>)
                .def("__sub__", &librapid::ArrayF32::operator-<librapid::internal::traits<librapid::ArrayF32>::Scalar>)
                .def("__mul__", &librapid::ArrayF32::operator*<librapid::ArrayF32>)
                .def("__mul__", &librapid::ArrayF32::operator*<librapid::internal::traits<librapid::ArrayF32>::Scalar>)
                .def("__div__", &librapid::ArrayF32::operator/<librapid::ArrayF32>)
                .def("__div__", &librapid::ArrayF32::operator/<librapid::internal::traits<librapid::ArrayF32>::Scalar>)
                .def("transpose", &librapid::ArrayF32::transpose)
                .def("transposed", &librapid::ArrayF32::transposed)
                .def("dot", &librapid::ArrayF32::dot<librapid::ArrayF32>)
                .def("str", &librapid::ArrayF32::str)
                .def("__str__", &librapid::ArrayF32::str)
                .def("__repr__", [](const librapid::ArrayF32 &arr) { return "<librapid.ArrayF32\n" + arr.str() + "\n>"; })
                .def("cast_ArrayB", &librapid::ArrayF32::cast<librapid::ArrayB>)
                .def("castMove_ArrayB_CPU", &librapid::ArrayF32::castMove<librapid::internal::traits<ArrayB>::Scalar, librapid::device::CPU>)
                .def("castMove_ArrayB_GPU", &librapid::ArrayF32::castMove<librapid::internal::traits<ArrayB>::Scalar, librapid::device::GPU>)
                .def("cast_ArrayF32", &librapid::ArrayF32::cast<librapid::ArrayF32>)
                .def("castMove_ArrayF32_CPU", &librapid::ArrayF32::castMove<librapid::internal::traits<ArrayF32>::Scalar, librapid::device::CPU>)
                .def("castMove_ArrayF32_GPU", &librapid::ArrayF32::castMove<librapid::internal::traits<ArrayF32>::Scalar, librapid::device::GPU>);

	// Include the Vector library
	#include "cpp/vecInterface.hpp"

	// py::implicitly_convertible<int64_t, librapid::Array>();
	// py::implicitly_convertible<double, librapid::Array>();
	// py::implicitly_convertible<py::tuple, librapid::Array>();
	// py::implicitly_convertible<py::list, librapid::Array>();
}
