#pragma once

#include <librapid/librapid.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <functional>
#include <string>

// Just remove these. They're pointless
#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

namespace lrc = librapid;
namespace py = pybind11;

void init_ArrayI64G(py::module &module) {
#if defined(LIBRAPID_HAS_CUDA)
py::class_<librapid::ArrayI64G>(module, "ArrayI64G")
	.def(py::init<>())
	.def(py::init<librapid::Extent>())
	.def(py::init<const librapid::ArrayI64G &>())
	.def(py::init<librapid::internal::traits<librapid::ArrayI64G>::Scalar>())
	.def("copy", [](const librapid::ArrayI64G & this_) { return this_.copy(); })
	.def("__getitem__", [](const librapid::ArrayI64G & this_, int64_t index) { return this_[index]; }, py::arg("index"))
	.def("__setitem__", [](librapid::ArrayI64G & this_, int64_t index, typename librapid::internal::traits<librapid::ArrayI64G>::Scalar val) { this_[index] = val; }, py::arg("index"), py::arg("val"))
	.def("__setitem__", [](librapid::ArrayI64G & this_, int64_t index, typename librapid::internal::traits<librapid::ArrayI64G>::Scalar val) { this_[index] = val; }, py::arg("index"), py::arg("val"))
	.def("move_CPU", [](const librapid::ArrayI64G & this_) { return this_.move<librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("move_GPU", [](const librapid::ArrayI64G & this_) { return this_.move<librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayBG", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayBG>::Scalar>(); })
	.def("cast_ArrayCG", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayCG>::Scalar>(); })
	.def("cast_ArrayF16G", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF16G>::Scalar>(); })
	.def("cast_ArrayF32G", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF32G>::Scalar>(); })
	.def("cast_ArrayF64G", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF64G>::Scalar>(); })
	.def("cast_ArrayI16G", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI16G>::Scalar>(); })
	.def("cast_ArrayI32G", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI32G>::Scalar>(); })
	.def("cast_ArrayI64G", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI64G>::Scalar>(); })
	.def("cast_ArrayB", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayB>::Scalar>(); })
	.def("cast_ArrayC", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayC>::Scalar>(); })
	.def("cast_ArrayF16", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF16>::Scalar>(); })
	.def("cast_ArrayF32", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF32>::Scalar>(); })
	.def("cast_ArrayF64", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF64>::Scalar>(); })
	.def("cast_ArrayI16", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI16>::Scalar>(); })
	.def("cast_ArrayI32", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI32>::Scalar>(); })
	.def("cast_ArrayI64", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI64>::Scalar>(); })
	.def("cast_ArrayMPZ", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayMPZ>::Scalar>(); })
	.def("cast_ArrayMPQ", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayMPQ>::Scalar>(); })
	.def("cast_ArrayMPFR", [](const librapid::ArrayI64G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayMPFR>::Scalar>(); })
	.def("__add__", [](const librapid::ArrayI64G & this_, const librapid::ArrayI64G & other) { return this_ + other; }, py::arg("other"))
	.def("__add__", [](const librapid::ArrayI64G & this_, typename librapid::internal::traits<librapid::ArrayI64G>::Scalar other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::ArrayI64G & this_, const librapid::ArrayI64G & other) { return this_ - other; }, py::arg("other"))
	.def("__sub__", [](const librapid::ArrayI64G & this_, typename librapid::internal::traits<librapid::ArrayI64G>::Scalar other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::ArrayI64G & this_, const librapid::ArrayI64G & other) { return this_ * other; }, py::arg("other"))
	.def("__mul__", [](const librapid::ArrayI64G & this_, typename librapid::internal::traits<librapid::ArrayI64G>::Scalar other) { return this_ * other; }, py::arg("other"))
	.def("__div__", [](const librapid::ArrayI64G & this_, const librapid::ArrayI64G & other) { return this_ / other; }, py::arg("other"))
	.def("__div__", [](const librapid::ArrayI64G & this_, typename librapid::internal::traits<librapid::ArrayI64G>::Scalar other) { return this_ / other; }, py::arg("other"))
	.def("__or__", [](const librapid::ArrayI64G & this_, const librapid::ArrayI64G & other) { return this_ | other; }, py::arg("other"))
	.def("__and__", [](const librapid::ArrayI64G & this_, const librapid::ArrayI64G & other) { return this_ & other; }, py::arg("other"))
	.def("__xor__", [](const librapid::ArrayI64G & this_, const librapid::ArrayI64G & other) { return this_ ^ other; }, py::arg("other"))
	.def("__neg__", [](const librapid::ArrayI64G & this_) { return -this_; })
	.def("transpose", [](librapid::ArrayI64G & this_, const librapid::Extent & order) { this_.transpose(order); }, py::arg("order") = librapid::Extent({}))
	.def("transposed", [](const librapid::ArrayI64G & this_, const librapid::Extent & order) { return this_.transposed(order); }, py::arg("order") = librapid::Extent({}))
	.def("dot", [](const librapid::ArrayI64G & this_, const librapid::ArrayI64G & other) { return this_.dot(other); }, py::arg("other"))
	.def("str", [](const librapid::ArrayI64G & this_, const std::string & format, const std::string & delim, int64_t stripWidth, int64_t beforePoint, int64_t afterPoint, int64_t depth) { return this_.str(format, delim, stripWidth, beforePoint, afterPoint, depth); }, py::arg("format") = std::string("{}"), py::arg("delim") = std::string(" "), py::arg("stripWidth") = int64_t(-1), py::arg("beforePoint") = int64_t(-1), py::arg("afterPoint") = int64_t(-1), py::arg("depth") = int64_t(0))
	.def("__str__", [](const librapid::ArrayI64G & this_) { return this_.str(); })
	.def("__repr__", [](const librapid::ArrayI64G & this_) { return "<librapid::ArrayI64G\n" + this_.str("{}", ",") + "\n>"; })
	.def("isScalar", [](const librapid::ArrayI64G & this_) { return this_.isScalar(); })
	.def("extent", [](const librapid::ArrayI64G & this_) { return this_.extent(); });


module.def("add", [](const librapid::ArrayI64G & lhs, const librapid::ArrayI64G & rhs, librapid::ArrayI64G & dst) { librapid::add(lhs, rhs, dst); }, py::arg("lhs"), py::arg("rhs"), py::arg("dst"));
module.def("sub", [](const librapid::ArrayI64G & lhs, const librapid::ArrayI64G & rhs, librapid::ArrayI64G & dst) { librapid::sub(lhs, rhs, dst); }, py::arg("lhs"), py::arg("rhs"), py::arg("dst"));
module.def("mul", [](const librapid::ArrayI64G & lhs, const librapid::ArrayI64G & rhs, librapid::ArrayI64G & dst) { librapid::mul(lhs, rhs, dst); }, py::arg("lhs"), py::arg("rhs"), py::arg("dst"));
module.def("div", [](const librapid::ArrayI64G & lhs, const librapid::ArrayI64G & rhs, librapid::ArrayI64G & dst) { librapid::div(lhs, rhs, dst); }, py::arg("lhs"), py::arg("rhs"), py::arg("dst"));
#endif

}