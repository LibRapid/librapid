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

void init_ArrayI32G(py::module &module) {
#if defined(LIBRAPID_HAS_CUDA)
py::class_<librapid::ArrayI32G>(module, "ArrayI32G")
	.def(py::init<>())
	.def(py::init<librapid::Extent>())
	.def(py::init<const librapid::ArrayI32G &>())
	.def(py::init<librapid::internal::traits<librapid::ArrayI32G>::Scalar>())
	.def("copy", [](const librapid::ArrayI32G & this_) { return this_.copy(); })
	.def("__getitem__", [](const librapid::ArrayI32G & this_, int64_t index) { return this_[index]; }, py::arg("index"))
	.def("__setitem__", [](librapid::ArrayI32G & this_, int64_t index, typename librapid::internal::traits<librapid::ArrayI32G>::Scalar val) { this_[index] = val; }, py::arg("index"), py::arg("val"))
	.def("__setitem__", [](librapid::ArrayI32G & this_, int64_t index, typename librapid::internal::traits<librapid::ArrayI32G>::Scalar val) { this_[index] = val; }, py::arg("index"), py::arg("val"))
	.def("move_CPU", [](const librapid::ArrayI32G & this_) { return this_.move<librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("move_GPU", [](const librapid::ArrayI32G & this_) { return this_.move<librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayBG>::Scalar>(); })
	.def("castMove_ArrayBG_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayBG>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayBG_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayBG>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayCG>::Scalar>(); })
	.def("castMove_ArrayCG_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayCG>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayCG_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayCG>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF16G>::Scalar>(); })
	.def("castMove_ArrayF16G_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF16G>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayF16G_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF16G>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF32G>::Scalar>(); })
	.def("castMove_ArrayF32G_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF32G>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayF32G_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF32G>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF64G>::Scalar>(); })
	.def("castMove_ArrayF64G_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF64G>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayF64G_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF64G>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI16G>::Scalar>(); })
	.def("castMove_ArrayI16G_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI16G>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayI16G_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI16G>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI32G>::Scalar>(); })
	.def("castMove_ArrayI32G_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI32G>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayI32G_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI32G>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI64G>::Scalar>(); })
	.def("castMove_ArrayI64G_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI64G>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayI64G_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI64G>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayB>::Scalar>(); })
	.def("castMove_ArrayB_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayB>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayB_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayB>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayC>::Scalar>(); })
	.def("castMove_ArrayC_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayC>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayC_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayC>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF16>::Scalar>(); })
	.def("castMove_ArrayF16_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF16>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayF16_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF16>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF32>::Scalar>(); })
	.def("castMove_ArrayF32_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF32>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayF32_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF32>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayF64>::Scalar>(); })
	.def("castMove_ArrayF64_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF64>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayF64_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayF64>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI16>::Scalar>(); })
	.def("castMove_ArrayI16_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI16>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayI16_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI16>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI32>::Scalar>(); })
	.def("castMove_ArrayI32_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI32>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayI32_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI32>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("cast_ArrayI32G", [](const librapid::ArrayI32G & this_) { return this_.cast<typename librapid::internal::traits<librapid::ArrayI64>::Scalar>(); })
	.def("castMove_ArrayI64_CPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI64>::Scalar, librapid::device::CPU>(); })
	#if defined(LIBRAPID_HAS_CUDA)
	.def("castMove_ArrayI64_GPU", [](const librapid::ArrayI32G & this_) { return this_.castMove<typename librapid::internal::traits<librapid::ArrayI64>::Scalar, librapid::device::GPU>(); })
	#endif // LIBRAPID_HAS_CUDA
	.def("__add__", [](const librapid::ArrayI32G & this_, const librapid::ArrayI32G & other) { return this_ + other; }, py::arg("other"))
	.def("__add__", [](const librapid::ArrayI32G & this_, typename librapid::internal::traits<librapid::ArrayI32G>::Scalar other) { return this_ + other; }, py::arg("other"))
	.def("__sub__", [](const librapid::ArrayI32G & this_, const librapid::ArrayI32G & other) { return this_ - other; }, py::arg("other"))
	.def("__sub__", [](const librapid::ArrayI32G & this_, typename librapid::internal::traits<librapid::ArrayI32G>::Scalar other) { return this_ - other; }, py::arg("other"))
	.def("__mul__", [](const librapid::ArrayI32G & this_, const librapid::ArrayI32G & other) { return this_ * other; }, py::arg("other"))
	.def("__mul__", [](const librapid::ArrayI32G & this_, typename librapid::internal::traits<librapid::ArrayI32G>::Scalar other) { return this_ * other; }, py::arg("other"))
	.def("__div__", [](const librapid::ArrayI32G & this_, const librapid::ArrayI32G & other) { return this_ / other; }, py::arg("other"))
	.def("__div__", [](const librapid::ArrayI32G & this_, typename librapid::internal::traits<librapid::ArrayI32G>::Scalar other) { return this_ / other; }, py::arg("other"))
	.def("__or__", [](const librapid::ArrayI32G & this_, const librapid::ArrayI32G & other) { return this_ | other; }, py::arg("other"))
	.def("__and__", [](const librapid::ArrayI32G & this_, const librapid::ArrayI32G & other) { return this_ & other; }, py::arg("other"))
	.def("__xor__", [](const librapid::ArrayI32G & this_, const librapid::ArrayI32G & other) { return this_ ^ other; }, py::arg("other"))
	.def("__neg__", [](const librapid::ArrayI32G & this_) { return -this_; })
	.def("transpose", [](librapid::ArrayI32G & this_, const librapid::Extent & order) { this_.transpose(order); }, py::arg("order") = librapid::Extent({}))
	.def("transposed", [](const librapid::ArrayI32G & this_, const librapid::Extent & order) { return this_.transposed(order); }, py::arg("order") = librapid::Extent({}))
	.def("dot", [](const librapid::ArrayI32G & this_, const librapid::ArrayI32G & other) { return this_.dot(other); }, py::arg("other"))
	.def("str", [](const librapid::ArrayI32G & this_, const std::string & format, const std::string & delim, int64_t stripWidth, int64_t beforePoint, int64_t afterPoint, int64_t depth) { return this_.str(format, delim, stripWidth, beforePoint, afterPoint, depth); }, py::arg("format") = std::string("{}"), py::arg("delim") = std::string(" "), py::arg("stripWidth") = int64_t(-1), py::arg("beforePoint") = int64_t(-1), py::arg("afterPoint") = int64_t(-1), py::arg("depth") = int64_t(0))
	.def("__str__", [](const librapid::ArrayI32G & this_) { return this_.str(); })
	.def("__repr__", [](const librapid::ArrayI32G & this_) { return "<librapid::ArrayI32G\n" + this_.str("{}", ",") + "\n>"; })
	.def("isScalar", [](const librapid::ArrayI32G & this_) { return this_.isScalar(); })
	.def("extent", [](const librapid::ArrayI32G & this_) { return this_.extent(); });


module.def("add", [](const librapid::ArrayI32G & lhs, const librapid::ArrayI32G & rhs, librapid::ArrayI32G & dst) { librapid::add(lhs, rhs, dst); }, py::arg("lhs"), py::arg("rhs"), py::arg("dst"));
module.def("sub", [](const librapid::ArrayI32G & lhs, const librapid::ArrayI32G & rhs, librapid::ArrayI32G & dst) { librapid::sub(lhs, rhs, dst); }, py::arg("lhs"), py::arg("rhs"), py::arg("dst"));
module.def("mul", [](const librapid::ArrayI32G & lhs, const librapid::ArrayI32G & rhs, librapid::ArrayI32G & dst) { librapid::mul(lhs, rhs, dst); }, py::arg("lhs"), py::arg("rhs"), py::arg("dst"));
module.def("div", [](const librapid::ArrayI32G & lhs, const librapid::ArrayI32G & rhs, librapid::ArrayI32G & dst) { librapid::div(lhs, rhs, dst); }, py::arg("lhs"), py::arg("rhs"), py::arg("dst"));
#endif

}