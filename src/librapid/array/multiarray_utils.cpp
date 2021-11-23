#include <librapid/array/multiarray.hpp>
#include <librapid/array/multiarray_operations.hpp>

namespace librapid {
	Array Array::clone(Datatype dtype, Accelerator locn) const {
		Array res(m_extent, dtype == Datatype::NONE ? m_dtype : dtype,
				  locn == Accelerator::NONE ? m_location : locn);
		res.m_isScalar = m_isScalar;
		res.m_isChild = false;
		applyUnaryOp(res, *this, ops::Copy());
		return res;
	}

	Array Array::clone(const std::string &dtype, Accelerator locn) const {
		return clone(stringToDatatype(dtype), locn);
	}

	Array Array::clone(Datatype dtype, const std::string &locn) const {
		return clone(dtype, stringToAccelerator(locn));
	}

	Array Array::clone(const std::string &dtype, const std::string &locn) const {
		return clone(stringToDatatype(dtype), stringToAccelerator(locn));
	}
}