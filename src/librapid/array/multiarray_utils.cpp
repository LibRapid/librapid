#include <librapid/array/multiarray.hpp>
#include <librapid/array/multiarray_operations.hpp>

namespace librapid
{
	Array Array::clone() const
	{
		Array res(m_extent, m_dtype, m_location);
		res.m_isScalar = m_isScalar;
		res.m_isChild = false;
		// applyBinaryOp(res, *this, res, ops::Copy());
		applyUnaryOp(res, *this, ops::Copy());
		return res;
	}
}