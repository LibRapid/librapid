#include <librapid/array/multiarray.hpp>

namespace librapid
{
	const Array Array::subscript(size_t index) const
	{
		if (index >= (size_t) m_extent[0])
		{
			std::string msg = "Index " + std::to_string(index) +
				" out of range for array with leading dimension "
				+ std::to_string(m_extent[0]);

			throw std::out_of_range(msg);
		}

		Array res;

		res.m_dataOrigin = m_dataOrigin;
		res.m_dataStart = (void *) (((char *) m_dataStart) + m_stride[0] * index);
		res.m_references = m_references;
		lr_int dims = ndim();

		if (dims == 1)
		{
			// Return a scalar value
			res.constructHollow(Extent({1}), Stride{1}, m_dtype, m_location);
			res.m_isScalar = true;
		}
		else
		{
			res.constructHollow(m_extent.subExtent(1, AUTO),
								m_stride.subStride(1, AUTO),
								m_dtype, m_location);

			res.m_isScalar = false;
		}

		res.m_isChild = true;

		increment();
		return res;
	}
}