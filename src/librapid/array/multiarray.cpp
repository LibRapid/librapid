#include <librapid/array/multiarray.hpp>
#include <librapid/array/multiarray_operations.hpp>

namespace librapid
{
	void Array::fill(double val)
	{
		// AUTOCAST_UNARY(Array::simpleFill, makeVoidPtr(), validVoidPtr,
		// 			   m_extent.size(), val);

		Array::applyBinaryOp(*this, *this, val, ops::Fill());
	}

	void Array::fill(const Complex<double> &val)
	{
		// AUTOCAST_UNARY(Array::simpleFill, makeVoidPtr(), validVoidPtr,
		// 			   m_extent.size(), val);

		Array::applyBinaryOp(*this, *this, val, ops::Fill());
	}

	RawArray Array::createRaw() const
	{
		return {m_dataStart, m_dtype, m_location};
	}

	Array Array::copy(const Datatype &dtype, const Accelerator &locn)
	{
		Datatype resDtype = (dtype == Datatype::NONE) ? m_dtype : dtype;
		Accelerator resLocn = (locn == Accelerator::NONE) ? locn : locn;

		Array res(m_extent, resDtype, resLocn);
		auto ptrDst = res.createRaw();
		res.m_isScalar = m_isScalar;

		if (m_stride.isTrivial() && m_stride.isContiguous())
		{
			// Trivial stride, so just memcpy
			// AUTOCAST_MEMCPY(res.makeVoidPtr(), makeVoidPtr(), m_extent.size());

			rawArrayMemcpy(ptrDst, createRaw(), m_extent.size());
			// static_assert(false, "Just break everything");
			throw std::runtime_error("This hasn't yet been implemented\n");
		}
		else if (m_location == Accelerator::CPU && locn == Accelerator::GPU)
		{
		#ifdef LIBRAPID_HAS_CUDA
			// Non-trivial stride, so apply more complicated algorithm
			applyUnaryOp(*this, res, ops::Copy());

		#else
			throw std::invalid_argument("CUDA support was not enabled, so cannot"
										" copy array to GPU");
		#endif
		}

		return res;
	}
}