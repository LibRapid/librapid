#include <librapid/array/multiarray.hpp>
#include <librapid/array/multiarray_operations.hpp>

namespace librapid
{
	void Array::fill(double val)
	{
		AUTOCAST_UNARY(Array::simpleFill, makeVoidPtr(), validVoidPtr,
					   m_extent.size(), val);
	}

	void Array::fill(const Complex<double> &val)
	{
		AUTOCAST_UNARY(Array::simpleFill, makeVoidPtr(), validVoidPtr,
					   m_extent.size(), val);
	}

	VoidPtr Array::makeVoidPtr() const
	{
		return {m_dataStart, m_dtype, m_location};
	}

	Array Array::copy(const Datatype &dtype, const Accelerator &locn)
	{
		Datatype resDtype = (dtype == Datatype::NONE) ? m_dtype : dtype;
		Accelerator resLocn = (locn == Accelerator::NONE) ? locn : locn;

		Array res(m_extent, resDtype, resLocn);
		res.m_isScalar = m_isScalar;

		if (m_stride.isTrivial() && m_stride.isContiguous())
		{
			// Trivial stride, so just memcpy
			AUTOCAST_MEMCPY(res.makeVoidPtr(), makeVoidPtr(), m_extent.size());
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

	template<typename A, typename B, typename C>
	void Array::simpleFill(librapid::Accelerator locnA,
						   librapid::Accelerator locnB,
						   A *data, B *, size_t size,
						   C val)
	{
		if (locnA == Accelerator::CPU)
		{
			for (size_t i = 0; i < size; ++i)
				data[i] = (A) val;
		}
	#ifdef LIBRAPID_HAS_CUDA
		else
		{
			auto tmp = (A *) malloc(sizeof(A) * size);
			if (tmp == nullptr)
				throw std::bad_alloc();

			for (size_t i = 0; i < size; ++i)
				tmp[i] = (A) val;

		#ifdef LIBRAPID_CUDA_STREAM
			cudaSafeCall(cudaMemcpyAsync(data, tmp, sizeof(A) * size, cudaMemcpyHostToDevice, cudaStream));
			cudaSafeCall(cudaStreamSynchronize(cudaStream));
		#else
			cudaSafeCall(cudaDeviceSynchronize());
			cudaSafeCall(cudaMemcpy(data, tmp, sizeof(A) * size, cudaMemcpyHostToDevice));
		#endif
			free(tmp);
		}
	#endif
	}
}