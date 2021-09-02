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

	Array Array::operator+(const Array &other) const
	{
		// Add two arrays together
		if (!(m_stride.isTrivial() && m_stride.isContiguous()))
			throw std::runtime_error("Yeah, you can't do this yet");

		// if (m_location != other.m_location)
		// 	throw std::runtime_error("Can't do this either, unfortunately");

		Accelerator newLoc = max(m_location, other.m_location);
		Datatype newType = max(m_dtype, other.m_dtype);
		Array res(m_extent, newType, newLoc);

		AUTOCAST_BINARY(imp::multiarrayBinaryOpTrivial, makeVoidPtr(),
						other.makeVoidPtr(), res.makeVoidPtr(),
						false, false, m_extent.size(), ops::Add());

		return res;
	}

	Array Array::operator-(const Array &other) const
	{
		// Add two arrays together
		if (!(m_stride.isTrivial() && m_stride.isContiguous()))
			throw std::runtime_error("Yeah, you can't do this yet");

		if (m_location != other.m_location)
			throw std::runtime_error("Can't do this either, unfortunately");

		Array res(m_extent, m_dtype, m_location);

		AUTOCAST_BINARY(imp::multiarrayBinaryOpTrivial, makeVoidPtr(),
						other.makeVoidPtr(), res.makeVoidPtr(),
						false, false, m_extent.size(), ops::Sub());

		return res;
	}

	void Array::add(const Array &other, Array &res) const
	{
		// Add two arrays together
		if (!(m_stride.isTrivial() && m_stride.isContiguous()))
			throw std::runtime_error("Yeah, you can't do this yet");

		if (m_location != other.m_location)
			throw std::runtime_error("Can't do this either, unfortunately");

		AUTOCAST_BINARY(imp::multiarrayBinaryOpTrivial, makeVoidPtr(),
						other.makeVoidPtr(), res.makeVoidPtr(),
						false, false, m_extent.size(), ops::Add());
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