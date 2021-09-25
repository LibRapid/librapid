#include <librapid/array/multiarray.hpp>
#include <librapid/utils/array_utils.hpp>

namespace librapid
{
	VoidPtr validVoidPtr = VoidPtr{nullptr, Datatype::VALIDNONE, Accelerator::CPU};

#ifdef LIBRAPID_CUDA_STREAM
	cudaStream_t cudaStream;
	bool streamCreated = false;
#endif // LIBRAPID_CUDA_STREAM

	Array::Array()
	{
		initializeCudaStream();
	}

	Array::Array(const Extent &extent, Datatype dtype, Accelerator location)
	{
		initializeCudaStream();

		if (extent.containsAutomatic())
			throw std::invalid_argument("Cannot create an Array from an Extent"
										" containing automatic values. "
										"Extent was " + extent.str());

		constructNew(extent, Stride::fromExtent(extent), dtype, location);
		m_stride.scaleBytes(datatypeBytes(dtype));
	}

	Array::Array(const Array &other)
	{
		// Quick return if possible
		if (other.m_references == nullptr)
			return;

		m_location = other.m_location;
		m_dtype = other.m_dtype;
		m_dataStart = other.m_dataStart;
		m_dataOrigin = other.m_dataOrigin;

		m_references = other.m_references;

		m_extent = other.m_extent;
		m_stride = other.m_stride;

		m_isScalar = other.m_isScalar;
		m_isChild = other.m_isChild;

		increment();
	}

	Array::Array(bool val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::BOOL, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::BOOL));
		m_isScalar = true;
		*((bool *) m_dataStart) = val;
	}

	Array::Array(int8_t val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::INT8, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::INT8));
		m_isScalar = true;
		*((int8_t *) m_dataStart) = val;
	}

	Array::Array(uint8_t val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::UINT8, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::UINT8));
		m_isScalar = true;
		*((uint8_t *) m_dataStart) = val;
	}

	Array::Array(int16_t val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::INT16, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::INT16));
		m_isScalar = true;
		*((int16_t *) m_dataStart) = val;
	}

	Array::Array(uint16_t val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::UINT16, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::UINT16));
		m_isScalar = true;
		*((uint16_t *) m_dataStart) = val;
	}

	Array::Array(int32_t val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::INT32, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::INT32));
		m_isScalar = true;
		*((int32_t *) m_dataStart) = val;
	}

	Array::Array(uint32_t val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::UINT32, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::UINT32));
		m_isScalar = true;
		*((uint32_t *) m_dataStart) = val;
	}

	Array::Array(int64_t val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::INT64, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::INT64));
		m_isScalar = true;
		*((int64_t *) m_dataStart) = val;
	}

	Array::Array(uint64_t val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::UINT64, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::UINT64));
		m_isScalar = true;
		*((uint64_t *) m_dataStart) = val;
	}

	Array::Array(float val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::FLOAT32, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::FLOAT32));
		m_isScalar = true;
		*((float *) m_dataStart) = val;
	}

	Array::Array(double val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::FLOAT64, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::FLOAT64));
		m_isScalar = true;
		*((double *) m_dataStart) = val;
	}

	Array::Array(const Complex<float> &val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::CFLOAT32, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::CFLOAT32));
		m_isScalar = true;
		*((Complex<float> *) m_dataStart) = val;
	}

	Array::Array(const Complex<double> &val)
	{
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), Datatype::CFLOAT64, Accelerator::CPU);
		m_stride.scaleBytes(datatypeBytes(Datatype::CFLOAT64));
		m_isScalar = true;
		*((Complex<double> *) m_dataStart) = val;
	}

	Array &Array::operator=(const Array &other)
	{
		// Quick return if possible
		if (other.m_references == nullptr)
			return *this;

		if (m_isChild && m_extent != other.m_extent)
		{
			throw std::invalid_argument("Cannot set child array with "
										+ m_extent.str() + " to "
										+ other.m_extent.str());
		}

		if (m_references == nullptr)
		{
			constructNew(other.m_extent, other.m_stride,
						 other.m_dtype, other.m_location);
		}
		else
		{
			// Array already exists, so check if it must be reallocated
			if (!m_isChild && m_extent != other.m_extent ||
				m_dtype != other.m_dtype)
			{
				// Extents are not equal, so memory can not be safely copied
				decrement();
				constructNew(other.m_extent, other.m_stride, other.m_dtype,
							 other.m_location);
				m_isScalar = other.m_isScalar;
			}
		}

		if (!m_isChild)
		{
			m_extent = other.m_extent;
			m_stride = other.m_stride;
		}

		// Attempt to copy the data from other into *this
		if (m_stride.isContiguous() && other.m_stride.isContiguous())
		{
			AUTOCAST_MEMCPY(makeVoidPtr(), other.makeVoidPtr(),
							m_extent.size());
		}
		else
		{
			throw std::runtime_error("Haven't gotten to this yet...");
		}

		return *this;
	}

	Array &Array::operator=(bool val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::BOOL, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::BOOL,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(int8_t val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::INT8, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::INT8,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(uint8_t val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::UINT8, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::UINT8,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(int16_t val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::INT16, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::INT16,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(uint16_t val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::UINT16, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::UINT16,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(int32_t val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::INT32, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::INT32,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(uint32_t val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::UINT32, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::UINT32,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(int64_t val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::INT64, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::INT64,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(uint64_t val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::UINT64, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::UINT64,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(float val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::FLOAT32, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::FLOAT32,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(double val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::FLOAT64, Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::FLOAT64,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(const Complex<float> &val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::CFLOAT32,
						 Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::CFLOAT32,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(const Complex<double> &val)
	{
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild)
		{
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::CFLOAT64,
						 Accelerator::CPU);
		}
		AUTOCAST_MEMCPY(makeVoidPtr(), VoidPtr{(void *) (&val), Datatype::CFLOAT64,
						Accelerator::CPU}, 1);
		m_isScalar = true;
		return *this;
	}

	Array::~Array()
	{
		decrement();
	}

	void Array::constructNew(const Extent &e, const Stride &s,
							 const Datatype &dtype,
							 const Accelerator &location)
	{
		// Is scalar if extent is [0]
		bool isScalar = (e.ndim() == 1) && (e[0] == 0);

		// Construct members
		m_location = location;
		m_dtype = dtype;

		// If array is scalar, allocate "sizeof(dtype)" bytes -- e.size() is 0
		m_dataStart = AUTOCAST_ALLOC(dtype, location, e.size() + isScalar).ptr;
		m_dataOrigin = m_dataStart;

		m_references = new std::atomic<size_t>(1);

		m_extent = e;
		m_stride = s;

		m_isScalar = isScalar;
		m_isChild = false;
	}

	void Array::constructHollow(const Extent &e, const Stride &s,
								const Datatype &dtype, const Accelerator &location)
	{
		m_extent = e;
		m_stride = s;
		m_dtype = dtype;
		m_location = location;
		m_stride.setContiguity(m_stride.checkContiguous(m_extent));

		if (ndim() > LIBRAPID_MAX_DIMS)
			throw std::domain_error("Cannot create an array with "
									+ std::to_string(ndim()) + " dimensions. The "
									"maximum allowed number of dimensions is "
									+ std::to_string(LIBRAPID_MAX_DIMS));
	}
}