#include <librapid/array/multiarray.hpp>

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
		m_isChild = false;

		increment();
	}

	Array &Array::operator=(const Array &other)
	{
		// Quick return if possible
		if (other.m_references == nullptr)
			return *this;

		if (m_references == nullptr)
			constructNew(other.m_extent, other.m_stride,
						 other.m_dtype, other.m_location);

		if (m_isChild && m_extent != other.m_extent)
			throw std::invalid_argument("Cannot set child array with "
										+ m_extent.str() + " to "
										+ other.m_extent.str());

		if (!m_isChild)
		{
			m_extent = other.m_extent;
			m_stride = other.m_stride;
		}

		// Attempt to copy the data from other into *this
		if (m_stride.isContiguous() && other.m_stride.isContiguous())
			AUTOCAST_MEMCPY(makeVoidPtr(), other.makeVoidPtr(),
							m_extent.size());
		else
			throw std::runtime_error("Haven't gotten to this yet...");

		increment();

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