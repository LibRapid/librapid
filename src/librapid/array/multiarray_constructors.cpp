#include <librapid/array/multiarray.hpp>
#include <librapid/utils/array_utils.hpp>

namespace librapid {
	RawArray validRawArray = RawArray{(int64_t *) nullptr, Datatype::VALIDNONE, Accelerator::CPU};

#ifdef LIBRAPID_HAS_CUDA
#ifdef LIBRAPID_CUDA_STREAM
	cudaStream_t cudaStream;
	bool streamCreated = false;
#endif // LIBRAPID_CUDA_STREAM
#endif // LIBRAPID_HAS_CUDA

	Array::Array() {
		initializeCudaStream();
	}

	Array::Array(const Extent &extent, Datatype dtype, Accelerator location) {
		initializeCudaStream();

		if (extent.containsAutomatic())
			throw std::invalid_argument("Cannot create an Array from an Extent"
										" containing automatic values. "
										"Extent was " + extent.str());

		constructNew(extent, Stride::fromExtent(extent), dtype, location);
	}

	Array::Array(const Array &other, Datatype dtype, Accelerator locn) {
		// Quick return if possible
		if (other.m_references == nullptr)
			return;

		if (dtype == Datatype::NONE) dtype = other.m_dtype;
		if (locn == Accelerator::NONE) locn = other.m_location;

		if (dtype != other.m_dtype || locn != other.m_location) {
			// Must cast other to the correct value
			Array tmp = other.clone(dtype, locn);

			m_location = tmp.m_location;
			m_dtype = tmp.m_dtype;
			m_dataStart = tmp.m_dataStart;
			m_dataOrigin = tmp.m_dataOrigin;

			m_references = tmp.m_references;

			m_extent = tmp.m_extent;
			m_stride = tmp.m_stride;

			m_isScalar = tmp.m_isScalar;
			m_isChild = tmp.m_isChild;

			increment();

			return;
		}

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

	Array::Array(bool val, Datatype dtype, Accelerator locn) {
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), dtype, locn);
		m_isScalar = true;
		if (locn == Accelerator::CPU) {
			std::visit([&](auto *data) {
				*data = val;
			}, m_dataStart);
		}
#ifdef LIBRAPID_HAS_CUDA
		else {
			int64_t tmpVal = val;
			RawArray temp = RawArray{m_dataStart, dtype, locn};
			rawArrayMemcpy(temp, {&tmpVal, Datatype::INT64, Accelerator::CPU}, 1);
		}
#else
		else {
			throw std::invalid_argument("CUDA support was not enabled, "
				"so a value cannot be created on the GPU");
		}
#endif // LIBRAPID_HAS_CUDA
	}

	Array::Array(float val, Datatype dtype, Accelerator locn) {
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), dtype, locn);
		m_isScalar = true;
		if (locn == Accelerator::CPU) {
			std::visit([&](auto *data) {
				*data = val;
			}, m_dataStart);
		}
#ifdef LIBRAPID_HAS_CUDA
		else {
			RawArray temp = RawArray{m_dataStart, dtype, locn};
			rawArrayMemcpy(temp, {&val, Datatype::FLOAT32, Accelerator::CPU}, 1);
		}
#else
		else {
			throw std::invalid_argument("CUDA support was not enabled, "
				"so a value cannot be created on the GPU");
		}
#endif // LIBRAPID_HAS_CUDA
	}

	Array::Array(double val, Datatype dtype, Accelerator locn) {
		initializeCudaStream();

		constructNew(Extent(1), Stride(1), dtype, locn);
		m_isScalar = true;
		if (locn == Accelerator::CPU) {
			std::visit([&](auto *data) {
				*data = val;
			}, m_dataStart);
		}
#ifdef LIBRAPID_HAS_CUDA
		else {
			RawArray temp = RawArray{m_dataStart, dtype, locn};
			rawArrayMemcpy(temp, {&val, Datatype::FLOAT64, Accelerator::CPU}, 1);
		}
#else
		else {
			throw std::invalid_argument("CUDA support was not enabled, "
				"so a value cannot be created on the GPU");
		}
#endif // LIBRAPID_HAS_CUDA
	}

	Array &Array::operator=(const Array &other) {
		// Quick return if possible
		if (other.m_references == nullptr)
			return *this;

		if (m_isChild && m_extent != other.m_extent) {
			throw std::invalid_argument("Cannot set child array with "
										+ m_extent.str() + " to "
										+ other.m_extent.str());
		}

		if (m_references == nullptr) {
			constructNew(other.m_extent, other.m_stride,
						 other.m_dtype, other.m_location);
		} else {
			// Array already exists, so check if it must be reallocated
			if (!m_isChild && m_extent != other.m_extent ||
				m_dtype != other.m_dtype) {
				// Extents are not equal, so memory can not be safely copied
				decrement();
				constructNew(other.m_extent, other.m_stride, other.m_dtype,
							 other.m_location);
				m_isScalar = other.m_isScalar;
			}
		}

		if (!m_isChild) {
			m_extent = other.m_extent;
			m_stride = other.m_stride;
		}

		// Attempt to copy the data from other into *this
		if (m_stride.isContiguous() && other.m_stride.isContiguous()) {
			auto raw = createRaw();
			rawArrayMemcpy(raw, other.createRaw(), m_extent.size());
		} else {
			throw std::runtime_error("Haven't gotten to this yet...");
		}

		m_extent = other.m_extent;
		m_stride = other.m_stride;
		m_isScalar = other.m_isScalar;
		m_dtype = other.m_dtype;
		m_location = other.m_location;

		return *this;
	}

	Array &Array::operator=(bool val) {
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild) {
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::INT64, Accelerator::CPU);
		}

		int64_t tmpVal = val;
		auto raw = createRaw();
		rawArrayMemcpy(raw, RawArray{&tmpVal, Datatype::INT64, Accelerator::CPU}, 1);

		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(float val) {
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild) {
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::FLOAT32, Accelerator::CPU);
		}

		auto raw = createRaw();
		rawArrayMemcpy(raw, RawArray{&val, Datatype::FLOAT32, Accelerator::CPU}, 1);

		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(double val) {
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild) {
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::FLOAT64, Accelerator::CPU);
		}

		auto raw = createRaw();
		rawArrayMemcpy(raw, RawArray{&val, Datatype::FLOAT64, Accelerator::CPU}, 1);

		m_isScalar = true;
		return *this;
	}

	Array &Array::operator=(Complex<double> val) {
		if (m_isChild && !m_isScalar)
			throw std::invalid_argument("Cannot set an array with more than zero"
										" dimensions to a scalar value. Array must"
										" have zero dimensions (i.e. scalar)");
		if (!m_isChild) {
			if (m_references != nullptr) decrement();
			constructNew(Extent(1), Stride(1), Datatype::CFLOAT64, Accelerator::CPU);
		}

		auto raw = createRaw();
		rawArrayMemcpy(raw, RawArray{&val, Datatype::FLOAT64, Accelerator::CPU}, 1);

		m_isScalar = true;
		return *this;
	}

	Array::~Array() {
		decrement();
	}

	void Array::constructNew(const Extent &e, const Stride &s,
							 const Datatype &dtype,
							 const Accelerator &location) {
		// Is scalar if extent is [0]
		bool isScalar = (e.ndim() == 1) && (e[0] == 0);

		// Construct members
		m_location = location;
		m_dtype = dtype;

		// If array is scalar, allocate "sizeof(dtype)" bytes -- e.size() is 0
		auto raw = RawArray{m_dataStart, dtype, location};
		m_dataStart = rawArrayMalloc(raw, e.size() + isScalar).data;
		m_dataOrigin = m_dataStart;

		m_references = new std::atomic<int64_t>(1);

		m_extent = e;
		m_stride = s;

		m_isScalar = isScalar;
		m_isChild = false;
	}

	void Array::constructHollow(const Extent &e, const Stride &s,
								const Datatype &dtype, const Accelerator &location) {
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

	Array zerosLike(const Array &other) {
		Array res(other.extent(), other.dtype(), other.location());
		res.fill(0);
		return res;
	}

	Array onesLike(const Array &other) {
		Array res(other.extent(), other.dtype(), other.location());
		res.fill(1);
		return res;
	}
}