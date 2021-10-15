#include <librapid/array/stride.hpp>

namespace librapid
{
	Stride::Stride(const std::initializer_list<int64_t> &data)
		: Stride(std::vector<int64_t>(data.begin(), data.end()))
	{}

	Stride::Stride(const std::vector<int64_t> &data)
	{
		// Initialize members
		m_isTrivial = true;
		m_isContiguous = true;
		m_dims = data.size();

		// Check for a valid number of dimensions
		if (m_dims > LIBRAPID_MAX_DIMS)
			throw std::runtime_error("Cannot create Stride with "
									 + std::to_string(m_dims)
									 + " dimensions. Maximum allowed is "
									 + std::to_string(LIBRAPID_MAX_DIMS));

		for (size_t i = 0; i < data.size(); ++i)
			m_stride[i] = data[i];
	}

	Stride::Stride(size_t dims)
	{
		m_dims = dims;
		if (m_dims > LIBRAPID_MAX_DIMS)
			throw std::runtime_error("Cannot create Stride with "
									 + std::to_string(m_dims)
									 + " dimensions. Limit is "
									 + std::to_string(LIBRAPID_MAX_DIMS));

		for (size_t i = 0; i < m_dims; ++i)
			m_stride[i] = 1;
	}

	Stride::Stride(const Stride &other)
	{
		m_isTrivial = other.m_isTrivial;
		m_isContiguous = other.m_isContiguous;
		m_dims = other.m_dims;
		m_one = other.m_one;

		for (size_t i = 0; i < m_dims; ++i)
			m_stride[i] = other.m_stride[i];
	}

#ifdef LIBRAPID_PYTHON
	Stride::Stride(py::args args)
	{
		m_dims = py::len(args);

		if (m_dims > LIBRAPID_MAX_DIMS)
			throw std::runtime_error("Cannot create Stride with "
									 + std::to_string(m_dims)
									 + " dimensions. Limit is "
									 + std::to_string(LIBRAPID_MAX_DIMS));

		size_t neg = 0;

		for (int64_t i = 0; i < m_dims; i++)
			m_stride[i] = py::cast<int64_t>(args[i]);
	}
#endif

	Stride &Stride::operator=(const Stride &other)
	{
		m_dims = other.m_dims;
		m_isTrivial = other.m_isTrivial;
		m_isContiguous = other.m_isContiguous;
		m_one = other.m_one;

		for (size_t i = 0; i < m_dims; ++i)
		{
			m_stride[i] = other.m_stride[i];
		}

		return *this;
	}

	Stride Stride::fromExtent(const Extent &extent)
	{
		Stride res;
		res.m_dims = extent.ndim();

		size_t prod = 1;
		for (size_t i = 0; i < extent.ndim(); ++i)
		{
			res.m_stride[res.m_dims - i - 1] = (int64_t) prod;
			prod *= extent[res.m_dims - i - 1];
		}

		return res;
	}

	void Stride::setContiguity(bool newVal)
	{
		m_isContiguous = newVal;
	}

	bool Stride::operator==(const Stride &other) const
	{
		if (m_dims != other.m_dims)
			return false;

		if (m_isTrivial != other.m_isTrivial)
			return false;

		if (m_isContiguous != other.m_isContiguous)
			return false;

		for (size_t i = 0; i < m_dims; ++i)
			if (m_stride[i] != other.m_stride[i])
				return false;

		return true;
	}

	const int64_t &Stride::operator[](const size_t index) const
	{
		if (index > m_dims)
			throw std::out_of_range("Cannot access index "
									+ std::to_string(index)
									+ " of Stride with "
									+ std::to_string(m_dims) + " dimensions");

		return m_stride[index];
	}

	int64_t &Stride::operator[](const size_t index)
	{
		if (index > m_dims)
			throw std::out_of_range("Cannot access index "
									+ std::to_string(index)
									+ " of Stride with "
									+ std::to_string(m_dims) + " dimensions");

		return m_stride[index];
	}

	void Stride::reorder(const std::vector<size_t> &order)
	{
		Stride temp = *this;

		for (size_t i = 0; i < order.size(); ++i)
			m_stride[i] = temp.m_stride[order[i]];

		m_isTrivial = checkTrivial();
	}

	void Stride::reorder(const std::vector<int64_t> &order)
	{
		Stride temp = *this;

		for (size_t i = 0; i < order.size(); ++i)
			m_stride[i] = temp.m_stride[order[i]];

		m_isTrivial = checkTrivial();
	}

	Stride Stride::subStride(int64_t start, int64_t end) const
	{
		if (start == -1) start = 0;
		if (end == -1) end = m_dims;

		if (start >= end)
			throw std::invalid_argument("Cannot create subStride from range ["
										+ std::to_string(start) + ", "
										+ std::to_string(end) + ")");

		Stride res(end - start);
		for (int64_t i = start; i < end; ++i)
			res.m_stride[i - start] = m_stride[i];
		res.m_one = m_one;
		res.m_isTrivial = res.checkTrivial();

		return res;
	}

	// void Stride::scaleBytes(size_t bytes)
	// {
	// 	for (size_t i = 0; i < m_dims; ++i)
	// 		m_stride[i] *= bytes;
	// 	m_one = bytes;
	// }
	//
	// Stride Stride::scaledBytes(size_t bytes) const
	// {
	// 	Stride res = *this;
	// 	res.scaleBytes(bytes);
	// 	return res;
	// }

	bool Stride::checkTrivial() const
	{
		// Ensure every stride is bigger than the next one
		bool foundOne = false;
		for (size_t i = 0; i < m_dims; ++i)
		{
			if (m_stride[i] <= m_stride[i + 1]) return false;
			if (m_stride[i] == m_one) foundOne = true;
		}

		return foundOne;
	}

	bool Stride::checkContiguous(const Extent &extent) const
	{
		if (m_dims != extent.ndim())
			throw std::domain_error("Stride and Extent must have the same "
									"dimensions for a contiguity test");

		Stride temp = fromExtent(extent);
		// temp.scaleBytes(m_one);
		size_t valid = 0;

		for (size_t i = 0; i < m_dims; ++i)
		{
			for (size_t j = 0; j < m_dims; j++)
			{
				if (temp[i] == m_stride[i])
				{
					++valid;
					break;
				}
			}
		}

		return valid == m_dims;
	}

	std::string Stride::str() const
	{
		std::stringstream res;
		res << "Stride(";
		for (size_t i = 0; i < m_dims; ++i)
		{
			res << m_stride[i];

			if (i < m_dims - 1)
				res << ", ";
		}
		res << ")";

		return res.str();
	}
}