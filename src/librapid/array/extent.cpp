#include <librapid/array/extent.hpp>
#include <sstream>

namespace librapid {
	Extent::Extent(const std::initializer_list<int64_t> &data) {
		// Initialize the dimensions
		m_dims = data.size();

		if (m_dims > LIBRAPID_MAX_DIMS)
			throw std::runtime_error("Cannot create Extent with "
									 + std::to_string(m_dims)
									 + " dimensions. Limit is "
									 + std::to_string(LIBRAPID_MAX_DIMS));

		uint64_t index = 0;
		for (const auto &val: data)
			m_extent[index++] = val;

		update();
	}

	Extent::Extent(const std::vector<int64_t> &data) {
		// Initialize the dimensions
		m_dims = data.size();

		if (m_dims > LIBRAPID_MAX_DIMS)
			throw std::runtime_error("Cannot create Extent with "
									 + std::to_string(m_dims)
									 + " dimensions. Limit is "
									 + std::to_string(LIBRAPID_MAX_DIMS));

		for (uint64_t i = 0; i < data.size(); i++)
			m_extent[i] = data[i];

		update();
	}

	Extent::Extent(const Extent &other) {
		// Initialize the dimensions
		m_dims = other.m_dims;

		if (m_dims > LIBRAPID_MAX_DIMS)
			throw std::runtime_error("Cannot create Extent with "
									 + std::to_string(m_dims)
									 + " dimensions. Limit is "
									 + std::to_string(LIBRAPID_MAX_DIMS));

		for (int64_t i = 0; i < other.m_dims; i++)
			m_extent[i] = other.m_extent[i];

		update();
	}

	Extent::Extent(int64_t dims) {
		m_dims = dims;
		m_size = dims;
		if (m_dims > LIBRAPID_MAX_DIMS)
			throw std::runtime_error("Cannot create Extent with "
									 + std::to_string(m_dims)
									 + " dimensions. Limit is "
									 + std::to_string(LIBRAPID_MAX_DIMS));

		for (int64_t i = 0; i < m_dims; ++i)
			m_extent[i] = 1;
	}

#ifdef LIBRAPID_PYTHON
	Extent::Extent(py::args args)
	{
		m_dims = py::len(args);

		if (m_dims > LIBRAPID_MAX_DIMS)
			throw std::runtime_error("Cannot create Extent with "
									 + std::to_string(m_dims)
									 + " dimensions. Limit is "
									 + std::to_string(LIBRAPID_MAX_DIMS));

		for (int64_t i = 0; i < m_dims; i++)
			m_extent[i] = py::cast<int64_t>(args[i]);

		update();
	}
#endif // LIBRAPID_PYTHON

	Extent &Extent::operator=(const Extent &other) {
		memcpy(m_extent, other.m_extent, sizeof(int64_t) * LIBRAPID_MAX_DIMS);
		m_dims = other.m_dims;
		m_size = other.m_size;
		m_containsAutomatic = other.m_containsAutomatic;

		update();

		return *this;
	}

	const int64_t &Extent::operator[](int64_t index) const {
		if (index >= m_dims)
			throw std::out_of_range("Index " + std::to_string(index)
									+ " is out of range for Extent with "
									+ std::to_string(m_dims) + " dimensions");

		return m_extent[index];
	}

	int64_t &Extent::operator[](int64_t index) {
		if (index >= m_dims)
			throw std::out_of_range("Index " + std::to_string(index)
									+ " is out of range for Extent with "
									+ std::to_string(m_dims) + " dimensions");

		if (m_isDirty)
			update();

		m_isDirty = true;

		return m_extent[index];
	}

	Extent Extent::fixed(int64_t target) const {
		int64_t neg = 0;
		for (int64_t i = 0; i < m_dims; ++i)
			if (m_extent[i] < 0) ++neg;

		if (neg > 1)
			throw std::invalid_argument("Cannot construct Extent with more than"
										" one automatic values. " +
										std::to_string(neg) +
										" automatic values were found.");

		// If no automatic dimensions exist, quick return
		if (!m_isDirty && !m_containsAutomatic)
			return *this;

		int64_t autoIndex = 0;
		int64_t prod = 1;

		for (int64_t i = 0; i < m_dims; ++i) {
			if (m_extent[i] < 1)
				autoIndex = i;
			else
				prod *= m_extent[i];
		}

		if (target % prod == 0) {
			Extent res = *this;
			res.m_extent[autoIndex] = target / prod;
			return res;
		}

		throw std::runtime_error("Could not resolve automatic dimension of "
								 " Extent to fit " + std::to_string(target)
								 + " elements");
	}

	Extent Extent::subExtent(int64_t start, int64_t end) const {
		if (start == -1) start = 0;
		if (end == -1) end = m_dims;

		if (start >= end)
			throw std::invalid_argument("Cannot create subExtent from range ["
										+ std::to_string(start) + ", "
										+ std::to_string(end) + ")");

		Extent res(end - start);
		for (int64_t i = start; i < end; ++i)
			res.m_extent[i - start] = m_extent[i];

		return res;
	}

	bool Extent::operator==(const Extent &other) const {
		if (m_dims != other.m_dims)
			return false;

		if (m_containsAutomatic != other.m_containsAutomatic)
			return false;

		for (int64_t i = 0; i < m_dims; ++i) {
			if (m_extent[i] != other.m_extent[i])
				return false;
		}

		return true;
	}

	void Extent::reorder(const std::vector<int64_t> &order) {
		Extent temp = *this;

		for (uint64_t i = 0; i < order.size(); ++i)
			m_extent[i] = temp.m_extent[order[i]];
	}

	std::string Extent::str() const {
		std::stringstream res;
		res << "Extent(";
		for (int64_t i = 0; i < m_dims; ++i) {
			if (m_extent[i] == librapid::AUTO)
				res << "librapid::AUTO";
			else
				res << m_extent[i];

			if (i < m_dims - 1)
				res << ", ";
		}
		res << ")";

		return res.str();
	}

	void Extent::update() {
		int64_t neg = 0;
		m_size = 1;
		for (int64_t i = 0; i < m_dims; i++) {
			m_size *= m_extent[i];

			if (m_extent[i] < 0) {
				neg++;
				m_extent[i] = AUTO;
			}
		}

		if (neg == 1)
			m_containsAutomatic = true;
		else if (neg > 1)
			throw std::invalid_argument("Cannot construct Extent with more than"
										" one automatic values. " +
										std::to_string(neg) +
										" automatic values were found.");
		else
			m_containsAutomatic = false;

		m_isDirty = false;
	}
}