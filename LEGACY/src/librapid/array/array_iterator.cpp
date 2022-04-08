#include <librapid/config.hpp>
#include <librapid/array/multiarray.hpp>
#include <librapid/array/array_iterator.hpp>
#include <librapid/math/rapid_math.hpp>

namespace librapid {
	AIterator::AIterator() = default;

	AIterator::~AIterator() = default;

	AIterator::AIterator(const valueType &arr, indexType index) :
			m_array(arr), m_index(index) {}

	AIterator::AIterator(indexType index) : m_index(index) {}

	AIterator &AIterator::operator=(const AIterator &other) = default;

	AIterator::operator bool() const {
		return m_index >= 0 && m_index < m_array.len();
	}

	bool AIterator::operator==(const AIterator &other) const {
		return m_index == other.m_index;
	}

	bool AIterator::operator!=(const AIterator &other) const {
		return !(*this == other);
	}

	AIterator &AIterator::operator+=(const differenceType &movement) {
		m_index += movement;
		return *this;
	}

	AIterator &AIterator::operator-=(const differenceType &movement) {
		m_index -= movement;
		return *this;
	}

	AIterator &AIterator::operator++() {
		++m_index;
		return *this;
	}

	AIterator &AIterator::operator--() {
		--m_index;
		return *this;
	}

	AIterator AIterator::operator++(int) {
		AIterator temp(m_array, m_index);
		++m_index;
		return temp;
	}

	AIterator AIterator::operator--(int) {
		AIterator temp(m_array, m_index);
		--m_index;
		return temp;
	}

	AIterator AIterator::operator+(const differenceType &movement) const {
		return AIterator(m_array, m_index + movement);
	}

	AIterator AIterator::operator-(const differenceType &movement) const {
		return AIterator(m_array, m_index - movement);
	}

	AIterator::differenceType
	AIterator::operator-(const AIterator &rawIterator) const {
		return abs(rawIterator.m_index - m_index);
	}

	AIterator::valueType AIterator::operator*() { return m_array[m_index]; }

	AIterator::valueType AIterator::operator*() const {
		return m_array[m_index];
	}

	AIterator Array::begin() const {
		return AIterator(*this);
	}

	AIterator Array::end() const { return AIterator(len()); }
} // namespace librapid