#include <librapid/config.hpp>
#include <librapid/array/multiarray.hpp>
#include <librapid/array/array_iterator.hpp>
#include <librapid/math/rapid_math.hpp>

namespace librapid {
	template<>
	AIterator<Array>::AIterator() = default;

	template<>
	AIterator<Array>::~AIterator() {
		// if (m_decrementOnFree) m_array._decrement();
	}

	template<>
	AIterator<Array>::AIterator(const valueType &arr, indexType index) :
			m_array(arr), m_index(index) {}

	template<>
	AIterator<Array>::AIterator(indexType index) : m_index(index) {}

	template<>
	AIterator<Array> &
	AIterator<Array>::operator=(const AIterator<Array> &other) = default;

	template<>
	AIterator<Array>::operator bool() const {
		return m_index >= 0 && m_index < m_array.len();
	}

	template<>
	bool AIterator<Array>::operator==(const AIterator<Array> &other) const {
		// return m_array.isSame(other.m_array) && m_index == other.m_index;
		return m_index == other.m_index;
	}

	template<>
	bool AIterator<Array>::operator!=(const AIterator<Array> &other) const {
		return !(*this == other);
	}

	template<>
	AIterator<Array> &
	AIterator<Array>::operator+=(const differenceType &movement) {
		m_index += movement;
		return *this;
	}

	template<>
	AIterator<Array> &
	AIterator<Array>::operator-=(const differenceType &movement) {
		m_index -= movement;
		return *this;
	}

	template<>
	AIterator<Array> &AIterator<Array>::operator++() {
		++m_index;
		return *this;
	}

	template<>
	AIterator<Array> &AIterator<Array>::operator--() {
		--m_index;
		return *this;
	}

	template<>
	AIterator<Array> AIterator<Array>::operator++(int) {
		auto temp(*this);
		++m_index;
		return temp;
	}

	template<>
	AIterator<Array> AIterator<Array>::operator--(int) {
		auto temp(*this);
		--m_index;
		return temp;
	}

	template<>
	AIterator<Array>
	AIterator<Array>::operator+(const differenceType &movement) const {
		return AIterator(m_array, m_index + movement);
	}

	template<>
	AIterator<Array>
	AIterator<Array>::operator-(const differenceType &movement) const {
		return AIterator(m_array, m_index - movement);
	}

	template<>
	AIterator<Array>::differenceType
	AIterator<Array>::operator-(const AIterator<Array> &rawIterator) const {
		return abs(rawIterator.m_index - m_index);
	}

	template<>
	AIterator<Array>::valueType AIterator<Array>::operator*() {
		return m_array[m_index];
	}

	template<>
	AIterator<Array>::valueType AIterator<Array>::operator*() const {
		return m_array[m_index];
	}
} // namespace librapid