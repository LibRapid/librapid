#include <librapid/array/iterators.hpp>

namespace librapid {
	ESIterator::ESIterator() = default;

	ESIterator::ESIterator(pointer start) : m_ptr(start) {}

	ESIterator::~ESIterator() = default;

	ESIterator &ESIterator::operator=(const ESIterator &other) = default;

	ESIterator &ESIterator::operator=(pointer ptr) {
		m_ptr = ptr;
		return *this;
	}

	ESIterator::operator bool() const {
		return m_ptr ? true : false;
	}

	bool ESIterator::operator==(const ESIterator &other) const {
		return m_ptr == other.getConstPointer();
	}

	bool ESIterator::operator!=(const ESIterator &other) const {
		return m_ptr != other.getConstPointer();
	}

	ESIterator &ESIterator::operator+=(const differenceType &movement) {
		m_ptr += movement;
		return (*this);
	}

	ESIterator &ESIterator::operator-=(const differenceType &movement) {
		m_ptr -= movement;
		return (*this);
	}

	ESIterator &ESIterator::operator++() {
		++m_ptr;
		return (*this);
	}

	ESIterator &ESIterator::operator--() {
		--m_ptr;
		return (*this);
	}

	ESIterator ESIterator::operator++(int) {
		auto temp(*this);
		++m_ptr;
		return temp;
	}

	ESIterator ESIterator::operator--(int) {
		auto temp(*this);
		--m_ptr;
		return temp;
	}

	ESIterator ESIterator::operator+(const differenceType &movement) const {
		return ESIterator(m_ptr + movement);
	}

	ESIterator ESIterator::operator-(const differenceType &movement) const {
		return ESIterator(m_ptr - movement);
	}

	ESIterator::differenceType ESIterator::operator-(const ESIterator &rawIterator) const {
		return std::distance(rawIterator.getPointer(), getPointer());
	}

	ESIterator::reference ESIterator::operator*() {
		return *m_ptr;
	}

	const ESIterator::reference ESIterator::operator*() const {
		return *m_ptr;
	}

	const ESIterator::pointer ESIterator::getConstPointer() const {
		return m_ptr;
	}

	ESIterator::pointer ESIterator::getPointer() const {
		return m_ptr;
	}
}