#ifndef LIBRAPID_ARRAY_ITERATORS
#define LIBRAPID_ARRAY_ITERATORS

#include <iterator>
#include <librapid/config.hpp>

namespace librapid {
	class Array;

	// An iterator for the Array class
	// template<typename T>
	class AIterator {
	public:
		using iteratorCategory = std::random_access_iterator_tag;
		using valueType		   = Array;
		using differenceType   = uint64_t;
		using indexType		   = int64_t;
		using reference		   = valueType &;

		/**
		 * \rst
		 *
		 * Default constructor for the AIterator object. It will not iterate
		 * over anything.
		 *
		 * \endrst
		 */
		AIterator();

		/**
		 * \rst
		 *
		 * Create an AIterator from an Array object
		 *
		 * \endrst
		 */
		explicit AIterator(const valueType &arr, indexType index = 0);

		explicit AIterator(indexType index = 0);

		~AIterator();

		/**
		 * \rst
		 *
		 * Set one AIterator equal to another
		 *
		 * \endrst
		 */
		AIterator &operator=(const AIterator &other);

		/**
		 * \rst
		 *
		 * Returns ``true`` if this AIterator's pointer is valid (i.e. not
		 * ``nullptr``), otherwise ``false``
		 *
		 *
		 * \endrst
		 */
		operator bool() const;

		/**
		 * \rst
		 *
		 * Returns true if this AIterator's pointer exactly matches the other's
		 * value.
		 *
		 * \endrst
		 */
		bool operator==(const AIterator &other) const;

		/**
		 * \rst
		 *
		 * Returns true if this AIterator's pointer does not match the other's
		 * value.
		 *
		 * \endrst
		 */
		bool operator!=(const AIterator &other) const;

		/**
		 * \rst
		 *
		 * Increment the AIterator by a specified offset
		 *
		 * \endrst
		 */
		AIterator &operator+=(const differenceType &movement);

		/**
		 * \rst
		 *
		 * Decrement the AIterator by a specified offset
		 *
		 * \endrst
		 */
		AIterator &operator-=(const differenceType &movement);

		/**
		 * \rst
		 *
		 * Increment this AIterator by one and return the result
		 *
		 * \endrst
		 */
		AIterator &operator++();

		/**
		 * \rst
		 *
		 * Decrement this AIterator by one and return the result
		 *
		 * \endrst
		 */
		AIterator &operator--();

		/**
		 * \rst
		 *
		 * Increment this AIterator by one and return the unaltered value
		 *
		 * \endrst
		 */
		AIterator operator++(int);

		/**
		 * \rst
		 *
		 * Decrement this AIterator by one and return the unaltered value
		 *
		 * \endrst
		 */
		AIterator operator--(int);

		/**
		 * \rst
		 *
		 * Return a new AIterator whose pointer is ``movement`` objects further
		 * through memory
		 *
		 * \endrst
		 */
		AIterator operator+(const differenceType &movement) const;

		/**
		 * \rst
		 *
		 * Return a new AIterator whose pointer is ``movement`` objects before
		 * ``*this->m_ptr`` in memory.
		 *
		 * \endrst
		 */
		AIterator operator-(const differenceType &movement) const;

		/**
		 * \rst
		 *
		 * Return the distance (in bytes) between two AIterator objects
		 *
		 * \endrst
		 */
		differenceType operator-(const AIterator &rawIterator) const;

		/**
		 * \rst
		 *
		 * Dereference this AIterator's pointer
		 *
		 * \endrst
		 */
		valueType operator*();

		/**
		 * \rst
		 *
		 * Dereference this AIterator's pointer (const)
		 *
		 * \endrst
		 */
		valueType operator*() const;

	private:
		indexType m_index = 0;
		valueType m_array;
	};
} // namespace librapid

#endif // LIBRAPID_ARRAY_ITERATORS