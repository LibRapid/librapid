#ifndef LIBRAPID_ARRAY_ITERATORS
#define LIBRAPID_ARRAY_ITERATORS

#include <librapid/config.hpp>
#include <iterator>

namespace librapid {
	class ESIterator {
	public:
		using iteratorCategory = std::random_access_iterator_tag;
		using valueType = int64_t;
		using differenceType = std::ptrdiff_t;
		using pointer = valueType *;
		using reference = valueType &;

		/**
		 * \rst
		 *
		 * Default constructor for the ESIterator object. It will not iterate
		 * over anything.
		 *
		 * \endrst
		 */
		ESIterator();

		/**
		 * \rst
		 *
		 * Create an ESIterator from a pointer
		 *
		 * \endrst
		 */
		ESIterator(pointer start);

		~ESIterator();

		/**
		 * \rst
		 *
		 * Set one ESIterator equal to another
		 *
		 * \endrst
		 */
		ESIterator &operator=(const ESIterator &other);

		/**
		 * \rst
		 *
		 * Set one ESIterator equal to a pointer
		 *
		 * \endrst
		 */
		ESIterator &operator=(pointer ptr);

		/**
		 * \rst
		 *
		 * Returns ``true`` if this ESIterator's pointer is valid (i.e. not
		 * ``nullptr``), otherwise ``false``
		 *
		 *
		 * \endrst
		 */
		operator bool() const;

		/**
		 * \rst
		 *
		 * Returns true if this ESIterator's pointer exactly matches the other's
		 * value.
		 *
		 * \endrst
		 */
		bool operator==(const ESIterator &other) const;

		/**
		 * \rst
		 *
		 * Returns true if this ESIterator's pointer does not match the other's
		 * value.
		 *
		 * \endrst
		 */
		bool operator!=(const ESIterator &other) const;

		/**
		 * \rst
		 *
		 * Increment the ESIterator by a specified offset
		 *
		 * \endrst
		 */
		ESIterator &operator+=(const differenceType &movement);

		/**
		 * \rst
		 *
		 * Decrement the ESIterator by a specified offset
		 *
		 * \endrst
		 */
		ESIterator &operator-=(const differenceType &movement);

		/**
		 * \rst
		 *
		 * Increment this ESIterator by one and return the result
		 *
		 * \endrst
		 */
		ESIterator &operator++();

		/**
		 * \rst
		 *
		 * Decrement this ESIterator by one and return the result
		 *
		 * \endrst
		 */
		ESIterator &operator--();

		/**
		 * \rst
		 *
		 * Increment this ESIterator by one and return the unaltered value
		 *
		 * \endrst
		 */
		ESIterator operator++(int);

		/**
		 * \rst
		 *
		 * Decrement this ESIterator by one and return the unaltered value
		 *
		 * \endrst
		 */
		ESIterator operator--(int);

		/**
		 * \rst
		 *
		 * Return a new ESIterator whose pointer is ``movement`` objects further
		 * through memory
		 *
		 * \endrst
		 */
		ESIterator operator+(const differenceType &movement) const;

		/**
		 * \rst
		 *
		 * Return a new ESIterator whose pointer is ``movement`` objects before
		 * ``*this->m_ptr`` in memory.
		 *
		 * \endrst
		 */
		ESIterator operator-(const differenceType &movement) const;

		/**
		 * \rst
		 *
		 * Return the distance (in bytes) between two ESIterator objects
		 *
		 * \endrst
		 */
		differenceType operator-(const ESIterator &rawIterator) const;

		/**
		 * \rst
		 *
		 * Dereference this ESIterator's pointer
		 *
		 * \endrst
		 */
		reference operator*();

		/**
		* \rst
		*
		* Dereference this ESIterator's pointer (const)
		*
		* \endrst
		*/
		const reference operator*() const;

		pointer operator->() {
			return m_ptr;
		}

		/**
		 * \rst
		 *
		 * Return the ESIterator's pointer in ``const`` form
		 *
		 * \endrst
		 */
		const pointer getConstPointer() const;

		/**
		 * \rst
		 *
		 * Return the ESIterator's pointer
		 *
		 * \endrst
		 */
		pointer getPointer() const;

	private:
		pointer m_ptr = nullptr;
	};
}

#endif // LIBRAPID_ARRAY_ITERATORS