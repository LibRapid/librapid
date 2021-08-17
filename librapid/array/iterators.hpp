#ifndef LIBRAPID_ARRAY_ITERATORS
#define LIBRAPID_ARRAY_ITERATORS

#include <librapid/config.hpp>
#include <iterator>

namespace librapid
{
	class ESIterator
	{
	public:
		using iteratorCategory = std::random_access_iterator_tag;
		using valueType = lr_int;
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
		ESIterator() = default;

		/**
		 * \rst
		 * 
		 * Create an ESIterator from a pointer
		 * 
		 * \endrst
		 */
		ESIterator(pointer start)
		{
			m_ptr = start;
		}

		~ESIterator() = default;

		/**
		 * \rst
		 * 
		 * Set one ESIterator equal to another
		 * 
		 * \endrst
		 */
		LR_INLINE ESIterator &operator=(const ESIterator &other) = default;

		/**
		 * \rst
		 * 
		 * Set one ESIterator equal to a pointer
		 * 
		 * \endrst
		 */
		LR_INLINE ESIterator &operator=(pointer ptr)
		{
			m_ptr = ptr;
			return *this;
		}

		/**
		 * \rst
		 * 
		 * Returns ``true`` if this ESIterator's pointer is valid (i.e. not
		 * ``nullptr``), otherwise ``false``
		 * 
		 * 
		 * \endrst
		 */
		LR_INLINE operator bool() const
		{
			return m_ptr ? true : false;
		}

		/**
		 * \rst
		 * 
		 * Returns true if this ESIterator's pointer exactly matches the other's
		 * value.
		 * 
		 * \endrst
		 */
		LR_INLINE bool operator==(const ESIterator &other) const
		{
			return m_ptr == other.getConstPointer();
		}

		/**
		 * \rst
		 * 
		 * Returns true if this ESIterator's pointer does not match the other's
		 * value.
		 * 
		 * \endrst
		 */
		LR_INLINE bool operator!=(const ESIterator &other) const
		{
			return m_ptr != other.getConstPointer();
		}

		/**
		 * \rst
		 *
		 * Increment the ESIterator by a specified offset
		 *
		 * \endrst
		 */
		LR_INLINE ESIterator &operator+=(const differenceType &movement)
		{
			m_ptr += movement;
			return (*this);
		}

		/**
		 * \rst
		 * 
		 * Decrement the ESIterator by a specified offset
		 * 
		 * \endrst
		 */
		LR_INLINE ESIterator &operator-=(const differenceType &movement)
		{
			m_ptr -= movement;
			return (*this);
		}

		/**
		 * \rst
		 * 
		 * Increment this ESIterator by one and return the result
		 * 
		 * \endrst
		 */
		LR_INLINE ESIterator &operator++()
		{
			++m_ptr;
			return (*this);
		}

		/**
		 * \rst
		 * 
		 * Decrement this ESIterator by one and return the result
		 * 
		 * \endrst
		 */
		LR_INLINE ESIterator &operator--()
		{
			--m_ptr;
			return (*this);
		}

		/**
		 * \rst
		 * 
		 * Increment this ESIterator by one and return the unaltered value
		 * 
		 * \endrst
		 */
		LR_INLINE ESIterator operator++(int)
		{
			auto temp(*this);
			++m_ptr;
			return temp;
		}

		/**
		 * \rst
		 * 
		 * Decrement this ESIterator by one and return the unaltered value
		 * 
		 * \endrst
		 */
		LR_INLINE ESIterator operator--(int)
		{
			auto temp(*this);
			--m_ptr;
			return temp;
		}

		/**
		 * \rst
		 * 
		 * Return a new ESIterator whose pointer is ``movement`` objects further
		 * through memory
		 * 
		 * \endrst
		 */
		LR_INLINE ESIterator operator+(const differenceType &movement) const
		{
			return ESIterator(m_ptr + movement);
		}

		/**
		 * \rst
		 * 
		 * Return a new ESIterator whose pointer is ``movement`` objects before
		 * ``*this->m_ptr`` in memory.
		 * 
		 * \endrst
		 */
		LR_INLINE ESIterator operator-(const differenceType &movement) const
		{
			return ESIterator(m_ptr - movement);
		}

		/**
		 * \rst
		 * 
		 * Return the distance (in bytes) between two ESIterator objects
		 * 
		 * \endrst
		 */
		LR_INLINE differenceType operator-(const ESIterator &rawIterator) const
		{
			return std::distance(rawIterator.getPointer(), getPointer());
		}

		/**
		 * \rst
		 *
		 * Dereference this ESIterator's pointer
		 *
		 * \endrst
		 */
		LR_INLINE reference operator*()
		{
			return *m_ptr;
		}

		/**
		* \rst
		*
		* Dereference this ESIterator's pointer (const)
		*
		* \endrst
		*/
		LR_INLINE const reference operator*() const
		{
			return *m_ptr;
		}

		LR_INLINE pointer operator->()
		{
			return m_ptr;
		}

		/**
		 * \rst
		 *
		 * Return the ESIterator's pointer in ``const`` form
		 *
		 * \endrst
		 */
		LR_INLINE const pointer getConstPointer() const
		{
			return m_ptr;
		}

		/**
		 * \rst
		 *
		 * Return the ESIterator's pointer
		 *
		 * \endrst
		 */
		LR_INLINE pointer getPointer() const
		{
			return m_ptr;
		}

	private:
		pointer m_ptr = nullptr;
	};
}

#endif // LIBRAPID_ARRAY_ITERATORS