#ifndef LIBRAPID_ARRAY_ITERATOR_HPP
#define LIBRAPID_ARRAY_ITERATOR_HPP

namespace librapid::detail {
	template<typename T>
	class ArrayIterator {
	public:
		using IndexType = int64_t;

		/// Default constructor should never be used
		ArrayIterator() = delete;

		explicit LIBRAPID_ALWAYS_INLINE ArrayIterator(const T &array);

		explicit LIBRAPID_ALWAYS_INLINE ArrayIterator(const T &array, IndexType index);

		/// Copy an ArrayIterator object (const)
		/// \param other The array to copy
		LIBRAPID_ALWAYS_INLINE ArrayIterator(const ArrayIterator &other) = default;

		/// Constructs an ArrayIterator from a temporary instance
		/// \param other The ArrayIterator to move
		LIBRAPID_ALWAYS_INLINE ArrayIterator(ArrayIterator &&other) = default;

		/// Assigns another ArrayIterator object to this ArrayIterator.
		/// \param other The ArrayIterator to assign.
		/// \return A reference to this
		LIBRAPID_ALWAYS_INLINE ArrayIterator &operator=(const ArrayIterator &other) = default;

		LIBRAPID_ALWAYS_INLINE ArrayIterator &operator++();
		LIBRAPID_ALWAYS_INLINE bool operator==(const ArrayIterator<T> &other) const;
		LIBRAPID_ALWAYS_INLINE bool operator!=(const ArrayIterator<T> &other) const;

		LIBRAPID_ALWAYS_INLINE auto operator*() const;
		LIBRAPID_ALWAYS_INLINE auto operator*();

		LIBRAPID_ALWAYS_INLINE ArrayIterator<ArrayIterator<T>> begin() const noexcept;
		LIBRAPID_ALWAYS_INLINE ArrayIterator<ArrayIterator<T>> end() const noexcept;

		LIBRAPID_ALWAYS_INLINE ArrayIterator<ArrayIterator<T>> begin();
		LIBRAPID_ALWAYS_INLINE ArrayIterator<ArrayIterator<T>> end();

	private:
		T m_array;
		IndexType m_index;
	};

	template<typename T>
	LIBRAPID_ALWAYS_INLINE ArrayIterator<T>::ArrayIterator(const T &array) :
			m_array(array), m_index(0) {}

	template<typename T>
	LIBRAPID_ALWAYS_INLINE ArrayIterator<T>::ArrayIterator(const T &array, IndexType index) :
			m_array(array), m_index(index) {}

	template<typename T>
	LIBRAPID_ALWAYS_INLINE ArrayIterator<T> &ArrayIterator<T>::operator++() {
		++m_index;
		return *this;
	}

	template<typename T>
	LIBRAPID_ALWAYS_INLINE bool ArrayIterator<T>::operator==(const ArrayIterator<T> &other) const {
		return m_index == other.m_index;
	}

	template<typename T>
	LIBRAPID_ALWAYS_INLINE bool ArrayIterator<T>::operator!=(const ArrayIterator<T> &other) const {
		return !(this->operator==(other));
	}

	template<typename T>
	LIBRAPID_ALWAYS_INLINE auto ArrayIterator<T>::operator*() const {
		return m_array[m_index];
	}

	template<typename T>
	LIBRAPID_ALWAYS_INLINE auto ArrayIterator<T>::operator*() {
		return m_array[m_index];
	}

	template<typename T>
	LIBRAPID_ALWAYS_INLINE auto ArrayIterator<T>::begin() const noexcept
	  -> ArrayIterator<ArrayIterator<T>> {
		return ArrayIterator<ArrayIterator<T>>(*this, 0);
	}

	template<typename T>
	LIBRAPID_ALWAYS_INLINE auto ArrayIterator<T>::end() const noexcept
	  -> ArrayIterator<ArrayIterator<T>> {
		return ArrayIterator<ArrayIterator<T>>(*this, m_array.shape()[0]);
	}

	template<typename T>
	LIBRAPID_ALWAYS_INLINE auto ArrayIterator<T>::begin() -> ArrayIterator<ArrayIterator<T>> {
		return ArrayIterator<ArrayIterator<T>>(*this, 0);
	}

	template<typename T>
	LIBRAPID_ALWAYS_INLINE auto ArrayIterator<T>::end() -> ArrayIterator<ArrayIterator<T>> {
		return ArrayIterator<ArrayIterator<T>>(*this, m_array.shape()[0]);
	}
} // namespace librapid::detail

#endif // LIBRAPID_ARRAY_ITERATOR_HPP