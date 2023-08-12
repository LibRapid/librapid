#ifndef LIBRAPID_ARRAY_ITERATOR_HPP
#define LIBRAPID_ARRAY_ITERATOR_HPP

namespace librapid::detail {
    template<typename T>
    class ArrayIterator {
    public:
        using IndexType = int64_t;

        /// Default constructor should never be used
        ArrayIterator() = delete;

        explicit ArrayIterator(const T &array);

        explicit ArrayIterator(const T &array, IndexType index);

        /// Copy an ArrayIterator object (const)
        /// \param other The array to copy
        ArrayIterator(const ArrayIterator &other) = default;

        /// Constructs an ArrayIterator from a temporary instance
        /// \param other The ArrayIterator to move
        ArrayIterator(ArrayIterator &&other) = default;

        /// Assigns another ArrayIterator object to this ArrayIterator.
        /// \param other The ArrayIterator to assign.
        /// \return A reference to this
        ArrayIterator &operator=(const ArrayIterator &other) = default;

        ArrayIterator &operator++();
        bool operator==(const ArrayIterator<T> &other) const;
        bool operator!=(const ArrayIterator<T> &other) const;

        auto operator*() const;
        auto operator*();

        ArrayIterator<ArrayIterator<T>> begin() const noexcept;
        ArrayIterator<ArrayIterator<T>> end() const noexcept;

        ArrayIterator<ArrayIterator<T>> begin();
        ArrayIterator<ArrayIterator<T>> end();

    private:
        T m_array;
        IndexType m_index;
    };

    template<typename T>
    ArrayIterator<T>::ArrayIterator(const T &array) : m_array(array), m_index(0) {}

    template<typename T>
    ArrayIterator<T>::ArrayIterator(const T &array, IndexType index) :
            m_array(array), m_index(index) {}

    template<typename T>
    ArrayIterator<T> &ArrayIterator<T>::operator++() {
        ++m_index;
        return *this;
    }

    template<typename T>
    bool ArrayIterator<T>::operator==(const ArrayIterator<T> &other) const {
        return m_index == other.m_index;
    }

    template<typename T>
    bool ArrayIterator<T>::operator!=(const ArrayIterator<T> &other) const {
        return !(this->operator==(other));
    }

    template<typename T>
    auto ArrayIterator<T>::operator*() const {
        return m_array[m_index];
    }

    template<typename T>
    auto ArrayIterator<T>::operator*() {
        return m_array[m_index];
    }

    template<typename T>
    auto ArrayIterator<T>::begin() const noexcept -> ArrayIterator<ArrayIterator<T>> {
        return ArrayIterator<ArrayIterator<T>>(*this, 0);
    }

    template<typename T>
    auto ArrayIterator<T>::end() const noexcept -> ArrayIterator<ArrayIterator<T>> {
        return ArrayIterator<ArrayIterator<T>>(*this, m_array.shape()[0]);
    }

    template<typename T>
    auto ArrayIterator<T>::begin() -> ArrayIterator<ArrayIterator<T>> {
        return ArrayIterator<ArrayIterator<T>>(*this, 0);
    }

    template<typename T>
    auto ArrayIterator<T>::end() -> ArrayIterator<ArrayIterator<T>> {
        return ArrayIterator<ArrayIterator<T>>(*this, m_array.shape()[0]);
    }
} // namespace librapid::detail

#endif // LIBRAPID_ARRAY_ITERATOR_HPP