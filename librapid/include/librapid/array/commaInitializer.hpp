#ifndef LIBRAPID_ARRAY_COMMA_INITIALIZER_HPP
#define LIBRAPID_ARRAY_COMMA_INITIALIZER_HPP

namespace librapid::detail {
    /// Allows for an Array object to be initialized with a comma separated list of values. While
    /// this is not particularly useful for large arrays, it is a very quick and easy way to
    /// initialize smaller arrays with a few values.
    /// \tparam ArrT The type of the Array object to be initialized.
    template<typename ArrT>
    class CommaInitializer {
    public:
        /// The scalar type of the Array object.
        using Scalar = typename typetraits::TypeInfo<ArrT>::Scalar;

        CommaInitializer() = delete;

        /// Construct a CommaInitializer from an Array object.
        /// \param dst The Array object to initialize.
        /// \param val The first value to initialize the Array object with.
        template<typename T>
        explicit CommaInitializer(ArrT &dst, const T &val) : m_array(dst) {
            next(static_cast<Scalar>(val));
        }

        /// Initialize the next element of the Array object.
        template<typename T>
        CommaInitializer &operator,(const T &val) {
            next(static_cast<Scalar>(val));
            return *this;
        }

    private:
        /// Initialize the current element of the Array and increment the index.
        /// \param other The value to initialize the current element with.
        void next(const Scalar &other) {
            m_array.storage()[m_index] = other;
            ++m_index;
        }

        /// The Array object to initialize.
        ArrT &m_array;

        /// The current index of the Array object.
        int64_t m_index = 0;
    };
} // namespace librapid::detail

#endif // LIBRAPID_ARRAY_COMMA_INITIALIZER_HPP