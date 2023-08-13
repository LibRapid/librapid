#ifndef LIBRAPID_ARRAY_STRIDE_TOOLS_HPP
#define LIBRAPID_ARRAY_STRIDE_TOOLS_HPP

namespace librapid {
    namespace typetraits {
        LIBRAPID_DEFINE_AS_TYPE(typename T COMMA size_t N, Stride<T COMMA N>);
    }

    /// A Stride is a vector of integers that describes the distance between elements in each
    /// dimension of an ArrayContainer object. This can be used to access elements in a non-trivial
    /// order, or to access a sub-array of an ArrayContainer object. The Stride class inherits from
    /// the Shape class.
    /// \tparam T The type of the Stride. Must be an integer type.
    /// \tparam N The number of dimensions in the Stride.
    /// \see Shape
    template<typename T = size_t, size_t N = 32>
    class Stride : public Shape<T, N> {
    public:
        /// Default Constructor
        Stride() = default;

        /// Construct a Stride from a Shape object. This will assume that the data represented by
        /// the Shape object is a contiguous block of memory, and will calculate the corresponding
        /// strides based on this.
        /// \param shape
        Stride(const Shape<T, N> &shape);

        /// Copy a Stride object
        /// \param other The Stride object to copy.
        Stride(const Stride &other) = default;

        /// Move a Stride object
        /// \param other The Stride object to move.
        Stride(Stride &&other) noexcept = default;

        /// Assign a Stride object to this Stride object.
        /// \param other The Stride object to assign.
        Stride &operator=(const Stride &other) = default;

        /// Move a Stride object to this Stride object.
        /// \param other The Stride object to move.
        Stride &operator=(Stride &&other) noexcept = default;
    };

    template<typename T, size_t N>
    Stride<T, N>::Stride(const Shape<T, N> &shape) : Shape<T, N>(shape) {
        if (this->m_dims == 0) {
            // Edge case for a zero-dimensional array
            this->m_data[0] = 1;
            return;
        }

        T tmp[N] {0};
        tmp[this->m_dims - 1] = 1;
        for (size_t i = this->m_dims - 1; i > 0; --i) tmp[i - 1] = tmp[i] * this->m_data[i];
        for (size_t i = 0; i < this->m_dims; ++i) this->m_data[i] = tmp[i];
    }
} // namespace librapid

// Support FMT printing
#ifdef FMT_API
LIBRAPID_SIMPLE_IO_IMPL(typename T COMMA size_t N, librapid::Stride<T COMMA N>)
#endif // FMT_API

#endif // LIBRAPID_ARRAY_STRIDE_TOOLS_HPP