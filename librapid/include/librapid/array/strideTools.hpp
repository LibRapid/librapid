#ifndef LIBRAPID_ARRAY_STRIDE_TOOLS_HPP
#define LIBRAPID_ARRAY_STRIDE_TOOLS_HPP

namespace librapid {
	template<typename T = size_t, size_t N = 32>
	class Stride : public Shape<T, N> {
	public:
		Stride() = default;
		Stride(const Shape<T, N> &shape);
		Stride(const Stride &other)		= default;
		Stride(Stride &&other) noexcept = default;

		Stride &operator=(const Stride &other)	   = default;
		Stride &operator=(Stride &&other) noexcept = default;

	private:
	};

	template<typename T, size_t N>
	Stride<T, N>::Stride(const Shape<T, N> &shape) : Shape(shape) {
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