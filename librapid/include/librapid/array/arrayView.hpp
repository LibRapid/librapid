#ifndef LIBRAPID_ARRAY_ARRAY_VIEW_HPP
#define LIBRAPID_ARRAY_ARRAY_VIEW_HPP

namespace librapid {
	namespace typetraits {
		template<typename T>
		struct TypeInfo<array::ArrayView<T>> {
			detail::LibRapidType type				 = detail::LibRapidType::ArrayView;
			using Scalar							 = typename TypeInfo<T>::Scalar;
			using Device							 = typename TypeInfo<T>::Device;
			static constexpr bool allowVectorisation = false;
		};
	} // namespace typetraits

	namespace array {
		template<typename T>
		class ArrayView {
		public:
			using ArrayType		 = T;
			using Reference		 = ArrayType &;
			using ConstReference = const ArrayType &;
			using ConstReference = ArrayType const &;
			using StrideType	 = typename ArrayType::StrideType;
			using ShapeType		 = typename ArrayType::ShapeType;
			using Scalar		 = typename typetraits::TypeInfo<ArrayType>::Scalar;

			ArrayView() = delete;
			ArrayView(const ArrayType &array);
			ArrayView(const ArrayView &other) = default;

			/// Constructs an ArrayView from a temporary ArrayView.
			/// \param other The ArrayView to move.
			ArrayView(ArrayView &&other) = default;

			/// Assigns an ArrayView to this ArrayView.
			/// \param other The ArrayView to assign.
			/// \return A reference to this ArrayView.
			ArrayView &operator=(const ArrayView &other) = default;

			/// Assigns a temporary ArrayView to this ArrayView.
			/// \param other The ArrayView to move.
			/// \return A reference to this ArrayView.
			ArrayView &operator=(ArrayView &&other) = default;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ShapeType shape() const;
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE StrideType stride() const;
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE int64_t offset() const;

			void setShape(const ShapeType &shape);
			void setStride(const StrideType &stride);
			void setOffset(const int64_t &offset);

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ArrayType eval() const;

		private:
			ConstReference m_ref;
			ShapeType m_shape;
			StrideType m_stride;
			int64_t m_offset = 0;
		};

		// template<typename T>
		// ArrayView<T>::ArrayView(ArrayType &array) :
		// 		m_ref(array), m_shape(array.shape()), m_stride(array.shape()) {}

		template<typename T>
		ArrayView<T>::ArrayView(const ArrayType &array) :
				m_ref(array), m_shape(array.shape()), m_stride(array.shape()) {}

		template<typename T>
		auto ArrayView<T>::shape() const -> ShapeType {
			return m_shape;
		}

		template<typename T>
		auto ArrayView<T>::stride() const -> StrideType {
			return m_stride;
		}

		template<typename T>
		auto ArrayView<T>::offset() const -> int64_t {
			return m_offset;
		}

		template<typename T>
		void ArrayView<T>::setShape(const ShapeType &shape) {
			m_shape = shape;
		}

		template<typename T>
		void ArrayView<T>::setStride(const StrideType &stride) {
			m_stride = stride;
		}

		template<typename T>
		void ArrayView<T>::setOffset(const int64_t &offset) {
			m_offset = offset;
		}

		template<typename T>
		auto ArrayView<T>::eval() const -> ArrayType {
			ArrayType res(m_shape);
			ShapeType coord = ShapeType::zeros(m_shape.ndim());
			int64_t d = 0, p = 0;
			int64_t idim = 0, adim = 0;
			const int64_t ndim = m_shape.ndim();

			do {
				res.storage()[d++] = m_ref.scalar(p + m_offset);

				for (idim = 0; idim < ndim; ++idim) {
					adim = ndim - idim - 1;
					if (++coord[adim] == m_shape[adim]) {
						coord[adim] = 0;
						p			= p - (m_shape[adim] - 1) * m_stride[adim];
					} else {
						p = p + m_stride[adim];
						break;
					}
				}
			} while (idim < ndim);

			return res;
		}
	} // namespace array
} // namespace librapid

#endif // LIBRAPID_ARRAY_ARRAY_VIEW_HPP