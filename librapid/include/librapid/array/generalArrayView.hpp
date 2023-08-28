#ifndef LIBRAPID_ARRAY_ARRAY_VIEW_HPP
#define LIBRAPID_ARRAY_ARRAY_VIEW_HPP

namespace librapid {
	namespace typetraits {
		template<typename T, typename S>
		struct TypeInfo<array::GeneralArrayView<T, S>> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::GeneralArrayView;
			using Scalar							   = typename TypeInfo<std::decay_t<T>>::Scalar;
			using Backend		= typename TypeInfo<std::decay_t<T>>::Backend;
			using ArrayViewType = std::decay_t<T>;
			using ShapeType		= typename TypeInfo<ArrayViewType>::ShapeType;
			using StorageType	= typename TypeInfo<ArrayViewType>::StorageType;
			static constexpr bool allowVectorisation = false;
		};

		LIBRAPID_DEFINE_AS_TYPE(typename T COMMA typename S, array::GeneralArrayView<T COMMA S>);
	} // namespace typetraits

	template<typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto createGeneralArrayView(T &&array) {
		using ShapeType = typename std::decay_t<T>::ShapeType;
		return array::GeneralArrayView<T, ShapeType>(std::forward<T>(array));
	}

	template<typename ShapeType, typename T>
	LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto createGeneralArrayViewShapeModifier(T &&array) {
		return array::GeneralArrayView<T, ShapeType>(std::forward<T>(array));
	}

	namespace array {
		template<typename ArrayViewType,
				 typename ArrayViewShapeType = typename std::decay_t<ArrayViewType>::ShapeType>
		class GeneralArrayView {
		public:
			using BaseType		 = typename std::decay_t<ArrayViewType>;
			using Scalar		 = typename typetraits::TypeInfo<BaseType>::Scalar;
			using Reference		 = BaseType &;
			using ConstReference = const BaseType &;
			using Backend		 = typename typetraits::TypeInfo<BaseType>::Backend;
			using ShapeType		 = ArrayViewShapeType;
			using StrideType	 = Stride<ShapeType>;
			using StorageType	 = typename typetraits::TypeInfo<BaseType>::StorageType;
			using ArrayType = array::ArrayContainer<ShapeType, StorageType>;
			using Iterator	= detail::ArrayIterator<GeneralArrayView>;

			/// Default constructor should never be used
			GeneralArrayView() = delete;

			/// Copy an ArrayView object
			/// \param array The array to copy
			// LIBRAPID_ALWAYS_INLINE GeneralArrayView(ArrayViewType &array);

			/// Copy an ArrayView object (not const)
			/// \param array The array to copy
			LIBRAPID_ALWAYS_INLINE GeneralArrayView(ArrayViewType &&array);

			/// Copy an ArrayView object (const)
			/// \param other The array to copy
			LIBRAPID_ALWAYS_INLINE GeneralArrayView(const GeneralArrayView &other) = default;

			/// Constructs an ArrayView from a temporary instance
			/// \param other The ArrayView to move
			LIBRAPID_ALWAYS_INLINE GeneralArrayView(GeneralArrayView &&other) = default;

			/// Assigns another ArrayView object to this ArrayView.
			/// \param other The ArrayView to assign.
			/// \return A reference to this
			LIBRAPID_ALWAYS_INLINE GeneralArrayView &
			operator=(const GeneralArrayView &other) = default;

			/// Assigns a temporary ArrayView to this ArrayView.
			/// \param other The ArrayView to move.
			/// \return A reference to this ArrayView.
			GeneralArrayView &operator=(GeneralArrayView &&other) noexcept = default;

			/// Assign a scalar value to this ArrayView. This function should only be used to
			/// assign to a zero-dimensional "scalar" ArrayView, and will throw an error if used
			/// incorrectly.
			/// \param scalar The scalar value to assign
			/// \return A reference to this
			LIBRAPID_ALWAYS_INLINE GeneralArrayView &operator=(const Scalar &scalar);

			template<typename ShapeType_, typename StorageType_>
			LIBRAPID_ALWAYS_INLINE GeneralArrayView &
			operator=(const ArrayContainer<ShapeType_, StorageType_> &other);

			/// Access a sub-array of this ArrayView.
			/// \param index The index of the sub-array.
			/// \return An ArrayView from this
			LIBRAPID_ALWAYS_INLINE const auto operator[](int64_t index) const;

			LIBRAPID_ALWAYS_INLINE auto operator[](int64_t index);

			/// Since even scalars are represented as an ArrayView object, it can be difficult to
			/// operate on them directly. This allows you to extract the scalar value stored by a
			/// zero-dimensional ArrayView object
			/// \tparam CAST Type to cast to
			/// \return The scalar represented by the ArrayView object
			template<typename CAST = Scalar>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE CAST get() const;

			/// Same functionality as "get", except slightly less robust for user-defined types.
			/// \tparam CAST Type to cast to
			/// \return The scalar represented by the ArrayView object
			template<typename CAST>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE explicit operator CAST() const;

			/// Access the underlying shape of this ArrayView
			/// \return Shape object
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ShapeType shape() const;

			/// Access the stride of this ArrayView
			/// \return Stride object
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE StrideType stride() const;

			/// Access the offset of this ArrayView. This is the offset, in elements, from the
			/// referenced Array's first element.
			/// \return Offset
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE int64_t offset() const;

			/// Set the Shape of this ArrayView to something else. Intended for internal use only.
			/// \param shape The new shape of this ArrayView
			LIBRAPID_ALWAYS_INLINE void setShape(const ShapeType &shape);

			/// Set the Stride of this ArrayView to something else. Intended for internal use only.
			/// \param stride The new stride of this ArrayView
			LIBRAPID_ALWAYS_INLINE void setStride(const StrideType &stride);

			/// Set the offset of this ArrayView object. Intended for internal use only.
			/// \param offset The new offset of this ArrayView
			LIBRAPID_ALWAYS_INLINE void setOffset(const int64_t &offset);

			/// Returns the number of dimensions of this ArrayView
			/// \return Number of dimensions
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE int64_t ndim() const;

			/// Return the Scalar at a given index in this ArrayView. This is intended for use
			/// internally, but can be used externally too.
			/// \param index The index of the Scalar to return
			/// \return Scalar at the given index
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto scalar(int64_t index) const;

			/// Evaluate the contents of this ArrayView object and return an Array instance from
			/// it. Depending on your use case, this may result in more performant code, but the new
			/// Array will not reference the original data in the ArrayView.
			/// \return A new Array instance
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ArrayType eval() const;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Iterator begin() const;
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Iterator end() const;

			template<typename T, typename Char, typename Ctx>
			void str(const fmt::formatter<T, Char> &format, char bracket, char separator,
					 Ctx &ctx) const;

		private:
			ArrayViewType m_ref;
			ShapeType m_shape;
			StrideType m_stride;
			int64_t m_offset = 0;
		};

		// template<typename ArrayViewType>
		// LIBRAPID_ALWAYS_INLINE
		// GeneralArrayView<ArrayViewType>::GeneralArrayView(ArrayViewType &array) :
		// 		m_ref(array),
		// 		m_shape(array.shape()), m_stride(array.shape()) {}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::GeneralArrayView(
		  ArrayViewType &&array) :
				m_ref(array),
				m_shape(array.shape()), m_stride(array.shape()) {}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE GeneralArrayView<ArrayViewType, ArrayViewShapeType> &
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::operator=(const Scalar &scalar) {
			LIBRAPID_ASSERT(m_shape.ndim() == 0, "Cannot assign to a non-scalar ArrayView.");
			m_ref.storage()[m_offset] = static_cast<Scalar>(scalar);
			return *this;
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE GeneralArrayView<ArrayViewType, ArrayViewShapeType> &
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::operator=(
		  const ArrayContainer<ShapeType_, StorageType_> &other) {
			LIBRAPID_ASSERT(m_shape.operator==(other.shape()),
							"Cannot assign to a non-scalar ArrayView.");

			ShapeType coord = ShapeType::zeros(m_shape.ndim());
			int64_t d = 0, p = 0;
			int64_t idim = 0, adim = 0;
			const int64_t ndim = m_shape.ndim();

			do {
				m_ref.storage()[p + m_offset] = other.scalar(d++);

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

			return *this;
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE const auto
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::operator[](int64_t index) const {
			LIBRAPID_ASSERT(
			  index >= 0 && index < static_cast<int64_t>(m_shape[0]),
			  "Index {} out of bounds in ArrayContainer::operator[] with leading dimension={}",
			  index,
			  m_shape[0]);
			auto view		  = createGeneralArrayViewShapeModifier<Shape>(m_ref);
			const auto stride = Stride(m_shape);
			view.setShape(m_shape.subshape(1, ndim()));
			if (ndim() == 1)
				view.setStride(Stride<Shape>({1}));
			else
				view.setStride(stride.substride(1, ndim()));
			view.setOffset(m_offset + index * stride[0]);
			return view;
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE auto
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::operator[](int64_t index) {
			LIBRAPID_ASSERT(
			  index >= 0 && index < static_cast<int64_t>(m_shape[0]),
			  "Index {} out of bounds in ArrayContainer::operator[] with leading dimension={}",
			  index,
			  m_shape[0]);
			auto view		  = createGeneralArrayViewShapeModifier<Shape>(m_ref);
			const auto stride = Stride(m_shape);
			view.setShape(m_shape.subshape(1, ndim()));
			if (ndim() == 1)
				view.setStride(Stride<Shape>({1}));
			else
				view.setStride(stride.substride(1, ndim()));
			view.setOffset(m_offset + index * stride[0]);
			return view;
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		template<typename CAST>
		LIBRAPID_ALWAYS_INLINE CAST
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::get() const {
			LIBRAPID_ASSERT(m_shape.ndim() == 0,
							"Can only cast a scalar ArrayView to a salar object");
			return scalar(0);
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		template<typename CAST>
		LIBRAPID_ALWAYS_INLINE
		  GeneralArrayView<ArrayViewType, ArrayViewShapeType>::operator CAST() const {
			return get();
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE auto
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::shape() const -> ShapeType {
			return m_shape;
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE auto
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::stride() const -> StrideType {
			return m_stride;
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE auto
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::offset() const -> int64_t {
			return m_offset;
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE void
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::setShape(const ShapeType &shape) {
			m_shape = shape;
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE void
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::setStride(const StrideType &stride) {
			m_stride = stride;
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE void
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::setOffset(const int64_t &offset) {
			m_offset = offset;
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE auto
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::ndim() const -> int64_t {
			return m_shape.ndim();
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE auto
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::scalar(int64_t index) const -> auto {
			if (ndim() == 0) return m_ref.scalar(m_offset);

			ShapeType tmp	= ShapeType::zeros(ndim());
			tmp[ndim() - 1] = index % m_shape[ndim() - 1];
			for (int64_t i = ndim() - 2; i >= 0; --i) {
				index /= m_shape[i + 1];
				tmp[i] = index % m_shape[i];
			}
			int64_t offset = 0;
			for (int64_t i = 0; i < ndim(); ++i) { offset += tmp[i] * m_stride[i]; }
			return m_ref.scalar(m_offset + offset);
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE auto
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::eval() const -> ArrayType {
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

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE auto
		GeneralArrayView<ArrayViewType, ArrayViewShapeType>::begin() const -> Iterator {
			return Iterator(*this, 0);
		}

		template<typename ArrayViewType, typename ArrayViewShapeType>
		LIBRAPID_ALWAYS_INLINE auto GeneralArrayView<ArrayViewType, ArrayViewShapeType>::end() const
		  -> Iterator {
			return Iterator(*this, m_shape[0]);
		}
	} // namespace array
} // namespace librapid

// Support FMT printing
ARRAY_TYPE_FMT_IML(typename T COMMA typename S, librapid::array::GeneralArrayView<T COMMA S>)
LIBRAPID_SIMPLE_IO_NORANGE(typename T COMMA typename S,
						   librapid::array::GeneralArrayView<T COMMA S>)

#endif // LIBRAPID_ARRAY_ARRAY_VIEW_HPP