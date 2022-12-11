#ifndef LIBRAPID_ARRAY_ARRAY_VIEW_HPP
#define LIBRAPID_ARRAY_ARRAY_VIEW_HPP

namespace librapid::array {
	template<typename T>
	class ArrayView {
	public:
		using ArrayType		 = ArrayRef<T>;
		using Reference		 = ArrayType &;
		using ConstReference = ArrayType const &;
		using StrideType	 = typename ArrayType::ShapeType;
		using ShapeType		 = typename ArrayType::ShapeType;

		ArrayView() = delete;
		ArrayView(ArrayType &array);
		ArrayView(const ArrayType &array);

	private:
		Reference m_ref;
		ShapeType m_shape;
		StrideType m_stride;
	};

	template<typename T>
	ArrayView<T>::ArrayView(ArrayType &array) : m_ref(array) {}

	template<typename T>
	ArrayView<T>::ArrayView(const ArrayType &array) : m_ref(array) {}
} // namespace librapid::array

#endif // LIBRAPID_ARRAY_ARRAY_VIEW_HPP