#ifndef LIBRAPID_ARRAY_TYPE_DEF_HPP
#define LIBRAPID_ARRAY_TYPE_DEF_HPP

namespace librapid {
	namespace detail {
		template<typename Scalar, typename T>
		struct TypeDefStorageEvaluator {
			using Type = T;
		};

		template<typename Scalar>
		struct TypeDefStorageEvaluator<Scalar, device::CPU> {
			using Type = Storage<Scalar>;
		};

		template<typename Scalar>
		struct TypeDefStorageEvaluator<Scalar, device::GPU> {
			using Type = CudaStorage<Scalar>;
		};
	} // namespace detail

	/// An easier to use definition than ArrayContainer. In this case, StorageType can be
	/// `device::CPU`, `device::GPU` or any Storage interface
	/// \tparam Scalar The scalar type of the array.
	/// \tparam StorageType The storage type of the array.
	template<typename Scalar, typename StorageType = device::CPU>
	using Array =
	  array::ArrayContainer<Shape<size_t, 32>,
							typename detail::TypeDefStorageEvaluator<Scalar, StorageType>::Type>;

	/// A definition for fixed-size array objects.
	/// \tparam Scalar The scalar type of the array.
	/// \tparam Dimensions The dimensions of the array.
	/// \see Array
	template<typename Scalar, size_t... Dimensions>
	using ArrayF = array::ArrayContainer<Shape<size_t, 32>, FixedStorage<Scalar, Dimensions...>>;

	/// A reference type for Array objects. Use this to accept Array objects as parameters since
	/// the compiler cannot determine the templates tingle for the Array typedef. For more
	/// granularity, you can also accept a raw ArrayContainer object. \tparam StorageType The
	/// storage type of the array. \see Array \see ArrayF \see Function \see FunctionRef
	template<typename StorageType>
	using ArrayRef = array::ArrayContainer<Shape<size_t, 32>, StorageType>;

	/// A reference type for Array Function objects. Use this to accept Function objects as
	/// parameters since the compiler cannot determine the templates for the typedef by default.
	/// Additionally, this can be used to store references to Function objects.
	/// \tparam Inputs The argument types to the function (template...)
	/// \see Array
	/// \see ArrayF
	/// \see ArrayRef
	/// \see Function
	template<typename... Inputs>
	using FunctionRef = detail::Function<Inputs...>;

	namespace array {
		/// An intermediate type to represent a slice or view of an array.
		/// \tparam T The type of the array.
		template<typename T>
		class ArrayView;
	}
} // namespace librapid

#endif // LIBRAPID_ARRAY_TYPE_DEF_HPP