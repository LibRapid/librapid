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
	  ArrayContainer<Shape<size_t, 32>,
					 typename detail::TypeDefStorageEvaluator<Scalar, StorageType>::Type>;

	/// A definition for fixed-size array objects.
	/// \tparam Scalar The scalar type of the array.
	/// \tparam Dimensions The dimensions of the array.
	/// \see Array
	template<typename Scalar, size_t... Dimensions>
	using ArrayF =
	  ArrayContainer<Shape<size_t, 32>, FixedStorage<Scalar, product<Dimensions...>()>>;

	/// A reference type for Array objects. Use this to accept Array objects as parameters since
	/// the compiler cannot determine the templates for the Array typedef. For more granularity,
	/// you can also accept a raw ArrayContainer object.
	/// \tparam StorageType The storage type of the array.
	/// \see Array
	/// \see ArrayF
	template<typename StorageType>
	using ArrayRef = ArrayContainer<Shape<size_t, 32>, StorageType>;
} // namespace librapid

#endif // LIBRAPID_ARRAY_TYPE_DEF_HPP