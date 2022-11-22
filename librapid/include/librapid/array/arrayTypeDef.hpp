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

	template<typename Scalar, typename StorageType = device::CPU>
	using Array =
	  ArrayContainer<Shape<size_t, 32>,
					 typename detail::TypeDefStorageEvaluator<Scalar, StorageType>::Type>;

	template<typename Scalar, size_t... Dimensions>
	using ArrayF = ArrayContainer<Shape<size_t, 32>, FixedStorage<Scalar, product<Dimensions...>()>>;
} // namespace librapid

#endif // LIBRAPID_ARRAY_TYPE_DEF_HPP