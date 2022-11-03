#ifndef LIBRAPID_ARRAY_CUDA_STORAGE_HPP
#define LIBRAPID_ARRAY_CUDA_STORAGE_HPP

/*
 * Implement a storage class for CUDA data structures. It will expose
 * the same functions as the `librapid::Storage` class, but the underlying
 * memory buffer will be allocated on the device.
 */

namespace librapid {
	namespace typetraits {
		template<typename Scalar_>
		struct TypeInfo<CudaStorage<Scalar_>> {
			static constexpr bool isLibRapidType = true;
			using Scalar						 = Scalar_;
		};
	} // namespace typetraits

	template<typename Scalar_>
	class CudaStorage {
	public:
		using Scalar		 = Scalar_;
		using Pointer		 = Scalar *__restrict;
		using ConstPointer	 = const Scalar *__restrict;
		using Reference		 = Scalar &;
		using ConstReference = const Scalar &;
		using DifferenceType = std::ptrdiff_t;
		using SizeType		 = std::size_t;

		/// Default constructor -- initializes with nullptr
		CudaStorage() = default;

		/// Create a CudaStorage object with \p elements. The data is not
		/// initialized.
		/// \param size Number of elements
		LIBRAPID_ALWAYS_INLINE explicit CudaStorage(SizeType size);

		/// Create a CudaStorage object with \p elements. The data is initialized
		/// to \p value.
		/// \param size Number of elements
		/// \param value Value to fill with
		LIBRAPID_ALWAYS_INLINE CudaStorage(SizeType size, ConstReference value);

		/// Create a new CudaStorage object from an existing one.
		/// \param other The CudaStorage to copy
		LIBRAPID_ALWAYS_INLINE CudaStorage(const CudaStorage &other);

		/// Create a new CudaStorage object from a temporary one, moving the
		/// data
		/// \param other The array to move
		LIBRAPID_ALWAYS_INLINE CudaStorage(const CudaStorage &&other) noexcept;

		/// Create a CudaStorage object from an std::initializer_list
		/// \tparam V Type of the elements in the initializer list
		/// \param list Initializer list of elements
		template<typename V>
		explicit CudaStorage(const std::initializer_list<V> &list);
	};
} // namespace librapid

#endif // LIBRAPID_ARRAY_CUDA_STORAGE_HPP