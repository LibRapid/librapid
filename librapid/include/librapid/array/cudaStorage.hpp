#ifndef LIBRAPID_ARRAY_CUDA_STORAGE_HPP
#define LIBRAPID_ARRAY_CUDA_STORAGE_HPP

#if defined(LIBRAPID_HAS_CUDA)

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
		LIBRAPID_ALWAYS_INLINE CudaStorage(CudaStorage &&other) noexcept = default;

		/// Create a CudaStorage object from an std::initializer_list
		/// \param list Initializer list of elements
		LIBRAPID_ALWAYS_INLINE CudaStorage(const std::initializer_list<Scalar> &list);

		/// Create a CudaStorage object from an std::vector of values
		/// \param vec The vector to fill with
		LIBRAPID_ALWAYS_INLINE explicit CudaStorage(const std::vector<Scalar> &vec);

		/// Assignment operator for a CudaStorage object
		/// \param other CudaStorage object to copy
		/// \return *this
		LIBRAPID_ALWAYS_INLINE CudaStorage &operator=(const CudaStorage &other);

		/// Move assignment operator for a CudaStorage object
		/// \param other CudaStorage object to move
		/// \return *this
		LIBRAPID_ALWAYS_INLINE CudaStorage &operator=(CudaStorage &&other) noexcept = default;

		/// Free a CudaStorage object
		~CudaStorage();

		/// Resize a CudaStorage object to \p size elements. Existing elements are preserved where
		/// possible.
		/// \param size Number of elements
		/// \see resize(SizeType, int)
		LIBRAPID_ALWAYS_INLINE void resize(SizeType newSize);

		/// Resize a CudaStorage object to \p size elements. Existing elements are not preserved.
		/// This method of resizing is faster and more efficient than the version which preserves
		/// the original data, but of course, this has the drawback that data will be lost.
		/// \param size Number of elements
		LIBRAPID_ALWAYS_INLINE void resize(SizeType newSize, int);

		/// Return the number of elements in the CudaStorage object.
		/// \return The number of elements
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SizeType size() const noexcept;

		/// Returns the pointer to the first element of the CudaStorage object
		/// \return Pointer to the first element of the CudaStorage object
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Pointer begin() const noexcept;

		/// Returns the pointer to the last element of the CudaStorage object
		/// \return A pointer to the last element of the CudaStorage
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Pointer end() const noexcept;

	private:
		// Copy data from \p begin to \p end into this Storage object
		/// \tparam P Pointer type
		/// \param begin Beginning of data to copy
		/// \param end End of data to copy
		template<typename P>
		LIBRAPID_ALWAYS_INLINE void initData(P begin, P end);

		/// Resize the Storage Object to \p newSize elements, retaining existing
		/// data.
		/// \param newSize New size of the Storage object
		LIBRAPID_ALWAYS_INLINE void resizeImpl(SizeType newSize, int);

		/// Resize the Storage object to \p newSize elements. Note this does not
		/// initialize the new elements or maintain existing data.
		/// \param newSize New size of the Storage object
		LIBRAPID_ALWAYS_INLINE void resizeImpl(SizeType newSize);

		Pointer m_begin = nullptr; // It is more efficient to store pointers to the start
		Pointer m_end	= nullptr; // and end of the data block than to store the size
	};

	namespace detail {
		/// Safely allocate memory for \p size elements of type on the GPU using CUDA.
		/// \tparam T Scalar type
		/// \param size Number of elements to allocate
		/// \return GPU pointer
		/// \see safeAllocate
		template<typename T>
		T *__restrict cudaSafeAllocate(size_t size) {
			static_assert(typetraits::TriviallyDefaultConstructible<T>::value,
						  "Data type must be trivially constructable for use with CUDA");
			T *result;
			cudaSafeCall(cudaMallocAsync(&result, sizeof(T) * size, global::cudaStream));
			return result;
		}

		/// Safely free memory for \p size elements of type on the GPU using CUDA.
		/// \tparam T Scalar type
		/// \param data The data to deallocate
		/// \return GPU pointer
		/// \see safeAllocate
		template<typename T>
		void cudaSafeDeallocate(T *__restrict data) {
			static_assert(typetraits::TriviallyDefaultConstructible<T>::value,
						  "Data type must be trivially constructable for use with CUDA");
			cudaSafeCall(cudaFreeAsync(data, global::cudaStream));
		}
	} // namespace detail

	template<typename T>
	CudaStorage<T>::CudaStorage(SizeType size) {
		m_begin = detail::cudaSafeAllocate<T>(size);
		m_end	= m_begin + size;
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(SizeType size, ConstReference value) {
		m_begin = detail::cudaSafeAllocate<T>(size);
		m_end	= m_begin + size;

		// Fill the data with "value"
		runKernel<T, T>("fill", "fillArray", size, size, m_begin, value);
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(const CudaStorage &other) {
		m_begin = detail::cudaSafeAllocate<T>(other.size());
		m_end	= m_begin + other.size();
		cudaSafeCall(cudaMemcpyAsync(
		  m_begin, other.begin(), sizeof(T) * other.size(), cudaMemcpyDeviceToDevice));
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(const std::initializer_list<T> &list) {
		m_begin = detail::cudaSafeAllocate<T>(list.size());
		m_end	= m_begin + list.size();
		cudaSafeCall(cudaMemcpyAsync(m_begin, list.begin(), sizeof(T)));
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(const std::vector<T> &list) {
		m_begin = detail::cudaSafeAllocate<T>(list.size());
		m_end	= m_begin + list.size();
		cudaSafeCall(cudaMemcpyAsync(m_begin, list.begin(), sizeof(T)));
	}

	template<typename T>
	auto CudaStorage<T>::operator=(const CudaStorage<T> &storage) -> CudaStorage & {
		m_begin = detail::cudaSafeAllocate<T>(storage.size());
		m_end	= m_begin + storage.size();
		cudaSafeCall(cudaMemcpyAsync(m_begin, storage.begin(), sizeof(T)));
	}

	template<typename T>
	CudaStorage<T>::~CudaStorage() {
		detail::cudaSafeDeallocate(m_begin);
	}

	template<typename T>
	template<typename P>
	void CudaStorage<T>::initData(P begin, P end) {
		auto size = std::distance(begin, end);
		m_begin	  = detail::cudaSafeAllocate<T>(size);
		m_end	  = m_begin + size;
		cudaSafeCall(cudaMemcpyAsync(m_begin, begin, sizeof(T) * size, global::cudaStream));
	}

	template<typename T>
	void CudaStorage<T>::resize(SizeType newSize) {
		resizeImpl(newSize);
	}

	template<typename T>
	void CudaStorage<T>::resize(SizeType newSize, int) {
		resizeImpl(newSize, 0);
	}

	template<typename T>
	void CudaStorage<T>::resizeImpl(SizeType newSize) {
		if (newSize == size()) { return; }

		Pointer oldBegin = m_begin;

		// Reallocate
		m_begin = detail::cudaSafeAllocate<T>(newSize);
		m_end	= m_begin + newSize;

		// Copy old data
		cudaSafeCall(cudaMemcpyAsync(m_begin, oldBegin, sizeof(T) * std::min(size(), newSize)));

		// Free old data
		detail::cudaSafeDeallocate(oldBegin);
	}

	template<typename T>
	void CudaStorage<T>::resizeImpl(SizeType newSize, int) {
		if (newSize == size()) { return; }

		Pointer oldBegin = m_begin;
		SizeType oldSize = size();
		m_begin			 = detail::cudaSafeAllocate<T>(newSize);
		m_end			 = m_begin + newSize;

		detail::cudaSafeDeallocate(oldBegin);
	}

	template<typename T>
	auto CudaStorage<T>::size() const noexcept -> SizeType {
		return std::distance(m_begin, m_end);
	}

	template<typename T>
	auto CudaStorage<T>::begin() const noexcept -> Pointer {
		return m_begin;
	}

	template<typename T>
	auto CudaStorage<T>::end() const noexcept -> Pointer {
		return m_end;
	}
} // namespace librapid

#endif // LIBRAPID_HAS_CUDA
#endif // LIBRAPID_ARRAY_CUDA_STORAGE_HPP