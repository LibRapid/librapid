#ifndef LIBRAPID_ARRAY_CUDA_STORAGE_HPP
#define LIBRAPID_ARRAY_CUDA_STORAGE_HPP

#if defined(LIBRAPID_HAS_CUDA)

/*
 * Implement a storage class for CUDA data structures. It will expose
 * the same functions as the `librapid::Storage` class, but the underlying
 * memory buffer will be allocated on the GPU.
 */

namespace librapid {
	namespace typetraits {
		template<typename Scalar_>
		struct TypeInfo<CudaStorage<Scalar_>> {
			static constexpr bool isLibRapidType = true;
			using Scalar						 = Scalar_;
			using Backend						 = backend::CUDA;
		};

		template<typename T>
		struct IsCudaStorage : std::false_type {};

		template<typename Scalar>
		struct IsCudaStorage<CudaStorage<Scalar>> : std::true_type {};

		LIBRAPID_DEFINE_AS_TYPE(typename Scalar_, CudaStorage<Scalar_>);
	} // namespace typetraits

	namespace detail {
		/// Safely allocate memory for \p size elements of type on the GPU using CUDA.
		/// \tparam T Scalar type
		/// \param size Number of elements to allocate
		/// \return GPU pointer
		/// \see safeAllocate
		template<typename T>
		T *__restrict cudaSafeAllocate(size_t size);

		/// Safely free memory for \p size elements of type on the GPU using CUDA.
		/// \tparam T Scalar type
		/// \param data The data to deallocate
		/// \return GPU pointer
		/// \see safeAllocate
		template<typename T>
		void cudaSafeDeallocate(T *__restrict data);

		template<typename T>
		std::shared_ptr<T> cudaSharedPtrAllocate(size_t size);

#	define CUDA_REF_OPERATOR(OP)                                                                  \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const CudaRef<LHS> &lhs, const RHS &rhs) {                                \
			return lhs.get() OP rhs;                                                               \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const LHS &lhs, const CudaRef<RHS> &rhs) {                                \
			return lhs OP rhs.get();                                                               \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const CudaRef<LHS> &lhs, const CudaRef<RHS> &rhs) {                       \
			return lhs.get() OP rhs.get();                                                         \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP##=(CudaRef<LHS> &lhs, const RHS &rhs) {                                   \
			lhs = lhs.get() OP rhs;                                                                \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP##=(CudaRef<LHS> &lhs, const CudaRef<RHS> &rhs) {                          \
			lhs = lhs.get() OP rhs.get();                                                          \
		}

#	define CUDA_REF_OPERATOR_NO_ASSIGN(OP)                                                        \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const CudaRef<LHS> &lhs, const RHS &rhs) {                                \
			return lhs.get() OP rhs;                                                               \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const LHS &lhs, const CudaRef<RHS> &rhs) {                                \
			return lhs OP rhs.get();                                                               \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const CudaRef<LHS> &lhs, const CudaRef<RHS> &rhs) {                       \
			return lhs.get() OP rhs.get();                                                         \
		}

		template<typename T>
		class CudaRef {
		public:
			using PtrType = std::shared_ptr<T>;

			CudaRef(const PtrType &ptr, size_t offset) : m_ptr(ptr), m_offset(offset) {}

			LIBRAPID_ALWAYS_INLINE CudaRef &operator=(const T &val) {
				cudaSafeCall(cudaMemcpyAsync(m_ptr.get() + m_offset,
											 &val,
											 sizeof(T),
											 cudaMemcpyHostToDevice,
											 global::cudaStream));
				return *this;
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T get() const {
				T tmp;
				cudaSafeCall(cudaMemcpyAsync(&tmp,
											 m_ptr.get() + m_offset,
											 sizeof(T),
											 cudaMemcpyDeviceToHost,
											 global::cudaStream));
				return tmp;
			}

			template<typename CAST>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE operator CAST() const {
				return static_cast<CAST>(get());
			}

			LIBRAPID_NODISCARD std::string str(const std::string &format = "{}") const {
				return fmt::format(format, get());
			}

		private:
			std::shared_ptr<T> m_ptr;
			size_t m_offset;
		};

		CUDA_REF_OPERATOR(+)
		CUDA_REF_OPERATOR(-)
		CUDA_REF_OPERATOR(*)
		CUDA_REF_OPERATOR(/)
		CUDA_REF_OPERATOR(%)
		CUDA_REF_OPERATOR(^)
		CUDA_REF_OPERATOR(&)
		CUDA_REF_OPERATOR(|)
		CUDA_REF_OPERATOR(<<)
		CUDA_REF_OPERATOR(>>)
		CUDA_REF_OPERATOR_NO_ASSIGN(==)
		CUDA_REF_OPERATOR_NO_ASSIGN(!=)
		CUDA_REF_OPERATOR_NO_ASSIGN(<)
		CUDA_REF_OPERATOR_NO_ASSIGN(>)
		CUDA_REF_OPERATOR_NO_ASSIGN(<=)
		CUDA_REF_OPERATOR_NO_ASSIGN(>=)
	} // namespace detail

	template<typename Scalar_>
	class CudaStorage {
	public:
		using Scalar		 = Scalar_;
		using Pointer		 = std::shared_ptr<Scalar>;		  // Scalar *__restrict;
		using ConstPointer	 = const std::shared_ptr<Scalar>; // const Scalar *__restrict;
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

		LIBRAPID_ALWAYS_INLINE CudaStorage(Scalar *begin, SizeType size, bool independent);

		/// Create a new CudaStorage object from an existing one.
		/// \param other The CudaStorage to copy
		LIBRAPID_ALWAYS_INLINE CudaStorage(const CudaStorage &other);

		/// Create a new CudaStorage object from a temporary one, moving the
		/// data
		/// \param other The array to move
		LIBRAPID_ALWAYS_INLINE CudaStorage(CudaStorage &&other) noexcept;

		/// Create a CudaStorage object from an std::initializer_list
		/// \param list Initializer list of elements
		LIBRAPID_ALWAYS_INLINE CudaStorage(const std::initializer_list<Scalar> &list);

		/// Create a CudaStorage object from an std::vector of values
		/// \param vec The vector to fill with
		LIBRAPID_ALWAYS_INLINE explicit CudaStorage(const std::vector<Scalar> &vec);

		void set(const CudaStorage &other);

		template<typename ShapeType>
		static ShapeType defaultShape();

		template<typename V>
		static CudaStorage fromData(const std::initializer_list<V> &vec);

		template<typename V>
		static CudaStorage fromData(const std::vector<V> &vec);

		/// Assignment operator for a CudaStorage object
		/// \param other CudaStorage object to copy
		/// \return *this
		LIBRAPID_ALWAYS_INLINE CudaStorage &operator=(const CudaStorage &other);

		/// Move assignment operator for a CudaStorage object
		/// \param other CudaStorage object to move
		/// \return *this
		LIBRAPID_ALWAYS_INLINE CudaStorage &operator=(CudaStorage &&other) noexcept;

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

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE detail::CudaRef<Scalar>
		operator[](SizeType index) const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE detail::CudaRef<Scalar>
		operator[](SizeType index);

		/// Return the underlying pointer to the data
		/// \return The underlying pointer to the data
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Pointer data() const noexcept;

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
		size_t m_size;
		bool m_ownsData = true;
	};

	namespace detail {
		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T *__restrict cudaSafeAllocate(size_t size) {
			static_assert(typetraits::TriviallyDefaultConstructible<T>::value,
						  "Data type must be trivially constructable for use with CUDA");
			T *result;
			// Round size up to nearest multiple of 32
			size = (size + size_t(31)) & ~size_t(31);
			cudaSafeCall(cudaMallocAsync(&result, sizeof(T) * size, global::cudaStream));
			return result;
		}

		template<>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE
		  Complex<float> *__restrict cudaSafeAllocate<Complex<float>>(size_t size) {
			Complex<float> *result;
			cudaSafeCall(
			  cudaMallocAsync(&result, sizeof(Complex<float>) * size, global::cudaStream));
			return result;
		}

		template<>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE
		  Complex<double> *__restrict cudaSafeAllocate<Complex<double>>(size_t size) {
			Complex<double> *result;
			cudaSafeCall(
			  cudaMallocAsync(&result, sizeof(Complex<double>) * size, global::cudaStream));
			return result;
		}

		template<typename T>
		LIBRAPID_ALWAYS_INLINE void cudaSafeDeallocate(T *__restrict data) {
			static_assert(typetraits::TriviallyDefaultConstructible<T>::value,
						  "Data type must be trivially constructable for use with CUDA");
			cudaSafeCall(cudaFreeAsync(data, global::cudaStream));
		}

		template<>
		LIBRAPID_ALWAYS_INLINE void
		cudaSafeDeallocate<Complex<float>>(Complex<float> *__restrict data) {
			cudaSafeCall(cudaFreeAsync(data, global::cudaStream));
		}

		template<>
		LIBRAPID_ALWAYS_INLINE void
		cudaSafeDeallocate<Complex<double>>(Complex<double> *__restrict data) {
			cudaSafeCall(cudaFreeAsync(data, global::cudaStream));
		}

		template<typename T>
		std::shared_ptr<T> cudaSharedPtrAllocate(size_t size) {
			return std::shared_ptr<T>(cudaSafeAllocate<T>(size), cudaSafeDeallocate<T>);
		}
	} // namespace detail

	template<typename T>
	CudaStorage<T>::CudaStorage(SizeType size) {
		m_begin = detail::cudaSharedPtrAllocate<T>(size);
		m_size	= size;
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(SizeType size, ConstReference value) {
		m_begin = detail::cudaSharedPtrAllocate<T>(size);
		m_size	= size;

		// Fill the data with "value"
		cuda::runKernel<T, T>("fill", "fillArray", size, size, m_begin, value);
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(Scalar *begin, SizeType size, bool independent) {
		if (independent)
			m_begin =
			  std::shared_ptr<Scalar>(begin, [](Scalar *ptr) { detail::cudaSafeDeallocate(ptr); });
		else
			m_begin = std::shared_ptr<Scalar>(begin, [](Scalar *ptr) {});
		m_size	   = size;
		m_ownsData = independent;
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(const CudaStorage &other) {
		m_begin = detail::cudaSharedPtrAllocate<T>(other.size());
		m_size	= other.size();
		cudaSafeCall(cudaMemcpyAsync(m_begin.get(),
									 other.begin().get(),
									 sizeof(T) * other.size(),
									 cudaMemcpyDeviceToDevice,
									 global::cudaStream));
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(CudaStorage &&other) noexcept :
			m_begin(other.m_begin), m_size(other.m_size), m_ownsData(other.m_ownsData) {
		other.m_begin = nullptr;
		other.m_size  = 0;
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(const std::initializer_list<T> &list) {
		m_begin = detail::cudaSharedPtrAllocate<T>(list.size());
		m_size	= list.size();
		cudaSafeCall(cudaMemcpyAsync(m_begin.get(),
									 list.begin(),
									 sizeof(T) * m_size,
									 cudaMemcpyHostToDevice,
									 global::cudaStream));
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(const std::vector<T> &list) {
		m_begin = detail::cudaSharedPtrAllocate<T>(list.size());
		m_size	= list.size();
		cudaSafeCall(cudaMemcpyAsync(m_begin.get(),
									 list.begin(),
									 sizeof(T) * m_size,
									 cudaMemcpyHostToDevice,
									 global::cudaStream));
	}

	template<typename T>
	void CudaStorage<T>::set(const CudaStorage &other) {
		m_begin	   = other.m_begin;
		m_size	   = other.m_size;
		m_ownsData = other.m_ownsData;
	}

	template<typename T>
	template<typename ShapeType>
	ShapeType CudaStorage<T>::defaultShape() {
		return ShapeType({0});
	}

	template<typename T>
	template<typename V>
	auto CudaStorage<T>::fromData(const std::initializer_list<V> &list) -> CudaStorage {
		CudaStorage ret;
		ret.initData(list.begin(), list.end());
		return ret;
	}

	template<typename T>
	template<typename V>
	auto CudaStorage<T>::fromData(const std::vector<V> &vec) -> CudaStorage {
		CudaStorage ret;
		ret.initData(vec.begin(), vec.end());
		return ret;
	}

	template<typename T>
	auto CudaStorage<T>::operator=(const CudaStorage<T> &storage) -> CudaStorage & {
		if (this != &storage) {
			LIBRAPID_ASSERT(
			  !m_ownsData || size() == storage.size(),
			  "Mismatched storage sizes. Cannot assign CUDA storage with {} elements to "
			  "dependent CUDA storage with {} elements",
			  storage.size(),
			  size());

			resizeImpl(storage.size(), 0); // Different sizes are handled here
			cudaSafeCall(cudaMemcpyAsync(m_begin.get(),
										 storage.begin().get(),
										 sizeof(T) * m_size,
										 cudaMemcpyDeviceToDevice,
										 global::cudaStream));
		}
		return *this;
	}

	template<typename T>
	auto CudaStorage<T>::operator=(CudaStorage &&other) noexcept -> CudaStorage & {
		if (this != &other) {
			if (m_ownsData) {
				m_begin		  = other.m_begin;
				m_size		  = other.m_size;
				other.m_begin = nullptr;
				other.m_size  = 0;
				m_ownsData	  = other.m_ownsData;
			} else {
				LIBRAPID_ASSERT(
				  !m_ownsData || size() == other.size(),
				  "Mismatched storage sizes. Cannot assign CUDA storage with {} elements to "
				  "dependent CUDA storage with {} elements",
				  other.size(),
				  size());

				resizeImpl(other.size(), 0); // Different sizes are handled here
				cudaSafeCall(cudaMemcpyAsync(m_begin.get(),
											 other.begin().get(),
											 sizeof(T) * m_size,
											 cudaMemcpyDeviceToDevice,
											 global::cudaStream));
			}
		}
		return *this;
	}

	template<typename T>
	CudaStorage<T>::~CudaStorage() {
		// Data is freed automatically by the shared_ptr. A custom deleter is used to ensure that
		// nothing happens if the storage is dependent.
	}

	template<typename T>
	template<typename P>
	void CudaStorage<T>::initData(P begin, P end) {
		auto size	  = std::distance(begin, end);
		m_begin		  = detail::cudaSharedPtrAllocate<T>(size);
		m_size		  = size;
		auto tmpBegin = [begin]() {
			if constexpr (std::is_pointer_v<P>)
				return begin;
			else
				return &(*begin);
		}();
		cudaSafeCall(cudaMemcpyAsync(
		  m_begin.get(), tmpBegin, sizeof(T) * size, cudaMemcpyDefault, global::cudaStream));
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
		LIBRAPID_ASSERT(m_ownsData, "Dependent CUDA storage cannot be resized");

		Pointer oldBegin = m_begin;

		// Reallocate
		m_begin = detail::cudaSharedPtrAllocate<T>(newSize);

		// Copy old data
		cudaSafeCall(cudaMemcpyAsync(m_begin.get(),
									 oldBegin.get(),
									 sizeof(T) * std::min(size(), newSize),
									 cudaMemcpyDeviceToDevice,
									 global::cudaStream));

		m_size	= newSize;
	}

	template<typename T>
	void CudaStorage<T>::resizeImpl(SizeType newSize, int) {
		if (newSize == size()) return;
		LIBRAPID_ASSERT(m_ownsData, "Dependent CUDA storage cannot be resized");
		m_begin = detail::cudaSharedPtrAllocate<T>(newSize);
		m_size	= newSize;
	}

	template<typename T>
	auto CudaStorage<T>::size() const noexcept -> SizeType {
		return m_size;
	}

	template<typename T>
	auto CudaStorage<T>::operator[](SizeType index) const -> detail::CudaRef<Scalar> {
		return {m_begin, index};
	}

	template<typename T>
	auto CudaStorage<T>::operator[](SizeType index) -> detail::CudaRef<Scalar> {
		return {m_begin, index};
	}

	template<typename T>
	auto CudaStorage<T>::data() const noexcept -> Pointer {
		return m_begin;
	}

	template<typename T>
	auto CudaStorage<T>::begin() const noexcept -> Pointer {
		return m_begin;
	}

	template<typename T>
	auto CudaStorage<T>::end() const noexcept -> Pointer {
		return m_begin + m_size;
	}
} // namespace librapid

#	if defined(FMT_API)
LIBRAPID_SIMPLE_IO_IMPL(typename T, librapid::detail::CudaRef<T>)
#	endif // FM_API
#else
// Trait implementations
namespace librapid::typetraits {
	// Define this so things still work correctly
	template<typename T>
	struct IsCudaStorage : std::false_type {};
} // namespace librapid::typetraits
#endif	   // LIBRAPID_HAS_CUDA
#endif	   // LIBRAPID_ARRAY_CUDA_STORAGE_HPP
