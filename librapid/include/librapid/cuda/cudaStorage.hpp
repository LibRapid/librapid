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
			using PtrType = T *;

			CudaRef(PtrType ptr, size_t offset) : m_ptr(ptr), m_offset(offset) {}

			LIBRAPID_ALWAYS_INLINE CudaRef &operator=(const T &val) {
				cudaSafeCall(cudaMemcpyAsync(
				  m_ptr + m_offset, &val, sizeof(T), cudaMemcpyHostToDevice, global::cudaStream));
				return *this;
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T get() const {
				T tmp;
				cudaSafeCall(cudaMemcpyAsync(
				  &tmp, m_ptr + m_offset, sizeof(T), cudaMemcpyDeviceToHost, global::cudaStream));
				return tmp;
			}

			template<typename CAST>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE operator CAST() const {
				return static_cast<CAST>(get());
			}

			template<typename T_, typename Char, typename Ctx>
			void str(const fmt::formatter<T_, Char> &format, Ctx &ctx) const {
				format.format(get(), ctx);
			}

		private:
			T *m_ptr;
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
		using Pointer		 = Scalar *;
		using ConstPointer	 = const Scalar *;
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

		LIBRAPID_ALWAYS_INLINE CudaStorage(Scalar *begin, SizeType size, bool ownsData);

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

		template<typename ShapeType>
		static ShapeType defaultShape();

		static CudaStorage fromData(const std::initializer_list<Scalar> &vec);

		static CudaStorage fromData(const std::vector<Scalar> &vec);

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

		/// \brief Create a deep copy of this CudaStorage object
		/// \return Deep copy of this CudaStorage object
		CudaStorage copy() const;

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

		Pointer m_begin = nullptr;
		size_t m_size;
		bool m_ownsData = true;
	};

	namespace detail {
		template<typename T>
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T *__restrict cudaSafeAllocate(size_t size) {
			LIBRAPID_ASSERT(size > 0, "Cannot allocate 0 bytes of memory");

			static_assert(typetraits::TriviallyDefaultConstructible<T>::value,
						  "Data type must be trivially constructable for use with CUDA");
			T *result;
			// Round size up to nearest multiple of 32
			size = (size + size_t(31)) & ~size_t(31);
			cudaSafeCall(cudaMallocAsync(&result, sizeof(T) * size, global::cudaStream));
			return result;
		}

		template<typename T>
		LIBRAPID_ALWAYS_INLINE void cudaSafeDeallocate(T *__restrict data) {
			static_assert(std::is_trivially_destructible_v<T>,
						  "Data type must be trivially constructable for use with CUDA");
			cudaSafeCall(cudaFreeAsync(data, global::cudaStream));
		}

	template<typename T>
	CudaStorage<T>::CudaStorage(SizeType size) :
			m_size(size), m_begin(detail::cudaSafeAllocate<T>(size)), m_ownsData(true) {
			LIBRAPID_ASSERT(m_size > 0, "Cannot allocate 0 bytes of memory");}
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(SizeType size, ConstReference value) :
			m_size(size), m_begin(detail::cudaSafeAllocate<T>(size)), m_ownsData(true) {
	LIBRAPID_ASSERT(m_size > 0, "Cannot allocate 0 bytes of memory");}
		// Fill the data with "value"
		cuda::runKernel<T, T>("fill", "fillArray", size, size, m_begin, value);
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(Scalar *begin, SizeType size, bool ownsData) :
			m_size(size), m_begin(begin), m_ownsData(ownsData) {
		LIBRAPID_ASSERT(m_size > 0, "Cannot allocate 0 bytes of memory");}
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(const CudaStorage &other) : m_size(other.m_size), m_ownsData(true) {
		LIBRAPID_ASSERT(m_size > 0, "Cannot allocate 0 bytes of memory");}
		// Copy the data
		initData(other.begin(), other.end());
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(CudaStorage &&other) noexcept :
			m_begin(std::move(other.begin())), m_size(std::move(other.size())),
			m_ownsData(std::move(other.m_ownsData)) {
		other.m_begin	 = nullptr;
		other.m_size	 = 0;
		other.m_ownsData = false;
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(const std::initializer_list<T> &list) :
			m_size(list.size()), m_begin(detail::cudaSafeAllocate<T>(list.size())),
			m_ownsData(true) {
		LIBRAPID_ASSERT(m_size > 0, "Cannot allocate 0 bytes of memory");}
		cudaSafeCall(cudaMemcpyAsync(
		  m_begin, list.begin(), sizeof(T) * m_size, cudaMemcpyHostToDevice, global::cudaStream));
	}

	template<typename T>
	CudaStorage<T>::CudaStorage(const std::vector<T> &list) :
			m_size(list.size()), m_begin(detail::cudaSafeAllocate<T>(list.size())),
			m_ownsData(true) {
		LIBRAPID_ASSERT(m_size > 0, "Cannot allocate 0 bytes of memory");}
		cudaSafeCall(cudaMemcpyAsync(
		  m_begin, &list[0], sizeof(T) * m_size, cudaMemcpyHostToDevice, global::cudaStream));
	}

	template<typename T>
	template<typename ShapeType>
	ShapeType CudaStorage<T>::defaultShape() {
		return ShapeType({0});
	}

	template<typename T>
	auto CudaStorage<T>::fromData(const std::initializer_list<T> &list) -> CudaStorage {
		CudaStorage ret;
		// ret.initData(list.begin(), list.end());
		ret.initData(static_cast<const T *>(list.begin()), static_cast<const T *>(list.end()));
		return ret;
	}

	template<typename T>
	auto CudaStorage<T>::fromData(const std::vector<T> &vec) -> CudaStorage {
		CudaStorage ret;
		// ret.initData(vec.begin(), vec.end());
		ret.initData(&vec[0], &vec[0] + vec.size());
		return ret;
	}

	template<typename T>
	auto CudaStorage<T>::operator=(const CudaStorage<T> &other) -> CudaStorage & {
		if (this != &other) {
			if (other.m_size == 0) return *this; // Quick return

			size_t oldSize = m_size;
			m_size		   = other.m_size;
			if (oldSize != m_size) LIBRAPID_UNLIKELY {
					if (m_ownsData) LIBRAPID_LIKELY {
							// Reallocate
							detail::cudaSafeDeallocate(m_begin);
							m_begin = detail::cudaSafeAllocate<T>(m_size);
						}
					else
						LIBRAPID_UNLIKELY {
							// We do not own this data, so we cannot reallocate it
							LIBRAPID_ASSERT(false, "Cannot reallocate dependent CUDA storage");
						}
				}

			// Copy the data
			cudaSafeCall(cudaMemcpyAsync(m_begin,
										 other.begin(),
										 sizeof(T) * m_size,
										 cudaMemcpyDeviceToDevice,
										 global::cudaStream));
		}
		return *this;
	}

	template<typename T>
	auto CudaStorage<T>::operator=(CudaStorage &&other) noexcept -> CudaStorage & {
		if (this != &other) {
			m_begin = std::move(other.m_begin);
			m_size  = std::move(other.m_size);
			m_ownsData = std::move(other.m_ownsData);

			other.m_begin	 = nullptr;
			other.m_size	 = 0;
			other.m_ownsData = false;
		}
		return *this;
	}

	template<typename T>
	CudaStorage<T>::~CudaStorage() {
		// If we own the data, we can free it
		if (m_ownsData) detail::cudaSafeDeallocate(m_begin);
	}

	template<typename T>
	auto CudaStorage<T>::copy() const -> CudaStorage {
		CudaStorage ret(m_size);

		cudaSafeCall(cudaMemcpyAsync(
		  ret.begin(), m_begin, sizeof(T) * m_size, cudaMemcpyDeviceToDevice, global::cudaStream));

		return ret;
	}

	template<typename T>
	template<typename P>
	void CudaStorage<T>::initData(P begin, P end) {
		// Quick return in the case of empty range
		if (begin == nullptr || end == nullptr || begin == end) return;

		auto size  = std::distance(begin, end);
		m_begin	   = detail::cudaSafeAllocate<T>(size);
		m_size	   = size;
		m_ownsData = true;
		cudaSafeCall(cudaMemcpyAsync(
		  m_begin, begin, sizeof(T) * size, cudaMemcpyHostToDevice, global::cudaStream));
	}

	template<typename T>
	void CudaStorage<T>::resize(SizeType newSize) {
		LIBRAPID_ASSERT(newSize > 0, "Cannot allocate 0 bytes of memory");}
		if (newSize == size()) { return; }

		LIBRAPID_ASSERT(m_ownsData, "Dependent CUDA storage cannot be resized");

		Pointer oldBegin = m_begin;
		SizeType oldSize = m_size;

		// Reallocate
		m_begin = detail::cudaSafeAllocate<T>(newSize);
		m_size	= newSize;

		// Copy old data
		cudaSafeCall(cudaMemcpyAsync(m_begin,
									 oldBegin,
									 sizeof(T) * std::min(oldSize, newSize),
									 cudaMemcpyDeviceToDevice,
									 global::cudaStream));

		// Free old data
		detail::cudaSafeDeallocate(oldBegin);
	}

	template<typename T>
	void CudaStorage<T>::resize(SizeType newSize, int) {
		LIBRAPID_ASSERT(newSize > 0, "Cannot allocate 0 bytes of memory");}
		if (newSize == size()) return;
		LIBRAPID_ASSERT(m_ownsData, "Dependent CUDA storage cannot be resized");
		detail::cudaSafeDeallocate(m_begin);
		m_begin = detail::cudaSafeAllocate<T>(newSize);
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
// LIBRAPID_SIMPLE_IO_IMPL(typename T, librapid::detail::CudaRef<T>)

template<typename T, typename Char>
struct fmt::formatter<librapid::detail::CudaRef<T>, Char> {
private:
	using Base = fmt::formatter<T, Char>;
	Base m_base;

public:
	template<typename ParseContext>
	FMT_CONSTEXPR auto parse(ParseContext &ctx) -> const char * {
		return m_base.parse(ctx);
	}

	template<typename FormatContext>
	FMT_CONSTEXPR auto format(const librapid::detail::CudaRef<T> &val, FormatContext &ctx) const
	  -> decltype(ctx.out()) {
		val.str(m_base, ctx);
		return ctx.out();
	}
};
#	endif // FM_API
#else
// Trait implementations
namespace librapid::typetraits {
	// Define this so things still work correctly
	template<typename T>
	struct IsCudaStorage : std::false_type {};
} // namespace librapid::typetraits
#endif // LIBRAPID_HAS_CUDA
#endif // LIBRAPID_ARRAY_CUDA_STORAGE_HPP
