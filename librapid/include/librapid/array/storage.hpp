#ifndef LIBRAPID_ARRAY_DENSE_STORAGE_HPP
#define LIBRAPID_ARRAY_DENSE_STORAGE_HPP

/*
 * This file defines the DenseStorage class, which contains a contiguous
 * block of memory of a single data type.
 */

namespace librapid {
	namespace typetraits {
		template<typename Scalar_, typename Allocator_>
		struct TypeInfo<Storage<Scalar_, Allocator_>> {
			static constexpr bool isLibRapidType = true;
			using Scalar						 = Scalar_;
			using Device						 = device::CPU;
		};

		template<typename Scalar_, size_t... Dims>
		struct TypeInfo<FixedStorage<Scalar_, Dims...>> {
			static constexpr bool isLibRapidType = true;
			using Scalar						 = Scalar_;
			using Device						 = device::CPU;
		};

		LIBRAPID_DEFINE_AS_TYPE(typename Scalar_ COMMA typename Allocator_,
								Storage<Scalar_ COMMA Allocator_>);
	} // namespace typetraits

	template<typename Scalar_, typename Allocator_ = std::allocator<Scalar_>>
	class Storage {
	public:
		using Allocator			   = Allocator_;
		using Scalar			   = Scalar_;
		using Pointer			   = typename std::allocator_traits<Allocator>::pointer;
		using ConstPointer		   = typename std::allocator_traits<Allocator>::const_pointer;
		using Reference			   = Scalar &;
		using ConstReference	   = const Scalar &;
		using SizeType			   = typename std::allocator_traits<Allocator>::size_type;
		using DifferenceType	   = typename std::allocator_traits<Allocator>::difference_type;
		using Iterator			   = Pointer;
		using ConstIterator		   = ConstPointer;
		using ReverseIterator	   = std::reverse_iterator<Iterator>;
		using ConstReverseIterator = std::reverse_iterator<ConstIterator>;

		/// Default constructor
		Storage() = default;

		/// Create a Storage object with \p size elements and,
		/// optionally, a custom allocator.
		/// \param size Number of elements to allocate
		/// \param alloc Allocator to use
		LIBRAPID_ALWAYS_INLINE explicit Storage(SizeType size,
												const Allocator &alloc = Allocator());

		LIBRAPID_ALWAYS_INLINE explicit Storage(Scalar *begin, Scalar *end, bool independent);

		/// Create a Storage object with \p size elements, each initialized
		/// to \p value. Optionally, a custom allocator can be used.
		/// \param size Number of elements to allocate
		/// \param value Value to initialize each element to
		/// \param alloc Allocator to use
		LIBRAPID_ALWAYS_INLINE Storage(SizeType size, ConstReference value,
									   const Allocator &alloc = Allocator());

		/// Create a Storage object from another Storage object. Additionally
		/// a custom allocator can be used.
		/// \param other Storage object to copy
		/// \param alloc Allocator to use
		LIBRAPID_ALWAYS_INLINE Storage(const Storage &other, const Allocator &alloc = Allocator());

		/// Move a Storage object into this object.
		/// \param other Storage object to move
		LIBRAPID_ALWAYS_INLINE Storage(Storage &&other) noexcept;

		/// Create a Storage object from an std::initializer_list
		/// \tparam V Type of the elements in the initializer list
		/// \param list Initializer list to copy
		/// \param alloc Allocator to use
		template<typename V>
		LIBRAPID_ALWAYS_INLINE Storage(const std::initializer_list<V> &list,
									   const Allocator &alloc = Allocator());

		/// Create a Storage object from a std::vector
		/// \tparam V Type of the elements in the vector
		/// \param vec Vector to copy
		/// \param alloc Allocator to use
		template<typename V>
		LIBRAPID_ALWAYS_INLINE explicit Storage(const std::vector<V> &vec,
												const Allocator &alloc = Allocator());

		template<typename V>
		static Storage fromData(const std::initializer_list<V> &vec);

		template<typename V>
		static Storage fromData(const std::vector<V> &vec);

		/// Assignment operator for a Storage object
		/// \param other Storage object to copy
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Storage &operator=(const Storage &other);

		/// Move assignment operator for a Storage object
		/// \param other Storage object to move
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Storage &operator=(Storage &&other) LIBRAPID_RELEASE_NOEXCEPT;

		/// Free a Storage object
		~Storage();

		template<typename ShapeType>
		static ShapeType defaultShape();

		/// Resize a Storage object to \p size elements. Existing elements
		/// are preserved.
		/// \param size New size of the Storage object
		LIBRAPID_ALWAYS_INLINE void resize(SizeType newSize);

		/// Resize a Storage object to \p size elements. Existing elements
		/// are not preserved
		/// \param size New size of the Storage object
		LIBRAPID_ALWAYS_INLINE void resize(SizeType newSize, int);

		/// Return the number of elements in the Storage object
		/// \return
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SizeType size() const noexcept;

		/// Const access to the element at index \p index
		/// \param index Index of the element to access
		/// \return Const reference to the element at index \p index
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstReference operator[](SizeType index) const;

		/// Access to the element at index \p index
		/// \param index Index of the element to access
		/// \return Reference to the element at index \p index
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Reference operator[](SizeType index);

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Iterator begin() noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Iterator end() noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstIterator begin() const noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstIterator end() const noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstIterator cbegin() const noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstIterator cend() const noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ReverseIterator rbegin() noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ReverseIterator rend() noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstReverseIterator rbegin() const noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstReverseIterator rend() const noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstReverseIterator crbegin() const noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstReverseIterator crend() const noexcept;

	private:
		/// Copy data from \p begin to \p end into this Storage object
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

		Allocator m_allocator;

		// Pointer m_begin	   = nullptr; // It is more efficient to store pointers to the start
		// Pointer m_end	   = nullptr; // and end of the data block than to store the size

#if defined(LIBRAPID_NATIVE_ARCH) && !defined(LIBRAPID_APPLE)
		alignas(LIBRAPID_DEFAULT_MEM_ALIGN) Pointer m_begin = nullptr;
		alignas(LIBRAPID_DEFAULT_MEM_ALIGN) Pointer m_end	= nullptr;
#else
		Pointer m_begin = nullptr;
		Pointer m_end	= nullptr;
#endif

		bool m_independent = true; // If true, m_begin will be freed on destruct
	};

	template<typename Scalar_, size_t... Size_>
	class FixedStorage {
	public:
		using Scalar				   = Scalar_;
		using Pointer				   = Scalar *;
		using ConstPointer			   = const Scalar *;
		using Reference				   = Scalar &;
		using ConstReference		   = const Scalar &;
		using SizeType				   = size_t;
		using DifferenceType		   = ptrdiff_t;
		static constexpr SizeType Size = product<Size_...>();
		using Iterator				   = typename std::array<Scalar, product<Size_...>()>::iterator;
		using ConstIterator	  = typename std::array<Scalar, product<Size_...>()>::const_iterator;
		using ReverseIterator = std::reverse_iterator<Iterator>;
		using ConstReverseIterator = std::reverse_iterator<ConstIterator>;

		/// Default constructor
		FixedStorage();

		/// Create a FixedStorage object filled with \p value
		/// \param value Value to fill the FixedStorage object with
		LIBRAPID_ALWAYS_INLINE explicit FixedStorage(const Scalar &value);

		/// Create a FixedStorage object from another FixedStorage object
		/// \param other FixedStorage object to copy
		LIBRAPID_ALWAYS_INLINE FixedStorage(const FixedStorage &other);

		/// Move constructor for a FixedStorage object
		/// \param other FixedStorage object to move
		LIBRAPID_ALWAYS_INLINE FixedStorage(FixedStorage &&other) noexcept;

		/// Create a FixedStorage object from a std::initializer_list
		/// \tparam V Type of the elements in the initializer list
		/// \param list Initializer list to copy
		LIBRAPID_ALWAYS_INLINE explicit FixedStorage(const std::initializer_list<Scalar> &list);

		/// Create a FixedStorage object from a std::vector
		/// \tparam V Type of the elements in the vector
		/// \param vec Vector to copy
		LIBRAPID_ALWAYS_INLINE explicit FixedStorage(const std::vector<Scalar> &vec);

		/// Assignment operator for a FixedStorage object
		/// \param other FixedStorage object to copy
		/// \return *this
		LIBRAPID_ALWAYS_INLINE FixedStorage &operator=(const FixedStorage &other);

		/// Move assignment operator for a FixedStorage object
		/// \param other FixedStorage object to move
		/// \return *this
		LIBRAPID_ALWAYS_INLINE FixedStorage &operator=(FixedStorage &&other) noexcept;

		/// Free a FixedStorage object
		~FixedStorage();

		template<typename ShapeType>
		static ShapeType defaultShape();

		/// Resize a Storage object to \p size elements. Existing elements
		/// are preserved.
		/// \param size New size of the Storage object
		LIBRAPID_ALWAYS_INLINE void resize(SizeType newSize);

		/// Resize a Storage object to \p size elements. Existing elements
		/// are not preserved
		/// \param size New size of the Storage object
		LIBRAPID_ALWAYS_INLINE void resize(SizeType newSize, int);

		/// Return the number of elements in the FixedStorage object
		/// \return Number of elements in the FixedStorage object
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SizeType size() const noexcept;

		/// Const access to the element at index \p index
		/// \param index Index of the element to access
		/// \return Const reference to the element at index \p index
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstReference operator[](SizeType index) const;

		/// Access to the element at index \p index
		/// \param index Index of the element to access
		/// \return Reference to the element at index \p index
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Reference operator[](SizeType index);

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Iterator begin() noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Iterator end() noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstIterator begin() const noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstIterator end() const noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstIterator cbegin() const noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstIterator cend() const noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ReverseIterator rbegin() noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ReverseIterator rend() noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstReverseIterator rbegin() const noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstReverseIterator rend() const noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstReverseIterator crbegin() const noexcept;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ConstReverseIterator crend() const noexcept;

	private:
#if defined(LIBRAPID_NATIVE_ARCH) && !defined(LIBRAPID_APPLE)
		alignas(LIBRAPID_DEFAULT_MEM_ALIGN) std::array<Scalar, Size> m_data;
#else
		// No memory alignment on Apple platforms or if it is disabled
		std::array<Scalar, Size> m_data;
#endif
	};

	// Trait implementations
	namespace typetraits {
		template<typename T>
		struct IsStorage : std::false_type {};

		template<typename Scalar, typename Allocator>
		struct IsStorage<Storage<Scalar, Allocator>> : std::true_type {};

		template<typename T>
		struct IsFixedStorage : std::false_type {};

		template<typename Scalar, size_t... Size>
		struct IsFixedStorage<FixedStorage<Scalar, Size...>> : std::true_type {};
	} // namespace typetraits

	namespace detail {
		/// Safely allocate memory for \p size elements using the allocator \p alloc. If the data
		/// can be trivially default constructed, then the constructor is not called and no data
		/// is initialized. Otherwise, the correct default constructor will be called for each
		/// element in the data, making sure the returned pointer is safe to use.
		/// \tparam A The allocator type to use
		/// \param alloc The allocator object to use
		/// \param size Number of elements to allocate
		/// \return Pointer to the first element
		/// \see safeDeallocate
		template<typename A>
		typename std::allocator_traits<A>::pointer
		safeAllocate(A &alloc, typename std::allocator_traits<A>::size_type size) {
			using Traits	= std::allocator_traits<A>;
			using Pointer	= typename Traits::pointer;
			using ValueType = typename Traits::value_type;

#if defined(LIBRAPID_BLAS_MKLBLAS)
			// MKL has its own memory allocation function
			auto ptr = static_cast<Pointer>(mkl_malloc(size * sizeof(ValueType), 64));
#else
#	if defined(LIBRAPID_NATIVE_ARCH)
			// Force aligned memory
#		if defined(LIBRAPID_APPLE)
			// No memory allignment. It breaks everything for some reason
			auto ptr = Traits::allocate(alloc, size);
#		elif defined(LIBRAPID_MSVC)
			auto ptr = static_cast<Pointer>(
			  _aligned_malloc(size * sizeof(ValueType), global::memoryAlignment));
#		else
			auto ptr = static_cast<Pointer>(
			  std::aligned_alloc(global::memoryAlignment, size * sizeof(ValueType)));
#		endif
#	else
			// No memory alignment
			auto ptr = Traits::allocate(alloc, size);
#	endif
#endif

			// If the type cannot be trivially constructed, we need to
			// initialize each value
			if constexpr (!typetraits::TriviallyDefaultConstructible<ValueType>::value &&
						  !std::is_array<ValueType>::value) {
				for (Pointer p = ptr; p != ptr + size; ++p) {
					Traits::construct(alloc, p, ValueType());
				}
			}

			return ptr;
		}

		/// Safely deallocate memory for \p size elements, using an std::allocator \p alloc. If the
		/// object cannot be trivially destroyed, the destructor will be called on each element of
		/// the data, ensuring that it is safe to free the allocated memory.
		/// \tparam A The allocator type
		/// \param alloc The allocator object
		/// \param ptr The pointer to free
		/// \param size The number of elements of type \p in the memory block
		template<typename A>
		void safeDeallocate(A &alloc, typename std::allocator_traits<A>::pointer ptr,
							typename std::allocator_traits<A>::size_type size) {
			using Traits	= std::allocator_traits<A>;
			using Pointer	= typename Traits::pointer;
			using ValueType = typename Traits::value_type;

			// If the type cannot be trivially destructed, we need to
			// destroy each value
			if (!typetraits::TriviallyDefaultConstructible<ValueType>::value) {
				for (Pointer p = ptr; p != ptr + size; ++p) { Traits::destroy(alloc, p); }
			}

#if defined(LIBRAPID_BLAS_MKLBLAS)
			mkl_free(ptr);
#else
#	if defined(LIBRAPID_NATIVE_ARCH) && defined(LIBRAPID_MSVC)
			_aligned_free(ptr);
#	else
			Traits::deallocate(alloc, ptr, size);
#	endif
#endif
		}
	} // namespace detail

	template<typename T, typename A>
	Storage<T, A>::Storage(SizeType size, const Allocator &alloc) :
			m_allocator(alloc), m_begin(detail::safeAllocate(m_allocator, size)),
			m_end(m_begin + size), m_independent(true) {}

	template<typename T, typename A>
	Storage<T, A>::Storage(Scalar *begin, Scalar *end, bool independent) :
			m_allocator(Allocator()), m_begin(begin), m_end(end), m_independent(independent) {}

	template<typename T, typename A>
	Storage<T, A>::Storage(SizeType size, ConstReference value, const Allocator &alloc) :
			m_allocator(alloc), m_begin(detail::safeAllocate(m_allocator, size)),
			m_end(m_begin + size), m_independent(true) {
		// std::fill(m_begin, m_end, value);
		for (auto it = m_begin; it != m_end; ++it) { *it = value; }
	}

	template<typename T, typename A>
	Storage<T, A>::Storage(const Storage &other, const Allocator &alloc) :
			m_allocator(alloc), m_begin(nullptr), m_end(nullptr),
			m_independent(other.m_independent) {
		initData(other.begin(), other.end());
	}

	template<typename T, typename A>
	Storage<T, A>::Storage(Storage &&other) noexcept :
			m_allocator(std::move(other.m_allocator)), m_begin(other.m_begin), m_end(other.m_end),
			m_independent(other.m_independent) {
		other.m_begin = nullptr;
		other.m_end	  = nullptr;
	}

	template<typename T, typename A>
	template<typename V>
	Storage<T, A>::Storage(const std::initializer_list<V> &list, const Allocator &alloc) :
			m_allocator(alloc), m_begin(nullptr), m_end(nullptr), m_independent(true) {
		initData(list.begin(), list.end());
	}

	template<typename T, typename A>
	template<typename V>
	Storage<T, A>::Storage(const std::vector<V> &vector, const Allocator &alloc) :
			m_allocator(alloc), m_begin(nullptr), m_end(nullptr), m_independent(true) {
		initData(vector.begin(), vector.end());
	}

	template<typename T, typename A>
	template<typename V>
	auto Storage<T, A>::fromData(const std::initializer_list<V> &list) -> Storage {
		Storage ret;
		ret.initData(list.begin(), list.end());
		return ret;
	}

	template<typename T, typename A>
	template<typename V>
	auto Storage<T, A>::fromData(const std::vector<V> &vec) -> Storage {
		Storage ret;
		ret.initData(vec.begin(), vec.end());
		return ret;
	}

	template<typename T, typename A>
	Storage<T, A> &Storage<T, A>::operator=(const Storage &other) {
		if (this != &other) {
			LIBRAPID_ASSERT(m_independent || size() == other.size(),
							"Mismatched storage sizes. Cannot assign storage with {} elements to "
							"dependent storage with {} elements",
							other.size(),
							size());

			m_allocator =
			  std::allocator_traits<A>::select_on_container_copy_construction(other.m_allocator);
			resizeImpl(other.size(), 0); // Different sizes are handled here
			if (typetraits::TriviallyDefaultConstructible<T>::value) {
				// Use a slightly faster memcpy if the type is trivially default constructible
				std::uninitialized_copy(other.begin(), other.end(), m_begin);
			} else {
				// Otherwise, use the standard copy algorithm
				std::copy(other.begin(), other.end(), m_begin);
			}
		}
		return *this;
	}

	template<typename T, typename A>
	Storage<T, A> &Storage<T, A>::operator=(Storage &&other) LIBRAPID_RELEASE_NOEXCEPT {
		if (this != &other) {
			if (m_independent) {
				m_allocator = std::move(other.m_allocator);
				std::swap(m_begin, other.m_begin);
				std::swap(m_end, other.m_end);
				m_independent = other.m_independent;
			} else {
				LIBRAPID_ASSERT(
				  size() == other.size(),
				  "Mismatched storage sizes. Cannot assign storage with {} elements to "
				  "dependent storage with {} elements",
				  other.size(),
				  size());

				m_allocator = std::allocator_traits<A>::select_on_container_copy_construction(
				  other.m_allocator);
				resizeImpl(other.size(), 0);
				if (typetraits::TriviallyDefaultConstructible<T>::value) {
					// Use a slightly faster memcpy if the type is trivially default constructible
					std::uninitialized_copy(other.begin(), other.end(), m_begin);
				} else {
					// Otherwise, use the standard copy algorithm
					std::copy(other.begin(), other.end(), m_begin);
				}
			}
		}
		return *this;
	}

	template<typename T, typename A>
	Storage<T, A>::~Storage() {
		if (!m_independent) return;
		detail::safeDeallocate(m_allocator, m_begin, size());
		m_begin = nullptr;
		m_end	= nullptr;
	}

	template<typename T, typename A>
	template<typename P>
	void Storage<T, A>::initData(P begin, P end) {
		auto size = static_cast<SizeType>(std::distance(begin, end));
		m_begin	  = detail::safeAllocate(m_allocator, size);
		m_end	  = m_begin + size;
		if (typetraits::TriviallyDefaultConstructible<T>::value) {
			// Use a slightly faster memcpy if the type is trivially default constructible
			std::uninitialized_copy(begin, end, m_begin);
		} else {
			// Otherwise, use the standard copy algorithm
			std::copy(begin, end, m_begin);
		}
	}

	template<typename T, typename A>
	template<typename ShapeType>
	auto Storage<T, A>::defaultShape() -> ShapeType {
		return ShapeType({0});
	}

	template<typename T, typename A>
	auto Storage<T, A>::size() const noexcept -> SizeType {
		return static_cast<SizeType>(std::distance(m_begin, m_end));
	}

	template<typename T, typename A>
	void Storage<T, A>::resize(SizeType newSize) {
		resizeImpl(newSize);
	}

	template<typename T, typename A>
	void Storage<T, A>::resize(SizeType newSize, int) {
		resizeImpl(newSize, 0);
	}

	template<typename T, typename A>
	LIBRAPID_ALWAYS_INLINE void Storage<T, A>::resizeImpl(SizeType newSize) {
		if (newSize == size()) return;
		LIBRAPID_ASSERT(m_independent, "Dependent storage cannot be resized");

		SizeType oldSize = size();
		Pointer oldBegin = m_begin;
		// Reallocate
		m_begin = detail::safeAllocate(m_allocator, newSize);
		m_end	= m_begin + newSize;

		if constexpr (typetraits::TriviallyDefaultConstructible<T>::value) {
			// Use a slightly faster memcpy if the type is trivially default constructible
			std::uninitialized_copy(oldBegin, oldBegin + std::min(oldSize, newSize), m_begin);
		} else {
			// Otherwise, use the standard copy algorithm
			std::copy(oldBegin, oldBegin + std::min(oldSize, newSize), m_begin);
		}

		detail::safeDeallocate(m_allocator, oldBegin, oldSize);
	}

	template<typename T, typename A>
	LIBRAPID_ALWAYS_INLINE void Storage<T, A>::resizeImpl(SizeType newSize, int) {
		if (size() == newSize) return;
		LIBRAPID_ASSERT(m_independent, "Dependent storage cannot be resized");

		SizeType oldSize = size();
		Pointer oldBegin = m_begin;
		// Reallocate
		m_begin = detail::safeAllocate(m_allocator, newSize);
		m_end	= m_begin + newSize;
		detail::safeDeallocate(m_allocator, oldBegin, oldSize);
	}

	template<typename T, typename A>
	auto Storage<T, A>::operator[](Storage<T, A>::SizeType index) const -> ConstReference {
		LIBRAPID_ASSERT(index < size(), "Index {} out of bounds for size {}", index, size());
		return m_begin[index];
	}

	template<typename T, typename A>
	auto Storage<T, A>::operator[](Storage<T, A>::SizeType index) -> Reference {
		LIBRAPID_ASSERT(index < size(), "Index {} out of bounds for size {}", index, size());
		return m_begin[index];
	}

	template<typename T, typename A>
	auto Storage<T, A>::begin() noexcept -> Iterator {
		return m_begin;
	}

	template<typename T, typename A>
	auto Storage<T, A>::end() noexcept -> Iterator {
		return m_end;
	}

	template<typename T, typename A>
	auto Storage<T, A>::begin() const noexcept -> ConstIterator {
		return m_begin;
	}

	template<typename T, typename A>
	auto Storage<T, A>::end() const noexcept -> ConstIterator {
		return m_end;
	}

	template<typename T, typename A>
	auto Storage<T, A>::cbegin() const noexcept -> ConstIterator {
		return begin();
	}

	template<typename T, typename A>
	auto Storage<T, A>::cend() const noexcept -> ConstIterator {
		return end();
	}

	template<typename T, typename A>
	auto Storage<T, A>::rbegin() noexcept -> ReverseIterator {
		return ReverseIterator(m_end);
	}

	template<typename T, typename A>
	auto Storage<T, A>::rend() noexcept -> ReverseIterator {
		return ReverseIterator(m_begin);
	}

	template<typename T, typename A>
	auto Storage<T, A>::rbegin() const noexcept -> ConstReverseIterator {
		return ConstReverseIterator(m_end);
	}

	template<typename T, typename A>
	auto Storage<T, A>::rend() const noexcept -> ConstReverseIterator {
		return ConstReverseIterator(m_begin);
	}

	template<typename T, typename A>
	auto Storage<T, A>::crbegin() const noexcept -> ConstReverseIterator {
		return rbegin();
	}

	template<typename T, typename A>
	auto Storage<T, A>::crend() const noexcept -> ConstReverseIterator {
		return rend();
	}

	template<typename T, size_t... D>
	FixedStorage<T, D...>::FixedStorage() = default;

	template<typename T, size_t... D>
	FixedStorage<T, D...>::FixedStorage(const Scalar &value) {
		for (size_t i = 0; i < Size; ++i) { m_data[i] = value; }
	}

	template<typename T, size_t... D>
	FixedStorage<T, D...>::FixedStorage(const FixedStorage &other) = default;

	template<typename T, size_t... D>
	FixedStorage<T, D...>::FixedStorage(FixedStorage &&other) noexcept = default;

	template<typename T, size_t... D>
	FixedStorage<T, D...>::FixedStorage(const std::initializer_list<Scalar> &list) {
		LIBRAPID_ASSERT(list.size() == size(), "Initializer list size does not match storage size");
		for (size_t i = 0; i < Size; ++i) { m_data[i] = list.begin()[i]; }
	}

	template<typename T, size_t... D>
	FixedStorage<T, D...>::FixedStorage(const std::vector<Scalar> &vec) {
		LIBRAPID_ASSERT(vec.size() == size(), "Initializer list size does not match storage size");
		for (size_t i = 0; i < Size; ++i) { m_data[i] = vec[i]; }
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::operator=(const FixedStorage &other) -> FixedStorage & {
		if (this != &other) {
			for (size_t i = 0; i < Size; ++i) { m_data[i] = other.m_data[i]; }
		}
		return *this;
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::operator=(FixedStorage &&other) noexcept
	  -> FixedStorage & = default;

	template<typename T, size_t... D>
	FixedStorage<T, D...>::~FixedStorage() = default;

	template<typename T, size_t... D>
	template<typename ShapeType>
	auto FixedStorage<T, D...>::defaultShape() -> ShapeType {
		return ShapeType({D...});
	}

	template<typename T, size_t... D>
	void FixedStorage<T, D...>::resize(SizeType newSize) {
		LIBRAPID_ASSERT(newSize == size(), "FixedStorage cannot be resized");
	}

	template<typename T, size_t... D>
	void FixedStorage<T, D...>::resize(SizeType newSize, int) {
		LIBRAPID_ASSERT(newSize == size(), "FixedStorage cannot be resized");
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::size() const noexcept -> SizeType {
		return Size;
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::operator[](SizeType index) const -> ConstReference {
		LIBRAPID_ASSERT(index < size(), "Index out of bounds");
		return m_data[index];
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::operator[](SizeType index) -> Reference {
		LIBRAPID_ASSERT(index < size(), "Index out of bounds");
		return m_data[index];
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::begin() noexcept -> Iterator {
		return m_data.begin();
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::end() noexcept -> Iterator {
		return m_data.end();
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::begin() const noexcept -> ConstIterator {
		return m_data.begin();
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::end() const noexcept -> ConstIterator {
		return m_data.end();
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::cbegin() const noexcept -> ConstIterator {
		return begin();
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::cend() const noexcept -> ConstIterator {
		return end();
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::rbegin() noexcept -> ReverseIterator {
		return ReverseIterator(end());
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::rend() noexcept -> ReverseIterator {
		return ReverseIterator(begin());
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::rbegin() const noexcept -> ConstReverseIterator {
		return ConstReverseIterator(end());
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::rend() const noexcept -> ConstReverseIterator {
		return ConstReverseIterator(begin());
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::crbegin() const noexcept -> ConstReverseIterator {
		return rbegin();
	}

	template<typename T, size_t... D>
	auto FixedStorage<T, D...>::crend() const noexcept -> ConstReverseIterator {
		return rend();
	}
} // namespace librapid

#endif // LIBRAPID_ARRAY_DENSE_STORAGE_HPP