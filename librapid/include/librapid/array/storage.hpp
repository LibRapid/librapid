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
		};

		template<typename Scalar_, size_t Dims>
		struct TypeInfo<FixedStorage<Scalar_, Dims>> {
			static constexpr bool isLibRapidType = true;
			using Scalar						 = Scalar_;
		};
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

		/// Assignment operator for a Storage object
		/// \param other Storage object to copy
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Storage &operator=(const Storage &other);

		/// Move assignment operator for a Storage object
		/// \param other Storage object to move
		/// \return *this
		LIBRAPID_ALWAYS_INLINE Storage &operator=(Storage &&other) noexcept;

		/// Free a Storage object
		~Storage();

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
		Pointer m_begin = nullptr; // It is more efficient to store pointers to the start
		Pointer m_end	= nullptr; // and end of the data block than to store the size
	};

	namespace detail {
		template<typename Scalar, size_t Dimensions>
		struct FixedStorageContainer {
			Scalar data[Dimensions];
		};
	} // namespace detail

	template<typename Scalar_, size_t Size_>
	class FixedStorage {
	public:
		using Scalar				   = Scalar_;
		using Pointer				   = Scalar *;
		using ConstPointer			   = const Scalar *;
		using Reference				   = Scalar &;
		using ConstReference		   = const Scalar &;
		using SizeType				   = size_t;
		using DifferenceType		   = ptrdiff_t;
		using Iterator				   = Pointer;
		using ConstIterator			   = ConstPointer;
		using ReverseIterator		   = std::reverse_iterator<Iterator>;
		using ConstReverseIterator	   = std::reverse_iterator<ConstIterator>;
		static constexpr SizeType Size = Size_;

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
		detail::FixedStorageContainer<Scalar, Size> *m_data = nullptr;
		Scalar *__restrict m_begin							= nullptr;
		Scalar *__restrict m_end							= nullptr;
	};

	// Trait implementations
	namespace typetraits {
		template<typename T>
		struct IsStorage : std::false_type {};

		template<typename Scalar, typename Allocator>
		struct IsStorage<Storage<Scalar, Allocator>> : std::true_type {};

		template<typename T>
		struct IsFixedStorage : std::false_type {};

		template<typename Scalar, size_t Size>
		struct IsFixedStorage<FixedStorage<Scalar, Size>> : std::true_type {};
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
			Pointer ptr		= alloc.allocate(size);

			// If the type cannot be trivially constructed, we need to
			// initialize each value
			if (!typetraits::TriviallyDefaultConstructible<ValueType>::value) {
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
			Traits::deallocate(alloc, ptr, size);
		}
	} // namespace detail

	template<typename T, typename A>
	Storage<T, A>::Storage(SizeType size, const Allocator &alloc) :
			m_allocator(alloc), m_begin(detail::safeAllocate(m_allocator, size)),
			m_end(m_begin + size) {}

	template<typename T, typename A>
	Storage<T, A>::Storage(SizeType size, ConstReference value, const Allocator &alloc) :
			m_allocator(alloc), m_begin(detail::safeAllocate(m_allocator, size)),
			m_end(m_begin + size) {
		std::fill(m_begin, m_end, value);
	}

	template<typename T, typename A>
	Storage<T, A>::Storage(const Storage &other, const Allocator &alloc) :
			m_allocator(alloc), m_begin(nullptr), m_end(nullptr) {
		initData(other.begin(), other.end());
	}

	template<typename T, typename A>
	Storage<T, A>::Storage(Storage &&other) noexcept :
			m_allocator(std::move(other.m_allocator)), m_begin(other.m_begin), m_end(other.m_end) {
		other.m_begin = nullptr;
		other.m_end	  = nullptr;
	}

	template<typename T, typename A>
	template<typename V>
	Storage<T, A>::Storage(const std::initializer_list<V> &list, const Allocator &alloc) :
			m_allocator(alloc), m_begin(nullptr), m_end(nullptr) {
		initData(list.begin(), list.end());
	}

	template<typename T, typename A>
	template<typename V>
	Storage<T, A>::Storage(const std::vector<V> &vector, const Allocator &alloc) :
			m_allocator(alloc), m_begin(nullptr), m_end(nullptr) {
		initData(vector.begin(), vector.end());
	}

	template<typename T, typename A>
	Storage<T, A> &Storage<T, A>::operator=(const Storage &other) {
		if (this != &other) {
			m_allocator =
			  std::allocator_traits<A>::select_on_container_copy_construction(other.m_allocator);
			resizeImpl(other.size());
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
	Storage<T, A> &Storage<T, A>::operator=(Storage &&other) noexcept {
		if (this != &other) {
			m_allocator = std::move(other.m_allocator);
			std::swap(m_begin, other.m_begin);
			std::swap(m_end, other.m_end);
		}
		return *this;
	}

	template<typename T, typename A>
	Storage<T, A>::~Storage() {
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
	auto Storage<T, A>::size() const noexcept -> SizeType {
		return static_cast<SizeType>(std::distance(m_begin, m_end));
	}

	template<typename T, typename A>
	void Storage<T, A>::resize(SizeType newSize) {
		resizeImpl(newSize);
	}

	template<typename T, typename A>
	void Storage<T, A>::resize(SizeType newSize, int) {
		resizeImpl(newSize);
	}

	template<typename T, typename A>
	LIBRAPID_ALWAYS_INLINE void Storage<T, A>::resizeImpl(SizeType newSize) {
		if (newSize == size()) { return; }

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
		SizeType oldSize = size();
		Pointer oldBegin = m_begin;
		if (oldSize != newSize) {
			// Reallocate
			m_begin = detail::safeAllocate(m_allocator, newSize);
			m_end	= m_begin + newSize;
			detail::safeDeallocate(m_allocator, oldBegin, oldSize);
		}
	}

	template<typename T, typename A>
	auto Storage<T, A>::operator[](Storage<T, A>::SizeType index) const -> ConstReference {
		LIBRAPID_ASSERT(index < size(), "Index out of bounds");
		return m_begin[index];
	}

	template<typename T, typename A>
	auto Storage<T, A>::operator[](Storage<T, A>::SizeType index) -> Reference {
		LIBRAPID_ASSERT(index < size(), "Index out of bounds");
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

	template<typename T, size_t D>
	FixedStorage<T, D>::FixedStorage() :
			m_data(new detail::FixedStorageContainer<Scalar, Size>), m_begin(m_data->data),
			m_end(m_data->data + Size) {}

	template<typename T, size_t D>
	FixedStorage<T, D>::FixedStorage(const Scalar &value) :
			m_data(new detail::FixedStorageContainer<Scalar, Size>), m_begin(m_data->data),
			m_end(m_data->data + Size) {
		std::fill(begin(), end(), value);
	}

	template<typename T, size_t D>
	FixedStorage<T, D>::FixedStorage(const FixedStorage &other) :
			m_data(new detail::FixedStorageContainer<Scalar, Size>), m_begin(m_data->data),
			m_end(m_data->data + Size) {
		if (typetraits::TriviallyDefaultConstructible<T>::value) {
			// Use a slightly faster memcpy if the type is trivially default constructible
			std::uninitialized_copy(other.begin(), other.end(), m_begin);
		} else {
			// Otherwise, use the standard copy algorithm
			std::copy(other.begin(), other.end(), m_begin);
		}
	}

	template<typename T, size_t D>
	FixedStorage<T, D>::FixedStorage(FixedStorage &&other) noexcept {
		m_data		  = other.m_data;
		m_begin		  = other.m_begin;
		m_end		  = other.m_end;
		other.m_data  = nullptr;
		other.m_begin = nullptr;
		other.m_end	  = nullptr;
	}

	template<typename T, size_t D>
	FixedStorage<T, D>::FixedStorage(const std::initializer_list<Scalar> &list) {
		LIBRAPID_ASSERT(list.size() == size(), "Initializer list size does not match storage size");
		m_data	= new detail::FixedStorageContainer<Scalar, Size>;
		m_begin = m_data->data;
		m_end	= m_data->data + Size;
		if (typetraits::TriviallyDefaultConstructible<T>::value) {
			// Use a slightly faster memcpy if the type is trivially default constructible
			std::uninitialized_copy(list.begin(), list.end(), m_begin);
		} else {
			// Otherwise, use the standard copy algorithm
			std::copy(list.begin(), list.end(), m_begin);
		}
	}

	template<typename T, size_t D>
	FixedStorage<T, D>::FixedStorage(const std::vector<Scalar> &vec) {
		LIBRAPID_ASSERT(vec.size() == size(), "Initializer list size does not match storage size");
		m_data	= new detail::FixedStorageContainer<Scalar, Size>;
		m_begin = m_data->data;
		m_end	= m_data->data + Size;
		if (typetraits::TriviallyDefaultConstructible<T>::value) {
			// Use a slightly faster memcpy if the type is trivially default constructible
			std::uninitialized_copy(vec.begin(), vec.end(), m_begin);
		} else {
			// Otherwise, use the standard copy algorithm
			std::copy(vec.begin(), vec.end(), m_begin);
		}
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::operator=(const FixedStorage &other) -> FixedStorage & {
		if (this != &other) std::copy(other.begin(), other.end(), begin());
		return *this;
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::operator=(FixedStorage &&other) noexcept -> FixedStorage & {
		if (this != &other) {
			delete m_data;
			m_data		  = other.m_data;
			m_begin		  = m_data->data;
			m_end		  = m_data->data + Size;
			other.m_data  = nullptr;
			other.m_begin = nullptr;
			other.m_end	  = nullptr;
		}
		return *this;
	}

	template<typename T, size_t D>
	FixedStorage<T, D>::~FixedStorage() {
		delete m_data;
	}

	template<typename T, size_t D>
	void FixedStorage<T, D>::resize(SizeType newSize) {
		LIBRAPID_ASSERT(newSize == size(), "FixedStorage cannot be resized");
	}

	template<typename T, size_t D>
	void FixedStorage<T, D>::resize(SizeType newSize, int) {
		LIBRAPID_ASSERT(newSize == size(), "FixedStorage cannot be resized");
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::size() const noexcept -> SizeType {
		return Size;
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::operator[](SizeType index) const -> ConstReference {
		LIBRAPID_ASSERT(index < size(), "Index out of bounds");
		return m_begin[index];
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::operator[](SizeType index) -> Reference {
		LIBRAPID_ASSERT(index < size(), "Index out of bounds");
		return m_begin[index];
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::begin() noexcept -> Iterator {
		return m_begin;
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::end() noexcept -> Iterator {
		return m_end;
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::begin() const noexcept -> ConstIterator {
		return m_data->data;
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::end() const noexcept -> ConstIterator {
		return m_data->data + size();
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::cbegin() const noexcept -> ConstIterator {
		return begin();
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::cend() const noexcept -> ConstIterator {
		return end();
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::rbegin() noexcept -> ReverseIterator {
		return ReverseIterator(end());
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::rend() noexcept -> ReverseIterator {
		return ReverseIterator(begin());
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::rbegin() const noexcept -> ConstReverseIterator {
		return ConstReverseIterator(end());
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::rend() const noexcept -> ConstReverseIterator {
		return ConstReverseIterator(begin());
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::crbegin() const noexcept -> ConstReverseIterator {
		return rbegin();
	}

	template<typename T, size_t D>
	auto FixedStorage<T, D>::crend() const noexcept -> ConstReverseIterator {
		return rend();
	}
} // namespace librapid

#endif // LIBRAPID_ARRAY_DENSE_STORAGE_HPP