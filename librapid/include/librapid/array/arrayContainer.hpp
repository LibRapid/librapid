#ifndef LIBRAPID_ARRAY_ARRAY_CONTAINER_HPP
#define LIBRAPID_ARRAY_ARRAY_CONTAINER_HPP

namespace librapid {
	namespace typetraits {
		template<typename ShapeType_, typename StorageType_>
		struct TypeInfo<ArrayContainer<ShapeType_, StorageType_>> {
			static constexpr bool isLibRapidType = true;
			using Scalar						 = typename TypeInfo<StorageType_>::Scalar;
		};
	} // namespace typetraits

	// template<typename Scalar, typename Device = device::CPU>
	// using Array =
	//   ArrayContainer<Shape<size_t, 32>,
	// 				 typename std::conditional_t<typetraits::IsSame<Device, device::GPU>,
	// 											 CudaStorage<Scalar>, Storage<Scalar>>>;

	template<typename ShapeType_, typename StorageType_>
	class ArrayContainer {
	public:
		using StorageType = StorageType_;
		using ShapeType	  = ShapeType_;
		using SizeType	  = typename ShapeType::SizeType;
		using Scalar	  = typename StorageType::Scalar;
		using Packet	  = typename typetraits::TypeInfo<Scalar>::Packet;

		/// Default constructor.
		ArrayContainer() = default;

		/// Constructs an array container from a shape
		/// \param shape The shape of the array container
		LIBRAPID_ALWAYS_INLINE explicit ArrayContainer(const ShapeType &shape);

		/// Create an array container from a shape and a scalar value. The scalar value represents
		/// the value the memory is initialized with.
		/// \param shape The shape of the array container
		/// \param value The value to initialize the memory with
		LIBRAPID_ALWAYS_INLINE ArrayContainer(const ShapeType &shape, const Scalar &value);

		/// Allows for a fixed-size array to be constructed with a fill value
		/// \param value The value to fill the array with
		LIBRAPID_ALWAYS_INLINE explicit ArrayContainer(const Scalar &value);

		/// Construct an array container from a shape, which is moved, not copied.
		/// \param shape The shape of the array container
		LIBRAPID_ALWAYS_INLINE explicit ArrayContainer(ShapeType &&shape);

		/// Construct an array container from another array container.
		/// \param other The array container to copy.
		LIBRAPID_ALWAYS_INLINE ArrayContainer(const ArrayContainer &other) = default;

		/// Construct an array container from a temporary array container.
		/// \param other The array container to move.
		LIBRAPID_ALWAYS_INLINE ArrayContainer(ArrayContainer &&other) noexcept = default;

		/// Construct an array container from a function object. This will assign the result of
		/// the function to the array container, evaluating it accordingly.
		/// \tparam desc The assignment descriptor
		/// \tparam Functor_ The function type
		/// \tparam Args The argument types of the function
		/// \param function The function to assign
		template<detail::Descriptor desc, typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE explicit ArrayContainer(
		  const detail::Function<desc, Functor_, Args...> &function) LIBRAPID_RELEASE_NOEXCEPT;

		/// Assign an array container to this array container.
		/// \param other The array container to copy.
		/// \return A reference to this array container.
		LIBRAPID_ALWAYS_INLINE ArrayContainer &operator=(const ArrayContainer &other) = default;

		/// Assign a temporary array container to this array container.
		/// \param other The array container to move.
		/// \return A reference to this array container.
		LIBRAPID_ALWAYS_INLINE ArrayContainer &operator=(ArrayContainer &&other) noexcept = default;

		/// Assign a function object to this array container. This will assign the result of
		/// the function to the array container, evaluating it accordingly.
		/// \tparam Functor_ The function type
		/// \tparam Args The argument types of the function
		/// \param function The function to assign
		/// \return A reference to this array container.
		template<detail::Descriptor desc, typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE ArrayContainer &
		operator=(const detail::Function<desc, Functor_, Args...> &function);

		/// Return the shape of the array container. This is an immutable reference.
		/// \return The shape of the array container.
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const ShapeType &shape() const noexcept;

		/// Return the StorageType object of the ArrayContainer
		/// \return The StorageType object of the ArrayContainer
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const StorageType &storage() const noexcept;

		/// Return the StorageType object of the ArrayContainer
		/// \return The StorageType object of the ArrayContainer
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE StorageType &storage() noexcept;

		/// Return a Packet object from the array's storage at a specific index.
		/// \param index The index to get the packet from
		/// \return A Packet object from the array's storage at a specific index
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Packet packet(size_t index) const;

		/// Return a Scalar from the array's storage at a specific index.
		/// \param index The index to get the scalar from
		/// \return A Scalar from the array's storage at a specific index
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar scalar(size_t index) const;

		/// Write a Packet object to the array's storage at a specific index
		/// \param index The index to write the packet to
		/// \param value The value to write to the array's storage
		LIBRAPID_ALWAYS_INLINE void writePacket(size_t index, const Packet &value);

		/// Write a Scalar to the array's storage at a specific index
		/// \param index The index to write the scalar to
		/// \param value The value to write to the array's storage
		LIBRAPID_ALWAYS_INLINE void write(size_t index, const Scalar &value);

	private:
		ShapeType m_shape;	   // The shape type of the array
		StorageType m_storage; // The storage container of the array
	};

	template<typename ShapeType_, typename StorageType_>
	ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const ShapeType &shape) :
			m_shape(shape), m_storage(shape.size()) {}

	template<typename ShapeType_, typename StorageType_>
	ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const ShapeType &shape,
															 const Scalar &value) :
			m_shape(shape),
			m_storage(shape.size(), value) {
		static_assert(typetraits::IsStorage<StorageType_>::value ||
						typetraits::IsCudaStorage<StorageType_>::value,
					  "For a runtime-defined shape, "
					  "the storage type must be "
					  "either a Storage or a "
					  "CudaStorage object");
	}

	template<typename ShapeType_, typename StorageType_>
	ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const Scalar &value) :
			m_shape(detail::shapeFromFixedStorage(m_storage)), m_storage(value) {
		static_assert(typetraits::IsFixedStorage<StorageType_>::value,
					  "For a compile-time-defined shape, "
					  "the storage type must be "
					  "a FixedStorage object");
	}

	template<typename ShapeType_, typename StorageType_>
	ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(ShapeType_ &&shape) :
			m_shape(std::forward<ShapeType_>(shape)), m_storage(m_shape.size()) {}

	template<typename ShapeType_, typename StorageType_>
	template<detail::Descriptor desc, typename Functor_, typename... Args>
	ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(
	  const detail::Function<desc, Functor_, Args...> &function) LIBRAPID_RELEASE_NOEXCEPT
			: m_shape(function.shape()),
			  m_storage(m_shape.size()) {
#if !defined(LIBRAPID_OPTIMISE_SMALL_ARRAYS)
		if (m_storage.size() > global::multithreadThreshold && global::numThreads > 1)
			detail::assignParallel(*this, function);
		else
#endif // LIBRAPID_OPTIMISE_SMALL_ARRAYS
			detail::assign(*this, function);
	}

	template<typename ShapeType_, typename StorageType_>
	template<detail::Descriptor desc, typename Functor_, typename... Args>
	ArrayContainer<ShapeType_, StorageType_> &ArrayContainer<ShapeType_, StorageType_>::operator=(
	  const detail::Function<desc, Functor_, Args...> &function) {
		m_storage.resize(function.shape().size(), 0);
#if !defined(LIBRAPID_OPTIMISE_SMALL_ARRAYS)
		if (m_storage.size() > global::multithreadThreshold && global::numThreads > 1)
			detail::assignParallel(*this, function);
		else
#endif // LIBRAPID_OPTIMISE_SMALL_ARRAYS
			detail::assign(*this, function);
		return *this;
	}

	template<typename ShapeType_, typename StorageType_>
	auto ArrayContainer<ShapeType_, StorageType_>::shape() const noexcept -> const ShapeType & {
		return m_shape;
	}

	template<typename ShapeType_, typename StorageType_>
	auto ArrayContainer<ShapeType_, StorageType_>::storage() const noexcept -> const StorageType & {
		return m_storage;
	}

	template<typename ShapeType_, typename StorageType_>
	auto ArrayContainer<ShapeType_, StorageType_>::storage() noexcept -> StorageType & {
		return m_storage;
	}

	template<typename ShapeType_, typename StorageType_>
	auto ArrayContainer<ShapeType_, StorageType_>::packet(size_t index) const -> Packet {
		Packet res;
		res.load(m_storage.begin() + index);
		return res;
	}

	template<typename ShapeType_, typename StorageType_>
	auto ArrayContainer<ShapeType_, StorageType_>::scalar(size_t index) const -> Scalar {
		return m_storage[index];
	}

	template<typename ShapeType_, typename StorageType_>
	void ArrayContainer<ShapeType_, StorageType_>::writePacket(size_t index, const Packet &value) {
		value.store(m_storage.begin() + index);
	}

	template<typename ShapeType_, typename StorageType_>
	void ArrayContainer<ShapeType_, StorageType_>::write(size_t index, const Scalar &value) {
		m_storage[index] = value;
	}

	namespace typetraits {
		template<typename T>
		struct IsArrayContainer : std::false_type {};

		template<typename SizeType, size_t dims, typename StorageScalar>
		struct IsArrayContainer<ArrayContainer<Shape<SizeType, dims>, StorageScalar>>
				: std::true_type {};
	} // namespace typetraits
} // namespace librapid

#endif // LIBRAPID_ARRAY_ARRAY_CONTAINER_HPP