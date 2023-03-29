#ifndef LIBRAPID_ARRAY_ARRAY_CONTAINER_HPP
#define LIBRAPID_ARRAY_ARRAY_CONTAINER_HPP

namespace librapid {
	namespace typetraits {
		template<typename ShapeType_, typename StorageType_>
		struct TypeInfo<array::ArrayContainer<ShapeType_, StorageType_>> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::ArrayContainer;
			using Scalar							   = typename TypeInfo<StorageType_>::Scalar;
			using Device							   = typename TypeInfo<StorageType_>::Device;
			static constexpr bool allowVectorisation   = TypeInfo<Scalar>::packetWidth > 1;
		};

		/// Evaluates as true if the input type is an ArrayContainer instance
		/// \tparam T Input type
		template<typename T>
		struct IsArrayContainer : std::false_type {};

		template<typename SizeType, size_t dims, typename StorageScalar>
		struct IsArrayContainer<array::ArrayContainer<Shape<SizeType, dims>, StorageScalar>>
				: std::true_type {};

		LIBRAPID_DEFINE_AS_TYPE(
		  typename SizeType COMMA size_t dims COMMA typename StorageScalar,
		  array::ArrayContainer<Shape<SizeType COMMA dims> COMMA StorageScalar>);
	} // namespace typetraits

	namespace array {
		template<typename ShapeType_, typename StorageType_>
		class ArrayContainer {
		public:
			using StorageType = StorageType_;
			using ShapeType	  = ShapeType_;
			using StrideType  = Stride<size_t, 32>;
			using SizeType	  = typename ShapeType::SizeType;
			using Scalar	  = typename StorageType::Scalar;
			using Packet	  = typename typetraits::TypeInfo<Scalar>::Packet;

			/// Default constructor
			ArrayContainer();

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer(const std::initializer_list<T> &data);

			template<typename T>
			explicit LIBRAPID_ALWAYS_INLINE ArrayContainer(const std::vector<T> &data);

			/// Constructs an array container from a shape
			/// \param shape The shape of the array container
			LIBRAPID_ALWAYS_INLINE explicit ArrayContainer(const ShapeType &shape);

			/// Create an array container from a shape and a scalar value. The scalar value
			/// represents the value the memory is initialized with. \param shape The shape of the
			/// array container \param value The value to initialize the memory with
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
			template<typename desc, typename Functor_, typename... Args>
			LIBRAPID_ALWAYS_INLINE ArrayContainer(
			  const detail::Function<desc, Functor_, Args...> &function) LIBRAPID_RELEASE_NOEXCEPT;

			/// Assign an array container to this array container.
			/// \param other The array container to copy.
			/// \return A reference to this array container.
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator=(const ArrayContainer &other) = default;

			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator=(const Scalar &value);

			/// Assign a temporary array container to this array container.
			/// \param other The array container to move.
			/// \return A reference to this array container.
			LIBRAPID_ALWAYS_INLINE ArrayContainer &
			operator=(ArrayContainer &&other) noexcept = default;

			/// Assign a function object to this array container. This will assign the result of
			/// the function to the array container, evaluating it accordingly.
			/// \tparam Functor_ The function type
			/// \tparam Args The argument types of the function
			/// \param function The function to assign
			/// \return A reference to this array container.
			template<typename desc, typename Functor_, typename... Args>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &
			operator=(const detail::Function<desc, Functor_, Args...> &function);

			template<typename TransposeType>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &
			operator=(const Transpose<TransposeType> &transpose);

			/// Allow ArrayContainer objects to be initialized with a comma separated list of
			/// values. This makes use of the CommaInitializer class
			/// \tparam T The type of the values
			/// \param value The value to set in the Array object
			/// \return The comma initializer object
			template<typename T>
			detail::CommaInitializer<ArrayContainer> operator<<(const T &value);

			/// Access a sub-array of this ArrayContainer instance. The sub-array will reference
			/// the same memory as this ArrayContainer instance.
			/// \param index The index of the sub-array
			/// \return A reference to the sub-array (ArrayView)
			/// \see ArrayView
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](int64_t index) const;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](int64_t index);

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar get() const;

			/// Return the number of dimensions of the ArrayContainer object
			/// \return Number of dimensions of the ArrayContainer
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE typename ShapeType::SizeType
			ndim() const noexcept;

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

			/// Return a string representation of the array container
			/// \format The format to use for the string representation
			/// \return A string representation of the array container
			LIBRAPID_NODISCARD std::string str(const std::string &format = "{}") const;

		private:
			ShapeType m_shape;	   // The shape type of the array
			StorageType m_storage; // The storage container of the array
		};

		template<typename ShapeType_, typename StorageType_>
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer() :
				m_shape(StorageType_::template defaultShape<ShapeType_>()) {}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(
		  const std::initializer_list<T> &data) :
				m_shape({data.size()}),
				m_storage(StorageType::fromData(data)) {}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const std::vector<T> &data) :
				m_shape({data.size()}), m_storage(StorageType::fromData(data)) {}

		template<typename ShapeType_, typename StorageType_>
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const ShapeType &shape) :
				m_shape(shape), m_storage(shape.size()) {
			static_assert(!typetraits::IsFixedStorage<StorageType_>::value,
						  "For a compile-time-defined shape, "
						  "the storage type must be "
						  "a FixedStorage object");
		}

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
			static_assert(!typetraits::IsFixedStorage<StorageType_>::value,
						  "For a compile-time-defined shape, "
						  "the storage type must be "
						  "a FixedStorage object");
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
		template<typename desc, typename Functor_, typename... Args>
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
		template<typename desc, typename Functor_, typename... Args>
		auto ArrayContainer<ShapeType_, StorageType_>::operator=(
		  const detail::Function<desc, Functor_, Args...> &function) -> ArrayContainer & {
			using FunctionType = detail::Function<desc, Functor_, Args...>;
			m_storage.resize(function.shape().size(), 0);
#if !defined(LIBRAPID_OPTIMISE_SMALL_ARRAYS)
			if (!std::is_same_v<typename FunctionType::Device, device::GPU> &&
				m_storage.size() > global::multithreadThreshold && global::numThreads > 1)
				detail::assignParallel(*this, function);
			else
#endif // LIBRAPID_OPTIMISE_SMALL_ARRAYS
				detail::assign(*this, function);
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename TransposeType>
		auto ArrayContainer<ShapeType_, StorageType_>::operator=(
		  const Transpose<TransposeType> &transpose) -> ArrayContainer & {
			m_shape = transpose.shape();
			m_storage.resize(m_shape.size(), 0);
			transpose.applyTo(*this);
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		auto ArrayContainer<ShapeType_, StorageType_>::operator=(const Scalar &value)
		  -> ArrayContainer & {
			LIBRAPID_ASSERT(m_shape.ndim() == 0, "Cannot assign a scalar to an array");
			m_storage[0] = value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		auto ArrayContainer<ShapeType_, StorageType_>::operator<<(const T &value)
		  -> detail::CommaInitializer<ArrayContainer> {
			return detail::CommaInitializer<ArrayContainer>(*this, static_cast<Scalar>(value));
		}

		template<typename ShapeType_, typename StorageType_>
		auto ArrayContainer<ShapeType_, StorageType_>::operator[](int64_t index) const {
			LIBRAPID_ASSERT(
			  index >= 0 && index < static_cast<int64_t>(m_shape[0]),
			  "Index {} out of bounds in ArrayContainer::operator[] with leading dimension={}",
			  index,
			  m_shape[0]);

			if constexpr (std::is_same_v<typename typetraits::TypeInfo<ArrayContainer>::Device,
										 device::GPU>) {
				ArrayContainer res;
				res.m_shape	  = m_shape.subshape(1, ndim());
				auto subSize  = res.shape().size();
				Scalar *begin = m_storage.begin().get() + index * subSize;
				res.m_storage = StorageType_(begin, subSize, false);

				return res;
			} else {
				ArrayContainer res;
				res.m_shape	  = m_shape.subshape(1, ndim());
				auto subSize  = res.shape().size();
				Scalar *begin = m_storage.begin() + index * subSize;
				Scalar *end	  = begin + subSize;
				res.m_storage = StorageType_(begin, end, false);

				return res;
			}
		}

		template<typename ShapeType_, typename StorageType_>
		auto ArrayContainer<ShapeType_, StorageType_>::operator[](int64_t index) {
			LIBRAPID_ASSERT(
			  index >= 0 && index < static_cast<int64_t>(m_shape[0]),
			  "Index {} out of bounds in ArrayContainer::operator[] with leading dimension={}",
			  index,
			  m_shape[0]);

			if constexpr (std::is_same_v<typename typetraits::TypeInfo<ArrayContainer>::Device,
										 device::GPU>) {
				ArrayContainer res;
				res.m_shape	  = m_shape.subshape(1, ndim());
				auto subSize  = res.shape().size();
				Scalar *begin = m_storage.begin().get() + index * subSize;
				res.m_storage = StorageType_(begin, subSize, false);

				return res;
			} else {
				if constexpr (typetraits::IsFixedStorage<StorageType_>::value) {
					return ArrayView(*this)[index];
				} else {
					ArrayContainer res;
					res.m_shape	  = m_shape.subshape(1, ndim());
					auto subSize  = res.shape().size();
					Scalar *begin = m_storage.begin() + index * subSize;
					Scalar *end	  = begin + subSize;
					res.m_storage = StorageType_(begin, end, false);

					return res;
				}
			}
		}

		template<typename ShapeType_, typename StorageType_>
		auto ArrayContainer<ShapeType_, StorageType_>::get() const -> Scalar {
			LIBRAPID_ASSERT(m_shape.ndim() == 0,
							"Can only cast a scalar ArrayView to a salar object");
			return scalar(0);
		}

		template<typename ShapeType_, typename StorageType_>
		auto ArrayContainer<ShapeType_, StorageType_>::ndim() const noexcept ->
		  typename ShapeType_::SizeType {
			return m_shape.ndim();
		}

		template<typename ShapeType_, typename StorageType_>
		auto ArrayContainer<ShapeType_, StorageType_>::shape() const noexcept -> const ShapeType & {
			return m_shape;
		}

		template<typename ShapeType_, typename StorageType_>
		auto ArrayContainer<ShapeType_, StorageType_>::storage() const noexcept
		  -> const StorageType & {
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
		void ArrayContainer<ShapeType_, StorageType_>::writePacket(size_t index,
																   const Packet &value) {
			value.store(m_storage.begin() + index);
		}

		template<typename ShapeType_, typename StorageType_>
		void ArrayContainer<ShapeType_, StorageType_>::write(size_t index, const Scalar &value) {
			m_storage[index] = value;
		}

		template<typename ShapeType_, typename StorageType_>
		std::string ArrayContainer<ShapeType_, StorageType_>::str(const std::string &format) const {
			return ArrayView(*this).str(format);
		}
	} // namespace array

	namespace detail {
		template<typename T>
		struct IsArrayType {
			static constexpr bool val = false;
		};

		template<typename T>
		struct IsArrayType<ArrayRef<T>> {
			static constexpr bool val = true;
		};

		template<typename... T>
		struct IsArrayType<FunctionRef<T...>> {
			static constexpr bool val = true;
		};

		template<typename T>
		struct IsArrayType<array::ArrayView<T>> {
			static constexpr bool val = true;
		};

		template<typename First, typename... Types>
		struct ContainsArrayType {
			static constexpr auto evaluator = []() {
				if constexpr (sizeof...(Types) == 0)
					return IsArrayType<First>::val;
				else
					return IsArrayType<First>::val || ContainsArrayType<Types...>::val;
			};

			static constexpr bool val = evaluator();
		};
	}; // namespace detail
} // namespace librapid

// Support FMT printing
#ifdef FMT_API
LIBRAPID_SIMPLE_IO_IMPL(typename ShapeType_ COMMA typename StorageType_,
						librapid::array::ArrayContainer<ShapeType_ COMMA StorageType_>)
#endif // FMT_API

#endif // LIBRAPID_ARRAY_ARRAY_CONTAINER_HPP