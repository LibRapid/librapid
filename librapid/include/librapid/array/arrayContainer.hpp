#ifndef LIBRAPID_ARRAY_ARRAY_CONTAINER_HPP
#define LIBRAPID_ARRAY_ARRAY_CONTAINER_HPP

namespace librapid {
	namespace detail {
		template<typename T>
		struct SubscriptType {
			using Scalar = T;
			using Direct = const Scalar &;
			using Ref	 = Scalar &;
		};

		template<typename T>
		struct SubscriptType<Storage<T>> {
			using Scalar = T;
			using Direct = const Scalar &;
			using Ref	 = Scalar &;
		};

		template<typename T, size_t... Dims>
		struct SubscriptType<FixedStorage<T, Dims...>> {
			using Scalar = T;
			using Direct = const Scalar &;
			using Ref	 = Scalar &;
		};

#if defined(LIBRAPID_HAS_OPENCL)
		template<typename T>
		struct SubscriptType<OpenCLStorage<T>> {
			using Scalar = T;
			using Direct = const OpenCLRef<Scalar>;
			using Ref	 = OpenCLRef<Scalar>;
		};
#endif // LIBRAPID_HAS_OPENCL

#if defined(LIBRAPID_HAS_CUDA)
		template<typename T>
		struct SubscriptType<CudaStorage<T>> {
			using Scalar = T;
			using Direct = const detail::CudaRef<Scalar>;
			using Ref	 = detail::CudaRef<Scalar>;
		};
#endif // LIBRAPID_HAS_CUDA
	}  // namespace detail

	namespace typetraits {
		template<typename ShapeType_, typename StorageType_>
		struct TypeInfo<array::ArrayContainer<ShapeType_, StorageType_>> {
			static constexpr detail::LibRapidType type = detail::LibRapidType::ArrayContainer;
			using Scalar							   = typename TypeInfo<StorageType_>::Scalar;
			using Packet							   = std::false_type;
			using Backend							   = typename TypeInfo<StorageType_>::Backend;
			using ShapeType							   = ShapeType_;
			using StorageType						   = StorageType_;
			static constexpr int64_t packetWidth	   = 1;
			static constexpr bool supportsArithmetic   = TypeInfo<Scalar>::supportsArithmetic;
			static constexpr bool supportsLogical	   = TypeInfo<Scalar>::supportsLogical;
			static constexpr bool supportsBinary	   = TypeInfo<Scalar>::supportsBinary;
			static constexpr bool allowVectorisation   = TypeInfo<Scalar>::packetWidth > 1;

#if defined(LIBRAPID_HAS_CUDA)
			static constexpr cudaDataType_t CudaType = TypeInfo<Scalar>::CudaType;
			static constexpr int64_t cudaPacketWidth = 1;
#endif // LIBRAPID_HAS_CUDA

			static constexpr bool canAlign	   = false;
			static constexpr int64_t canMemcpy = false;
		};

		/// Evaluates as true if the input type is an ArrayContainer instance
		/// \tparam T Input type
		template<typename T>
		struct IsArrayContainer : std::false_type {};

		template<typename ShapeType, typename StorageScalar>
		struct IsArrayContainer<array::ArrayContainer<ShapeType, StorageScalar>> : std::true_type {
		};

		LIBRAPID_DEFINE_AS_TYPE(typename StorageScalar,
								array::ArrayContainer<Shape COMMA StorageScalar>);

		LIBRAPID_DEFINE_AS_TYPE(typename StorageScalar,
								array::ArrayContainer<MatrixShape COMMA StorageScalar>);
	} // namespace typetraits

	namespace array {
		template<typename ShapeType_, typename StorageType_>
		class ArrayContainer {
		public:
			using StorageType = StorageType_;
			using ShapeType	  = ShapeType_;
			using StrideType  = Stride<ShapeType>;
			using SizeType	  = typename ShapeType::SizeType;
			using Scalar	  = typename StorageType::Scalar;
			using Packet	  = typename typetraits::TypeInfo<Scalar>::Packet;
			using Backend	  = typename typetraits::TypeInfo<ArrayContainer>::Backend;

			using DirectSubscriptType	 = typename detail::SubscriptType<StorageType>::Direct;
			using DirectRefSubscriptType = typename detail::SubscriptType<StorageType>::Ref;

			/// Default constructor
			ArrayContainer();

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer(const std::initializer_list<T> &data);

			template<typename T>
			explicit LIBRAPID_ALWAYS_INLINE ArrayContainer(const std::vector<T> &data);

			// clang-format off
#define SINIT(SUB_TYPE) std::initializer_list<SUB_TYPE>
#define SVEC(SUB_TYPE)	std::vector<SUB_TYPE>

#define ARRAY_FROM_DATA_DEF(TYPE_INIT, TYPE_VEC)                                                   \
	LIBRAPID_NODISCARD static LIBRAPID_ALWAYS_INLINE auto fromData(const TYPE_INIT &data)          \
	  -> ArrayContainer;                                                                           \
	LIBRAPID_NODISCARD static LIBRAPID_ALWAYS_INLINE auto fromData(const TYPE_VEC &data)           \
	  -> ArrayContainer

			ARRAY_FROM_DATA_DEF(SINIT(Scalar), SVEC(Scalar));
			ARRAY_FROM_DATA_DEF(SINIT(SINIT(Scalar)), SVEC(SVEC(Scalar)));
			ARRAY_FROM_DATA_DEF(SINIT(SINIT(SINIT(Scalar))), SVEC(SVEC(SVEC(Scalar))));
			ARRAY_FROM_DATA_DEF(SINIT(SINIT(SINIT(SINIT(Scalar)))), SVEC(SVEC(SVEC(SVEC(Scalar)))));
			ARRAY_FROM_DATA_DEF(SINIT(SINIT(SINIT(SINIT(SINIT(Scalar))))), SVEC(SVEC(SVEC(SVEC(SVEC(Scalar))))));
			ARRAY_FROM_DATA_DEF(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(Scalar)))))), SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(Scalar)))))));
			ARRAY_FROM_DATA_DEF(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(Scalar))))))), SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(Scalar))))))));
			ARRAY_FROM_DATA_DEF(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(SINIT(Scalar)))))))), SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(SVEC(Scalar)))))))));

#undef SINIT
#undef SVEC

			// clang-format on

			/// Constructs an array container from a shape
			/// \param shape The shape of the array container
			LIBRAPID_ALWAYS_INLINE explicit ArrayContainer(const Shape &shape);
			LIBRAPID_ALWAYS_INLINE explicit ArrayContainer(const MatrixShape &shape);
			LIBRAPID_ALWAYS_INLINE explicit ArrayContainer(const VectorShape &shape);

			/// Create an array container from a shape and a scalar value. The scalar value
			/// represents the value the memory is initialized with. \param shape The shape of the
			/// array container \param value The value to initialize the memory with
			LIBRAPID_ALWAYS_INLINE ArrayContainer(const Shape &shape, const Scalar &value);
			LIBRAPID_ALWAYS_INLINE ArrayContainer(const MatrixShape &shape, const Scalar &value);
			LIBRAPID_ALWAYS_INLINE ArrayContainer(const VectorShape &shape, const Scalar &value);

			/// Allows for a fixed-size array to be constructed with a fill value
			/// \param value The value to fill the array with
			LIBRAPID_ALWAYS_INLINE explicit ArrayContainer(const Scalar &value);

			/// Construct an array container from a shape, which is moved, not copied.
			/// \param shape The shape of the array container
			LIBRAPID_ALWAYS_INLINE explicit ArrayContainer(ShapeType &&shape);

			/// \brief Reference an existing array container
			///
			/// This constructor does not copy the data, but instead references the data of the
			/// input array container. This means that the input array container must outlive the
			/// constructed array container. Please use ``ArrayContainer::copy()`` if you want to
			/// copy the data.
			/// \param other The array container to reference
			LIBRAPID_ALWAYS_INLINE ArrayContainer(const ArrayContainer &other) = default;

			/// Construct an array container from a temporary array container.
			/// \param other The array container to move.
			LIBRAPID_ALWAYS_INLINE ArrayContainer(ArrayContainer &&other) noexcept = default;

			template<typename TransposeType>
			LIBRAPID_ALWAYS_INLINE ArrayContainer(const Transpose<TransposeType> &trans);

			template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
					 typename StorageTypeB, typename Alpha, typename Beta>
			LIBRAPID_ALWAYS_INLINE
			ArrayContainer(const linalg::ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB,
													   StorageTypeB, Alpha, Beta> &multiply);

			template<typename desc, typename Functor_, typename... Args>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &
			assign(const detail::Function<desc, Functor_, Args...> &function);

			/// Construct an array container from a function object. This will assign the result of
			/// the function to the array container, evaluating it accordingly.
			/// \tparam desc The assignment descriptor
			/// \tparam Functor_ The function type
			/// \tparam Args The argument types of the function
			/// \param function The function to assign
			template<typename desc, typename Functor_, typename... Args>
			LIBRAPID_ALWAYS_INLINE ArrayContainer(
			  const detail::Function<desc, Functor_, Args...> &function) LIBRAPID_RELEASE_NOEXCEPT;

			/// \brief Reference an existing array container
			///
			/// This assignment operator does not copy the data, but instead references the data of
			/// the input array container. This means that the input array container must outlive
			/// the constructed array container. Please use ``ArrayContainer::copy()`` if you want
			/// to copy the data.
			///
			/// \param other The array container to reference
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

			template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
					 typename StorageTypeB, typename Alpha, typename Beta>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &
			operator=(const linalg::ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB,
												  StorageTypeB, Alpha, Beta> &multiply);

			/// Allow ArrayContainer objects to be initialized with a comma separated list of
			/// values. This makes use of the CommaInitializer class
			/// \tparam T The type of the values
			/// \param value The value to set in the Array object
			/// \return The comma initializer object
			template<typename T>
			LIBRAPID_ALWAYS_INLINE detail::CommaInitializer<ArrayContainer>
			operator<<(const T &value);

			// template<typename ScalarTo = Scalar, typename BackendTo = Backend>
			// LIBRAPID_NODISCARD auto cast() const;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE ArrayContainer copy() const;

			/// Access a sub-array of this ArrayContainer instance. The sub-array will reference
			/// the same memory as this ArrayContainer instance.
			/// \param index The index of the sub-array
			/// \return A reference to the sub-array (ArrayView)
			/// \see ArrayView
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](int64_t index) const;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator[](int64_t index);

			template<typename... Indices>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE DirectSubscriptType
			operator()(Indices... indices) const;

			template<typename... Indices>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE DirectRefSubscriptType
			operator()(Indices... indices);

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar get() const;

			/// Return the number of dimensions of the ArrayContainer object
			/// \return Number of dimensions of the ArrayContainer
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE typename ShapeType::SizeType
			ndim() const noexcept;

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto size() const noexcept -> size_t;

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

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator+=(const T &other);

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator-=(const T &other);

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator*=(const T &other);

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator/=(const T &other);

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator%=(const T &other);

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator&=(const T &other);

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator|=(const T &other);

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator^=(const T &other);

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator<<=(const T &other);

			template<typename T>
			LIBRAPID_ALWAYS_INLINE ArrayContainer &operator>>=(const T &other);

			/// \brief Return an iterator to the beginning of the array container
			/// \return Iterator
			LIBRAPID_ALWAYS_INLINE auto begin() const noexcept;

			/// \brief Return an iterator to the end of the array container
			/// \return Iterator
			LIBRAPID_ALWAYS_INLINE auto end() const noexcept;

			/// \brief Return an iterator to the beginning of the array container
			/// \return Iterator
			LIBRAPID_ALWAYS_INLINE auto begin();

			/// \brief Return an iterator to the end of the array container
			/// \return Iterator
			LIBRAPID_ALWAYS_INLINE auto end();

			template<typename T, typename Char, typename Ctx>
			void str(const fmt::formatter<T, Char> &format, char bracket, char separator,
					 Ctx &ctx) const;

		private:
			ShapeType m_shape;	   // The shape type of the array
			size_t m_size;		   // The size of the array
			StorageType m_storage; // The storage container of the array
		};

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE ArrayContainer<ShapeType_, StorageType_>::ArrayContainer() :
				m_shape(StorageType_::template defaultShape<ShapeType_>()), m_size(0) {}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(
		  const std::initializer_list<T> &data) :
				m_shape({data.size()}),
				m_size(data.size()), m_storage(StorageType::fromData(data)) {}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const std::vector<T> &data) :
				m_shape({data.size()}),
				m_size(data.size()), m_storage(StorageType::fromData(data)) {}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const Shape &shape) :
				m_shape(shape),
				m_size(shape.size()), m_storage(m_size) {
			static_assert(!typetraits::IsFixedStorage<StorageType_>::value,
						  "For a compile-time-defined shape, "
						  "the storage type must be "
						  "a FixedStorage object");
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const MatrixShape &shape) :
				m_shape(shape),
				m_size(shape.size()), m_storage(m_size) {
			static_assert(!typetraits::IsFixedStorage<StorageType_>::value,
						  "For a compile-time-defined shape, "
						  "the storage type must be "
						  "a FixedStorage object");
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const VectorShape &shape) :
				m_shape(shape),
				m_size(shape.size()), m_storage(m_size) {
			static_assert(!typetraits::IsFixedStorage<StorageType_>::value,
						  "For a compile-time-defined shape, "
						  "the storage type must be "
						  "a FixedStorage object");
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const Shape &shape,
																 const Scalar &value) :
				m_shape(shape),
				m_size(shape.size()), m_storage(m_size, value) {
			static_assert(typetraits::IsStorage<StorageType_>::value ||
							typetraits::IsOpenCLStorage<StorageType_>::value ||
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
		LIBRAPID_ALWAYS_INLINE
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const MatrixShape &shape,
																 const Scalar &value) :
				m_shape(shape),
				m_size(shape.size()), m_storage(m_size, value) {
			static_assert(typetraits::IsStorage<StorageType_>::value ||
							typetraits::IsOpenCLStorage<StorageType_>::value ||
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
		LIBRAPID_ALWAYS_INLINE
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const VectorShape &shape,
																 const Scalar &value) :
				m_shape(shape),
				m_size(shape.size()), m_storage(m_size, value) {
			static_assert(typetraits::IsStorage<StorageType_>::value ||
							typetraits::IsOpenCLStorage<StorageType_>::value ||
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
		LIBRAPID_ALWAYS_INLINE
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const Scalar &value) :
				m_shape(detail::shapeFromFixedStorage(m_storage)),
				m_size(m_shape.size()), m_storage(value) {
			static_assert(typetraits::IsFixedStorage<StorageType_>::value,
						  "For a compile-time-defined shape, "
						  "the storage type must be "
						  "a FixedStorage object");
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE
		ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(ShapeType_ &&shape) :
				m_shape(std::forward<ShapeType_>(shape)),
				m_size(m_shape.size()), m_storage(m_size) {}

		template<typename ShapeType_, typename StorageType_>
		template<typename TransposeType>
		LIBRAPID_ALWAYS_INLINE ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(
		  const array::Transpose<TransposeType> &trans) {
			*this = trans;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		LIBRAPID_ALWAYS_INLINE ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(
		  const linalg::ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha,
									  Beta> &multiply) {
			*this = multiply;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename desc, typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::assign(
		  const detail::Function<desc, Functor_, Args...> &function) -> ArrayContainer & {
			using FunctionType = detail::Function<desc, Functor_, Args...>;
			m_storage.resize(function.size(), 0);
			if constexpr (std::is_same_v<typename FunctionType::Backend, backend::OpenCL> ||
						  std::is_same_v<typename FunctionType::Backend, backend::CUDA>) {
				detail::assign(*this, function);
			} else {
#if !defined(LIBRAPID_OPTIMISE_SMALL_ARRAYS)
				if (m_storage.size() > global::multithreadThreshold && global::numThreads > 1)
					detail::assignParallel(*this, function);
				else
#endif // LIBRAPID_OPTIMISE_SMALL_ARRAYS
					detail::assign(*this, function);
			}
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename desc, typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(
		  const detail::Function<desc, Functor_, Args...> &function) LIBRAPID_RELEASE_NOEXCEPT
				: m_shape(function.shape()),
				  m_size(function.size()),
				  m_storage(m_shape.size()) {
			assign(function);
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename desc, typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::operator=(
		  const detail::Function<desc, Functor_, Args...> &function) -> ArrayContainer & {
			return assign(function);
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename TransposeType>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::operator=(
		  const Transpose<TransposeType> &transpose) -> ArrayContainer & {
			m_shape = transpose.shape();
			m_size	= transpose.size();
			m_storage.resize(m_shape.size(), 0);
			transpose.applyTo(*this);
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename ShapeTypeA, typename StorageTypeA, typename ShapeTypeB,
				 typename StorageTypeB, typename Alpha, typename Beta>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::operator=(
		  const linalg::ArrayMultiply<ShapeTypeA, StorageTypeA, ShapeTypeB, StorageTypeB, Alpha,
									  Beta> &arrayMultiply) -> ArrayContainer & {
			m_shape = arrayMultiply.shape();
			m_size	= arrayMultiply.size();
			m_storage.resize(m_shape.size(), 0);
			arrayMultiply.applyTo(*this);
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator=(const Scalar &value)
		  -> ArrayContainer & {
			LIBRAPID_ASSERT(m_shape.ndim() == 0, "Cannot assign a scalar to an array");
			m_storage[0] = value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator<<(const T &value)
		  -> detail::CommaInitializer<ArrayContainer> {
			return detail::CommaInitializer<ArrayContainer>(*this, static_cast<Scalar>(value));
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::copy() const
		  -> ArrayContainer {
			ArrayContainer res(m_shape);
			res.m_storage = m_storage.copy();
			return res;
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator[](int64_t index) const {
			LIBRAPID_ASSERT(
			  index >= 0 && index < static_cast<int64_t>(m_shape[0]),
			  "Index {} out of bounds in ArrayContainer::operator[] with leading dimension={}",
			  index,
			  m_shape[0]);

			return createGeneralArrayView(*this)[index];

			//			if constexpr (typetraits::IsOpenCLStorage<StorageType_>::value) {
			// #if defined(LIBRAPID_HAS_OPENCL)
			//				ArrayContainer res;
			//				res.m_shape			= m_shape.subshape(1, ndim());
			//				auto subSize		= res.shape().size();
			//				int64_t storageSize = sizeof(typename StorageType_::Scalar);
			//				cl_buffer_region region {index * subSize * storageSize, subSize *
			// storageSize}; 				res.m_storage =
			// StorageType_(m_storage.data().createSubBuffer(
			// StorageType_::bufferFlags,
			// CL_BUFFER_CREATE_TYPE_REGION, &region), 							   subSize,
			// false); 				return res; #else 				LIBRAPID_ERROR("OpenCL support
			// not enabled"); #endif // LIBRAPID_HAS_OPENCL 			} else if constexpr
			//(typetraits::IsCudaStorage<StorageType_>::value) { #if defined(LIBRAPID_HAS_CUDA)
			//				ArrayContainer res;
			//				res.m_shape	  = m_shape.subshape(1, ndim());
			//				auto subSize  = res.shape().size();
			//				Scalar *begin = m_storage.begin().get() + index * subSize;
			//				res.m_storage = StorageType_(begin, subSize, false);
			//				return res;
			// #else
			//				LIBRAPID_ERROR("CUDA support not enabled");
			// #endif // LIBRAPID_HAS_CUDA
			//			} else if constexpr (typetraits::IsFixedStorage<StorageType_>::value) {
			//				return GeneralArrayView(*this)[index];
			//			} else {
			//				ArrayContainer res;
			//				res.m_shape	  = m_shape.subshape(1, ndim());
			//				auto subSize  = res.shape().size();
			//				Scalar *begin = m_storage.begin() + index * subSize;
			//				Scalar *end	  = begin + subSize;
			//				res.m_storage = StorageType_(begin, end, false);
			//				return res;
			//			}
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator[](int64_t index) {
			LIBRAPID_ASSERT(
			  index >= 0 && index < static_cast<int64_t>(m_shape[0]),
			  "Index {} out of bounds in ArrayContainer::operator[] with leading dimension={}",
			  index,
			  m_shape[0]);

			return createGeneralArrayView(*this)[index];

			//			if constexpr (typetraits::IsOpenCLStorage<StorageType_>::value) {
			// #if defined(LIBRAPID_HAS_OPENCL)
			//				ArrayContainer res;
			//				res.m_shape			= m_shape.subshape(1, ndim());
			//				auto subSize		= res.shape().size();
			//				int64_t storageSize = sizeof(typename StorageType_::Scalar);
			//				cl_buffer_region region {index * subSize * storageSize, subSize *
			// storageSize}; 				res.m_storage =
			// StorageType_(m_storage.data().createSubBuffer(
			// StorageType_::bufferFlags,
			// CL_BUFFER_CREATE_TYPE_REGION, &region), 							   subSize,
			// false); 				return res; #else 				LIBRAPID_ERROR("OpenCL support
			// not enabled"); #endif // LIBRAPID_HAS_OPENCL 			} else if constexpr
			//(typetraits::IsCudaStorage<StorageType_>::value) { #if defined(LIBRAPID_HAS_CUDA)
			//				ArrayContainer res;
			//				res.m_shape	  = m_shape.subshape(1, ndim());
			//				auto subSize  = res.shape().size();
			//				Scalar *begin = m_storage.begin().get() + index * subSize;
			//				res.m_storage = StorageType_(begin, subSize, false);
			//				return res;
			// #else
			//				LIBRAPID_ERROR("CUDA support not enabled");
			// #endif // LIBRAPID_HAS_CUDA
			//			} else if constexpr (typetraits::IsFixedStorage<StorageType_>::value) {
			//				return GeneralArrayView(*this)[index];
			//			} else {
			//				ArrayContainer res;
			//				res.m_shape	  = m_shape.subshape(1, ndim());
			//				auto subSize  = res.shape().size();
			//				Scalar *begin = m_storage.begin() + index * subSize;
			//				Scalar *end	  = begin + subSize;
			//				res.m_storage = StorageType_(begin, end, false);
			//				return res;
			//			}
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename... Indices>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator()(Indices... indices) const
		  -> DirectSubscriptType {
			LIBRAPID_ASSERT(
			  m_shape.ndim() == sizeof...(Indices),
			  "ArrayContainer::operator() called with {} indices, but array has {} dimensions",
			  sizeof...(Indices),
			  m_shape.ndim());

			int64_t index = 0;
			for (int64_t i : {indices...}) {
				LIBRAPID_ASSERT(
				  i >= 0 && i < static_cast<int64_t>(m_shape[index]),
				  "Index {} out of bounds in ArrayContainer::operator() with dimension={}",
				  i,
				  m_shape[index]);
				index = index * m_shape[index] + i;
			}
			return m_storage[index];
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename... Indices>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator()(Indices... indices)
		  -> DirectRefSubscriptType {
			LIBRAPID_ASSERT(
			  m_shape.ndim() == sizeof...(Indices),
			  "ArrayContainer::operator() called with {} indices, but array has {} dimensions",
			  sizeof...(Indices),
			  m_shape.ndim());

			int64_t index = 0;
			int64_t count = 0;
			for (int64_t i : {indices...}) {
				LIBRAPID_ASSERT(
				  i >= 0 && i < static_cast<int64_t>(m_shape[count]),
				  "Index {} out of bounds in ArrayContainer::operator() with dimension={}",
				  i,
				  m_shape[index]);
				index = index * m_shape[count++] + i;
			}
			return m_storage[index];
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::get() const
		  -> Scalar {
			LIBRAPID_ASSERT(m_shape.ndim() == 0,
							"Can only cast a scalar ArrayView to a salar object");
			return scalar(0);
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::ndim() const noexcept
		  -> typename ShapeType_::SizeType {
			return m_shape.ndim();
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::size() const noexcept
		  -> size_t {
			return m_size;
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::shape() const noexcept
		  -> const ShapeType & {
			return m_shape;
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::storage() const noexcept -> const StorageType & {
			return m_storage;
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::storage() noexcept
		  -> StorageType & {
			return m_storage;
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::packet(size_t index) const -> Packet {
			auto ptr = LIBRAPID_ASSUME_ALIGNED(m_storage.begin());

#if defined(LIBRAPID_NATIVE_ARCH)
			LIBRAPID_ASSERT(
			  reinterpret_cast<uintptr_t>(ptr) % typetraits::TypeInfo<Scalar>::packetWidth == 0,
			  "ArrayContainer::packet called on unaligned storage");

			return xsimd::load_aligned(ptr + index);
#else
			return xsimd::load_unaligned(ptr + index);
#endif
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::scalar(size_t index) const -> Scalar {
			return m_storage[index];
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE void
		ArrayContainer<ShapeType_, StorageType_>::writePacket(size_t index, const Packet &value) {
			auto ptr = LIBRAPID_ASSUME_ALIGNED(m_storage.begin());

#if defined(LIBRAPID_NATIVE_ARCH)
			LIBRAPID_ASSERT(
			  reinterpret_cast<uintptr_t>(ptr) % typetraits::TypeInfo<Scalar>::packetWidth == 0,
			  "ArrayContainer::packet called on unaligned storage");
			value.store_aligned(ptr + index);
#else
			value.store_unaligned(ptr + index);
#endif
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE void
		ArrayContainer<ShapeType_, StorageType_>::write(size_t index, const Scalar &value) {
			m_storage[index] = value;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator+=(const T &value) -> ArrayContainer & {
			*this = *this + value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator-=(const T &value) -> ArrayContainer & {
			*this = *this - value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator*=(const T &value) -> ArrayContainer & {
			*this = *this * value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator/=(const T &value) -> ArrayContainer & {
			*this = *this / value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator%=(const T &value) -> ArrayContainer & {
			*this = *this % value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator&=(const T &value) -> ArrayContainer & {
			*this = *this & value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator|=(const T &value) -> ArrayContainer & {
			*this = *this | value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator^=(const T &value) -> ArrayContainer & {
			*this = *this ^ value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator<<=(const T &value) -> ArrayContainer & {
			*this = *this << value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T>
		LIBRAPID_ALWAYS_INLINE auto
		ArrayContainer<ShapeType_, StorageType_>::operator>>=(const T &value) -> ArrayContainer & {
			*this = *this >> value;
			return *this;
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::begin() const noexcept
		  -> auto {
			return detail::ArrayIterator(createGeneralArrayView(*this), 0);
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::end() const noexcept
		  -> auto {
			return detail::ArrayIterator(createGeneralArrayView(*this), m_shape[0]);
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::begin() -> auto {
			return detail::ArrayIterator(createGeneralArrayView(*this), 0);
		}

		template<typename ShapeType_, typename StorageType_>
		LIBRAPID_ALWAYS_INLINE auto ArrayContainer<ShapeType_, StorageType_>::end() -> auto {
			return detail::ArrayIterator(createGeneralArrayView(*this), m_shape[0]);
		}

		template<typename ShapeType_, typename StorageType_>
		template<typename T, typename Char, typename Ctx>
		LIBRAPID_ALWAYS_INLINE void ArrayContainer<ShapeType_, StorageType_>::str(
		  const fmt::formatter<T, Char> &format, char bracket, char separator, Ctx &ctx) const {
			createGeneralArrayView(*this).str(format, bracket, separator, ctx);
		}
	} // namespace array

	namespace detail {
		template<typename T>
		struct IsArrayType {
			static constexpr bool val = false;
		};

		template<typename T, typename V>
		struct IsArrayType<ArrayRef<T, V>> {
			static constexpr bool val = true;
		};

		template<typename... T>
		struct IsArrayType<FunctionRef<T...>> {
			static constexpr bool val = true;
		};

		template<typename T, typename S>
		struct IsArrayType<array::GeneralArrayView<T, S>> {
			static constexpr bool val = true;
		};

		template<typename First, typename... Types>
		struct ContainsArrayType {
			static constexpr auto evaluator() {
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
ARRAY_TYPE_FMT_IML(typename ShapeType_ COMMA typename StorageType_,
				   librapid::array::ArrayContainer<ShapeType_ COMMA StorageType_>)

LIBRAPID_SIMPLE_IO_NORANGE(typename ShapeType_ COMMA typename StorageType_,
						   librapid::array::ArrayContainer<ShapeType_ COMMA StorageType_>)

#endif // LIBRAPID_ARRAY_ARRAY_CONTAINER_HPP