#ifndef LIBRAPID_ARRAY_OPENCL_STORAGE_HPP
#define LIBRAPID_ARRAY_OPENCL_STORAGE_HPP

/*
 * This file defines the OpenCLStorage class, which is used to store the data of an Array
 * on an OpenCL device. It implements the same functions as the Storage class, but with
 * OpenCL-specific implementations and potentially different function signatures (often
 * regarding the return type).
 */

#if defined(LIBRAPID_HAS_OPENCL)

#	define LIBRAPID_CHECK_OPENCL                                                                  \
		LIBRAPID_ASSERT(global::openCLConfigured,                                                  \
						"OpenCL has not been configured. Please call configureOpenCL() before "    \
						"creating any Arrays with the OpenCL backend.")

namespace librapid {
	namespace typetraits {
		template<typename Scalar_>
		struct TypeInfo<OpenCLStorage<Scalar_>> {
			static constexpr bool isLibRapidType = true;
			using Scalar						 = Scalar_;
			using Backend						 = backend::OpenCL;
		};

		template<typename T>
		struct IsOpenCLStorage : std::false_type {};

		template<typename Scalar_>
		struct IsOpenCLStorage<OpenCLStorage<Scalar_>> : std::true_type {};

		LIBRAPID_DEFINE_AS_TYPE(typename Scalar_, OpenCLStorage<Scalar_>);
	} // namespace typetraits

	namespace detail {
#	define OPENCL_REF_OPERATOR(OP)                                                                \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const OpenCLRef<LHS> &lhs, const RHS &rhs) {                              \
			return lhs.get() OP rhs;                                                               \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const LHS &lhs, const OpenCLRef<RHS> &rhs) {                              \
			return lhs OP rhs.get();                                                               \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const OpenCLRef<LHS> &lhs, const OpenCLRef<RHS> &rhs) {                   \
			return lhs.get() OP rhs.get();                                                         \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP##=(OpenCLRef<LHS> &lhs, const RHS &rhs) {                                 \
			lhs = lhs.get() OP rhs;                                                                \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP##=(OpenCLRef<LHS> &lhs, const OpenCLRef<RHS> &rhs) {                      \
			lhs = lhs.get() OP rhs.get();                                                          \
		}

#	define OPENCL_REF_OPERATOR_NO_ASSIGN(OP)                                                      \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const OpenCLRef<LHS> &lhs, const RHS &rhs) {                              \
			return lhs.get() OP rhs;                                                               \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const LHS &lhs, const OpenCLRef<RHS> &rhs) {                              \
			return lhs OP rhs.get();                                                               \
		}                                                                                          \
                                                                                                   \
		template<typename LHS, typename RHS>                                                       \
		auto operator OP(const OpenCLRef<LHS> &lhs, const OpenCLRef<RHS> &rhs) {                   \
			return lhs.get() OP rhs.get();                                                         \
		}

		template<typename T>
		class OpenCLRef {
		public:
			OpenCLRef(const cl::Buffer &buffer, size_t offset) :
					m_buffer(buffer), m_offset(offset) {}

			LIBRAPID_ALWAYS_INLINE OpenCLRef &operator=(const T &val) {
				global::openCLQueue.enqueueWriteBuffer(
				  m_buffer, CL_TRUE, m_offset * sizeof(T), sizeof(T), &val);
				return *this;
			}

			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T get() const {
				T tmp;
				global::openCLQueue.enqueueReadBuffer(
				  m_buffer, CL_TRUE, m_offset * sizeof(T), sizeof(T), &tmp);
				global::openCLQueue.finish();
				return tmp;
			}

			template<typename CAST>
			LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE operator CAST() const {
				return static_cast<CAST>(get());
			}

			template<typename T_, typename Char, typename Ctx>
			void str(const fmt::formatter<T_, Char> &format, Ctx &ctx) {
				format.format(get(), ctx);
			}

		private:
			cl::Buffer m_buffer;
			size_t m_offset;
		};

		OPENCL_REF_OPERATOR(+)
		OPENCL_REF_OPERATOR(-)
		OPENCL_REF_OPERATOR(*)
		OPENCL_REF_OPERATOR(/)
		OPENCL_REF_OPERATOR(%)
		OPENCL_REF_OPERATOR(^)
		OPENCL_REF_OPERATOR(&)
		OPENCL_REF_OPERATOR(|)
		OPENCL_REF_OPERATOR(<<)
		OPENCL_REF_OPERATOR(>>)
		OPENCL_REF_OPERATOR_NO_ASSIGN(==)
		OPENCL_REF_OPERATOR_NO_ASSIGN(!=)
		OPENCL_REF_OPERATOR_NO_ASSIGN(<)
		OPENCL_REF_OPERATOR_NO_ASSIGN(>)
		OPENCL_REF_OPERATOR_NO_ASSIGN(<=)
		OPENCL_REF_OPERATOR_NO_ASSIGN(>=)
	} // namespace detail

	template<typename Scalar_>
	class OpenCLStorage {
	public:
		using Scalar						= Scalar_;
		using Pointer						= Scalar *;
		using ConstPointer					= const Scalar *;
		using Reference						= Scalar &;
		using ConstReference				= const Scalar &;
		using SizeType						= size_t;
		using DifferenceType				= ptrdiff_t;
		static constexpr cl_int bufferFlags = CL_MEM_READ_WRITE;

		/// \brief Default constructor
		OpenCLStorage() = default;

		/// \brief Construct an OpenCLStorage with the given size. The data is not initialised.
		/// \param size The size of the OpenCLStorage
		LIBRAPID_ALWAYS_INLINE explicit OpenCLStorage(SizeType size);

		/// \brief Construct an OpenCLStorage with the given size and initialise it with the given
		/// value.
		/// \param size The size of the OpenCLStorage
		/// \param value The value to initialise the OpenCLStorage with
		LIBRAPID_ALWAYS_INLINE OpenCLStorage(SizeType size, Scalar value);

		LIBRAPID_ALWAYS_INLINE OpenCLStorage(const cl::Buffer &buffer, SizeType size,
											 bool ownsData);

		/// \brief Construct an OpenCLStorage from another instance
		/// \param other The other instance
		LIBRAPID_ALWAYS_INLINE OpenCLStorage(const OpenCLStorage &other);

		/// \brief Move-construct an OpenCLStorage from another instance
		/// \param other The other instance
		LIBRAPID_ALWAYS_INLINE OpenCLStorage(OpenCLStorage &&other) noexcept;

		/// \brief Initialize an OpenCLStorage instance from an initializer-list
		/// \param list Values to populate with
		LIBRAPID_ALWAYS_INLINE OpenCLStorage(std::initializer_list<Scalar> list);

		/// \brief Initialize an OpenCLStorage instance from a vector
		/// \param vec Values to populate with
		LIBRAPID_ALWAYS_INLINE explicit OpenCLStorage(const std::vector<Scalar> &vec);

		LIBRAPID_ALWAYS_INLINE OpenCLStorage &operator=(const OpenCLStorage &other);

		LIBRAPID_ALWAYS_INLINE OpenCLStorage &operator=(OpenCLStorage &&other) noexcept;

		void set(const OpenCLStorage &other);

		OpenCLStorage copy() const;

		template<typename ShapeType>
		static ShapeType defaultShape();

		static OpenCLStorage fromData(const std::initializer_list<Scalar> &list);

		static OpenCLStorage fromData(const std::vector<Scalar> &vec);

		~OpenCLStorage();

		/// Resize a CudaStorage object to \p size elements. Existing elements are preserved where
		/// possible.
		/// \param size Number of elements
		/// \see resize(SizeType, int)
		LIBRAPID_ALWAYS_INLINE void resize(SizeType newSize);

		/// Resize a CudaStorage object to \p size elements. Existing elements are not preserved.
		/// This method of resizing is faster and more efficient than the version which preserves
		/// the original data, but of course, this has the drawback that data will be lost.
		/// \param size Number of elements
		LIBRAPID_ALWAYS_INLINE void resize(SizeType newSize, int value);

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE SizeType size() const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE detail::OpenCLRef<Scalar>
		operator[](SizeType index);

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const detail::OpenCLRef<Scalar>
		operator[](SizeType index) const;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE const cl::Buffer &data() const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE cl::Buffer &data();

	private:
		SizeType m_size;
		cl::Buffer m_buffer;
		bool m_ownsData = true;
	};

	template<typename Scalar>
	OpenCLStorage<Scalar>::OpenCLStorage(SizeType size) :
			m_size(size), m_buffer(global::openCLContext, bufferFlags, size * sizeof(Scalar)),
			m_ownsData(true) {
		LIBRAPID_CHECK_OPENCL;
	}

	template<typename Scalar>
	OpenCLStorage<Scalar>::OpenCLStorage(SizeType size, Scalar value) :
			m_size(size), m_buffer(global::openCLContext, bufferFlags, size * sizeof(Scalar)),
			m_ownsData(true) {
		LIBRAPID_CHECK_OPENCL;
		global::openCLQueue.enqueueFillBuffer(m_buffer, value, 0, size * sizeof(Scalar));
	}

	template<typename Scalar>
	OpenCLStorage<Scalar>::OpenCLStorage(const cl::Buffer &buffer, SizeType size, bool ownsData) :
			m_size(size), m_buffer(buffer), m_ownsData(ownsData) {
		LIBRAPID_CHECK_OPENCL;
	}

	template<typename Scalar>
	OpenCLStorage<Scalar>::OpenCLStorage(const OpenCLStorage &other) :
			m_size(other.m_size), m_ownsData(true) {
		LIBRAPID_CHECK_OPENCL;

		// Copy data
		m_buffer = cl::Buffer(global::openCLContext, bufferFlags, m_size * sizeof(Scalar));
		global::openCLQueue.enqueueCopyBuffer(
		  other.m_buffer, m_buffer, 0, 0, m_size * sizeof(Scalar));
	}

	template<typename Scalar>
	OpenCLStorage<Scalar>::OpenCLStorage(OpenCLStorage &&other) noexcept :
			m_size(std::move(other.m_size)), m_buffer(std::move(other.m_buffer)),
			m_ownsData(std::move(other.m_ownsData)) {
		other.m_size	 = 0;
		other.m_buffer	 = cl::Buffer();
		other.m_ownsData = false;
	}

	template<typename Scalar>
	OpenCLStorage<Scalar>::OpenCLStorage(std::initializer_list<Scalar> list) :
			m_size(list.size()),
			m_buffer(global::openCLContext, bufferFlags, list.size() * sizeof(Scalar)),
			m_ownsData(true) {
		LIBRAPID_CHECK_OPENCL;
		global::openCLQueue.enqueueWriteBuffer(
		  m_buffer, CL_TRUE, 0, m_size * sizeof(Scalar), list.begin());
	}

	template<typename Scalar>
	OpenCLStorage<Scalar>::OpenCLStorage(const std::vector<Scalar> &vec) :
			m_size(vec.size()),
			m_buffer(global::openCLContext, bufferFlags, m_size * sizeof(Scalar)),
			m_ownsData(true) {
		LIBRAPID_CHECK_OPENCL;
		global::openCLQueue.enqueueWriteBuffer(
		  m_buffer, CL_TRUE, 0, m_size * sizeof(Scalar), vec.data());
	}

	template<typename Scalar>
	OpenCLStorage<Scalar> &OpenCLStorage<Scalar>::operator=(const OpenCLStorage &other) {
		LIBRAPID_CHECK_OPENCL;
		if (this != &other) {
			if (other.m_size == 0) return *this; // Quick return

			size_t oldSize = m_size;
			m_size		   = other.m_size;
			if (oldSize != m_size) LIBRAPID_UNLIKELY {
					if (m_ownsData) LIBRAPID_LIKELY {
							m_buffer = cl::Buffer(
							  global::openCLContext, bufferFlags, m_size * sizeof(Scalar));
						}
					else {
						// We do not own the data and therefore cannot resize it.
						LIBRAPID_ASSERT(false, "Cannot resize dependent OpenCLStorage");
					}
				}

			// Copy data
			global::openCLQueue.enqueueCopyBuffer(
			  other.m_buffer, m_buffer, 0, 0, m_size * sizeof(Scalar));
		}
		return *this;
	}

	template<typename Scalar>
	OpenCLStorage<Scalar> &OpenCLStorage<Scalar>::operator=(OpenCLStorage &&other) noexcept {
		LIBRAPID_CHECK_OPENCL;
		if (this != &other) {
			m_size	   = std::move(other.m_size);
			m_buffer   = std::move(other.m_buffer);
			m_ownsData = std::move(other.m_ownsData);

			other.m_size	 = 0;
			other.m_buffer	 = cl::Buffer();
			other.m_ownsData = false;
		}
		return *this;
	}

	template<typename Scalar>
	auto OpenCLStorage<Scalar>::copy() const -> OpenCLStorage {
		LIBRAPID_CHECK_OPENCL;
		OpenCLStorage<Scalar> result(m_size);
		global::openCLQueue.enqueueCopyBuffer(
		  m_buffer, result.m_buffer, 0, 0, m_size * sizeof(Scalar));
		return result;
	}

	template<typename Scalar>
	template<typename ShapeType>
	ShapeType OpenCLStorage<Scalar>::defaultShape() {
		return ShapeType({0});
	}

	template<typename Scalar>
	OpenCLStorage<Scalar>
	OpenCLStorage<Scalar>::fromData(const std::initializer_list<Scalar> &list) {
		return OpenCLStorage<Scalar>(list);
	}

	template<typename Scalar>
	OpenCLStorage<Scalar> OpenCLStorage<Scalar>::fromData(const std::vector<Scalar> &vec) {
		return OpenCLStorage<Scalar>(vec);
	}

	template<typename Scalar>
	OpenCLStorage<Scalar>::~OpenCLStorage() {
		// cl::Buffer is reference counted, so we do not need to worry about whether the array
		// owns the buffer or not. If it does, the buffer will be deleted when the array is
		// destroyed. If it does not, the buffer will be deleted when the last array referencing
		// it is destroyed.
	}

	template<typename Scalar>
	void OpenCLStorage<Scalar>::resize(SizeType newSize) {
		if (newSize == m_size) return;
		LIBRAPID_ASSERT(newSize > 0, "Cannot resize to zero-size array");
		LIBRAPID_ASSERT(m_ownsData, "Cannot resize dependent OpenCLStorage");

		cl::Buffer oldBuffer = m_buffer;
		size_t oldSize		 = m_size;

		m_size	 = newSize;
		m_buffer = cl::Buffer(global::openCLContext, bufferFlags, newSize * sizeof(Scalar));

		// Copy data
		global::openCLQueue.enqueueCopyBuffer(
		  oldBuffer, m_buffer, 0, 0, std::min(m_size, oldSize) * sizeof(Scalar));
	}

	template<typename Scalar>
	void OpenCLStorage<Scalar>::resize(SizeType newSize, int) {
		if (newSize == m_size) return;
		LIBRAPID_ASSERT(newSize > 0, "Cannot resize to zero-size array");
		LIBRAPID_ASSERT(m_ownsData, "Cannot resize dependent OpenCLStorage");
		m_size	 = newSize;
		m_buffer = cl::Buffer(global::openCLContext, bufferFlags, newSize * sizeof(Scalar));
	}

	template<typename Scalar>
	auto OpenCLStorage<Scalar>::size() const -> SizeType {
		return m_size;
	}

	template<typename Scalar>
	auto OpenCLStorage<Scalar>::operator[](SizeType index) const
	  -> const detail::OpenCLRef<Scalar> {
		LIBRAPID_ASSERT(index >= 0 && index < m_size,
						"Index {} is out of range for OpenCLStorage with {} elements",
						index,
						m_size);
		return detail::OpenCLRef<Scalar>(m_buffer, index);
	}

	template<typename Scalar>
	auto OpenCLStorage<Scalar>::operator[](SizeType index) -> detail::OpenCLRef<Scalar> {
		LIBRAPID_ASSERT(index >= 0 && index < m_size,
						"Index {} is out of range for OpenCLStorage with {} elements",
						index,
						m_size);
		return detail::OpenCLRef<Scalar>(m_buffer, index);
	}

	template<typename Scalar>
	auto OpenCLStorage<Scalar>::data() const -> const cl::Buffer & {
		return m_buffer;
	}

	template<typename Scalar>
	auto OpenCLStorage<Scalar>::data() -> cl::Buffer & {
		return m_buffer;
	}
} // namespace librapid

template<typename T, typename Char>
struct fmt::formatter<librapid::detail::OpenCLRef<T>, Char> {
private:
	using Base = fmt::formatter<T, Char>;
	Base m_base;

public:
	template<typename ParseContext>
	FMT_CONSTEXPR auto parse(ParseContext &ctx) -> const char * {
		return m_base.parse(ctx);
	}

	template<typename FormatContext>
	FMT_CONSTEXPR auto format(const librapid::detail::OpenCLRef<T> &val, FormatContext &ctx) const
	  -> decltype(ctx.out()) {
		val.str(m_base, ctx);
		return ctx.out();
	}
};

#endif // LIBRAPID_HAS_OPENCL

#endif // LIBRAPID_ARRAY_OPENCL_STORAGE_HPP