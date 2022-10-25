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

	template<typename ShapeType_, typename StorageType_>
	class ArrayContainer {
	public:
		using StorageType = StorageType_;
		using ShapeType	  = ShapeType_;
		using SizeType	  = typename ShapeType::SizeType;
		using Scalar	  = typename StorageType::Scalar;
		using Packet	  = typename typetraits::TypeInfo<Scalar>::Packet;

		ArrayContainer() = default;
		explicit ArrayContainer(const ShapeType &shape);
		ArrayContainer(const ShapeType &shape, const Scalar &value);
		explicit ArrayContainer(ShapeType &&shape);

		ArrayContainer(const ArrayContainer &other)		= default;
		ArrayContainer(ArrayContainer &&other) noexcept = default;

		template<typename Functor_, typename... Args>
		explicit ArrayContainer(const detail::Function<Functor_, Args...> &function)
		  LIBRAPID_RELEASE_NOEXCEPT;

		ArrayContainer &operator=(const ArrayContainer &other)	   = default;
		ArrayContainer &operator=(ArrayContainer &&other) noexcept = default;

		template<typename Functor_, typename... Args>
		LIBRAPID_ALWAYS_INLINE ArrayContainer &
		operator=(const detail::Function<Functor_, Args...> &function);

		LIBRAPID_NODISCARD LIBRAPID_INLINE const ShapeType &shape() const noexcept;

		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Packet packet(size_t index) const;
		LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Scalar scalar(size_t index) const;
		LIBRAPID_ALWAYS_INLINE void writePacket(size_t index, const Packet &value);
		LIBRAPID_ALWAYS_INLINE void write(size_t index, const Scalar &value);

	public:
		ShapeType m_shape;
		StorageType m_storage;
	};

	template<typename ShapeType_, typename StorageType_>
	ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const ShapeType &shape) :
			m_shape(shape), m_storage(shape.size()) {}

	template<typename ShapeType_, typename StorageType_>
	ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(const ShapeType &shape,
															 const Scalar &value) :
			m_shape(shape),
			m_storage(shape.size(), value) {}

	template<typename ShapeType_, typename StorageType_>
	ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(ShapeType &&shape) :
			m_shape(std::move(shape)), m_storage(m_shape.size()) {}

	template<typename ShapeType_, typename StorageType_>
	template<typename Functor_, typename... Args>
	ArrayContainer<ShapeType_, StorageType_>::ArrayContainer(
	  const detail::Function<Functor_, Args...> &function) LIBRAPID_RELEASE_NOEXCEPT
			: m_shape(function.shape()),
			  m_storage(m_shape.size()) {
		detail::assign(*this, function);
	}

	template<typename ShapeType_, typename StorageType_>
	template<typename Functor_, typename... Args>
	ArrayContainer<ShapeType_, StorageType_> &ArrayContainer<ShapeType_, StorageType_>::operator=(
	  const detail::Function<Functor_, Args...> &function) {
		m_storage.resize(function.shape().size(), 0);
		detail::assign(*this, function);
		return *this;
	}

	template<typename ShapeType_, typename StorageType_>
	auto ArrayContainer<ShapeType_, StorageType_>::shape() const noexcept -> const ShapeType & {
		return m_shape;
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
} // namespace librapid

#endif // LIBRAPID_ARRAY_ARRAY_CONTAINER_HPP