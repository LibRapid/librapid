#pragma once

// TODO: Optimise this for GPU accesses
#define IMPL_BINOP(NAME, ASSIGN, OP)                                                               \
	template<typename Other>                                                                       \
	auto NAME(const Other &other) const {                                                          \
		return get() OP((T)other);                                                                 \
	}                                                                                              \
                                                                                                   \
	template<typename Other>                                                                       \
	void ASSIGN(const Other &other) {                                                              \
		set(get() OP(T) other);                                                                    \
	}

#define IMPL_BINOP_EXTERN(NAME, ASSIGN, OP)                                                        \
	template<typename Other,                                                                       \
			 typename T,                                                                           \
			 typename d,                                                                           \
			 typename std::enable_if_t<!is_same_v<Other, ValueReference<T, d>>, int> = 0>          \
	LR_INLINE auto NAME(const Other &other, const ValueReference<T, d> &val) {                     \
		return other OP((T)val);                                                                   \
	}                                                                                              \
                                                                                                   \
	template<typename Other,                                                                       \
			 typename T,                                                                           \
			 typename d,                                                                           \
			 typename std::enable_if_t<!is_same_v<Other, ValueReference<T, d>>, int> = 0>          \
	void ASSIGN(Other &other, const ValueReference<T, d> &val) {                                   \
		other = other OP((T)val);                                                                  \
	}

// TODO: Optimise this for GPU accesses
#define IMPL_BINOP2(NAME, OP)                                                                      \
	template<typename Other>                                                                       \
	auto NAME(const Other &other) const {                                                          \
		return get() OP((T)other);                                                                 \
	}

#define IMPL_BINOP2_EXTERN(NAME, OP)                                                               \
	template<typename Other,                                                                       \
			 typename T,                                                                           \
			 typename d,                                                                           \
			 typename std::enable_if_t<!is_same_v<Other, ValueReference<T, d>>, int> = 0>          \
	LR_INLINE auto NAME(const Other &other, const ValueReference<T, d> &val) {                     \
		return other OP((T)val);                                                                   \
	}

#define IMPL_UNOP(NAME, OP)                                                                        \
	template<typename Other>                                                                       \
	auto NAME() const {                                                                            \
		return OP get();                                                                           \
	}

namespace librapid {
	namespace internal {
		template<typename T, typename d>
		struct traits<memory::ValueReference<T, d>> {
			using Scalar = T;
			using Device = d;
			using Packet = typename traits<Scalar>::Packet;
		};
	} // namespace internal

	namespace memory {
		template<typename T, typename d>
		class ValueReference {
		public:
			ValueReference() = delete;

			explicit ValueReference(T *val) : m_value(val) {}

			explicit ValueReference(T &val) : m_value(&val) {
				static_assert(std::is_same<d, device::CPU>::value,
							  "Cannot construct Device ValueReference from Host scalar");
			}

			ValueReference(const ValueReference<T, d> &other) : m_value(other.m_value) {}

			ValueReference &operator=(const ValueReference<T, d> &other) {
				if (&other == this) return *this;
				m_value = other.m_value;
				return *this;
			}

			template<typename Type, typename Device>
			ValueReference &operator=(const ValueReference<Type, Device> &other) {
				if constexpr (std::is_same<d, device::CPU>::value)
					*m_value = *other.get_();
				else
					memcpy<T, d, Type, Device>(m_value, other.get_(), 1);
				return *this;
			}

			ValueReference &operator=(const T &val) {
				if constexpr (std::is_same<d, device::CPU>::value) {
					*m_value = val;
				} else {
					T tmp = val;
					memcpy<T, d, T, device::CPU>(m_value, &tmp, 1);
				}
				return *this;
			}

			template<typename Type>
			LR_NODISCARD("")
			LR_INLINE operator Type() const {
				if constexpr (std::is_same<d, device::CPU>::value) {
					return internal::traits<T>::template cast<Type>(get());
				} else {
					T res;
					memcpy<T, device::CPU, T, d>(&res, m_value, 1);
					return internal::traits<T>::template cast<Type>(res);
				}
			}

			LR_NODISCARD("") T get() const {
				if constexpr (std::is_same_v<d, device::CPU>)
					return *m_value;
				else {
					T res;
					memcpy<T, device::CPU, T, d>(&res, m_value, 1);
					return res;
				}
			}

			void set(T value) { operator=(value); }

			IMPL_BINOP2(operator==, ==);
			IMPL_BINOP2(operator!=, !=);
			IMPL_BINOP2(operator>, >);
			IMPL_BINOP2(operator>=, >=);
			IMPL_BINOP2(operator<, <);
			IMPL_BINOP2(operator<=, <=);

			IMPL_BINOP(operator+, operator+=, +);
			IMPL_BINOP(operator-, operator-=, -);
			IMPL_BINOP(operator*, operator*=, *);
			IMPL_BINOP(operator/, operator/=, /);

			IMPL_BINOP(operator|, operator|=, |);
			IMPL_BINOP(operator&, operator&=, &);
			IMPL_BINOP(operator^, operator^=, ^);

			LR_NODISCARD("") T *get_() const { return m_value; }

		protected:
			T *m_value = nullptr;
		};

		IMPL_BINOP2_EXTERN(operator==, ==)
		IMPL_BINOP2_EXTERN(operator!=, !=)
		IMPL_BINOP2_EXTERN(operator>, >)
		IMPL_BINOP2_EXTERN(operator>=, >=)
		IMPL_BINOP2_EXTERN(operator<, <)
		IMPL_BINOP2_EXTERN(operator<=, <=)

		IMPL_BINOP_EXTERN(operator+, operator+=, +)
		IMPL_BINOP_EXTERN(operator-, operator-=, -)
		IMPL_BINOP_EXTERN(operator*, operator*=, *)
		IMPL_BINOP_EXTERN(operator/, operator/=, /)

		IMPL_BINOP_EXTERN(operator|, operator|=, |)
		IMPL_BINOP_EXTERN(operator&, operator&=, &)
		IMPL_BINOP_EXTERN(operator^, operator^=, ^)
	} // namespace memory
} // namespace librapid

#ifdef FMT_API
template<typename T, typename d>
struct fmt::formatter<librapid::memory::ValueReference<T, d>> {
	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		return ctx.begin();
	}

	template<typename FormatContext>
	auto format(const librapid::memory::ValueReference<T, d> &val, FormatContext &ctx) {
		return fmt::format_to(ctx.out(), fmt::format("{}", val.get()));
	}
};
#endif // FMT_API

#undef IMPL_BINOP
#undef IMPL_BINOP2
#undef IMPL_BINOP_EXTERN
#undef IMPL_BINOP2_EXTERN
#undef IMPL_UNOP
