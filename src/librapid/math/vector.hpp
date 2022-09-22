#ifndef LIBRAPID_VECTOR
#define LIBRAPID_VECTOR

namespace librapid {
	template<typename Scalar, i64 Dims>
	using Vec = VecImpl<Scalar, Dims, Vc::SimdArray<Scalar, Dims>>;

	template<typename Scalar, i64 Dims, typename StorageType>
	class VecImpl {
	public:
		using Mask = Vc::Mask<Scalar, struct Vc::simd_abi::fixed_size<Dims>>;

		VecImpl() = default;

		explicit VecImpl(const StorageType &arr) : m_data {arr} {}

		template<typename T, typename ABI>
		explicit VecImpl(const Vc::Vector<T, ABI> &arr) : m_data {arr} {}

		template<typename... Args>
		VecImpl(Args... args) : m_data {static_cast<Scalar>(args)...} {
			static_assert(sizeof...(Args) <= Dims, "Invalid number of arguments");
		}

		VecImpl(const VecImpl &other)				 = default;
		VecImpl(VecImpl &&other) noexcept			 = default;
		VecImpl &operator=(const VecImpl &other)	 = default;
		VecImpl &operator=(VecImpl &&other) noexcept = default;

		// Implement conversion to and from GLM datatypes
#ifdef GLM_VERSION

		template<glm::qualifier p = glm::defaultp>
		VecImpl(const glm::vec<Dims, Scalar, p> &vec) {
			for (i64 i = 0; i < Dims; ++i) { m_data[i] = vec[i]; }
		}

		template<glm::qualifier p = glm::defaultp>
		operator glm::vec<Dims, Scalar, p>() const {
			glm::vec<Dims, Scalar, p> res;
			for (i64 i = 0; i < Dims; ++i) { res[i] = m_data[i]; }
			return res;
		}

#endif // GLM_VERSION

		LR_NODISCARD("") auto operator[](i64 index) const { return m_data[index]; }
		LR_NODISCARD("") auto &operator[](i64 index) { return m_data[index]; }

		LR_FORCE_INLINE void operator+=(const VecImpl &other) { m_data += other.m_data; }
		LR_FORCE_INLINE void operator-=(const VecImpl &other) { m_data -= other.m_data; }
		LR_FORCE_INLINE void operator*=(const VecImpl &other) { m_data *= other.m_data; }
		LR_FORCE_INLINE void operator/=(const VecImpl &other) { m_data /= other.m_data; }

		LR_FORCE_INLINE void operator+=(const Scalar &value) { m_data += value; }
		LR_FORCE_INLINE void operator-=(const Scalar &value) { m_data -= value; }
		LR_FORCE_INLINE void operator*=(const Scalar &value) { m_data *= value; }
		LR_FORCE_INLINE void operator/=(const Scalar &value) { m_data /= value; }

		LR_FORCE_INLINE VecImpl operator-() const {
			VecImpl res(*this);
			res *= -1;
			return res;
		}

		LR_FORCE_INLINE
		VecImpl<Scalar, Dims, Mask> operator>(const VecImpl &other) {
			return VecImpl<Scalar, Dims, Mask>(m_data > other.m_data);
		}

		LR_FORCE_INLINE
		VecImpl<Scalar, Dims, Mask> operator<(const VecImpl &other) {
			return VecImpl<Scalar, Dims, Mask>(m_data < other.m_data);
		}

		LR_FORCE_INLINE
		VecImpl<Scalar, Dims, Mask> operator>=(const VecImpl &other) {
			return VecImpl<Scalar, Dims, Mask>(m_data >= other.m_data);
		}

		LR_FORCE_INLINE
		VecImpl<Scalar, Dims, Mask> operator<=(const VecImpl &other) {
			return VecImpl<Scalar, Dims, Mask>(m_data <= other.m_data);
		}

		LR_FORCE_INLINE
		VecImpl<Scalar, Dims, Mask> operator==(const VecImpl &other) {
			return VecImpl<Scalar, Dims, Mask>(m_data == other.m_data);
		}

		LR_FORCE_INLINE
		VecImpl<Scalar, Dims, Mask> operator!=(const VecImpl &other) { return !(*this == other); }

		LR_NODISCARD("") LR_INLINE Scalar mag2() const { return (m_data * m_data).sum(); }
		LR_NODISCARD("") LR_INLINE Scalar mag() const { return ::librapid::sqrt(mag2()); }
		LR_NODISCARD("") LR_INLINE Scalar invMag() const { return 1.0 / mag(); }

		LR_NODISCARD("") LR_INLINE VecImpl norm() const {
			VecImpl res(*this);
			res /= mag();
			return res;
		}

		LR_NODISCARD("") LR_INLINE Scalar dot(const VecImpl &other) const {
			return (m_data * other.m_data).sum();
		}

		LR_NODISCARD("") LR_INLINE VecImpl cross(const VecImpl &other) const {
			static_assert(Dims == 3, "Cross product is only defined for 3D VecImpltors");
			return VecImpl {m_data[1] * other.m_data[2] - m_data[2] * other.m_data[1],
							m_data[2] * other.m_data[0] - m_data[0] * other.m_data[2],
							m_data[0] * other.m_data[1] - m_data[1] * other.m_data[0]};
		}

		LR_NODISCARD("") LR_FORCE_INLINE explicit operator bool() const {
			for (i64 i = 0; i < Dims; ++i)
				if (m_data[i] != 0) return true;
			return false;
		}

		LR_FORCE_INLINE Vec<Scalar, 2> xy() const { return {x(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 2> yx() const { return {y(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> xyz() const { return {x(), y(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> xzy() const { return {x(), z(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> yxz() const { return {y(), x(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> yzx() const { return {y(), z(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> zxy() const { return {z(), x(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> zyx() const { return {z(), y(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> xyzw() const { return {x(), y(), z(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> xywz() const { return {x(), y(), w(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> xzyw() const { return {x(), z(), y(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> xzwy() const { return {x(), z(), w(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> xwyz() const { return {x(), w(), y(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> xwzy() const { return {x(), w(), z(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> yxzw() const { return {y(), x(), z(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> yxwz() const { return {y(), x(), w(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> yzxw() const { return {y(), z(), x(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> yzwx() const { return {y(), z(), w(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> ywxz() const { return {y(), w(), x(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> ywzx() const { return {y(), w(), z(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> zxyw() const { return {z(), x(), y(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> zxwy() const { return {z(), x(), w(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> zyxw() const { return {z(), y(), x(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> zywx() const { return {z(), y(), w(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> zwxy() const { return {z(), w(), x(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> zwyx() const { return {z(), w(), y(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> wxyz() const { return {w(), x(), y(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> wxzy() const { return {w(), x(), z(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> wyxz() const { return {w(), y(), x(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> wyzx() const { return {w(), y(), z(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> wzxy() const { return {w(), z(), x(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 4> wzyx() const { return {w(), z(), y(), x()}; }

		LR_FORCE_INLINE const auto &data() const { return m_data; }
		LR_FORCE_INLINE auto &data() { return m_data; }

		LR_FORCE_INLINE Scalar x() const {
			if constexpr (Dims < 1)
				return 0;
			else
				return m_data[0];
		}

		LR_FORCE_INLINE Scalar y() const {
			if constexpr (Dims < 2)
				return 0;
			else
				return m_data[1];
		}

		LR_FORCE_INLINE Scalar z() const {
			if constexpr (Dims < 3)
				return 0;
			else
				return m_data[2];
		}

		LR_FORCE_INLINE Scalar w() const {
			if constexpr (Dims < 4)
				return 0;
			else
				return m_data[3];
		}

		LR_NODISCARD("") std::string str() const {
			std::string res = "(";
			for (i64 i = 0; i < Dims; ++i) {
				res += std::to_string(m_data[i]);
				if (i != Dims - 1) { res += ", "; }
			}
			return res + ")";
		}

	protected:
		StorageType m_data {};
	};

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator+(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res += rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator-(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res -= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator*(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res *= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator/(const VecImpl<Scalar, Dims, StorageType> &lhs,
			  const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res /= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType, typename S>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator+(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res += rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType, typename S>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator-(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res -= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType, typename S>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator*(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res *= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType, typename S>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator/(const VecImpl<Scalar, Dims, StorageType> &lhs, const S &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(lhs);
		res /= rhs;
		return res;
	}

	template<typename S, typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator+(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res += rhs;
		return res;
	}

	template<typename S, typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator-(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res -= rhs;
		return res;
	}

	template<typename S, typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator*(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res *= rhs;
		return res;
	}

	template<typename S, typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	operator/(const S &lhs, const VecImpl<Scalar, Dims, StorageType> &rhs) {
		VecImpl<Scalar, Dims, StorageType> res(StorageType(static_cast<Scalar>(lhs)));
		res /= rhs;
		return res;
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE Scalar dist2(const VecImpl<Scalar, Dims, StorageType> &lhs,
								 const VecImpl<Scalar, Dims, StorageType> &rhs) {
		return (lhs - rhs).mag2();
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE Scalar dist(const VecImpl<Scalar, Dims, StorageType> &lhs,
								const VecImpl<Scalar, Dims, StorageType> &rhs) {
		return (lhs - rhs).mag();
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	sin(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::sin(vec.data()));
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	cos(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::cos(vec.data()));
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	tan(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(sin(vec) / cos(vec));
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	asin(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::asin(vec.data()));
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	acos(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(HALFPI - asin(vec));
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	atan(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::atan(vec.data()));
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	exp(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::exp(vec.data()));
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	log(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::log(vec.data()));
	}

	template<typename Scalar, i64 Dims, typename StorageType>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType>
	sqrt(const VecImpl<Scalar, Dims, StorageType> &vec) {
		return VecImpl<Scalar, Dims, StorageType>(Vc::sqrt(vec.data()));
	}

	using Vec2i = Vec<i32, 2>;
	using Vec3i = Vec<i32, 3>;
	using Vec4i = Vec<i32, 4>;
	using Vec2f = Vec<f32, 2>;
	using Vec3f = Vec<f32, 3>;
	using Vec4f = Vec<f32, 4>;
	using Vec2d = Vec<f64, 2>;
	using Vec3d = Vec<f64, 3>;
	using Vec4d = Vec<f64, 4>;
} // namespace librapid
#ifdef FMT_API
template<typename Scalar, librapid::i64 D, typename StorageType>
struct fmt::formatter<librapid::VecImpl<Scalar, D, StorageType>> {
	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		return ctx.begin();
	}

	template<typename FormatContext>
	auto format(const librapid::VecImpl<Scalar, D, StorageType> &arr, FormatContext &ctx) {
		return fmt::format_to(ctx.out(), arr.str());
	}
};
#endif // FMT_API

#endif // LIBRAPID_VECTOR
