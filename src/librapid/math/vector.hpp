#pragma once

namespace librapid {
	template<typename Scalar, i64 Dims>
	using Vec = VecImpl<Scalar, Dims, Vc::SimdArray<Scalar, Dims>>;

	template<typename Scalar, i64 Dims, typename StorageType>
	class VecImpl {
	public:
		VecImpl() = default;

		explicit VecImpl(const StorageType &arr) : m_data {arr} {}

		template<typename T, typename ABI>
		explicit VecImpl(const Vc::Vector<T, ABI> &arr) : m_data {arr} {}

		template<typename... Args, typename std::enable_if_t<sizeof...(Args) == Dims, int> = 0>
		VecImpl(Args... args) : m_data {static_cast<Scalar>(args)...} {
			static_assert(sizeof...(Args) <= Dims, "Invalid number of arguments");
		}

		template<typename... Args, i64 size = sizeof...(Args),
				 typename std::enable_if_t<size != Dims, int> = 0>
		VecImpl(Args... args) {
			static_assert(sizeof...(Args) <= Dims, "Invalid number of arguments");
			const Scalar expanded[] = {static_cast<Scalar>(args)...};
			for (i64 i = 0; i < size; i++) { m_data[i] = expanded[i]; }
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

		LR_NODISCARD("") auto operator[](i64 index) const {
			LR_ASSERT(0 <= index < Dims,
					  "Index {} out of range for vector with {} dimensions",
					  index,
					  Dims);
			return m_data[index];
		}

		LR_NODISCARD("") auto operator[](i64 index) {
			LR_ASSERT(0 <= index < Dims,
					  "Index {} out of range for vector with {} dimensions",
					  index,
					  Dims);
			return m_data[index];
		}

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
		VecImpl cmp(const VecImpl &other, const char *mode) const {
			// Mode:
			// 0: ==
			// 1: !=
			// 2: <
			// 3: <=
			// 4: >
			// 5: >=

			VecImpl res(*this);
			i16 modeInt = *(i16 *)mode;
			fmt::print("Info: {:Lb}\n", modeInt);
			fmt::print("Info: {:Lb}\n", ('g' << 8) | 't');
			for (i64 i = 0; i < Dims; ++i) {
				switch (modeInt) {
					case 'e' | ('q' << 8):
						if (res[i] == other[i]) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					case 'n' | ('e' << 8):
						if (res[i] != other[i]) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					case 'l' | ('t' << 8):
						if (res[i] < other[i]) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					case 'l' | ('e' << 8):
						if (res[i] <= other[i]) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					case 'g' | ('t' << 8):
						if (res[i] > other[i]) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					case 'g' | ('e' << 8):
						if (res[i] >= other[i]) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					default: LR_ASSERT(false, "Invalid mode {}", mode);
				}
			}
			return res;
		}

		LR_FORCE_INLINE
		VecImpl cmp(const Scalar &value, const char *mode) const {
			// Mode:
			// 0: ==
			// 1: !=
			// 2: <
			// 3: <=
			// 4: >
			// 5: >=

			VecImpl res(*this);
			i16 modeInt = *(i16 *)mode;
			fmt::print("Info: {:Lb}\n", modeInt);
			fmt::print("Info: {:Lb}\n", ('g' << 8) | 't');
			for (i64 i = 0; i < Dims; ++i) {
				switch (modeInt) {
					case 'e' | ('q' << 8):
						if (res[i] == value) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					case 'n' | ('e' << 8):
						if (res[i] != value) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					case 'l' | ('t' << 8):
						if (res[i] < value) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					case 'l' | ('e' << 8):
						if (res[i] <= value) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					case 'g' | ('t' << 8):
						if (res[i] > value) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					case 'g' | ('e' << 8):
						if (res[i] >= value) {
							res[i] = 1;
						} else {
							res[i] = 0;
						}
						break;
					default: LR_ASSERT(false, "Invalid mode {}", mode);
				}
			}
			return res;
		}

		LR_FORCE_INLINE VecImpl operator<(const VecImpl &other) const { return cmp(other, "lt"); }
		LR_FORCE_INLINE VecImpl operator<=(const VecImpl &other) const { return cmp(other, "le"); }
		LR_FORCE_INLINE VecImpl operator>(const VecImpl &other) const { return cmp(other, "gt"); }
		LR_FORCE_INLINE VecImpl operator>=(const VecImpl &other) const { return cmp(other, "ge"); }
		LR_FORCE_INLINE VecImpl operator==(const VecImpl &other) const { return cmp(other, "eq"); }
		LR_FORCE_INLINE VecImpl operator!=(const VecImpl &other) const { return cmp(other, "ne"); }

		LR_FORCE_INLINE VecImpl operator<(const Scalar &other) const { return cmp(other, "lt"); }
		LR_FORCE_INLINE VecImpl operator<=(const Scalar &other) const { return cmp(other, "le"); }
		LR_FORCE_INLINE VecImpl operator>(const Scalar &other) const { return cmp(other, "gt"); }
		LR_FORCE_INLINE VecImpl operator>=(const Scalar &other) const { return cmp(other, "ge"); }
		LR_FORCE_INLINE VecImpl operator==(const Scalar &other) const { return cmp(other, "eq"); }
		LR_FORCE_INLINE VecImpl operator!=(const Scalar &other) const { return cmp(other, "ne"); }

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
			static_assert(Dims == 3, "Cross product is only defined for 3D Vectors");
			return VecImpl(y() * other.z() - z() * other.y(),
						   z() * other.x() - x() * other.z(),
						   x() * other.y() - y() * other.x());
		}

		LR_NODISCARD("") LR_FORCE_INLINE explicit operator bool() const {
			for (i64 i = 0; i < Dims; ++i)
				if (m_data[i] != 0) return true;
			return false;
		}

		LR_FORCE_INLINE Vec<Scalar, 2> xy() const { return {x(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 2> yx() const { return {y(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 2> xz() const { return {x(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 2> zx() const { return {z(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 2> yz() const { return {y(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 2> zy() const { return {z(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> xyz() const { return {x(), y(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> xzy() const { return {x(), z(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> yxz() const { return {y(), x(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> yzx() const { return {y(), z(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> zxy() const { return {z(), x(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> zyx() const { return {z(), y(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> xyw() const { return {x(), y(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> xwy() const { return {x(), w(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> yxw() const { return {y(), x(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> ywx() const { return {y(), w(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> wxy() const { return {w(), x(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> wyx() const { return {w(), y(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> xzw() const { return {x(), z(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> xwz() const { return {x(), w(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> zxw() const { return {z(), x(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> zwx() const { return {z(), w(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> wxz() const { return {w(), x(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> wzx() const { return {w(), z(), x()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> yzw() const { return {y(), z(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> ywz() const { return {y(), w(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> zyw() const { return {z(), y(), w()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> zwy() const { return {z(), w(), y()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> wyz() const { return {w(), y(), z()}; }
		LR_FORCE_INLINE Vec<Scalar, 3> wzy() const { return {w(), z(), y()}; }
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

		LR_FORCE_INLINE void x(Scalar val) {
			if constexpr (Dims >= 1) m_data[0] = val;
		}

		LR_FORCE_INLINE Scalar y() const {
			if constexpr (Dims < 2)
				return 0;
			else
				return m_data[1];
		}

		LR_FORCE_INLINE void y(Scalar val) {
			if constexpr (Dims >= 2) m_data[1] = val;
		}

		LR_FORCE_INLINE Scalar z() const {
			if constexpr (Dims < 3)
				return 0;
			else
				return m_data[2];
		}

		LR_FORCE_INLINE void z(Scalar val) {
			if constexpr (Dims >= 3) m_data[2] = val;
		}

		LR_FORCE_INLINE Scalar w() const {
			if constexpr (Dims < 4)
				return 0;
			else
				return m_data[3];
		}

		LR_FORCE_INLINE void w(Scalar val) {
			if constexpr (Dims >= 4) m_data[3] = val;
		}

#if defined(LIBRAPID_PYTHON)
		LR_FORCE_INLINE Scalar getX() const { return x(); }
		LR_FORCE_INLINE Scalar getY() const { return y(); }
		LR_FORCE_INLINE Scalar getZ() const { return z(); }
		LR_FORCE_INLINE Scalar getW() const { return w(); }
		LR_FORCE_INLINE void setX(Scalar val) { x(val); }
		LR_FORCE_INLINE void setY(Scalar val) { y(val); }
		LR_FORCE_INLINE void setZ(Scalar val) { z(val); }
		LR_FORCE_INLINE void setW(Scalar val) { w(val); }
#endif // LIBRAPID_PYTHON

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

	template<typename Scalar, i64 Dims, typename StorageType1, typename StorageType2>
	LR_FORCE_INLINE VecImpl<Scalar, Dims, StorageType1>
	operator&(const VecImpl<Scalar, Dims, StorageType1> &vec,
			  const VecImpl<Scalar, Dims, StorageType2> &mask) {
		VecImpl<Scalar, Dims, StorageType1> res(vec);
		for (i64 i = 0; i < Dims; ++i) {
			if (!mask[i]) { res[i] = 0; }
		}
		return res;
	}

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

	using VecMask2i = VecImpl<i32, 2, Vc::SimdMaskArray<i32, 2>>;
	using VecMask3i = VecImpl<i32, 3, Vc::SimdMaskArray<i32, 3>>;
	using VecMask4i = VecImpl<i32, 4, Vc::SimdMaskArray<i32, 4>>;
	using VecMask2f = VecImpl<f32, 2, Vc::SimdMaskArray<f32, 2>>;
	using VecMask3f = VecImpl<f32, 3, Vc::SimdMaskArray<f32, 3>>;
	using VecMask4f = VecImpl<f32, 4, Vc::SimdMaskArray<f32, 4>>;
	using VecMask2d = VecImpl<f64, 2, Vc::SimdMaskArray<f64, 2>>;
	using VecMask3d = VecImpl<f64, 3, Vc::SimdMaskArray<f64, 3>>;
	using VecMask4d = VecImpl<f64, 4, Vc::SimdMaskArray<f64, 4>>;

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
