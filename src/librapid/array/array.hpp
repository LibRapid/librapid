#pragma once

#include "../internal/config.hpp"
#include "helpers/extent.hpp"
#include "arrayBase.hpp"
#include "cwisebinop.hpp"
#include "cwisemap.hpp"
#include "denseStorage.hpp"
#include "commaInitializer.hpp"

namespace librapid {
	namespace internal {
		template<typename Scalar_, typename Device_>
		struct traits<Array<Scalar_, Device_>> {
			static constexpr bool IsScalar	  = false;
			static constexpr bool IsEvaluated = true;
			using Valid						  = std::true_type;
			using Scalar					  = Scalar_;
			using BaseScalar				  = typename traits<Scalar>::BaseScalar;
			using Device					  = Device_;
			using Packet					  = typename traits<Scalar>::Packet;
			using StorageType				  = memory::DenseStorage<Scalar, Device>;
			static constexpr int64_t Flags	  = flags::Evaluated | flags::PythonFlags;
		};
	} // namespace internal

	template<typename Scalar_, typename Device_ = device::CPU>
	class Array : public ArrayBase<Array<Scalar_, Device_>, Device_> {
	public:
#if !defined(LIBRAPID_HAS_CUDA)
		static_assert(is_same_v<Device_, device::CPU>, "CUDA support was not enabled");
#endif

		using Scalar					= Scalar_;
		using Device					= Device_;
		using Packet					= typename internal::traits<Scalar>::Packet;
		using Type						= Array<Scalar, Device>;
		using Base						= ArrayBase<Type, Device>;
		using StorageType				= typename internal::traits<Type>::StorageType;
		static constexpr uint64_t Flags = internal::traits<Base>::Flags;

		Array() = default;

		explicit Array(const Extent &extent) : Base(extent) {}

		Array(const Array &other) : Base(other) {}

		template<typename OtherDerived,
				 typename std::enable_if_t<!internal::traits<OtherDerived>::IsScalar, int> = 0>
		Array(const OtherDerived &other) : Base(other.extent()) {
			Base::assign(other);
		}

		Array(const Scalar &other) : Base(other) {}

		Array &operator=(const Array &other) { return Base::assign(other); }

		Array &operator=(const Scalar &other) { return Base::assign(other); }

		template<typename OtherDerived,
				 typename std::enable_if_t<!internal::traits<OtherDerived>::IsScalar, int> = 0>
		Array &operator=(const OtherDerived &other) {
			using ScalarOther = typename internal::traits<OtherDerived>::Scalar;
			static_assert(is_same_v<Scalar, ScalarOther>,
						  "Cannot assign Arrays with different types. Please use Array::cast<T>()");

			return Base::assign(other);
		}

		template<typename T>
		internal::CommaInitializer<Type> operator<<(const T &value) {
			return internal::CommaInitializer<Type>(*this, value);
		}

		Array copy() const {
			Array res(Base::extent());
			memory::memcpy(res.storage(), Base::storage());
			return res;
		}

		LR_NODISCARD("") Array<Scalar, Device> operator[](int64_t index) const {
			LR_ASSERT(!Base::isScalar(), "Cannot subscript a scalar value");

			int64_t memIndex = this->m_isScalar ? 0 : Base::extent().indexAdjusted(index);
			Array<Scalar, Device> res;
			res.m_extent   = Base::extent().partial(1);
			res.m_isScalar = Base::extent().dims() == 1;
			res.m_storage  = Base::storage();
			res.m_storage.offsetMemory(memIndex);

			return res;
		}

		LR_NODISCARD("") Array<Scalar, Device> operator[](int64_t index) {
			return const_cast<const Type *>(this)->operator[](index);
		}

		template<typename... T>
		LR_NODISCARD("")
		auto operator()(T... indices) const {
			LR_ASSERT((this->m_isScalar && sizeof...(T) == 1) ||
						sizeof...(T) == Base::extent().dims(),
					  "Array with {0} dimensions requires {0} access indices. Received {1}",
					  Base::extent().dims(),
					  sizeof...(indices));

			int64_t index = Base::isScalar() ? 0 : Base::extent().index(indices...);
			return Base::storage()[index];
		}

		template<typename... T>
		LR_NODISCARD("")
		auto operator()(T... indices) {
			return const_cast<const Type *>(this)->operator()(indices...);
		}

		void transpose(const Extent &order = {}) { *this = Base::transposed(order); }

		LR_FORCE_INLINE void writePacket(int64_t index, const Packet &p) {
			LR_ASSERT(index >= 0 && index < Base::extent().sizeAdjusted(),
					  "Index {} is out of range",
					  index);
			p.store(Base::storage().heap() + index);
		}

		LR_FORCE_INLINE void writeScalar(int64_t index, const Scalar &s) {
			Base::storage()[index] = s;
		}

		template<typename T>
		LR_FORCE_INLINE operator T() const {
			LR_ASSERT(Base::isScalar(), "Cannot cast non-scalar Array to scalar value");
			return operator()(0);
		}

		template<typename Other, bool forceTemporary = false>
		LR_NODISCARD("")
		auto filled(const Other &other) const {
			using BaseScalar = typename internal::traits<Scalar>::BaseScalar;
			using RetType	 = unop::CWiseUnop<functors::misc::FillArray<Scalar>, Type>;
			static constexpr uint64_t Flags	   = internal::traits<Scalar>::Flags;
			static constexpr uint64_t Required = RetType::Flags & internal::flags::OperationMask;

			static_assert(!(Required & ~(Flags & Required)),
						  "Scalar type is incompatible with Functor");

			BaseScalar tmp;
			if constexpr (std::is_same_v<Scalar, bool>) {
				// Booleans are mapped to an unsigned integer, so set all bits to 1 or 0
				tmp = other ? static_cast<BaseScalar>(-1) : static_cast<BaseScalar>(0);
			} else {
				tmp = other;
			}

			if constexpr (!forceTemporary && // If we REQUIRE a temporary value, don't evaluate it
						  ((bool)(Flags & internal::flags::RequireEval)))
				return RetType(*this, tmp).eval();
			else
				return RetType(*this, tmp);
		}

		template<typename Other>
		void fill(const Other &other) {
			Base::assign(filled<Other, true>(other));
		}

		void findLongest(const std::string &format, bool strip, int64_t stripWidth,
						 int64_t &longestInteger, int64_t &longestFloating) const {
			int64_t dims	= Base::extent().dims();
			int64_t zeroDim = Base::extent()[0];
			if (dims > 1) {
				for (int64_t i = 0; i < zeroDim; ++i) {
					if (stripWidth != 0 && strip && i == stripWidth && zeroDim > stripWidth * 2)
						i = zeroDim - stripWidth;

					this->operator[](i).findLongest(
					  format, strip, stripWidth, longestInteger, longestFloating);
				}
			} else {
				// Stringify vector
				for (int64_t i = 0; i < zeroDim; ++i) {
					if (stripWidth != 0 && strip && i == stripWidth && zeroDim > stripWidth * 2)
						i = zeroDim - stripWidth;

					auto val			  = this->operator()(i).get();
					std::string formatted = fmt::format(format, val);
					auto findIter		  = std::find(formatted.begin(), formatted.end(), '.');
					int64_t pointPos	  = findIter - formatted.begin();
					if (findIter == formatted.end()) {
						// No decimal point present
						if (formatted.length() > longestInteger)
							longestInteger = formatted.length();
					} else {
						// Decimal point present
						auto integer  = formatted.substr(0, pointPos);
						auto floating = formatted.substr(pointPos);
						if (integer.length() > longestInteger) longestInteger = integer.length();
						if (floating.length() - 1 > longestFloating)
							longestFloating = floating.length() - 1;
					}
				}
			}
		}

		// Strip modes:
		// stripWidth == -1 => Default
		// stripWidth == 0  => Never strip
		// stripWidth >= 1  => n values are shown
		LR_NODISCARD("")
		std::string str(std::string format = "", const std::string &delim = " ",
						int64_t stripWidth = -1, int64_t beforePoint = -1, int64_t afterPoint = -1,
						int64_t depth = 0) const {
			bool strip = stripWidth > 0;
			if (depth == 0) {
				strip = false;
				// Configure the strip width

				// Always print the full vector if the array has all dimensions as 1 except a single
				// axis, unless specified otherwise
				int64_t nonOneDims = 0;
				for (int64_t i = 0; i < Base::extent().dims(); ++i)
					if (Base::extent()[i] != 1) ++nonOneDims;

				if (nonOneDims == 1 && stripWidth == -1) stripWidth = 0;

				if (stripWidth == -1) {
					if (Base::extent().size() >= 1000) {
						// Strip the middle values
						strip	   = true;
						stripWidth = 3;
					}
				} else if (stripWidth > 0) {
					strip = true;
				}

				if (format.empty()) {
					if constexpr (std::is_floating_point_v<Scalar>)
						format = "{:.6f}";
					else
						format = "{}";
				}

				// Scalars
				if (Base::isScalar()) return fmt::format(format, this->operator()(0));

				int64_t tmpBeforePoint = 0, tmpAfterPoint = 0;
				findLongest(format, strip, stripWidth, tmpBeforePoint, tmpAfterPoint);
				if (beforePoint == -1) beforePoint = tmpBeforePoint;
				if (afterPoint == -1) afterPoint = tmpAfterPoint;
			}

			std::string res = "[";
			int64_t dims	= Base::extent().dims();
			int64_t zeroDim = Base::extent()[0];
			if (dims > 1) {
				for (int64_t i = 0; i < zeroDim; ++i) {
					if (stripWidth != 0 && strip && i == stripWidth && zeroDim > stripWidth * 2) {
						i = zeroDim - stripWidth;
						res += std::string(depth + 1, ' ') + "...\n";
						if (dims > 2) res += "\n";
					}

					if (i > 0) res += std::string(depth + 1, ' ');
					res += this->operator[](i).str(
					  format, delim, stripWidth, beforePoint, afterPoint, depth + 1);
					if (i + 1 < zeroDim) res += "\n";
					if (i + 1 < zeroDim && dims > 2) res += "\n";
				}
			} else {
				// Stringify vector
				for (int64_t i = 0; i < zeroDim; ++i) {
					if (stripWidth != 0 && strip && i == stripWidth && zeroDim > stripWidth * 2) {
						i = zeroDim - stripWidth;
						res += "... ";
					}

					auto val			  = this->operator()(i).get();
					std::string formatted = fmt::format(format, val);
					auto findIter		  = std::find(formatted.begin(), formatted.end(), '.');
					int64_t pointPos	  = findIter - formatted.begin();
					if (afterPoint == 0) {
						// No decimal point present
						res += fmt::format("{:>{}}", formatted, beforePoint);
					} else {
						// Decimal point present
						auto integer  = formatted.substr(0, pointPos);
						auto floating = formatted.substr(pointPos);
						// Add a space to account for the missing decimal point in some
						// cases
						res += fmt::format("{:>{}}{:<{}}",
										   integer,
										   beforePoint,
										   floating,
										   afterPoint + (findIter == formatted.end()));
					}
					if (i + 1 < zeroDim) res += delim;
				}
			}
			return res + "]";
		}
	};

#define FORCE_TMP_FUNC_BINOP(NAME, FUNC)                                                           \
	template<typename T, typename D>                                                               \
	LR_FORCE_INLINE void NAME(const Array<T, D> &lhs, const Array<T, D> &rhs, Array<T, D> &dst) {  \
		dst.assign(lhs.template operator FUNC<true, Array<T, D>>(rhs));                            \
	}                                                                                              \
                                                                                                   \
	template<typename T, typename D>                                                               \
	LR_FORCE_INLINE void NAME(const Array<T, D> &lhs, const T &rhs, Array<T, D> &dst) {            \
		dst.assign(lhs.template operator FUNC<true, Array<T, D>>(rhs));                            \
	}                                                                                              \
                                                                                                   \
	template<typename T, typename D>                                                               \
	LR_FORCE_INLINE void NAME(const T &lhs, const Array<T, D> &rhs, Array<T, D> &dst) {            \
		dst.assign(lhs FUNC rhs);                                                                  \
	}

#define FORCE_TMP_FUNC_BINOP_EXTERNAL(NAME)                                                        \
	template<typename T, typename D>                                                               \
	LR_FORCE_INLINE void NAME(const Array<T, D> &lhs, const Array<T, D> &rhs, Array<T, D> &dst) {  \
		dst.assign(::librapid::NAME<true>(lhs, rhs));                                              \
	}                                                                                              \
                                                                                                   \
	template<typename T, typename D>                                                               \
	LR_FORCE_INLINE void NAME(const Array<T, D> &lhs, const T &rhs, Array<T, D> &dst) {            \
		dst.assign(::librapid::NAME<true>(lhs, rhs));                                              \
	}                                                                                              \
                                                                                                   \
	template<typename T, typename D>                                                               \
	LR_FORCE_INLINE void NAME(const T &lhs, const Array<T, D> &rhs, Array<T, D> &dst) {            \
		dst.assign(::librapid::NAME<true>(lhs, rhs));                                              \
	}

#define FORCE_TMP_FUNC_UNOP(NAME, FUNC)                                                            \
	template<typename T, typename D>                                                               \
	LR_FORCE_INLINE void NAME(const Array<T, D> &lhs, Array<T, D> &dst) {                          \
		dst.assign(lhs.template FUNC<true>());                                                     \
	}

#define FORCE_TMP_FUNC_UNOP_EXTERNAL(NAME)                                                         \
	template<typename T, typename D>                                                               \
	LR_FORCE_INLINE void NAME(const Array<T, D> &lhs, Array<T, D> &dst) {                          \
		dst.assign(::librapid::NAME<true>(lhs));                                                   \
	}

	FORCE_TMP_FUNC_BINOP(add, +)
	FORCE_TMP_FUNC_BINOP(sub, -)
	FORCE_TMP_FUNC_BINOP(mul, *)
	FORCE_TMP_FUNC_BINOP(div, /)

	FORCE_TMP_FUNC_BINOP_EXTERNAL(pow)
	FORCE_TMP_FUNC_BINOP_EXTERNAL(log)

	FORCE_TMP_FUNC_UNOP_EXTERNAL(sin)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(cos)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(tan)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(asin)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(acos)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(atan)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(sinh)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(cosh)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(tanh)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(asinh)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(acosh)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(atanh)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(exp)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(log)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(log10)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(sqrt)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(abs)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(floor)
	FORCE_TMP_FUNC_UNOP_EXTERNAL(ceil)

	FORCE_TMP_FUNC_BINOP(bitwiseOr, |)
	FORCE_TMP_FUNC_BINOP(bitwiseAnd, &)
	FORCE_TMP_FUNC_BINOP(bitwiseXor, ^)

	FORCE_TMP_FUNC_UNOP(negate, operator-)
	FORCE_TMP_FUNC_UNOP(bitwiseNot, operator~)
	FORCE_TMP_FUNC_UNOP(logicalNot, operator!)

	template<typename T, typename D>
	inline std::string str(const Array<T, D> &val, const StrOpt &options = DEFAULT_STR_OPT) {
		return val.str();
	}

	using ArrayB   = Array<bool, device::CPU>;
	using ArrayC   = Array<char, device::CPU>;
	using ArrayF16 = Array<extended::float16_t, device::CPU>;
	using ArrayF32 = Array<float, device::CPU>;
	using ArrayF64 = Array<double, device::CPU>;
	using ArrayI16 = Array<int16_t, device::CPU>;
	using ArrayI32 = Array<int32_t, device::CPU>;
	using ArrayI64 = Array<int64_t, device::CPU>;

#if defined(LIBRAPID_USE_MULTIPREC)
	using ArrayMPZ	= Array<mpz, device::CPU>;
	using ArrayMPQ	= Array<mpq, device::CPU>;
	using ArrayMPFR = Array<mpfr, device::CPU>;
#endif

	using ArrayCF32 = Array<Complex<float>, device::CPU>;
	using ArrayCF64 = Array<Complex<double>, device::CPU>;
#if defined(LIBRAPID_USE_MULTIPREC)
	using ArrayCMPFR = Array<Complex<mpfr>, device::CPU>;
#endif

	// GPU array aliases will default to CPU arrays if CUDA is not enabled
#if defined(LIBRAPID_HAS_CUDA)
	using ArrayBG	= Array<bool, device::GPU>;
	using ArrayCG	= Array<char, device::GPU>;
	using ArrayF16G = Array<extended::float16_t, device::GPU>;
	using ArrayF32G = Array<float, device::GPU>;
	using ArrayF64G = Array<double, device::GPU>;
	using ArrayI16G = Array<int16_t, device::GPU>;
	using ArrayI32G = Array<int32_t, device::GPU>;
	using ArrayI64G = Array<int64_t, device::GPU>;
#else
	using ArrayBG	= Array<bool, device::CPU>;
	using ArrayCG	= Array<char, device::CPU>;
	using ArrayF16G = Array<extended::float16_t, device::CPU>;
	using ArrayF32G = Array<float, device::CPU>;
	using ArrayF64G = Array<double, device::CPU>;
	using ArrayI16G = Array<int16_t, device::CPU>;
	using ArrayI32G = Array<int32_t, device::CPU>;
	using ArrayI64G = Array<int64_t, device::CPU>;
#endif

#undef FORCE_TMP_FUNC
} // namespace librapid

// Provide {fmt} printing capabilities
#ifdef FMT_API
template<typename Scalar, typename Device>
struct fmt::formatter<librapid::Array<Scalar, Device>> {
	std::string formatStr = "{}";

	template<typename ParseContext>
	constexpr auto parse(ParseContext &ctx) {
		formatStr = "{:";
		auto it	  = ctx.begin();
		for (; it != ctx.end(); ++it) {
			if (*it == '}') break;
			formatStr += *it;
		}
		formatStr += "}";
		return it;
	}

	template<typename FormatContext>
	auto format(const librapid::Array<Scalar, Device> &arr, FormatContext &ctx) {
		try {
			return fmt::format_to(ctx.out(), arr.str(formatStr));
		} catch (std::exception &e) { return fmt::format_to(ctx.out(), e.what()); }
	}
};
#endif // FMT_API