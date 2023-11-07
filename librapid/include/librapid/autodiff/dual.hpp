#ifndef LIBRAPID_AUTODIFF_DUAL
#define LIBRAPID_AUTODIFF_DUAL

#if defined(LIBRAPID_IN_JITIFY)
#    define IS_SCALAR(TYPE) false
#else
#    define IS_SCALAR(TYPE) std::is_scalar_v<TYPE>
#endif // LIBRAPID_IN_JITIFY

#if !defined(LIBRAPID_IN_JITIFY)
namespace librapid {
#endif

    template<typename T>
    class Dual {
    public:
#if defined(LIBRAPID_IN_JITIFY)
        using Scalar                          = T;
        using Packet                          = T;
        static constexpr uint64_t packetWidth = 1;
#else
    using Scalar                          = typename typetraits::TypeInfo<T>::Scalar;
    using Packet                          = typename typetraits::TypeInfo<T>::Packet;
    static constexpr uint64_t packetWidth = typetraits::TypeInfo<T>::packetWidth;
#endif

        Scalar value;
        Scalar derivative;

        Dual() = default;
        explicit Dual(Scalar value) : value(value), derivative(Scalar()) {}
        Dual(Scalar value, Scalar derivative) : value(value), derivative(derivative) {}

        template<typename U>
        explicit Dual(const Dual<U> &other) : value(other.value), derivative(other.derivative) {}

        template<typename U>
        explicit Dual(Dual<U> &&other) :
                value(std::move(other.value)), derivative(std::move(other.derivative)) {}

        template<typename U>
        auto operator=(const Dual<U> &other) -> Dual & {
            value      = other.value;
            derivative = other.derivative;
            return *this;
        }

        template<typename U>
        auto operator=(Dual<U> &&other) -> Dual & {
            value      = std::move(other.value);
            derivative = std::move(other.derivative);
            return *this;
        }

        static constexpr auto size() -> size_t { return typetraits::TypeInfo<Dual>::packetWidth; }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator+=(const Dual &other) -> Dual & {
            value += other.value;
            derivative += other.derivative;
            return *this;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator-=(const Dual &other) -> Dual & {
            value -= other.value;
            derivative -= other.derivative;
            return *this;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator*=(const Dual &other) -> Dual & {
            value *= other.value;
            derivative = derivative * other.value + value * other.derivative;
            return *this;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator/=(const Dual &other) -> Dual & {
            value /= other.value;
            derivative =
              (derivative * other.value - value * other.derivative) / (other.value * other.value);
            return *this;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator+=(const T &other) -> Dual & {
            value += other;
            return *this;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator-=(const T &other) -> Dual & {
            value -= other;
            return *this;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator*=(const T &other) -> Dual & {
            value *= other;
            derivative *= other;
            return *this;
        }

        LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto operator/=(const T &other) -> Dual & {
            value /= other;
            derivative /= other;
            return *this;
        }

#if !defined(LIBRAPID_IN_JITIFY)
        template<typename T_, typename Char, typename Ctx>
        void str(const fmt::formatter<T_, Char> &format, Ctx &ctx) const {
            fmt::format_to(ctx.out(), "Dual(");
            format.format(value, ctx);
            fmt::format_to(ctx.out(), ", ");
            format.format(derivative, ctx);
            fmt::format_to(ctx.out(), ")");
        }
#endif // !defined(LIBRAPID_IN_JITIFY)
    };

    template<typename T, typename V>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() + V())>
    operator+(const Dual<T> &lhs, const Dual<V> &rhs) {
        return {lhs.value + rhs.value, lhs.derivative + rhs.derivative};
    }

    template<typename T, typename V> requires(IS_SCALAR(V))
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() + V())>
    operator+(const Dual<T> &lhs, const V &rhs) {
        return {lhs.value + rhs, lhs.derivative};
    }

    template<typename T, typename V> requires(IS_SCALAR(V))
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() + V())>
    operator+(const V &lhs, const Dual<T> &rhs) {
        return {lhs + rhs.value, rhs.derivative};
    }

    template<typename T, typename V>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() - V())>
    operator-(const Dual<T> &lhs, const Dual<V> &rhs) {
        return {lhs.value - rhs.value, lhs.derivative - rhs.derivative};
    }

    template<typename T, typename V> requires(IS_SCALAR(V))
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() - V())>
    operator-(const Dual<T> &lhs, const V &rhs) {
        return {lhs.value - rhs, lhs.derivative};
    }

    template<typename T, typename V> requires(IS_SCALAR(V))
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() - V())>
    operator-(const V &lhs, const Dual<T> &rhs) {
        return {lhs - rhs.value, -rhs.derivative};
    }

    template<typename T, typename V>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() * V())>
    operator*(const Dual<T> &lhs, const Dual<V> &rhs) {
        return {lhs.value * rhs.value, lhs.derivative * rhs.value + lhs.value * rhs.derivative};
    }

    template<typename T, typename V> requires(IS_SCALAR(V))
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() * V())>
    operator*(const Dual<T> &lhs, const V &rhs) {
        return {lhs.value * rhs, lhs.derivative * rhs};
    }

    template<typename T, typename V> requires(IS_SCALAR(V))
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() * V())>
    operator*(const V &lhs, const Dual<T> &rhs) {
        return {lhs * rhs.value, lhs * rhs.derivative};
    }

    template<typename T, typename V>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() / V())>
    operator/(const Dual<T> &lhs, const Dual<V> &rhs) {
        return {lhs.value / rhs.value,
                (lhs.derivative * rhs.value - lhs.value * rhs.derivative) /
                  (rhs.value * rhs.value)};
    }

    template<typename T, typename V> requires(IS_SCALAR(V))
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() / V())>
    operator/(const Dual<T> &lhs, const V &rhs) {
        return {lhs.value / rhs, lhs.derivative / rhs};
    }

    template<typename T, typename V> requires(IS_SCALAR(V))
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T() / V())>
    operator/(const V &lhs, const Dual<T> &rhs) {
        return {lhs / rhs.value, -lhs * rhs.derivative / (rhs.value * rhs.value)};
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(-T())> operator-(const Dual<T> &lhs) {
        return {-lhs.value, -lhs.derivative};
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<decltype(T())> operator+(const Dual<T> &lhs) {
        return {lhs.value, lhs.derivative};
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> sin(const Dual<T> &x) {
        using Ret = decltype(::librapid::sin(x.value));
        return Dual<Ret>(::librapid::sin(x.value), ::librapid::cos(x.value) * x.derivative);
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> cos(const Dual<T> &x) {
        using Ret = decltype(::librapid::cos(x.value));
        return Dual<Ret>(::librapid::cos(x.value), -::librapid::sin(x.value) * x.derivative);
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> tan(const Dual<T> &x) {
        using Ret = decltype(::librapid::tan(x.value));
        auto cosX = ::librapid::cos(x.value);
        return Dual<Ret>(::librapid::tan(x.value), x.derivative / (cosX * cosX));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> asin(const Dual<T> &x) {
        using Ret = decltype(::librapid::asin(x.value));
        return Dual<Ret>(::librapid::asin(x.value),
                         x.derivative / ::librapid::sqrt(1 - x.value * x.value));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> acos(const Dual<T> &x) {
        using Ret = decltype(::librapid::acos(x.value));
        return Dual<Ret>(::librapid::acos(x.value),
                         -x.derivative / ::librapid::sqrt(1 - x.value * x.value));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> atan(const Dual<T> &x) {
        using Ret = decltype(::librapid::atan(x.value));
        return Dual<Ret>(::librapid::atan(x.value), x.derivative / (1 + x.value * x.value));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> sinh(const Dual<T> &x) {
        using Ret = decltype(::librapid::sinh(x.value));
        return Dual<Ret>(::librapid::sinh(x.value), ::librapid::cosh(x.value) * x.derivative);
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> cosh(const Dual<T> &x) {
        using Ret = decltype(::librapid::cosh(x.value));
        return Dual<Ret>(::librapid::cosh(x.value), ::librapid::sinh(x.value) * x.derivative);
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> tanh(const Dual<T> &x) {
        using Ret  = decltype(::librapid::tanh(x.value));
        auto coshX = ::librapid::cosh(x.value);
        return Dual<Ret>(::librapid::tanh(x.value), x.derivative / (coshX * coshX));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> asinh(const Dual<T> &x) {
        using Ret = decltype(::librapid::asinh(x.value));
        return Dual<Ret>(::librapid::asinh(x.value),
                         x.derivative / ::librapid::sqrt(1 + x.value * x.value));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> acosh(const Dual<T> &x) {
        using Ret = decltype(::librapid::acosh(x.value));
        return Dual<Ret>(::librapid::acosh(x.value),
                         x.derivative / ::librapid::sqrt(x.value * x.value - 1));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> atanh(const Dual<T> &x) {
        using Ret = decltype(::librapid::atanh(x.value));
        return Dual<Ret>(::librapid::atanh(x.value), x.derivative / (1 - x.value * x.value));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> exp(const Dual<T> &x) {
        using Ret = decltype(::librapid::exp(x.value));
        auto expX = ::librapid::exp(x.value);
        return Dual<Ret>(expX, expX * x.derivative);
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> exp2(const Dual<T> &x) {
        using Ret  = decltype(::librapid::exp2(x.value));
        auto exp2X = ::librapid::exp2(x.value);
        return Dual<Ret>(exp2X, exp2X * ::librapid::log(2) * x.derivative);
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> exp10(const Dual<T> &x) {
        using Ret  = decltype(::librapid::exp2(x.value));
        auto exp2X = ::librapid::exp10(x.value);
        return Dual<Ret>(exp2X, exp2X * ::librapid::log(10) * x.derivative);
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> log(const Dual<T> &x) {
        using Ret = decltype(::librapid::log(x.value));
        return Dual<Ret>(::librapid::log(x.value), x.derivative / x.value);
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> log10(const Dual<T> &x) {
        using Ret = decltype(::librapid::log10(x.value));
        return Dual<Ret>(::librapid::log10(x.value),
                         x.derivative / (x.value * ::librapid::log(10)));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> log2(const Dual<T> &x) {
        using Ret = decltype(::librapid::log2(x.value));
        return Dual<Ret>(::librapid::log2(x.value), x.derivative / (x.value * ::librapid::log(2)));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> sqrt(const Dual<T> &x) {
        using Ret = decltype(::librapid::sqrt(x.value));
        return Dual<Ret>(::librapid::sqrt(x.value), x.derivative / (2 * ::librapid::sqrt(x.value)));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> cbrt(const Dual<T> &x) {
        using Ret = decltype(::librapid::cbrt(x.value));
        return Dual<Ret>(::librapid::cbrt(x.value),
                         x.derivative / (3 * ::librapid::cbrt(x.value * x.value)));
    }

    template<typename T, typename V> requires(IS_SCALAR(V))
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> pow(const Dual<T> &x, const V &y) {
        using Ret = decltype(::librapid::pow(x.value, y));
        return Dual<Ret>(::librapid::pow(x.value, y),
                         y * ::librapid::pow(x.value, y - 1) * x.derivative);
    }

    template<typename T, typename V> requires(IS_SCALAR(V))
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> pow(const V &x, const Dual<T> &y) {
        using Ret = decltype(::librapid::pow(x, y.value));
        return Dual<Ret>(::librapid::pow(x, y.value),
                         ::librapid::log(x) * ::librapid::pow(x, y.value) * y.derivative);
    }

    template<typename T, typename V>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Dual<T> pow(const Dual<T> &x, const Dual<V> &y) {
        using Ret = decltype(::librapid::pow(x.value, y.value));
        return Dual<Ret>(
          ::librapid::pow(x.value, y.value),
          ::librapid::pow(x.value, y.value) *
            (y.derivative * ::librapid::log(x.value) + y.value * x.derivative / x.value));
    }

#if !defined(LIBRAPID_IN_JITIFY)
    namespace typetraits {
        template<typename T>
        struct TypeInfo<Dual<T>> {
            static constexpr detail::LibRapidType type = detail::LibRapidType::Dual;
            using Scalar                               = T;
            using Packet = std::false_type; // Dual<typename TypeInfo<T>::Packet>;
            static constexpr int64_t packetWidth =
              0; // TypeInfo<typename TypeInfo<T>::Scalar>::packetWidth;
            using Backend = backend::CPU;

            static constexpr char name[] = "Dual_T";

            static constexpr bool supportsArithmetic = TypeInfo<T>::supportsArithmetic;
            static constexpr bool supportsLogical    = TypeInfo<T>::supportsLogical;
            static constexpr bool supportsBinary     = TypeInfo<T>::supportsBinary;
            static constexpr bool allowVectorisation = false; // TypeInfo<T>::allowVectorisation;

#    if defined(LIBRAPID_HAS_CUDA)
            static constexpr cudaDataType_t CudaType = TypeInfo<T>::CudaType;
            static constexpr int64_t cudaPacketWidth = 1;
#    endif // LIBRAPID_HAS_CUDA

            static constexpr bool canAlign     = TypeInfo<T>::canAlign;
            static constexpr int64_t canMemcpy = TypeInfo<T>::canMemcpy;

            LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
            LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
            LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
            LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
            LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
            LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
            LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
            LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
        };

        template<>
        struct TypeInfo<Dual<float>> {
            static constexpr detail::LibRapidType type = detail::LibRapidType::Dual;
            using Scalar                               = float;
            using Packet = std::false_type; // Dual<typename TypeInfo<float>::Packet>;
            static constexpr int64_t packetWidth =
              0; // TypeInfo<typename TypeInfo<float>::Scalar>::packetWidth;
            using Backend = backend::CPU;

            static constexpr char name[] = "Dual_float";

            static constexpr bool supportsArithmetic = TypeInfo<float>::supportsArithmetic;
            static constexpr bool supportsLogical    = TypeInfo<float>::supportsLogical;
            static constexpr bool supportsBinary     = TypeInfo<float>::supportsBinary;
            static constexpr bool allowVectorisation =
              false; // TypeInfo<float>::allowVectorisation;

#    if defined(LIBRAPID_HAS_CUDA)
            static constexpr cudaDataType_t CudaType = TypeInfo<float>::CudaType;
            static constexpr int64_t cudaPacketWidth = 1;
#    endif // LIBRAPID_HAS_CUDA

            static constexpr bool canAlign     = TypeInfo<float>::canAlign;
            static constexpr int64_t canMemcpy = TypeInfo<float>::canMemcpy;

            LIMIT_IMPL_CONSTEXPR(min) { return NUM_LIM(min); }
            LIMIT_IMPL_CONSTEXPR(max) { return NUM_LIM(max); }
            LIMIT_IMPL_CONSTEXPR(epsilon) { return NUM_LIM(epsilon); }
            LIMIT_IMPL_CONSTEXPR(roundError) { return NUM_LIM(round_error); }
            LIMIT_IMPL_CONSTEXPR(denormMin) { return NUM_LIM(denorm_min); }
            LIMIT_IMPL_CONSTEXPR(infinity) { return NUM_LIM(infinity); }
            LIMIT_IMPL_CONSTEXPR(quietNaN) { return NUM_LIM(quiet_NaN); }
            LIMIT_IMPL_CONSTEXPR(signalingNaN) { return NUM_LIM(signaling_NaN); }
        };
    }  // namespace typetraits
#endif // !LIBRAPID_IN_JITIFY

#if !defined(LIBRAPID_IN_JITIFY)
} // namespace librapid
#endif

#if defined(LIBRAPID_HAS_CUDA)
namespace jitify::reflection::detail {
    template<typename T>
    struct type_reflection<::librapid::Dual<T>> {
        inline static std::string name() {
            return fmt::format("Dual<{}>", type_reflection<T>::name());
        }
    };
} // namespace jitify::reflection::detail
#endif // LIBRAPID_HAS_CUDA

#ifdef FMT_API

template<typename T, typename Char>
struct fmt::formatter<librapid::Dual<T>, Char> {
private:
    using Type   = librapid::Dual<T>;
    using Scalar = typename Type::Scalar;
    using Base   = fmt::formatter<Scalar, Char>;
    Base m_base;

public:
    template<typename ParseContext>
    FMT_CONSTEXPR auto parse(ParseContext &ctx) -> const char * {
        return m_base.parse(ctx);
    }

    template<typename FormatContext>
    FMT_CONSTEXPR auto format(const Type &val, FormatContext &ctx) const -> decltype(ctx.out()) {
        val.str(m_base, ctx);
        return ctx.out();
    }
};

#endif // FMT_API

#undef IS_SCALAR

#endif // LIBRAPID_AUTODIFF_DUAL