#ifndef LIBRAPID_MATH_ROUND_HPP
#define LIBRAPID_MATH_ROUND_HPP

namespace librapid {
    enum class RoundingMode {
        // Rounding Mode Information:
        // Bit mask:
        // [0] -> Round up if difference >= 0.5
        // [1] -> Round up if difference < 0.5
        // [2] -> Round to nearest even
        // [3] -> Round to nearest odd
        // [4] -> Round only if difference == 0.5

        UP        = 0b00000011,
        DOWN      = 0b00000000,
        TRUNC     = 0b00000000,
        HALF_EVEN = 0b00010100,
        MATH      = 0b00000001,
    };

    template<typename T = double>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto round(T num, int64_t dp = 0,
                                                         RoundingMode mode = RoundingMode::MATH) {
        int8_t mode_        = static_cast<int8_t>(mode);
        const double alpha  = fastmath::pow10(dp);
        const double beta   = fastmath::pow10(-dp);
        const double absNum = ::librapid::abs(static_cast<double>(num) * alpha);
        double y            = ::librapid::floor(absNum);
        double diff         = absNum - y;
        if (mode_ & (1 << 0) && diff >= 0.5) y += 1;
        if (mode_ & (1 << 2)) {
            auto integer     = (uint64_t)y;
            auto nearestEven = (integer & 1) ? (y + 1) : (double)integer;
            if (mode_ & (1 << 4) && diff == 0.5) y = nearestEven;
        }

        return static_cast<T>(::librapid::copySign(y * beta, num));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto round(const Complex<T> &num, int64_t dp = 0,
                                                         RoundingMode mode = RoundingMode::MATH) {
        return Complex<T>(round(real(num), dp, mode), round(imag(num), dp, mode));
    }

#if defined(LIBRAPID_USE_MULTIPREC)
    template<>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE auto round(const mpfr &num, int64_t dp,
                                                         RoundingMode mode) {
        using Scalar        = mpfr;
        int8_t mode_        = static_cast<int8_t>(mode);
        const Scalar alpha  = ::librapid::exp10(mpfr(dp));
        const Scalar beta   = ::librapid::exp10(mpfr(-dp));
        const Scalar absNum = ::librapid::abs(num * alpha);
        Scalar y            = ::librapid::floor(absNum);
        Scalar diff         = absNum - y;
        if (mode_ & (1 << 0) && diff >= 0.5) y += 1;
        if (mode_ & (1 << 2)) {
            auto integer     = (uint64_t)y;
            auto nearestEven = (integer & 1) ? (y + 1) : (Scalar)integer;
            if (mode_ & (1 << 4) && diff == 0.5) y = nearestEven;
        }
        return (num >= 0 ? y : -y) * beta;
    }
#endif

    template<typename T1 = double, typename T2 = double>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T2 roundTo(T1 num, T2 val) {
        if (num == static_cast<T1>(0)) return 0;
        T2 rem = ::librapid::mod(::librapid::abs(static_cast<T2>(num)), val);
        if (rem >= val / static_cast<T2>(2))
            return ::librapid::copySign((::librapid::abs(static_cast<T2>(num)) + val) - rem, num);
        return ::librapid::copySign(static_cast<T2>(num) - rem, num);
    }

    template<typename T1, typename T2>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T2> roundTo(const Complex<T1> &num, T2 val) {
        return Complex<T2>(roundTo(real(num), val), roundTo(imag(num), val));
    }

    template<typename T1 = double, typename T2 = double>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T2> roundTo(const Complex<T1> &num,
                                                                  const Complex<T2> &val) {
        return Complex<T2>(roundTo(real(num), real(val)), roundTo(imag(num), imag(val)));
    }

    template<typename T1 = double, typename T2 = double>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T2 roundUpTo(T1 num, T2 val) {
        T2 rem = ::librapid::mod(T2(num), val);
        if (rem == T2(0)) return static_cast<T2>(num);
        return (static_cast<T2>(num) + val) - rem;
    }

    template<typename T1 = double, typename T2 = double>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T1> roundUpTo(const Complex<T1> &num,
                                                                    T2 val) {
        return Complex<T1>(roundUpTo(real(num), val), roundUpTo(imag(num), val));
    }

    template<typename T1 = double, typename T2 = double>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T1> roundUpTo(const Complex<T1> &num,
                                                                    const Complex<T2> &val) {
        return Complex<T1>(roundUpTo(real(num), real(val)), roundUpTo(imag(num), imag(val)));
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE T roundSigFig(T num, int64_t figs = 3) {
        LIBRAPID_ASSERT(figs > 0,
                        "Cannot round to {} significant figures. Value must be greater than zero",
                        figs);

        using Scalar = std::conditional_t<std::is_floating_point_v<T>, double, T>;

        if (num == static_cast<T>(0)) return static_cast<T>(0);

        auto tmp  = ::librapid::abs(static_cast<Scalar>(num));
        int64_t n = 0;

        const auto ten = static_cast<Scalar>(10);
        const auto one = static_cast<Scalar>(1);
        while (tmp > ten) {
            tmp /= ten;
            ++n;
        }

        while (tmp < one) {
            tmp *= ten;
            --n;
        }

        return ::librapid::copySign(static_cast<T>(round(tmp, figs - 1) * fastmath::pow10(n)), num);
    }

    template<typename T>
    LIBRAPID_NODISCARD LIBRAPID_ALWAYS_INLINE Complex<T> roundSigFig(const Complex<T> &num,
                                                                     int64_t figs = 3) {
        return Complex<T>(roundSigFig(real(num), figs), roundSigFig(imag(num), figs));
    }
} // namespace librapid

#endif // LIBRAPID_MATH_ROUND_HPP