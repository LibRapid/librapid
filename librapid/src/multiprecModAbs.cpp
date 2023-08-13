#include <librapid/librapid.hpp>

#if defined(LIBRAPID_USE_MULTIPREC)

namespace librapid {
    mpfr abs(const mpfr &val) { return ::mpfr::abs(val); }

    mpf abs(const mpf &val) {
        if (val >= 0)
            return val;
        else
            return -val;
    }

    mpz abs(const mpz &val) {
        if (val >= 0)
            return val;
        else
            return -val;
    }

    mpq abs(const mpq &val) {
        if (val >= 0)
            return val;
        else
            return -val;
    }

    mpfr mod(const mpfr &val, const mpfr &mod) { return ::mpfr::fmod(val, mod); }
} // namespace librapid

#endif // LIBRAPID_USE_MULTIPREC
