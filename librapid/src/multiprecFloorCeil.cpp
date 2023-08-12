#include <librapid/librapid.hpp>

#if defined(LIBRAPID_USE_MULTIPREC)

namespace librapid {
    mpfr floor(const mpfr &val) { return ::mpfr::floor(val); }
    mpfr ceil(const mpfr &val) { return ::mpfr::ceil(val); }
} // namespace librapid

#endif // LIBRAPID_USE_MULTIPREC
