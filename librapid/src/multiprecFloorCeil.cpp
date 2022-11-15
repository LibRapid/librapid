#include <librapid/librapid.hpp>

#if defined(LIBRAPID_USE_MULTIPREC)

namespace librapid {
	mpfr floor(const mpfr &val) { return ::mpfr::floor(val); }
	mpfr ceil(const mpfr &val) { return ::mpfr::ceil(val); }
}

#endif // LIBRAPID_USE_MULTIPREC
