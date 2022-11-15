#include <librapid/librapid.hpp>

#if defined(LIBRAPID_USE_MULTIPREC)

namespace librapid {
	mpfr abs(const mpfr &val) { return ::mpfr::abs(val); }
	mpfr mod(const mpfr &val, const mpfr &mod) { return ::mpfr::fmod(val, mod); }
} // namespace librapid

#endif // LIBRAPID_USE_MULTIPREC
