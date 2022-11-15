#include <librapid/librapid.hpp>

# if defined(LIBRAPID_USE_MULTIPREC)

namespace librapid {
	mpfr hypot(const mpfr &a, const mpfr &b) { return ::mpfr::hypot(a, b); }
} // namespace librapid

#endif // LIBRAPID_USE_MULTIPREC
