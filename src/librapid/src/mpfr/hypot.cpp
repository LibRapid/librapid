#include <librapid>

namespace librapid {
	mpfr hypot(const mpfr &a, const mpfr &b) { return ::mpfr::hypot(a, b); }
} // namespace librapid
