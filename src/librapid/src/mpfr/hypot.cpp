#include <librapid/math/mpfr.hpp>

namespace librapid {
	mpfr hypot(const mpfr &a, const mpfr &b) { return ::mpfr::hypot(a, b); }
} // namespace librapid
